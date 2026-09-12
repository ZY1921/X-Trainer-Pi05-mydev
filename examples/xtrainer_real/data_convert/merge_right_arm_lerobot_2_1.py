"""Merge single-task right-arm LeRobot v2.1 datasets into one multi-task dataset.

The input datasets are never modified. Parquet indices and metadata are rewritten
for the merged dataset, while videos are hard-linked when possible and copied as
a fallback.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from lerobot.common.datasets.compute_stats import aggregate_stats
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.common.datasets.utils import serialize_dict
except ModuleNotFoundError:
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.datasets.utils import serialize_dict


CODEBASE_VERSION = "v2.1"
DEFAULT_CHUNK_SIZE = 1000
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
REQUIRED_CAMERA_KEYS = ("observation.images.top", "observation.images.right_wrist")
REQUIRED_META_FILES = (
    "meta/info.json",
    "meta/stats.json",
    "meta/tasks.jsonl",
    "meta/episodes.jsonl",
    "meta/episodes_stats.jsonl",
)


@dataclasses.dataclass(frozen=True)
class SourceDataset:
    root: Path
    info: dict[str, Any]
    prompt: str
    task_index: int
    episodes: tuple[dict[str, Any], ...]
    episode_stats: dict[int, dict[str, Any]]


@dataclasses.dataclass(frozen=True)
class MergeArgs:
    input_roots: list[str]
    output_root: str
    media_mode: str = "auto"
    overwrite_output: bool = False


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required metadata file does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}, got {type(value).__name__}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required metadata file does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=4, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _format_episode_path(
    root: Path,
    template: str,
    episode_index: int,
    chunk_size: int,
    *,
    video_key: str | None = None,
) -> Path:
    values: dict[str, Any] = {
        "episode_chunk": episode_index // chunk_size,
        "episode_index": episode_index,
    }
    if video_key is not None:
        values["video_key"] = video_key
    try:
        relative_path = template.format(**values)
    except (KeyError, ValueError) as error:
        raise ValueError(f"Unsupported LeRobot path template {template!r}") from error
    return root / relative_path


def _validate_right_arm_features(info: dict[str, Any], root: Path) -> None:
    features = info.get("features")
    if not isinstance(features, dict):
        raise TypeError(f"Dataset features must be a JSON object: {root}")

    for key in ("action", "observation.state"):
        feature = features.get(key)
        if not isinstance(feature, dict) or feature.get("dtype") != "float32" or feature.get("shape") != [7]:
            raise ValueError(f"{root}: feature {key!r} must be float32 with shape [7], got {feature}")

    camera_keys = tuple(sorted(key for key in features if key.startswith("observation.images.")))
    if camera_keys != tuple(sorted(REQUIRED_CAMERA_KEYS)):
        raise ValueError(f"{root}: expected cameras {list(REQUIRED_CAMERA_KEYS)}, got {list(camera_keys)}")
    for key in REQUIRED_CAMERA_KEYS:
        feature = features[key]
        if feature.get("dtype") != "video":
            raise ValueError(f"{root}: camera {key!r} must use video storage, got {feature.get('dtype')!r}")


def _load_source(root_value: str) -> SourceDataset:
    root = Path(root_value).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Input dataset directory does not exist: {root}")
    for relative_path in REQUIRED_META_FILES:
        if not (root / relative_path).is_file():
            raise FileNotFoundError(f"Input is not a complete LeRobot v2.1 dataset; missing {root / relative_path}")

    info = _read_json(root / "meta/info.json")
    if info.get("codebase_version") != CODEBASE_VERSION:
        raise ValueError(
            f"{root}: expected codebase_version={CODEBASE_VERSION!r}, got {info.get('codebase_version')!r}"
        )
    if not isinstance(info.get("fps"), int) or info["fps"] <= 0:
        raise ValueError(f"{root}: invalid FPS {info.get('fps')!r}")
    if not isinstance(info.get("chunks_size"), int) or info["chunks_size"] <= 0:
        raise ValueError(f"{root}: invalid chunks_size {info.get('chunks_size')!r}")
    if not isinstance(info.get("data_path"), str) or not isinstance(info.get("video_path"), str):
        raise ValueError(f"{root}: both data_path and video_path must be present")
    _validate_right_arm_features(info, root)

    tasks = _read_jsonl(root / "meta/tasks.jsonl")
    if info.get("total_tasks") != 1 or len(tasks) != 1:
        raise ValueError(f"{root}: each input dataset must contain exactly one task")
    task_index = tasks[0].get("task_index")
    prompt = tasks[0].get("task")
    if not isinstance(task_index, int) or not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{root}: invalid single-task metadata {tasks[0]}")

    episodes = _read_jsonl(root / "meta/episodes.jsonl")
    episode_stats_rows = _read_jsonl(root / "meta/episodes_stats.jsonl")
    total_episodes = info.get("total_episodes")
    if not isinstance(total_episodes, int) or total_episodes <= 0:
        raise ValueError(f"{root}: invalid total_episodes {total_episodes!r}")
    if len(episodes) != total_episodes or len(episode_stats_rows) != total_episodes:
        raise ValueError(
            f"{root}: episode metadata count mismatch: info={total_episodes}, "
            f"episodes={len(episodes)}, episode_stats={len(episode_stats_rows)}"
        )

    episodes_by_index = {row.get("episode_index"): row for row in episodes}
    stats_by_index = {row.get("episode_index"): row for row in episode_stats_rows}
    expected_indices = set(range(total_episodes))
    if set(episodes_by_index) != expected_indices or set(stats_by_index) != expected_indices:
        raise ValueError(f"{root}: episode indices must be contiguous in [0, {total_episodes - 1}]")

    ordered_episodes: list[dict[str, Any]] = []
    total_frames = 0
    for episode_index in range(total_episodes):
        episode = episodes_by_index[episode_index]
        length = episode.get("length")
        if not isinstance(length, int) or length <= 0:
            raise ValueError(f"{root}: invalid length for episode {episode_index}: {length!r}")
        if episode.get("tasks") != [prompt]:
            raise ValueError(f"{root}: episode {episode_index} does not reference the sole task prompt")
        stats = stats_by_index[episode_index].get("stats")
        if not isinstance(stats, dict):
            raise TypeError(f"{root}: missing stats for episode {episode_index}")
        ordered_episodes.append(episode)
        total_frames += length

    if info.get("total_frames") != total_frames:
        raise ValueError(f"{root}: total_frames={info.get('total_frames')!r}, but episodes contain {total_frames}")

    expected_video_count = total_episodes * len(REQUIRED_CAMERA_KEYS)
    if info.get("total_videos") != expected_video_count:
        raise ValueError(f"{root}: expected total_videos={expected_video_count}, got {info.get('total_videos')!r}")

    return SourceDataset(
        root=root,
        info=info,
        prompt=prompt.strip(),
        task_index=task_index,
        episodes=tuple(ordered_episodes),
        episode_stats={index: stats_by_index[index] for index in range(total_episodes)},
    )


def _validate_compatibility(sources: list[SourceDataset]) -> None:
    if len(sources) < 2:
        raise ValueError("At least two input datasets are required")
    if len({source.root for source in sources}) != len(sources):
        raise ValueError("Input dataset directories must be unique")
    prompts = [source.prompt for source in sources]
    if len(set(prompts)) != len(prompts):
        raise ValueError(f"Every input dataset must have a distinct task prompt, got {prompts}")

    reference = sources[0]
    for source in sources[1:]:
        for key in ("robot_type", "fps"):
            if source.info.get(key) != reference.info.get(key):
                raise ValueError(
                    f"Dataset mismatch for {key}: {reference.root}={reference.info.get(key)!r}, "
                    f"{source.root}={source.info.get(key)!r}"
                )
        if source.info["features"] != reference.info["features"]:
            raise ValueError(f"Dataset feature schemas differ: {reference.root} and {source.root}")


def _validate_output_path(output_root: Path, sources: list[SourceDataset], *, overwrite: bool) -> None:
    for source in sources:
        if output_root == source.root or output_root in source.root.parents or source.root in output_root.parents:
            raise ValueError(f"Output directory must not overlap an input dataset: {output_root} and {source.root}")
    if output_root.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_root}. Pass --overwrite_output to replace it.")


def _replace_int_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    column_index = table.schema.get_field_index(name)
    if column_index < 0:
        raise ValueError(f"Parquet table is missing required column {name!r}")
    column_type = table.schema.field(column_index).type
    return table.set_column(column_index, name, pa.array(values, type=column_type))


def _validate_episode_table(
    table: pa.Table,
    source: SourceDataset,
    local_episode_index: int,
    expected_length: int,
) -> int:
    if table.num_rows != expected_length:
        raise ValueError(
            f"{source.root}: episode {local_episode_index} has {table.num_rows} parquet rows, "
            f"metadata says {expected_length}"
        )
    required_columns = ("frame_index", "episode_index", "index", "task_index")
    values: dict[str, np.ndarray] = {}
    for name in required_columns:
        if name not in table.column_names:
            raise ValueError(f"{source.root}: episode {local_episode_index} is missing parquet column {name!r}")
        values[name] = np.asarray(table[name].combine_chunks())

    expected_frames = np.arange(expected_length, dtype=np.int64)
    if not np.array_equal(values["frame_index"], expected_frames):
        raise ValueError(f"{source.root}: episode {local_episode_index} frame_index is not contiguous from zero")
    if not np.all(values["episode_index"] == local_episode_index):
        raise ValueError(f"{source.root}: episode_index column mismatch in episode {local_episode_index}")
    if not np.all(values["task_index"] == source.task_index):
        raise ValueError(f"{source.root}: task_index column mismatch in episode {local_episode_index}")
    if not np.array_equal(values["index"], np.arange(values["index"][0], values["index"][0] + expected_length)):
        raise ValueError(f"{source.root}: index column is not contiguous in episode {local_episode_index}")
    return int(values["index"][0])


def _reindex_episode_stats(
    stats_row: dict[str, Any],
    *,
    new_episode_index: int,
    new_task_index: int,
    new_start_index: int,
    old_start_index: int,
    episode_length: int,
) -> dict[str, Any]:
    result = copy.deepcopy(stats_row)
    result["episode_index"] = new_episode_index
    stats = result["stats"]

    def set_constant(name: str, value: int) -> None:
        if name not in stats:
            raise ValueError(f"Episode stats are missing required feature {name!r}")
        stats[name] = {
            "min": [value],
            "max": [value],
            "mean": [float(value)],
            "std": [0.0],
            "count": [episode_length],
        }

    set_constant("episode_index", new_episode_index)
    set_constant("task_index", new_task_index)

    index_stats = stats.get("index")
    if not isinstance(index_stats, dict):
        raise ValueError("Episode stats are missing required feature 'index'")
    offset = new_start_index - old_start_index
    for key in ("min", "max", "mean"):
        values = index_stats.get(key)
        if not isinstance(values, list) or len(values) != 1:
            raise ValueError(f"Invalid episode index statistics for {key!r}: {values!r}")
        values[0] += offset
    index_stats["count"] = [episode_length]
    return result


def _stats_to_numpy(stats: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    converted: dict[str, dict[str, np.ndarray]] = {}
    for feature_name, feature_stats in stats.items():
        if not isinstance(feature_stats, dict):
            raise TypeError(f"Statistics for {feature_name!r} must be an object")
        converted[feature_name] = {key: np.asarray(value) for key, value in feature_stats.items()}
    return converted


def _copy_media(source: Path, destination: Path, mode: str) -> str:
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(f"Required video is missing or empty: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode in ("auto", "hardlink"):
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            if mode == "hardlink":
                raise
    shutil.copy2(source, destination)
    return "copy"


def _merge_into(sources: list[SourceDataset], output_root: Path, media_mode: str) -> dict[str, Any]:
    reference = sources[0]
    chunk_size = DEFAULT_CHUNK_SIZE
    tasks_rows: list[dict[str, Any]] = []
    episodes_rows: list[dict[str, Any]] = []
    episodes_stats_rows: list[dict[str, Any]] = []
    all_episode_stats: list[dict[str, dict[str, np.ndarray]]] = []
    total_frames = 0
    total_videos = 0
    media_counts = {"hardlink": 0, "copy": 0}
    new_episode_index = 0

    for new_task_index, source in enumerate(sources):
        tasks_rows.append({"task_index": new_task_index, "task": source.prompt})
        source_chunk_size = source.info["chunks_size"]
        for local_episode_index, episode in enumerate(source.episodes):
            episode_length = episode["length"]
            source_parquet = _format_episode_path(
                source.root,
                source.info["data_path"],
                local_episode_index,
                source_chunk_size,
            )
            if not source_parquet.is_file():
                raise FileNotFoundError(f"Required parquet file does not exist: {source_parquet}")
            table = pq.read_table(source_parquet)
            old_start_index = _validate_episode_table(
                table,
                source,
                local_episode_index,
                episode_length,
            )
            table = _replace_int_column(
                table,
                "episode_index",
                np.full(episode_length, new_episode_index, dtype=np.int64),
            )
            table = _replace_int_column(
                table,
                "task_index",
                np.full(episode_length, new_task_index, dtype=np.int64),
            )
            table = _replace_int_column(
                table,
                "index",
                np.arange(total_frames, total_frames + episode_length, dtype=np.int64),
            )

            output_parquet = _format_episode_path(
                output_root,
                DATA_PATH,
                new_episode_index,
                chunk_size,
            )
            output_parquet.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, output_parquet, compression="snappy")

            episodes_rows.append(
                {
                    "episode_index": new_episode_index,
                    "tasks": [source.prompt],
                    "length": episode_length,
                }
            )
            stats_row = _reindex_episode_stats(
                source.episode_stats[local_episode_index],
                new_episode_index=new_episode_index,
                new_task_index=new_task_index,
                new_start_index=total_frames,
                old_start_index=old_start_index,
                episode_length=episode_length,
            )
            episodes_stats_rows.append(stats_row)
            all_episode_stats.append(_stats_to_numpy(stats_row["stats"]))

            for camera_key in REQUIRED_CAMERA_KEYS:
                source_video = _format_episode_path(
                    source.root,
                    source.info["video_path"],
                    local_episode_index,
                    source_chunk_size,
                    video_key=camera_key,
                )
                output_video = _format_episode_path(
                    output_root,
                    VIDEO_PATH,
                    new_episode_index,
                    chunk_size,
                    video_key=camera_key,
                )
                used_mode = _copy_media(source_video, output_video, media_mode)
                media_counts[used_mode] += 1
                total_videos += 1

            total_frames += episode_length
            new_episode_index += 1

    dataset_stats = serialize_dict(aggregate_stats(all_episode_stats))
    info = copy.deepcopy(reference.info)
    info.update(
        {
            "total_episodes": new_episode_index,
            "total_frames": total_frames,
            "total_tasks": len(tasks_rows),
            "total_videos": total_videos,
            "total_chunks": (new_episode_index + chunk_size - 1) // chunk_size,
            "chunks_size": chunk_size,
            "splits": {"train": f"0:{new_episode_index}"},
            "data_path": DATA_PATH,
            "video_path": VIDEO_PATH,
        }
    )
    _write_json(output_root / "meta/info.json", info)
    _write_json(output_root / "meta/stats.json", dataset_stats)
    _write_jsonl(output_root / "meta/tasks.jsonl", tasks_rows)
    _write_jsonl(output_root / "meta/episodes.jsonl", episodes_rows)
    _write_jsonl(output_root / "meta/episodes_stats.jsonl", episodes_stats_rows)

    return {
        "total_episodes": new_episode_index,
        "total_frames": total_frames,
        "total_tasks": len(tasks_rows),
        "total_videos": total_videos,
        "media_counts": media_counts,
    }


def _validate_merged_dataset(output_root: Path, summary: dict[str, Any]) -> None:
    repo_id = "local/xtrainer_right_arm_merged_validation"
    metadata = LeRobotDatasetMetadata(repo_id, root=output_root)
    expected = {
        "total_tasks": metadata.total_tasks,
        "total_episodes": metadata.total_episodes,
        "total_frames": metadata.total_frames,
    }
    mismatches = {key: (summary[key], actual) for key, actual in expected.items() if summary[key] != actual}
    if mismatches:
        raise ValueError(f"Merged LeRobot metadata validation failed: {mismatches}")

    task_stats = metadata.stats.get("task_index")
    if task_stats is None:
        raise ValueError("Merged LeRobot metadata is missing task_index statistics")
    if int(np.asarray(task_stats["min"]).item()) != 0:
        raise ValueError("Merged task_index statistics must start at zero")
    if int(np.asarray(task_stats["max"]).item()) != summary["total_tasks"] - 1:
        raise ValueError("Merged task_index statistics do not cover every task")

    dataset = LeRobotDataset(repo_id, root=output_root, download_videos=False)
    if len(dataset) != summary["total_frames"]:
        raise ValueError(
            f"Merged LeRobot parquet validation failed: expected {summary['total_frames']} rows, got {len(dataset)}"
        )


def merge(args: MergeArgs) -> dict[str, Any]:
    if args.media_mode not in {"auto", "hardlink", "copy"}:
        raise ValueError(f"Unsupported media_mode {args.media_mode!r}")
    sources = [_load_source(value) for value in args.input_roots]
    _validate_compatibility(sources)

    output_root = Path(args.output_root).expanduser().resolve()
    _validate_output_path(output_root, sources, overwrite=args.overwrite_output)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.merge-", dir=output_root.parent))

    try:
        summary = _merge_into(sources, temporary_root, args.media_mode)
        _validate_merged_dataset(temporary_root, summary)
        if output_root.exists():
            if output_root.is_dir():
                shutil.rmtree(output_root)
            else:
                output_root.unlink()
        temporary_root.replace(output_root)
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise

    print(f"Merged dataset: {output_root}")
    print(
        f"Tasks={summary['total_tasks']}, episodes={summary['total_episodes']}, "
        f"frames={summary['total_frames']}, videos={summary['total_videos']}"
    )
    print(f"Video files: hardlink={summary['media_counts']['hardlink']}, copy={summary['media_counts']['copy']}")
    return summary


def parse_args() -> MergeArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_roots",
        nargs="+",
        required=True,
        help="Two or more complete single-task right-arm LeRobot v2.1 dataset directories, in task-index order.",
    )
    parser.add_argument("--output_root", required=True, help="Output directory for the merged dataset.")
    parser.add_argument(
        "--media_mode",
        choices=("auto", "hardlink", "copy"),
        default="auto",
        help="How to materialize videos. auto tries hard links first and copies when unavailable.",
    )
    parser.add_argument("--overwrite_output", action="store_true")
    namespace = parser.parse_args()
    return MergeArgs(**vars(namespace))


if __name__ == "__main__":
    merge(parse_args())
