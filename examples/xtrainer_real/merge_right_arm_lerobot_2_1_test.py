import json
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from examples.xtrainer_real import merge_right_arm_lerobot_2_1 as merger


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _scalar_stats(value: float, count: int) -> dict:
    return {"min": [value], "max": [value], "mean": [value], "std": [0.0], "count": [count]}


def _vector_stats(value: float, count: int) -> dict:
    values = [value] * 7
    return {"min": values, "max": values, "mean": values, "std": [0.0] * 7, "count": [count]}


def _image_stats(value: float) -> dict:
    pixels = [[[value]], [[value]], [[value]]]
    return {"min": pixels, "max": pixels, "mean": pixels, "std": [[[0.0]], [[0.0]], [[0.0]]], "count": [1]}


def _episode_stats(episode_index: int, task_index: int, length: int, value: float) -> dict:
    start = 0
    stop = length - 1
    midpoint = (start + stop) / 2
    return {
        "episode_index": episode_index,
        "stats": {
            "action": _vector_stats(value, length),
            "observation.state": _vector_stats(value, length),
            "frame_index": {
                "min": [start],
                "max": [stop],
                "mean": [midpoint],
                "std": [0.0],
                "count": [length],
            },
            "timestamp": _scalar_stats(0.0, length),
            "episode_index": _scalar_stats(float(episode_index), length),
            "index": {
                "min": [start],
                "max": [stop],
                "mean": [midpoint],
                "std": [0.0],
                "count": [length],
            },
            "task_index": _scalar_stats(float(task_index), length),
            "observation.images.top": _image_stats(value),
            "observation.images.right_wrist": _image_stats(value),
        },
    }


def _features() -> dict:
    joint_feature = {"dtype": "float32", "shape": [7], "names": [f"joint_{index}" for index in range(7)]}
    video_feature = {"dtype": "video", "shape": [8, 8, 3], "names": ["height", "width", "channels"]}
    return {
        "action": joint_feature,
        "observation.state": joint_feature,
        "observation.images.top": video_feature,
        "observation.images.right_wrist": video_feature,
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }


def _make_dataset(root: Path, prompt: str, length: int, value: float, *, fps: int = 22) -> None:
    info = {
        "codebase_version": "v2.1",
        "robot_type": "dobot_xtrainer_right_arm",
        "total_episodes": 1,
        "total_frames": length,
        "total_tasks": 1,
        "total_videos": 2,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": "0:1"},
        "data_path": merger.DATA_PATH,
        "video_path": merger.VIDEO_PATH,
        "features": _features(),
    }
    _write_json(root / "meta/info.json", info)
    _write_json(root / "meta/stats.json", {})
    _write_jsonl(root / "meta/tasks.jsonl", [{"task_index": 0, "task": prompt}])
    _write_jsonl(root / "meta/episodes.jsonl", [{"episode_index": 0, "tasks": [prompt], "length": length}])
    _write_jsonl(root / "meta/episodes_stats.jsonl", [_episode_stats(0, 0, length, value)])

    vectors = pa.array([[value] * 7 for _ in range(length)], type=pa.list_(pa.float32(), 7))
    table = pa.table(
        {
            "action": vectors,
            "observation.state": vectors,
            "timestamp": pa.array([index / fps for index in range(length)], type=pa.float32()),
            "frame_index": pa.array(range(length), type=pa.int64()),
            "episode_index": pa.array([0] * length, type=pa.int64()),
            "index": pa.array(range(length), type=pa.int64()),
            "task_index": pa.array([0] * length, type=pa.int64()),
        }
    )
    parquet_path = root / merger.DATA_PATH.format(episode_chunk=0, episode_index=0)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, parquet_path)

    for camera_key in merger.REQUIRED_CAMERA_KEYS:
        video_path = root / merger.VIDEO_PATH.format(episode_chunk=0, episode_index=0, video_key=camera_key)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(f"{prompt}:{camera_key}".encode())


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_merges_and_reindexes_two_single_task_datasets(tmp_path):
    red = tmp_path / "red"
    green = tmp_path / "green"
    output = tmp_path / "merged"
    _make_dataset(red, "Press the red button.", 2, 1.0)
    _make_dataset(green, "Press the green button.", 3, 2.0)

    summary = merger.merge(
        merger.MergeArgs(
            input_roots=[str(red), str(green)],
            output_root=str(output),
            media_mode="copy",
        )
    )

    assert summary == {
        "total_episodes": 2,
        "total_frames": 5,
        "total_tasks": 2,
        "total_videos": 4,
        "media_counts": {"hardlink": 0, "copy": 4},
    }
    info = json.loads((output / "meta/info.json").read_text())
    assert info["splits"] == {"train": "0:2"}
    assert info["total_tasks"] == 2
    assert _read_jsonl(output / "meta/tasks.jsonl") == [
        {"task_index": 0, "task": "Press the red button."},
        {"task_index": 1, "task": "Press the green button."},
    ]

    second = pq.read_table(output / merger.DATA_PATH.format(episode_chunk=0, episode_index=1)).to_pydict()
    assert second["frame_index"] == [0, 1, 2]
    assert second["episode_index"] == [1, 1, 1]
    assert second["index"] == [2, 3, 4]
    assert second["task_index"] == [1, 1, 1]

    episode_stats = _read_jsonl(output / "meta/episodes_stats.jsonl")
    assert episode_stats[1]["stats"]["episode_index"]["mean"] == [1.0]
    assert episode_stats[1]["stats"]["index"]["min"] == [2]
    assert episode_stats[1]["stats"]["task_index"]["mean"] == [1.0]
    dataset_stats = json.loads((output / "meta/stats.json").read_text())
    assert dataset_stats["task_index"]["mean"] == pytest.approx([0.6])
    metadata = LeRobotDatasetMetadata("local/right_arm_multitask_test", root=output)
    assert metadata.tasks == {0: "Press the red button.", 1: "Press the green button."}


def test_rejects_incompatible_fps_without_creating_output(tmp_path):
    red = tmp_path / "red"
    green = tmp_path / "green"
    output = tmp_path / "merged"
    _make_dataset(red, "Press the red button.", 2, 1.0, fps=22)
    _make_dataset(green, "Press the green button.", 2, 2.0, fps=30)

    with pytest.raises(ValueError, match="Dataset mismatch for fps"):
        merger.merge(merger.MergeArgs([str(red), str(green)], str(output)))

    assert not output.exists()


def test_rejects_incomplete_dataset(tmp_path):
    red = tmp_path / "red"
    incomplete = tmp_path / "incomplete"
    _make_dataset(red, "Press the red button.", 2, 1.0)
    incomplete.mkdir()

    with pytest.raises(FileNotFoundError, match="not a complete LeRobot v2.1 dataset"):
        merger.merge(merger.MergeArgs([str(red), str(incomplete)], str(tmp_path / "merged")))


def test_rejects_duplicate_task_prompts(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _make_dataset(first, "Press the button.", 2, 1.0)
    _make_dataset(second, "Press the button.", 2, 2.0)

    with pytest.raises(ValueError, match="distinct task prompt"):
        merger.merge(merger.MergeArgs([str(first), str(second)], str(tmp_path / "merged")))
