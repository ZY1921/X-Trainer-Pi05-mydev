"""
Convert raw dobot_xtrainer-style recordings to LeRobot v2.1 dataset format.

Raw input layout (from examples/dobot_xtrainer/record.py raw mode):
    collect_data/
      <episode_id>/
        topImg/*.jpg
        leftImg/*.jpg
        rightImg/*.jpg
        observation/*.pkl   # contains {"joint_positions": ..., "control": ...}

Output layout (v2.1):
    <output_root>/
      data/chunk-000/episode_000000.parquet
      videos/chunk-000/<camera_key>/episode_000000.mp4      # when --use_videos
      meta/info.json
      meta/stats.json
      meta/episodes.jsonl
      meta/episodes_stats.jsonl
      meta/tasks.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import shutil
import subprocess
import sys

import cv2
import datasets
import numpy as np
from PIL import Image

try:
    # LeRobot >= 0.1.0
    from lerobot.common.datasets.compute_stats import aggregate_stats
    from lerobot.common.datasets.compute_stats import compute_episode_stats
    from lerobot.common.datasets.utils import DEFAULT_FEATURES
    from lerobot.common.datasets.utils import embed_images
    from lerobot.common.datasets.utils import get_hf_features_from_features
    from lerobot.common.datasets.utils import serialize_dict
except ModuleNotFoundError:
    # Backward compatibility for older LeRobot releases.
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.compute_stats import compute_episode_stats
    from lerobot.datasets.utils import DEFAULT_FEATURES
    from lerobot.datasets.utils import embed_images
    from lerobot.datasets.utils import get_hf_features_from_features
    from lerobot.datasets.utils import serialize_dict

CODEBASE_VERSION = "v2.1"
CHUNK_SIZE = 1000

LEGACY_INFO_PATH = "meta/info.json"
LEGACY_STATS_PATH = "meta/stats.json"
LEGACY_EPISODES_PATH = "meta/episodes.jsonl"
LEGACY_EPISODES_STATS_PATH = "meta/episodes_stats.jsonl"
LEGACY_TASKS_PATH = "meta/tasks.jsonl"

LEGACY_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
LEGACY_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
TEMP_IMAGE_PATH = "images/{image_key}/episode_{episode_index:06d}/frame_{frame_index:06d}.png"


def _frame_id(path: Path) -> int | None:
    stem = path.stem
    return int(stem) if stem.isdigit() else None


def _sorted_frame_ids(folder: Path, patterns: list[str]) -> list[int]:
    ids = set()
    for pattern in patterns:
        for path in folder.glob(pattern):
            idx = _frame_id(path)
            if idx is not None:
                ids.add(idx)
    return sorted(ids)


def _build_joint_names(dim: int) -> list[str]:
    if dim == 14:
        left = [f"left_joint{i}.pos" for i in range(1, 7)] + ["left_gripper.pos"]
        right = [f"right_joint{i}.pos" for i in range(1, 7)] + ["right_gripper.pos"]
        return left + right
    return [f"joint_{i}.pos" for i in range(dim)]


def _read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb)


def _verify_image_file(path: Path) -> None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(f"Failed to decode image: {path}")


def _resolve_image_path(folder: Path, idx: int) -> Path:
    for ext in (".jpg", ".jpeg", ".png"):
        p = folder / f"{idx}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Image for frame {idx} not found in {folder}")


def _find_first_valid_observation(episode_dirs: list[Path]) -> dict:
    for episode_dir in episode_dirs:
        obs_dir = episode_dir / "observation"
        if not obs_dir.is_dir():
            continue
        for pkl_path in sorted(obs_dir.glob("*.pkl")):
            with pkl_path.open("rb") as f:
                payload = pickle.load(f)
            if "joint_positions" in payload and "control" in payload:
                return payload
    raise RuntimeError("No valid observation pkl found under raw_root.")


def _find_first_valid_image_shape(episode_dirs: list[Path], folder_name: str) -> tuple[int, int, int]:
    for episode_dir in episode_dirs:
        cam_dir = episode_dir / folder_name
        if not cam_dir.is_dir():
            continue
        candidates = sorted(list(cam_dir.glob("*.jpg")) + list(cam_dir.glob("*.jpeg")) + list(cam_dir.glob("*.png")))
        for img_path in candidates:
            img = _read_rgb(img_path)
            if img.ndim == 3 and img.shape[2] == 3:
                return int(img.shape[0]), int(img.shape[1]), 3
    raise RuntimeError(f"No valid images found for {folder_name}.")


def _build_features(
    action_dim: int,
    state_dim: int,
    top_shape: tuple[int, int, int],
    left_shape: tuple[int, int, int],
    right_shape: tuple[int, int, int],
    use_videos: bool,
) -> dict:
    if action_dim != state_dim:
        raise ValueError(f"Action/state dim mismatch: {action_dim} vs {state_dim}")

    names = _build_joint_names(action_dim)
    image_dtype = "video" if use_videos else "image"
    return {
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": names,
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": names,
        },
        "observation.images.top": {
            "dtype": image_dtype,
            "shape": top_shape,
            "names": ["height", "width", "channels"],
        },
        "observation.images.left_wrist": {
            "dtype": image_dtype,
            "shape": left_shape,
            "names": ["height", "width", "channels"],
        },
        "observation.images.right_wrist": {
            "dtype": image_dtype,
            "shape": right_shape,
            "names": ["height", "width", "channels"],
        },
    }


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _episode_chunk(episode_index: int) -> int:
    return episode_index // CHUNK_SIZE


def _episode_data_path(root: Path, episode_index: int) -> Path:
    return root / LEGACY_DATA_PATH.format(
        episode_chunk=_episode_chunk(episode_index),
        episode_index=episode_index,
    )


def _episode_video_path(root: Path, episode_index: int, video_key: str) -> Path:
    return root / LEGACY_VIDEO_PATH.format(
        episode_chunk=_episode_chunk(episode_index),
        video_key=video_key,
        episode_index=episode_index,
    )


def _frame_image_path(root: Path, episode_index: int, image_key: str, frame_index: int) -> Path:
    return root / TEMP_IMAGE_PATH.format(
        image_key=image_key,
        episode_index=episode_index,
        frame_index=frame_index,
    )


def _cleanup_temp_images(root: Path, episode_index: int, image_keys: list[str]) -> None:
    for image_key in image_keys:
        ep_img_dir = _frame_image_path(root, episode_index, image_key, 0).parent
        if ep_img_dir.is_dir():
            shutil.rmtree(ep_img_dir, ignore_errors=True)


def _write_png_atomic(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    Image.fromarray(rgb).save(tmp_path, format="PNG")
    tmp_path.replace(path)


def _repair_episode_images(
    image_paths_by_key: dict[str, list[str]],
    source_paths_by_key: dict[str, list[str]],
    repair_retries: int,
) -> int:
    repaired = 0
    for key in image_paths_by_key:
        image_paths = image_paths_by_key[key]
        source_paths = source_paths_by_key[key]
        if len(image_paths) != len(source_paths):
            raise RuntimeError(
                f"Mismatched image/source path length for {key}: {len(image_paths)} vs {len(source_paths)}"
            )

        for dst_str, src_str in zip(image_paths, source_paths, strict=True):
            dst = Path(dst_str)
            src = Path(src_str)
            try:
                _verify_image_file(dst)
                continue
            except Exception:
                pass

            ok = False
            for _ in range(max(1, repair_retries)):
                rgb = _read_rgb(src)
                _write_png_atomic(dst, rgb)
                try:
                    _verify_image_file(dst)
                    ok = True
                    repaired += 1
                    break
                except Exception:
                    continue

            if not ok:
                raise RuntimeError(f"Failed to repair temp image {dst} from source {src}")

    return repaired


def _encode_video_once(
    imgs_dir: Path,
    video_path: Path,
    fps: int,
    vcodec: str,
    use_subprocess: bool,
    quiet_encoder: bool,
) -> tuple[bool, str]:
    if not use_subprocess:
        try:
            from lerobot.common.datasets.video_utils import encode_video_frames
        except ModuleNotFoundError:
            from lerobot.datasets.video_utils import encode_video_frames

        encode_video_frames(
            imgs_dir=imgs_dir,
            video_path=video_path,
            fps=fps,
            vcodec=vcodec,
            overwrite=True,
            log_level=None,
        )
        return True, ""

    child_code = (
        "from pathlib import Path\n"
        "import sys\n"
        "try:\n"
        "    from lerobot.common.datasets.video_utils import encode_video_frames\n"
        "except ModuleNotFoundError:\n"
        "    from lerobot.datasets.video_utils import encode_video_frames\n"
        "imgs_dir = Path(sys.argv[1])\n"
        "video_path = Path(sys.argv[2])\n"
        "fps = int(sys.argv[3])\n"
        "vcodec = sys.argv[4]\n"
        "encode_video_frames(imgs_dir=imgs_dir, video_path=video_path, fps=fps, vcodec=vcodec, overwrite=True, log_level=None)\n"
    )
    cmd = [sys.executable, "-c", child_code, str(imgs_dir), str(video_path), str(fps), vcodec]
    if quiet_encoder:
        completed = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        completed = subprocess.run(cmd, check=False)

    if completed.returncode == 0:
        return True, ""

    if completed.returncode in (-11, 139):
        return False, f"encoder subprocess segfault (returncode={completed.returncode})"
    return False, f"encoder subprocess failed (returncode={completed.returncode})"


def _get_video_info_cv2(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    cap.release()

    codec = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)).strip()
    return {
        "video.height": height,
        "video.width": width,
        "video.codec": codec if codec else "unknown",
        "video.pix_fmt": "unknown",
        "video.is_depth_map": False,
        "video.fps": round(fps) if fps > 0 else 0,
        "video.channels": 3,
        "has_audio": False,
    }


def convert(args: argparse.Namespace) -> None:
    raw_root = Path(args.raw_root)
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw_root not found: {raw_root}")

    episode_dirs = sorted([p for p in raw_root.iterdir() if p.is_dir()])
    if not episode_dirs:
        raise RuntimeError(f"No episode dirs found under {raw_root}")

    first_obs = _find_first_valid_observation(episode_dirs)
    action_dim = int(np.asarray(first_obs["control"]).reshape(-1).shape[0])
    state_dim = int(np.asarray(first_obs["joint_positions"]).reshape(-1).shape[0])
    top_shape = _find_first_valid_image_shape(episode_dirs, "topImg")
    left_shape = _find_first_valid_image_shape(episode_dirs, "leftImg")
    right_shape = _find_first_valid_image_shape(episode_dirs, "rightImg")
    user_features = _build_features(action_dim, state_dim, top_shape, left_shape, right_shape, args.use_videos)
    features = {**user_features, **DEFAULT_FEATURES}
    hf_features = get_hf_features_from_features(features)

    output_root = Path(args.output_root)
    if output_root.exists() and any(output_root.iterdir()):
        if args.overwrite_output:
            shutil.rmtree(output_root)
        else:
            raise FileExistsError(
                f"output_root already exists and is not empty: {output_root}. " "Use --overwrite_output to replace it."
            )
    output_root.mkdir(parents=True, exist_ok=True)

    camera_keys = [k for k, ft in features.items() if ft["dtype"] in ["image", "video"]]
    episodes_rows: list[dict] = []
    episodes_stats_rows: list[dict] = []
    all_episode_stats: list[dict] = []
    total_frames = 0
    total_videos = 0
    saved_episodes = 0

    task_to_task_index = {args.task: 0}
    tasks_rows = [{"task_index": 0, "task": args.task}]

    for episode_dir in episode_dirs:
        obs_dir = episode_dir / "observation"
        top_dir = episode_dir / "topImg"
        left_dir = episode_dir / "leftImg"
        right_dir = episode_dir / "rightImg"
        if not (obs_dir.is_dir() and top_dir.is_dir() and left_dir.is_dir() and right_dir.is_dir()):
            print(f"[Skip] Missing required folders: {episode_dir}")
            continue

        obs_ids = _sorted_frame_ids(obs_dir, ["*.pkl"])
        top_ids = _sorted_frame_ids(top_dir, ["*.jpg", "*.jpeg", "*.png"])
        left_ids = _sorted_frame_ids(left_dir, ["*.jpg", "*.jpeg", "*.png"])
        right_ids = _sorted_frame_ids(right_dir, ["*.jpg", "*.jpeg", "*.png"])
        common_ids = sorted(set(obs_ids) & set(top_ids) & set(left_ids) & set(right_ids))
        if len(common_ids) < args.min_frames:
            print(f"[Skip] Too few valid frames ({len(common_ids)}): {episode_dir.name}")
            continue

        frame_ids = common_ids[args.skip_first_frames :]
        if len(frame_ids) < args.min_frames:
            print(f"[Skip] Too few frames after skip ({len(frame_ids)}): {episode_dir.name}")
            continue

        episode_index = saved_episodes
        actions: list[np.ndarray] = []
        states: list[np.ndarray] = []
        timestamps: list[float] = []
        image_paths: dict[str, list[str]] = {k: [] for k in camera_keys}
        source_paths: dict[str, list[str]] = {k: [] for k in camera_keys}
        skipped_bad = 0

        for raw_idx in frame_ids:
            try:
                with (obs_dir / f"{raw_idx}.pkl").open("rb") as f:
                    payload = pickle.load(f)

                action = np.asarray(payload["control"], dtype=np.float32).reshape(-1)
                state = np.asarray(payload["joint_positions"], dtype=np.float32).reshape(-1)
                if action.shape[0] != action_dim or state.shape[0] != state_dim:
                    raise ValueError(
                        f"Unexpected action/state dims at {episode_dir.name}/{raw_idx}: "
                        f"{action.shape[0]}/{state.shape[0]}"
                    )

                top_path = _resolve_image_path(top_dir, raw_idx)
                left_path = _resolve_image_path(left_dir, raw_idx)
                right_path = _resolve_image_path(right_dir, raw_idx)

                frame_index = len(actions)
                cam_inputs = {
                    "observation.images.top": top_path,
                    "observation.images.left_wrist": left_path,
                    "observation.images.right_wrist": right_path,
                }

                for cam_key, src_path in cam_inputs.items():
                    image = _read_rgb(src_path)
                    dst = _frame_image_path(output_root, episode_index, cam_key, frame_index)
                    _write_png_atomic(dst, image)
                    image_paths[cam_key].append(str(dst))
                    source_paths[cam_key].append(str(src_path))

                actions.append(action)
                states.append(state)
                timestamps.append(frame_index / float(args.fps))

            except Exception as e:
                skipped_bad += 1
                if args.skip_bad_frames:
                    print(f"[Warn] Skip bad frame {episode_dir.name}/{raw_idx}: {type(e).__name__}: {e}")
                    continue
                raise

        episode_length = len(actions)
        if episode_length < args.min_frames:
            print(
                f"[Skip] Too few valid frames after filtering ({episode_length}), "
                f"skipped_bad={skipped_bad}: {episode_dir.name}"
            )
            _cleanup_temp_images(output_root, episode_index, camera_keys)
            continue

        action_array = np.stack(actions).astype(np.float32)
        state_array = np.stack(states).astype(np.float32)
        frame_index_array = np.arange(episode_length, dtype=np.int64)
        timestamp_array = np.asarray(timestamps, dtype=np.float32)
        episode_index_array = np.full((episode_length,), episode_index, dtype=np.int64)
        index_array = np.arange(total_frames, total_frames + episode_length, dtype=np.int64)
        task_index_array = np.full((episode_length,), task_to_task_index[args.task], dtype=np.int64)

        episode_buffer = {
            "action": action_array,
            "observation.state": state_array,
            "frame_index": frame_index_array,
            "timestamp": timestamp_array,
            "episode_index": episode_index_array,
            "index": index_array,
            "task_index": task_index_array,
        }
        for cam_key in camera_keys:
            episode_buffer[cam_key] = image_paths[cam_key]

        repaired_count = _repair_episode_images(
            image_paths_by_key=image_paths,
            source_paths_by_key=source_paths,
            repair_retries=args.repair_retries,
        )
        if repaired_count > 0:
            print(f"[Warn] Repaired {repaired_count} temporary frame files: {episode_dir.name}")

        parquet_payload = {k: episode_buffer[k] for k in hf_features}
        ep_hf_ds = datasets.Dataset.from_dict(parquet_payload, features=hf_features, split="train")
        if not args.use_videos:
            ep_hf_ds = embed_images(ep_hf_ds)

        ep_data_path = _episode_data_path(output_root, episode_index)
        ep_data_path.parent.mkdir(parents=True, exist_ok=True)
        ep_hf_ds.to_parquet(ep_data_path)

        # Keep temporary frame images alive until stats are computed, because
        # compute_episode_stats samples image files from their paths.
        ep_stats = compute_episode_stats(episode_buffer, features)
        all_episode_stats.append(ep_stats)
        episodes_stats_rows.append(
            {
                "episode_index": episode_index,
                "stats": serialize_dict(ep_stats),
            }
        )
        episodes_rows.append(
            {
                "episode_index": episode_index,
                "tasks": [args.task],
                "length": episode_length,
            }
        )

        if args.use_videos:
            for cam_key in camera_keys:
                ep_img_dir = Path(image_paths[cam_key][0]).parent
                ep_video_path = _episode_video_path(output_root, episode_index, cam_key)
                for attempt in range(args.encode_retries + 1):
                    try:
                        ok, encode_err = _encode_video_once(
                            imgs_dir=ep_img_dir,
                            video_path=ep_video_path,
                            fps=args.fps,
                            vcodec=args.vcodec,
                            use_subprocess=args.encode_in_subprocess,
                            quiet_encoder=args.quiet_encoder,
                        )
                        if ok:
                            break
                        if attempt >= args.encode_retries:
                            raise RuntimeError(encode_err)
                        repaired = _repair_episode_images(
                            image_paths_by_key={cam_key: image_paths[cam_key]},
                            source_paths_by_key={cam_key: source_paths[cam_key]},
                            repair_retries=args.repair_retries,
                        )
                        print(
                            f"[Warn] Video encode retry {attempt + 1}/{args.encode_retries} "
                            f"for {episode_dir.name}/{cam_key}, repaired={repaired}, reason={encode_err}"
                        )
                    except OSError as e:
                        is_stream_error = "broken data stream" in str(e).lower()
                        if not is_stream_error or attempt >= args.encode_retries:
                            raise
                        repaired = _repair_episode_images(
                            image_paths_by_key={cam_key: image_paths[cam_key]},
                            source_paths_by_key={cam_key: source_paths[cam_key]},
                            repair_retries=args.repair_retries,
                        )
                        print(
                            f"[Warn] Video encode retry {attempt + 1}/{args.encode_retries} "
                            f"for {episode_dir.name}/{cam_key}, repaired={repaired}, reason={type(e).__name__}"
                        )
                total_videos += 1
                if "info" not in features[cam_key]:
                    features[cam_key]["info"] = _get_video_info_cv2(ep_video_path)

            if not args.keep_images_for_video:
                _cleanup_temp_images(output_root, episode_index, camera_keys)

        total_frames += episode_length
        saved_episodes += 1
        print(
            f"[OK] Saved episode {saved_episodes}: {episode_dir.name} "
            f"({episode_length} frames, skipped_bad={skipped_bad})"
        )

    if saved_episodes == 0:
        raise RuntimeError("No episodes converted. Please check raw data folders and frame files.")

    dataset_stats = aggregate_stats(all_episode_stats)
    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": args.robot_type,
        "total_episodes": saved_episodes,
        "total_frames": total_frames,
        "total_tasks": len(tasks_rows),
        "total_videos": total_videos,
        "total_chunks": (saved_episodes + CHUNK_SIZE - 1) // CHUNK_SIZE,
        "chunks_size": CHUNK_SIZE,
        "fps": int(args.fps),
        "splits": {"train": f"0:{saved_episodes}"},
        "data_path": LEGACY_DATA_PATH,
        "video_path": LEGACY_VIDEO_PATH if args.use_videos else None,
        "features": features,
    }

    _write_json(output_root / LEGACY_INFO_PATH, info)
    _write_json(output_root / LEGACY_STATS_PATH, serialize_dict(dataset_stats))
    _write_jsonl(output_root / LEGACY_TASKS_PATH, tasks_rows)
    _write_jsonl(output_root / LEGACY_EPISODES_PATH, episodes_rows)
    _write_jsonl(output_root / LEGACY_EPISODES_STATS_PATH, episodes_stats_rows)

    print(f"Done. Converted episodes: {saved_episodes}")
    print(f"LeRobot v2.1 dataset root: {output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert dobot raw collect_data to LeRobot v2.1 format.")
    parser.add_argument("--raw_root", type=str, required=True, help="Path to raw collect_data directory.")
    parser.add_argument("--output_root", type=str, required=True, help="Output path for LeRobot dataset root.")
    parser.add_argument("--repo_id", type=str, default="local/dobot_xtrainer_converted_v21")
    parser.add_argument("--robot_type", type=str, default="dobot_xtrainer_bimanual")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--task", type=str, default="Insert the test tube on the desktop into the rack.")
    parser.add_argument("--use_videos", dest="use_videos", action="store_true")
    parser.add_argument("--no_videos", dest="use_videos", action="store_false")
    parser.set_defaults(use_videos=True)
    parser.add_argument("--vcodec", type=str, default="h264", choices=["h264", "hevc", "libsvtav1"])
    parser.add_argument("--repair_retries", type=int, default=2, help="Retries when repairing temporary png frames.")
    parser.add_argument(
        "--encode_retries",
        type=int,
        default=1,
        help="Retries for video encoding when broken temporary image stream is detected.",
    )
    parser.add_argument("--encode_in_subprocess", dest="encode_in_subprocess", action="store_true")
    parser.add_argument("--encode_in_process", dest="encode_in_subprocess", action="store_false")
    parser.set_defaults(encode_in_subprocess=True)
    parser.add_argument("--quiet_encoder", dest="quiet_encoder", action="store_true")
    parser.add_argument("--verbose_encoder", dest="quiet_encoder", action="store_false")
    parser.set_defaults(quiet_encoder=True)
    parser.add_argument("--keep_images_for_video", action="store_true")
    parser.add_argument("--skip_first_frames", type=int, default=0)
    parser.add_argument("--min_frames", type=int, default=10)
    parser.add_argument("--skip_bad_frames", dest="skip_bad_frames", action="store_true")
    parser.add_argument("--fail_on_bad_frames", dest="skip_bad_frames", action="store_false")
    parser.set_defaults(skip_bad_frames=True)
    parser.add_argument("--overwrite_output", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
