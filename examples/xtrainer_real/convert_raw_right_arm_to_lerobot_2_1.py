"""Convert raw bimanual X-Trainer recordings into a right-arm LeRobot v2.1 dataset.

The source recording remains bimanual, but the converted dataset contains only:

* the right arm state/action (source indices 7:14),
* the top camera, and
* the right-wrist camera.

The legacy bimanual converter is intentionally kept unchanged. This module reuses
its image/video helpers while defining an independent conversion entry point.
"""

# Reusing the legacy converter's private helpers keeps that entry point unchanged.
# ruff: noqa: SLF001

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import shutil

import numpy as np

from examples.xtrainer_real import convert_raw_to_lerobot_2_1 as _base

RIGHT_ARM_SOURCE_SLICE = slice(7, 14)
RIGHT_ARM_DIM = 7
RIGHT_ARM_NAMES = [f"right_joint{i}.pos" for i in range(1, 7)] + ["right_gripper.pos"]


def _select_right_arm(values: object, *, field_name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    if vector.shape[0] != 14:
        raise ValueError(f"Expected raw {field_name} length 14, got {vector.shape[0]}")
    return np.ascontiguousarray(vector[RIGHT_ARM_SOURCE_SLICE])


def _build_features(
    top_shape: tuple[int, int, int],
    right_shape: tuple[int, int, int],
    *,
    use_videos: bool,
) -> dict:
    image_dtype = "video" if use_videos else "image"
    return {
        "action": {
            "dtype": "float32",
            "shape": (RIGHT_ARM_DIM,),
            "names": RIGHT_ARM_NAMES,
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (RIGHT_ARM_DIM,),
            "names": RIGHT_ARM_NAMES,
        },
        "observation.images.top": {
            "dtype": image_dtype,
            "shape": top_shape,
            "names": ["height", "width", "channels"],
        },
        "observation.images.right_wrist": {
            "dtype": image_dtype,
            "shape": right_shape,
            "names": ["height", "width", "channels"],
        },
    }


def convert(args: argparse.Namespace) -> None:
    raw_root = Path(args.raw_root)
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw_root not found: {raw_root}")

    episode_dirs = sorted(path for path in raw_root.iterdir() if path.is_dir())
    if not episode_dirs:
        raise RuntimeError(f"No episode dirs found under {raw_root}")

    first_obs = _base._find_first_valid_observation(episode_dirs)
    _select_right_arm(first_obs["control"], field_name="control")
    _select_right_arm(first_obs["joint_positions"], field_name="joint_positions")
    top_shape = _base._find_first_valid_image_shape(episode_dirs, "topImg")
    right_shape = _base._find_first_valid_image_shape(episode_dirs, "rightImg")
    user_features = _build_features(top_shape, right_shape, use_videos=args.use_videos)
    features = {**user_features, **_base.DEFAULT_FEATURES}
    hf_features = _base.get_hf_features_from_features(features)

    output_root = Path(args.output_root)
    if output_root.exists() and any(output_root.iterdir()):
        if args.overwrite_output:
            shutil.rmtree(output_root)
        else:
            raise FileExistsError(
                f"output_root already exists and is not empty: {output_root}. Use --overwrite_output to replace it."
            )
    output_root.mkdir(parents=True, exist_ok=True)

    camera_keys = [key for key, feature in features.items() if feature["dtype"] in ("image", "video")]
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
        right_dir = episode_dir / "rightImg"
        if not (obs_dir.is_dir() and top_dir.is_dir() and right_dir.is_dir()):
            print(f"[Skip] Missing observation/topImg/rightImg folders: {episode_dir}")
            continue

        obs_ids = _base._sorted_frame_ids(obs_dir, ["*.pkl"])
        top_ids = _base._sorted_frame_ids(top_dir, ["*.jpg", "*.jpeg", "*.png"])
        right_ids = _base._sorted_frame_ids(right_dir, ["*.jpg", "*.jpeg", "*.png"])
        common_ids = sorted(set(obs_ids) & set(top_ids) & set(right_ids))
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
        image_paths: dict[str, list[str]] = {key: [] for key in camera_keys}
        source_paths: dict[str, list[str]] = {key: [] for key in camera_keys}
        skipped_bad = 0

        for raw_idx in frame_ids:
            written_frame_paths: list[Path] = []
            try:
                with (obs_dir / f"{raw_idx}.pkl").open("rb") as file_obj:
                    payload = pickle.load(file_obj)

                action = _select_right_arm(payload["control"], field_name="control")
                state = _select_right_arm(payload["joint_positions"], field_name="joint_positions")
                top_path = _base._resolve_image_path(top_dir, raw_idx)
                right_path = _base._resolve_image_path(right_dir, raw_idx)

                frame_index = len(actions)
                cam_inputs = {
                    "observation.images.top": top_path,
                    "observation.images.right_wrist": right_path,
                }
                cam_images = {cam_key: _base._read_rgb(src_path) for cam_key, src_path in cam_inputs.items()}
                frame_image_paths: dict[str, Path] = {}
                for cam_key, image in cam_images.items():
                    dst = _base._frame_image_path(output_root, episode_index, cam_key, frame_index)
                    _base._write_png_atomic(dst, image)
                    written_frame_paths.append(dst)
                    frame_image_paths[cam_key] = dst

                # Commit the frame to the in-memory episode only after both camera
                # images have been decoded and written successfully.
                for cam_key, src_path in cam_inputs.items():
                    dst = frame_image_paths[cam_key]
                    image_paths[cam_key].append(str(dst))
                    source_paths[cam_key].append(str(src_path))

                actions.append(action)
                states.append(state)
                timestamps.append(frame_index / float(args.fps))
            except Exception as exc:
                for written_path in written_frame_paths:
                    written_path.unlink(missing_ok=True)
                skipped_bad += 1
                if args.skip_bad_frames:
                    print(f"[Warn] Skip bad frame {episode_dir.name}/{raw_idx}: {type(exc).__name__}: {exc}")
                    continue
                raise

        episode_length = len(actions)
        if episode_length < args.min_frames:
            print(
                f"[Skip] Too few valid frames after filtering ({episode_length}), "
                f"skipped_bad={skipped_bad}: {episode_dir.name}"
            )
            _base._cleanup_temp_images(output_root, episode_index, camera_keys)
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

        repaired_count = _base._repair_episode_images(
            image_paths_by_key=image_paths,
            source_paths_by_key=source_paths,
            repair_retries=args.repair_retries,
        )
        if repaired_count > 0:
            print(f"[Warn] Repaired {repaired_count} temporary frame files: {episode_dir.name}")

        parquet_payload = {key: episode_buffer[key] for key in hf_features}
        ep_hf_ds = _base.datasets.Dataset.from_dict(parquet_payload, features=hf_features, split="train")
        if not args.use_videos:
            ep_hf_ds = _base.embed_images(ep_hf_ds)

        ep_data_path = _base._episode_data_path(output_root, episode_index)
        ep_data_path.parent.mkdir(parents=True, exist_ok=True)
        ep_hf_ds.to_parquet(ep_data_path)

        ep_stats = _base.compute_episode_stats(episode_buffer, features)
        all_episode_stats.append(ep_stats)
        episodes_stats_rows.append(
            {
                "episode_index": episode_index,
                "stats": _base.serialize_dict(ep_stats),
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
                ep_video_path = _base._episode_video_path(output_root, episode_index, cam_key)
                for attempt in range(args.encode_retries + 1):
                    try:
                        ok, encode_err = _base._encode_video_once(
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
                        repaired = _base._repair_episode_images(
                            image_paths_by_key={cam_key: image_paths[cam_key]},
                            source_paths_by_key={cam_key: source_paths[cam_key]},
                            repair_retries=args.repair_retries,
                        )
                        print(
                            f"[Warn] Video encode retry {attempt + 1}/{args.encode_retries} "
                            f"for {episode_dir.name}/{cam_key}, repaired={repaired}, reason={encode_err}"
                        )
                    except OSError as exc:
                        is_stream_error = "broken data stream" in str(exc).lower()
                        if not is_stream_error or attempt >= args.encode_retries:
                            raise
                        repaired = _base._repair_episode_images(
                            image_paths_by_key={cam_key: image_paths[cam_key]},
                            source_paths_by_key={cam_key: source_paths[cam_key]},
                            repair_retries=args.repair_retries,
                        )
                        print(
                            f"[Warn] Video encode retry {attempt + 1}/{args.encode_retries} "
                            f"for {episode_dir.name}/{cam_key}, repaired={repaired}, reason={type(exc).__name__}"
                        )
                total_videos += 1
                if "info" not in features[cam_key]:
                    features[cam_key]["info"] = _base._get_video_info_cv2(ep_video_path)

            if not args.keep_images_for_video:
                _base._cleanup_temp_images(output_root, episode_index, camera_keys)

        total_frames += episode_length
        saved_episodes += 1
        print(
            f"[OK] Saved episode {saved_episodes}: {episode_dir.name} "
            f"({episode_length} frames, skipped_bad={skipped_bad})"
        )

    if saved_episodes == 0:
        raise RuntimeError("No episodes converted. Please check raw data folders and frame files.")

    dataset_stats = _base.aggregate_stats(all_episode_stats)
    info = {
        "codebase_version": _base.CODEBASE_VERSION,
        "robot_type": args.robot_type,
        "total_episodes": saved_episodes,
        "total_frames": total_frames,
        "total_tasks": len(tasks_rows),
        "total_videos": total_videos,
        "total_chunks": (saved_episodes + _base.CHUNK_SIZE - 1) // _base.CHUNK_SIZE,
        "chunks_size": _base.CHUNK_SIZE,
        "fps": int(args.fps),
        "splits": {"train": f"0:{saved_episodes}"},
        "data_path": _base.LEGACY_DATA_PATH,
        "video_path": _base.LEGACY_VIDEO_PATH if args.use_videos else None,
        "features": features,
    }

    _base._write_json(output_root / _base.LEGACY_INFO_PATH, info)
    _base._write_json(output_root / _base.LEGACY_STATS_PATH, _base.serialize_dict(dataset_stats))
    _base._write_jsonl(output_root / _base.LEGACY_TASKS_PATH, tasks_rows)
    _base._write_jsonl(output_root / _base.LEGACY_EPISODES_PATH, episodes_rows)
    _base._write_jsonl(output_root / _base.LEGACY_EPISODES_STATS_PATH, episodes_stats_rows)

    print(f"Done. Converted right-arm episodes: {saved_episodes}")
    print(f"LeRobot v2.1 dataset root: {output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert bimanual X-Trainer raw data to a right-arm LeRobot v2.1 dataset."
    )
    parser.add_argument("--raw_root", type=str, required=True, help="Path to raw collect_data directory.")
    parser.add_argument("--output_root", type=str, required=True, help="Output path for LeRobot dataset root.")
    parser.add_argument("--robot_type", type=str, default="dobot_xtrainer_right_arm")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--task", type=str, default="Insert the test tube on the desktop into the rack.")
    parser.add_argument("--use_videos", dest="use_videos", action="store_true")
    parser.add_argument("--no_videos", dest="use_videos", action="store_false")
    parser.set_defaults(use_videos=True)
    parser.add_argument("--vcodec", type=str, default="h264", choices=["h264", "hevc", "libsvtav1"])
    parser.add_argument("--repair_retries", type=int, default=2)
    parser.add_argument("--encode_retries", type=int, default=1)
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
