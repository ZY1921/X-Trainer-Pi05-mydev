# ruff: noqa: SLF001

import json
import pickle
from types import SimpleNamespace

import datasets
import numpy as np
from PIL import Image
import pytest

from examples.xtrainer_real import convert_raw_right_arm_to_lerobot_2_1 as converter


def test_select_right_arm_uses_source_dimensions_7_through_13():
    raw = np.arange(14, dtype=np.float64)
    selected = converter._select_right_arm(raw, field_name="control")

    assert selected.dtype == np.float32
    np.testing.assert_array_equal(selected, np.arange(7, 14, dtype=np.float32))


def test_select_right_arm_rejects_non_bimanual_source():
    with pytest.raises(ValueError, match="length 14"):
        converter._select_right_arm(np.zeros(7), field_name="joint_positions")


def test_right_arm_features_contain_only_top_and_right_wrist():
    features = converter._build_features((480, 640, 3), (480, 640, 3), use_videos=True)

    assert set(features) == {
        "action",
        "observation.state",
        "observation.images.top",
        "observation.images.right_wrist",
    }
    assert features["action"]["shape"] == (7,)
    assert features["observation.state"]["shape"] == (7,)
    assert features["action"]["names"][-1] == "right_gripper.pos"


def test_conversion_succeeds_without_left_camera_folder(tmp_path):
    raw_root = tmp_path / "raw"
    episode = raw_root / "episode_0"
    obs_dir = episode / "observation"
    top_dir = episode / "topImg"
    right_dir = episode / "rightImg"
    for directory in (obs_dir, top_dir, right_dir):
        directory.mkdir(parents=True)

    for frame_id in (1, 2):
        state = np.arange(14, dtype=np.float32) + frame_id * 100
        action = np.arange(14, dtype=np.float32) + frame_id * 1000
        with (obs_dir / f"{frame_id}.pkl").open("wb") as file_obj:
            pickle.dump({"joint_positions": state, "control": action}, file_obj)
        Image.fromarray(np.full((12, 16, 3), frame_id, dtype=np.uint8)).save(top_dir / f"{frame_id}.jpg")
        Image.fromarray(np.full((12, 16, 3), frame_id + 10, dtype=np.uint8)).save(right_dir / f"{frame_id}.jpg")

    output_root = tmp_path / "converted"
    converter.convert(
        SimpleNamespace(
            raw_root=str(raw_root),
            output_root=str(output_root),
            robot_type="dobot_xtrainer_right_arm",
            fps=30,
            task="test task",
            use_videos=False,
            overwrite_output=False,
            min_frames=2,
            skip_first_frames=0,
            skip_bad_frames=False,
            repair_retries=1,
            encode_retries=0,
            vcodec="h264",
            encode_in_subprocess=True,
            quiet_encoder=True,
            keep_images_for_video=False,
        )
    )

    info = json.loads((output_root / "meta/info.json").read_text())
    assert "observation.images.left_wrist" not in info["features"]
    assert {key for key in info["features"] if key.startswith("observation.images.")} == {
        "observation.images.top",
        "observation.images.right_wrist",
    }
    dataset = datasets.Dataset.from_parquet(str(output_root / "data/chunk-000/episode_000000.parquet"))
    np.testing.assert_array_equal(dataset[0]["observation.state"], np.arange(7, 14) + 100)
    np.testing.assert_array_equal(dataset[0]["action"], np.arange(7, 14) + 1000)


def test_skipped_partial_camera_frame_does_not_leave_misaligned_images(tmp_path, monkeypatch):
    raw_root = tmp_path / "raw"
    episode = raw_root / "episode_0"
    obs_dir = episode / "observation"
    top_dir = episode / "topImg"
    right_dir = episode / "rightImg"
    for directory in (obs_dir, top_dir, right_dir):
        directory.mkdir(parents=True)

    for frame_id in (1, 2, 3):
        values = np.arange(14, dtype=np.float32) + frame_id * 100
        with (obs_dir / f"{frame_id}.pkl").open("wb") as file_obj:
            pickle.dump({"joint_positions": values, "control": values}, file_obj)
        Image.fromarray(np.full((12, 16, 3), frame_id, dtype=np.uint8)).save(top_dir / f"{frame_id}.jpg")
        Image.fromarray(np.full((12, 16, 3), frame_id + 10, dtype=np.uint8)).save(right_dir / f"{frame_id}.jpg")

    original_write = converter._base._write_png_atomic
    failed_once = False

    def fail_first_right_write(path, image):
        nonlocal failed_once
        if not failed_once and "right_wrist" in str(path):
            failed_once = True
            raise OSError("simulated right camera write failure")
        original_write(path, image)

    monkeypatch.setattr(converter._base, "_write_png_atomic", fail_first_right_write)
    output_root = tmp_path / "converted"
    converter.convert(
        SimpleNamespace(
            raw_root=str(raw_root),
            output_root=str(output_root),
            robot_type="dobot_xtrainer_right_arm",
            fps=30,
            task="test task",
            use_videos=False,
            overwrite_output=False,
            min_frames=2,
            skip_first_frames=0,
            skip_bad_frames=True,
            repair_retries=1,
            encode_retries=0,
            vcodec="h264",
            encode_in_subprocess=True,
            quiet_encoder=True,
            keep_images_for_video=False,
        )
    )

    dataset = datasets.Dataset.from_parquet(str(output_root / "data/chunk-000/episode_000000.parquet"))
    assert len(dataset) == 2
    np.testing.assert_array_equal(dataset[0]["observation.state"], np.arange(7, 14) + 200)
    for camera in ("observation.images.top", "observation.images.right_wrist"):
        assert len(list((output_root / "images" / camera / "episode_000000").glob("*.png"))) == 2
