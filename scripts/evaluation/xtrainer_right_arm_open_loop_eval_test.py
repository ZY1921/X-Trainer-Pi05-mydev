# ruff: noqa: SLF001

import json
from pathlib import Path

import pytest

from scripts.evaluation import xtrainer_right_arm_open_loop_eval as right_arm_eval


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _dataset_info() -> dict:
    return {
        "codebase_version": "v2.1",
        "fps": 22,
        "total_episodes": 3,
        "features": {
            "observation.state": {"shape": [7]},
            "action": {"shape": [7]},
            "observation.images.top": {"shape": [480, 640, 3], "dtype": "video"},
            "observation.images.right_wrist": {"shape": [480, 640, 3], "dtype": "video"},
        },
    }


def _norm_stats(dim: int = 7) -> dict:
    values = [0.0] * dim
    return {"norm_stats": {key: dict.fromkeys(("mean", "std", "q01", "q99"), values) for key in ("state", "actions")}}


def test_validates_single_right_arm_dataset(tmp_path):
    _write_json(tmp_path / "meta/info.json", _dataset_info())

    info = right_arm_eval._validate_dataset(tmp_path, expected_fps=22, traj_ids=[0, 2])

    assert info["total_episodes"] == 3


def test_rejects_legacy_left_camera_and_wrong_action_dimension(tmp_path):
    info = _dataset_info()
    info["features"]["action"]["shape"] = [14]
    info["features"]["observation.images.left_wrist"] = {"shape": [480, 640, 3], "dtype": "video"}
    _write_json(tmp_path / "meta/info.json", info)

    with pytest.raises(ValueError, match=r"must have shape \[7\]"):
        right_arm_eval._validate_dataset(tmp_path, expected_fps=22, traj_ids=[0])


def test_rejects_non_7d_checkpoint_stats(tmp_path):
    for relative_path in ("_CHECKPOINT_METADATA", "params/_METADATA", "params/manifest.ocdbt"):
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    _write_json(tmp_path / "assets/xtrainer_right_arm/norm_stats.json", _norm_stats(dim=14))

    with pytest.raises(ValueError, match="must contain 7 values"):
        right_arm_eval._validate_checkpoint(tmp_path)


def test_main_builds_right_arm_base_configuration(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "output"
    _write_json(dataset / "meta/info.json", _dataset_info())
    for relative_path in ("_CHECKPOINT_METADATA", "params/_METADATA", "params/manifest.ocdbt"):
        path = checkpoint / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    _write_json(checkpoint / "assets/xtrainer_right_arm/norm_stats.json", _norm_stats())

    captured = []
    monkeypatch.setattr(right_arm_eval._base, "main", captured.append)
    right_arm_eval.main(
        right_arm_eval.Args(
            dataset_path=str(dataset),
            checkpoint_path=str(checkpoint),
            metrics_output_dir=str(output),
            traj_ids=[0, 2],
        )
    )

    assert len(captured) == 1
    base_args = captured[0]
    assert base_args.config_name == "pi05_xtrainer_right_arm_lora_finetune"
    assert base_args.asset_id == "xtrainer_right_arm"
    assert base_args.action_start_index == 0
    assert base_args.camera_keys == ["observation.images.top", "observation.images.right_wrist"]
