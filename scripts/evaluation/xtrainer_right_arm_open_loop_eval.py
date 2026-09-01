"""Open-loop evaluation entry point for the 7D X-Trainer right-arm policy.

This keeps the generic/legacy evaluator unchanged while pinning the dataset,
camera, normalization, and model-space assumptions used by the single-right-arm
training and deployment pipeline.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import tyro

from openpi.training import config as _config

if __package__:
    from scripts.evaluation import open_loop_eval as _base
else:
    import open_loop_eval as _base

PHYSICAL_ACTION_DIM = 7
CONFIG_NAME = "pi05_xtrainer_right_arm_lora_finetune"
ASSET_ID = "xtrainer_right_arm"
CAMERA_KEYS = ["observation.images.top", "observation.images.right_wrist"]


@dataclasses.dataclass
class Args:
    """Configuration for single-right-arm baseline and RTC open-loop evaluation."""

    dataset_path: str = "/home/user/Dobot/Xtrainer_dataset/right_arm_red_101_lerobot"
    checkpoint_path: str = "/home/user/Dobot/checkpoints/pi05_right_lora/19999"
    dataset_repo_id: str = "local/xtrainer_right_arm_open_loop"

    traj_ids: list[int] = dataclasses.field(default_factory=lambda: [0, 50, 100])
    steps: int = 200
    execution_horizon: int = 15

    rtc_enabled: bool = True
    rtc_inference_delay: int = 6
    rtc_execution_horizon: int = 15
    rtc_max_guidance_weight: float = 10.0
    rtc_prefix_attention_schedule: str = "linear"

    denoising_steps: int = 10
    seed: int = 42
    prompt: str | None = None
    expected_dataset_fps: int = 22

    metrics_output_dir: str = "output/xtrainer_right_arm_open_loop"
    overwrite_output: bool = False


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}, got {type(payload).__name__}.")
    return payload


def _validate_dataset(dataset_path: Path, *, expected_fps: int, traj_ids: list[int]) -> dict[str, Any]:
    info = _read_json(dataset_path / "meta" / "info.json")
    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"Expected a LeRobot v2.1 dataset, got {info.get('codebase_version')!r}.")
    if info.get("fps") != expected_fps:
        raise ValueError(f"Expected dataset FPS {expected_fps}, got {info.get('fps')!r}.")

    features = info.get("features")
    if not isinstance(features, dict):
        raise TypeError("Dataset info.json does not contain a features object.")
    for key in ("observation.state", "action"):
        feature = features.get(key)
        if not isinstance(feature, dict) or feature.get("shape") != [PHYSICAL_ACTION_DIM]:
            raise ValueError(f"Dataset feature {key!r} must have shape [{PHYSICAL_ACTION_DIM}], got {feature}.")

    camera_keys = [key for key in features if key.startswith("observation.images.")]
    if camera_keys != CAMERA_KEYS:
        raise ValueError(f"Expected camera keys {CAMERA_KEYS}, got {camera_keys}.")

    total_episodes = info.get("total_episodes")
    if not isinstance(total_episodes, int) or total_episodes <= 0:
        raise ValueError(f"Invalid total_episodes in dataset metadata: {total_episodes!r}.")
    invalid_traj_ids = [traj_id for traj_id in traj_ids if not 0 <= traj_id < total_episodes]
    if invalid_traj_ids:
        raise ValueError(f"Trajectory IDs are outside [0, {total_episodes - 1}]: {invalid_traj_ids}.")
    return info


def _validate_checkpoint(checkpoint_path: Path) -> None:
    required_paths = [
        checkpoint_path / "_CHECKPOINT_METADATA",
        checkpoint_path / "params" / "_METADATA",
        checkpoint_path / "params" / "manifest.ocdbt",
    ]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoint is incomplete; missing: {missing}")

    stats_path = checkpoint_path / "assets" / ASSET_ID / "norm_stats.json"
    stats_payload = _read_json(stats_path)
    norm_stats = stats_payload.get("norm_stats")
    if not isinstance(norm_stats, dict):
        raise TypeError(f"Missing norm_stats object in {stats_path}.")
    for key in ("state", "actions"):
        key_stats = norm_stats.get(key)
        if not isinstance(key_stats, dict):
            raise TypeError(f"Missing norm_stats[{key!r}] in {stats_path}.")
        for statistic in ("mean", "std", "q01", "q99"):
            values = key_stats.get(statistic)
            if not isinstance(values, list) or len(values) != PHYSICAL_ACTION_DIM:
                raise ValueError(
                    f"norm_stats[{key!r}][{statistic!r}] must contain {PHYSICAL_ACTION_DIM} values, got {values}."
                )


def _validate_training_config() -> None:
    train_config = _config.get_config(CONFIG_NAME)
    expected_metadata = {
        "robot_side": "right",
        "physical_action_dim": PHYSICAL_ACTION_DIM,
        "model_action_start_index": 7,
        "model_action_end_index": 14,
        "camera_keys": CAMERA_KEYS,
    }
    mismatches = {
        key: (train_config.policy_metadata.get(key), expected)
        for key, expected in expected_metadata.items()
        if train_config.policy_metadata.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Training config {CONFIG_NAME!r} is incompatible: {mismatches}")


def _prepare_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {path}. Pass --overwrite-output to reuse it.")
    path.mkdir(parents=True, exist_ok=True)


def main(args: Args) -> None:
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    output_dir = Path(args.metrics_output_dir).expanduser().resolve()

    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_path}")
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_path}")
    _validate_dataset(dataset_path, expected_fps=args.expected_dataset_fps, traj_ids=args.traj_ids)
    _validate_checkpoint(checkpoint_path)
    _validate_training_config()
    _prepare_output_dir(output_dir, overwrite=args.overwrite_output)

    base_args = _base.ArgsConfig(
        steps=args.steps,
        traj_ids=args.traj_ids,
        execution_horizon=args.execution_horizon,
        rtc_enabled=args.rtc_enabled,
        rtc_inference_delay=args.rtc_inference_delay,
        rtc_execution_horizon=args.rtc_execution_horizon,
        rtc_max_guidance_weight=args.rtc_max_guidance_weight,
        rtc_prefix_attention_schedule=args.rtc_prefix_attention_schedule,
        action_start_index=0,
        seed=args.seed,
        dataset_path=str(dataset_path),
        dataset_repo_id=args.dataset_repo_id,
        config_name=CONFIG_NAME,
        model_path=str(checkpoint_path),
        asset_id=ASSET_ID,
        denoising_steps=args.denoising_steps,
        prompt=args.prompt,
        camera_keys=CAMERA_KEYS,
        save_plot_path=str(output_dir / "trajectory_{traj_id}.jpeg"),
        metrics_output_dir=str(output_dir),
    )
    _base.main(base_args)


if __name__ == "__main__":
    main(tyro.cli(Args))
