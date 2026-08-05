import csv
import dataclasses
import json
import logging
from pathlib import Path
from typing import Any
import warnings

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from matplotlib import pyplot as plt
import numpy as np
import tyro

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

warnings.simplefilter("ignore", category=FutureWarning)

"""
Run open-loop inference on LeRobot episodes and compare predicted actions with
ground-truth actions.

Provide --model-path to load a checkpoint in this process. If --model-path is
omitted, the script connects to an OpenPI policy server using --host/--port.
"""


@dataclasses.dataclass(frozen=True)
class ErrorMetrics:
    mse: float
    mae: float
    rmse: float
    error_std: float
    max_abs_error: float

    @classmethod
    def from_errors(cls, errors: np.ndarray) -> "ErrorMetrics":
        return cls(
            mse=float(np.mean(errors**2)),
            mae=float(np.mean(np.abs(errors))),
            rmse=float(np.sqrt(np.mean(errors**2))),
            error_std=float(np.std(errors)),
            max_abs_error=float(np.max(np.abs(errors))),
        )

    def as_dict(self) -> dict[str, float]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class TrajectoryEvaluation:
    policy_mode: str
    traj_id: int
    steps: int
    num_inferences: int
    pred_actions: np.ndarray
    errors: np.ndarray
    metrics: ErrorMetrics
    continuity: "ContinuityMetrics"


@dataclasses.dataclass(frozen=True)
class ContinuityMetrics:
    switch_count: int
    boundary_action_jump_mae: float
    boundary_velocity_error_mae: float
    boundary_acceleration_error_mae: float
    overlap_mae: float
    inference_mean_ms: float
    inference_p95_ms: float

    def as_dict(self) -> dict[str, int | float]:
        return dataclasses.asdict(self)


def plot_trajectory_results(
    state_joints_across_time: np.ndarray,
    gt_action_across_time: np.ndarray,
    baseline_action_across_time: np.ndarray,
    rtc_action_across_time: np.ndarray | None,
    traj_id: int,
    state_keys: list[str],
    action_keys: list[str],
    execution_horizon: int,
    action_start_index: int,
    save_plot_path: str,
) -> None:
    """Plot state, ground-truth actions, baseline predictions, and RTC predictions."""
    actual_steps = len(gt_action_across_time)
    action_dim = gt_action_across_time.shape[1]
    indices_to_plot = list(range(action_start_index, action_dim))

    num_plots = len(indices_to_plot)
    if num_plots == 0:
        logging.warning("No valid indices to plot")
        return

    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(8, 4 * num_plots))
    if num_plots == 1:
        axes = [axes]

    for plot_idx, action_idx in enumerate(indices_to_plot):
        ax = axes[plot_idx]
        if state_joints_across_time.shape == gt_action_across_time.shape:
            ax.plot(state_joints_across_time[:, action_idx], color="tab:blue", label="state joints")
        ax.plot(gt_action_across_time[:, action_idx], color="tab:orange", label="gt action")
        ax.plot(
            baseline_action_across_time[:, action_idx],
            color="tab:green",
            label="baseline prediction",
        )
        if rtc_action_across_time is not None:
            ax.plot(rtc_action_across_time[:, action_idx], color="tab:purple", label="RTC prediction")

        for step in range(0, actual_steps, execution_horizon):
            ax.plot(
                step,
                gt_action_across_time[step, action_idx],
                "ro",
                label="inference point" if step == 0 else None,
            )

        ax.set_title(f"Action {action_idx}")
        ax.legend()

    plt.tight_layout()
    Path(save_plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_plot_path)
    plt.close()


def _to_numpy(value: Any) -> np.ndarray:
    """Convert a LeRobot value to a NumPy array without requiring torch here."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _make_observation(
    sample: dict[str, Any],
    *,
    state_key: str,
    camera_keys: list[str],
    prompt: str | None,
) -> dict[str, Any]:
    observation = {
        state_key: _to_numpy(sample[state_key]),
        **{key: _to_numpy(sample[key]) for key in camera_keys},
    }
    observation["prompt"] = prompt or sample["task"]
    return observation


def _resolve_plot_path(save_plot_path: str | None, traj_id: int) -> str:
    if save_plot_path is None:
        return f"/tmp/open_loop_eval/traj_{traj_id}.jpeg"
    if "{traj_id}" in save_plot_path:
        return save_plot_path.format(traj_id=traj_id)
    return save_plot_path


def _validate_action_chunk(action_chunk: np.ndarray, policy_name: str) -> None:
    if action_chunk.ndim != 2:
        raise ValueError(f"Expected {policy_name} actions with shape (horizon, action_dim), got {action_chunk.shape}")


def _make_policy_noise(policy: Any, rng: np.random.Generator) -> np.ndarray | None:
    """Create explicit model-space noise so baseline and RTC are directly comparable."""
    if not hasattr(policy, "action_horizon") or not hasattr(policy, "action_dim"):
        return None
    return rng.standard_normal((policy.action_horizon, policy.action_dim)).astype(np.float32)


def _infer_baseline(policy: Any, observation: dict[str, Any], noise: np.ndarray | None) -> dict[str, Any]:
    if noise is None:
        return policy.infer(observation)
    return policy.infer(observation, noise=noise)


def _mean_or_zero(values: np.ndarray | list[float]) -> float:
    return float(np.mean(values)) if len(values) else 0.0


def _continuity_metrics(
    pred_actions: np.ndarray,
    gt_actions: np.ndarray,
    switch_steps: list[int],
    overlap_errors: list[np.ndarray],
    inference_times_ms: list[float],
) -> ContinuityMetrics:
    valid_switches = np.asarray([step for step in switch_steps if 0 < step < len(pred_actions)], dtype=np.int64)
    if len(valid_switches):
        pred_velocity = pred_actions[valid_switches] - pred_actions[valid_switches - 1]
        gt_velocity = gt_actions[valid_switches] - gt_actions[valid_switches - 1]
        action_jump = np.abs(pred_velocity)
        velocity_error = np.abs(pred_velocity - gt_velocity)

        acceleration_switches = valid_switches[valid_switches >= 2]
        pred_acceleration = (
            pred_actions[acceleration_switches]
            - 2 * pred_actions[acceleration_switches - 1]
            + pred_actions[acceleration_switches - 2]
        )
        gt_acceleration = (
            gt_actions[acceleration_switches]
            - 2 * gt_actions[acceleration_switches - 1]
            + gt_actions[acceleration_switches - 2]
        )
        acceleration_error = np.abs(pred_acceleration - gt_acceleration)
    else:
        action_jump = np.asarray([])
        velocity_error = np.asarray([])
        acceleration_error = np.asarray([])

    overlap_values = np.concatenate([np.abs(error).reshape(-1) for error in overlap_errors]) if overlap_errors else []
    inference_times = np.asarray(inference_times_ms, dtype=np.float64)
    return ContinuityMetrics(
        switch_count=len(valid_switches),
        boundary_action_jump_mae=_mean_or_zero(action_jump),
        boundary_velocity_error_mae=_mean_or_zero(velocity_error),
        boundary_acceleration_error_mae=_mean_or_zero(acceleration_error),
        overlap_mae=_mean_or_zero(overlap_values),
        inference_mean_ms=_mean_or_zero(inference_times),
        inference_p95_ms=float(np.percentile(inference_times, 95)) if len(inference_times) else 0.0,
    )


def evaluate_single_trajectory(
    policy: Any,
    dataset: LeRobotDataset,
    episode_indices: np.ndarray,
    traj_id: int,
    *,
    state_key: str,
    action_key: str,
    camera_keys: list[str],
    prompt: str | None = None,
    steps: int = 300,
    execution_horizon: int = 16,
    rtc_enabled: bool = True,
    rtc_inference_delay: int = 4,
    rtc_execution_horizon: int = 10,
    rtc_max_guidance_weight: float = 10.0,
    rtc_prefix_attention_schedule: str = "exp",
    action_start_index: int = 7,
    seed: int = 42,
    save_plot_path: str | None = None,
) -> list[TrajectoryEvaluation]:
    trajectory_indices = np.flatnonzero(episode_indices == traj_id)
    traj_length = len(trajectory_indices)
    actual_steps = min(steps, traj_length)
    logging.info(
        "Using %d steps (requested: %d, trajectory length: %d)",
        actual_steps,
        steps,
        traj_length,
    )

    state_joints_across_time = []
    gt_action_across_time = []
    for dataset_index in trajectory_indices[:actual_steps]:
        raw_step = dataset.hf_dataset[int(dataset_index)]
        state_joints_across_time.append(_to_numpy(raw_step[state_key]))
        gt_action_across_time.append(_to_numpy(raw_step[action_key]))

    state_joints = np.stack(state_joints_across_time)
    gt_actions = np.stack(gt_action_across_time)
    if not 0 <= action_start_index < gt_actions.shape[1]:
        raise ValueError(f"action_start_index must be in [0, {gt_actions.shape[1] - 1}], got {action_start_index}.")
    metric_gt_actions = gt_actions[:, action_start_index:]
    rng = np.random.default_rng(seed + traj_id)

    def observation_at(step_count: int) -> dict[str, Any]:
        dataset_index = int(trajectory_indices[step_count])
        sample = dataset[dataset_index]
        return _make_observation(
            sample,
            state_key=state_key,
            camera_keys=camera_keys,
            prompt=prompt,
        )

    logging.info("inferencing at step: 0 (initial shared chunk)")
    initial_noise = _make_policy_noise(policy, rng)
    initial_result = _infer_baseline(policy, observation_at(0), initial_noise)
    initial_chunk = np.asarray(initial_result["actions"])
    _validate_action_chunk(initial_chunk, "initial")
    required_chunk_steps = execution_horizon + rtc_inference_delay if rtc_enabled else execution_horizon
    if initial_chunk.shape[0] < required_chunk_steps:
        raise ValueError(
            f"The policy action chunk must contain at least {required_chunk_steps} steps, got {initial_chunk.shape[0]}."
        )

    baseline_active_chunk = initial_chunk
    baseline_action_index = 0
    pending_baseline_chunk: tuple[int, np.ndarray] | None = None
    rtc_active_chunk = initial_chunk.copy()
    rtc_action_index = 0
    pending_rtc_chunk: tuple[int, np.ndarray] | None = None

    baseline_predictions: list[np.ndarray] = []
    rtc_predictions: list[np.ndarray] = []
    baseline_switch_steps = [0]
    rtc_switch_steps = [0]
    baseline_overlap_errors: list[np.ndarray] = []
    rtc_overlap_errors: list[np.ndarray] = []
    initial_inference_ms = float(initial_result.get("policy_timing", {}).get("infer_ms", 0.0))
    baseline_inference_times = [initial_inference_ms]
    rtc_inference_times = [initial_inference_ms]
    num_inferences = 1

    for step_count in range(actual_steps):
        if step_count > 0 and step_count % execution_horizon == 0:
            logging.info("inferencing at step: %d", step_count)
            observation = observation_at(step_count)
            noise = _make_policy_noise(policy, rng)

            baseline_previous = baseline_active_chunk[baseline_action_index:]
            baseline_result = _infer_baseline(policy, observation, noise)
            generated_baseline_chunk = np.asarray(baseline_result["actions"])
            _validate_action_chunk(generated_baseline_chunk, "baseline")
            if rtc_enabled:
                if pending_baseline_chunk is not None:
                    raise RuntimeError(
                        "A new baseline inference started before the previous simulated inference finished. "
                        "Use rtc_inference_delay < execution_horizon."
                    )
                pending_baseline_chunk = (step_count + rtc_inference_delay, generated_baseline_chunk)
            else:
                baseline_active_chunk = generated_baseline_chunk
                baseline_action_index = 0
                baseline_switch_steps.append(step_count)
            baseline_inference_times.append(float(baseline_result.get("policy_timing", {}).get("infer_ms", 0.0)))
            baseline_overlap_steps = min(len(baseline_previous), len(generated_baseline_chunk), rtc_execution_horizon)
            if baseline_overlap_steps:
                baseline_overlap_errors.append(
                    (generated_baseline_chunk[:baseline_overlap_steps] - baseline_previous[:baseline_overlap_steps])[
                        :, action_start_index:
                    ]
                )

            if rtc_enabled:
                if pending_rtc_chunk is not None:
                    raise RuntimeError(
                        "A new RTC inference started before the previous simulated inference finished. "
                        "Use rtc_inference_delay < execution_horizon."
                    )
                rtc_previous = rtc_active_chunk[rtc_action_index:]
                rtc_result = policy.infer(
                    observation,
                    noise=noise,
                    prev_chunk_left_over=rtc_previous,
                    inference_delay=rtc_inference_delay,
                    execution_horizon=rtc_execution_horizon,
                    rtc_prefix_attention_schedule=rtc_prefix_attention_schedule,
                    rtc_max_guidance_weight=rtc_max_guidance_weight,
                )
                generated_rtc_chunk = np.asarray(rtc_result["actions"])
                _validate_action_chunk(generated_rtc_chunk, "RTC")
                pending_rtc_chunk = (step_count + rtc_inference_delay, generated_rtc_chunk)
                rtc_inference_times.append(float(rtc_result.get("policy_timing", {}).get("infer_ms", 0.0)))
                rtc_overlap_steps = min(len(rtc_previous), len(generated_rtc_chunk), rtc_execution_horizon)
                if rtc_overlap_steps:
                    rtc_overlap_errors.append(
                        (generated_rtc_chunk[:rtc_overlap_steps] - rtc_previous[:rtc_overlap_steps])[
                            :, action_start_index:
                        ]
                    )
            else:
                rtc_active_chunk = baseline_active_chunk
                rtc_action_index = 0

            num_inferences += 1

        if pending_baseline_chunk is not None and pending_baseline_chunk[0] == step_count:
            _, baseline_active_chunk = pending_baseline_chunk
            baseline_action_index = rtc_inference_delay
            baseline_switch_steps.append(step_count)
            pending_baseline_chunk = None

        if pending_rtc_chunk is not None and pending_rtc_chunk[0] == step_count:
            _, rtc_active_chunk = pending_rtc_chunk
            rtc_action_index = rtc_inference_delay
            rtc_switch_steps.append(step_count)
            pending_rtc_chunk = None

        if baseline_action_index >= len(baseline_active_chunk):
            raise RuntimeError(f"Baseline action chunk exhausted at trajectory step {step_count}.")
        baseline_predictions.append(baseline_active_chunk[baseline_action_index])
        baseline_action_index += 1

        if rtc_action_index >= len(rtc_active_chunk):
            raise RuntimeError(f"RTC action chunk exhausted at trajectory step {step_count}.")
        rtc_predictions.append(rtc_active_chunk[rtc_action_index])
        rtc_action_index += 1

    baseline_actions = np.asarray(baseline_predictions)
    rtc_actions = np.asarray(rtc_predictions)
    for policy_mode, predicted in (("baseline", baseline_actions), ("rtc", rtc_actions)):
        if gt_actions.shape != predicted.shape:
            raise ValueError(
                f"gt_action: {gt_actions.shape}, {policy_mode}_action: {predicted.shape}. "
                "Check that the dataset action dimensions match the policy output dimensions."
            )

    logging.info("state_joints vs time %s", state_joints.shape)
    logging.info("gt_action_joints vs time %s", gt_actions.shape)
    logging.info("baseline_action_joints vs time %s", baseline_actions.shape)
    if rtc_enabled:
        logging.info("rtc_action_joints vs time %s", rtc_actions.shape)

    plot_trajectory_results(
        state_joints_across_time=state_joints,
        gt_action_across_time=gt_actions,
        baseline_action_across_time=baseline_actions,
        rtc_action_across_time=rtc_actions if rtc_enabled else None,
        traj_id=traj_id,
        state_keys=[state_key],
        action_keys=[action_key],
        execution_horizon=execution_horizon,
        action_start_index=action_start_index,
        save_plot_path=_resolve_plot_path(save_plot_path, traj_id),
    )

    results = []
    result_inputs = [
        (
            "baseline",
            baseline_actions,
            baseline_switch_steps,
            baseline_overlap_errors,
            baseline_inference_times,
        )
    ]
    if rtc_enabled:
        result_inputs.append(("rtc", rtc_actions, rtc_switch_steps, rtc_overlap_errors, rtc_inference_times))

    for policy_mode, predicted, switch_steps, overlap_errors, inference_times in result_inputs:
        metric_predictions = predicted[:, action_start_index:]
        errors = metric_predictions - metric_gt_actions
        metrics = ErrorMetrics.from_errors(errors)
        continuity = _continuity_metrics(
            metric_predictions,
            metric_gt_actions,
            switch_steps,
            overlap_errors,
            inference_times,
        )
        logging.info(
            "%s trajectory %d: MSE=%s MAE=%s boundary_velocity_error=%s overlap_MAE=%s",
            policy_mode,
            traj_id,
            metrics.mse,
            metrics.mae,
            continuity.boundary_velocity_error_mae,
            continuity.overlap_mae,
        )
        results.append(
            TrajectoryEvaluation(
                policy_mode=policy_mode,
                traj_id=traj_id,
                steps=actual_steps,
                num_inferences=num_inferences,
                pred_actions=metric_predictions,
                errors=errors,
                metrics=metrics,
                continuity=continuity,
            )
        )
    return results


def _mean_trajectory_metrics(results: list[TrajectoryEvaluation]) -> ErrorMetrics:
    return ErrorMetrics(
        **{
            field.name: float(np.mean([getattr(result.metrics, field.name) for result in results]))
            for field in dataclasses.fields(ErrorMetrics)
        }
    )


def _get_action_names(feature: dict[str, Any], action_dim: int) -> list[str]:
    names = feature.get("names")
    if isinstance(names, list) and len(names) == action_dim:
        return [str(name) for name in names]
    return [f"action_{index}" for index in range(action_dim)]


def _metric_row(
    policy_mode: str,
    trajectory: int | str,
    steps: int | str,
    num_inferences: int | str,
    metrics: ErrorMetrics,
) -> dict[str, int | float | str]:
    return {
        "policy": policy_mode,
        "trajectory": trajectory,
        "steps": steps,
        "inferences": num_inferences,
        **metrics.as_dict(),
    }


def _dimension_metric_rows(
    results: list[TrajectoryEvaluation],
    action_names: list[str],
    action_indices: list[int],
) -> list[dict[str, int | float | str]]:
    rows = []
    for result in results:
        for local_action_index, (action_index, action_name) in enumerate(
            zip(action_indices, action_names, strict=True)
        ):
            metrics = ErrorMetrics.from_errors(result.errors[:, local_action_index])
            rows.append(
                {
                    "policy": result.policy_mode,
                    "trajectory": result.traj_id,
                    "action_index": action_index,
                    "action_name": action_name,
                    "steps": result.steps,
                    **metrics.as_dict(),
                }
            )

    for policy_mode in dict.fromkeys(result.policy_mode for result in results):
        mode_results = [result for result in results if result.policy_mode == policy_mode]
        all_errors = np.concatenate([result.errors for result in mode_results], axis=0)
        for local_action_index, (action_index, action_name) in enumerate(
            zip(action_indices, action_names, strict=True)
        ):
            metrics = ErrorMetrics.from_errors(all_errors[:, local_action_index])
            rows.append(
                {
                    "policy": policy_mode,
                    "trajectory": "ALL_WEIGHTED",
                    "action_index": action_index,
                    "action_name": action_name,
                    "steps": len(all_errors),
                    **metrics.as_dict(),
                }
            )
    return rows


def _continuity_row(result: TrajectoryEvaluation) -> dict[str, int | float | str]:
    return {
        "policy": result.policy_mode,
        "trajectory": result.traj_id,
        **result.continuity.as_dict(),
    }


def _mean_continuity(results: list[TrajectoryEvaluation]) -> ContinuityMetrics:
    return ContinuityMetrics(
        switch_count=sum(result.continuity.switch_count for result in results),
        **{
            field.name: float(np.mean([getattr(result.continuity, field.name) for result in results]))
            for field in dataclasses.fields(ContinuityMetrics)
            if field.name != "switch_count"
        },
    )


def _format_value(value: int | float | str) -> str:
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def _format_table(headers: list[str], rows: list[list[int | float | str]]) -> str:
    formatted_rows = [[_format_value(value) for value in row] for row in rows]
    widths = [max(len(header), *(len(row[column]) for row in formatted_rows)) for column, header in enumerate(headers)]

    def format_row(row: list[str]) -> str:
        return " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))

    separator = "-+-".join("-" * width for width in widths)
    return "\n".join([format_row(headers), separator, *(format_row(row) for row in formatted_rows)])


def save_metrics(
    results: list[TrajectoryEvaluation],
    action_names: list[str],
    action_indices: list[int],
    args: "ArgsConfig",
) -> None:
    output_dir = Path(args.metrics_output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectory_rows = [
        _metric_row(result.policy_mode, result.traj_id, result.steps, result.num_inferences, result.metrics)
        for result in results
    ]
    continuity_rows = [_continuity_row(result) for result in results]
    aggregate_summary = {}
    for policy_mode in dict.fromkeys(result.policy_mode for result in results):
        mode_results = [result for result in results if result.policy_mode == policy_mode]
        average_metrics = _mean_trajectory_metrics(mode_results)
        all_errors = np.concatenate([result.errors for result in mode_results], axis=0)
        weighted_metrics = ErrorMetrics.from_errors(all_errors)
        total_steps = sum(result.steps for result in mode_results)
        total_inferences = sum(result.num_inferences for result in mode_results)
        average_continuity = _mean_continuity(mode_results)
        trajectory_rows.extend(
            [
                _metric_row(policy_mode, "AVERAGE", "-", "-", average_metrics),
                _metric_row(
                    policy_mode,
                    "ALL_WEIGHTED",
                    total_steps,
                    total_inferences,
                    weighted_metrics,
                ),
            ]
        )
        continuity_rows.append(
            {
                "policy": policy_mode,
                "trajectory": "AVERAGE",
                **average_continuity.as_dict(),
            }
        )
        aggregate_summary[policy_mode] = {
            "mean_across_trajectories": average_metrics.as_dict(),
            "all_steps_weighted": weighted_metrics.as_dict(),
            "continuity_mean_across_trajectories": average_continuity.as_dict(),
        }

    dimension_rows = _dimension_metric_rows(results, action_names, action_indices)

    trajectory_csv_path = output_dir / "trajectory_metrics.csv"
    dimension_csv_path = output_dir / "action_dimension_metrics.csv"
    continuity_csv_path = output_dir / "continuity_metrics.csv"
    summary_json_path = output_dir / "summary.json"

    with trajectory_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(trajectory_rows[0]))
        writer.writeheader()
        writer.writerows(trajectory_rows)

    with dimension_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(dimension_rows[0]))
        writer.writeheader()
        writer.writerows(dimension_rows)

    with continuity_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(continuity_rows[0]))
        writer.writeheader()
        writer.writerows(continuity_rows)

    summary = {
        "config": dataclasses.asdict(args),
        "metrics_definition": {
            "evaluated_action_indices": action_indices,
            "mse": "mean((predicted_action - ground_truth_action) ** 2) over evaluated action indices",
            "mae": "mean(abs(predicted_action - ground_truth_action)) over evaluated action indices",
            "rmse": "sqrt(mse)",
            "error_std": "standard deviation of signed action error",
            "max_abs_error": "maximum absolute action error",
            "boundary_action_jump_mae": "mean absolute predicted action jump at chunk switches",
            "boundary_velocity_error_mae": "MAE of predicted versus ground-truth first difference at switches",
            "boundary_acceleration_error_mae": "MAE of predicted versus ground-truth second difference at switches",
            "overlap_mae": "MAE between a newly generated chunk and the previous chunk at aligned prefix steps",
        },
        "trajectory_metrics": trajectory_rows,
        "action_dimension_metrics": dimension_rows,
        "continuity_metrics": continuity_rows,
        "aggregate_by_policy": aggregate_summary,
    }
    with summary_json_path.open("w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, indent=2, ensure_ascii=False)

    trajectory_table_rows = [
        [
            row["policy"],
            row["trajectory"],
            row["steps"],
            row["inferences"],
            row["mse"],
            row["mae"],
            row["rmse"],
            row["error_std"],
            row["max_abs_error"],
        ]
        for row in trajectory_rows
    ]
    logging.info(
        "Trajectory metrics table:\n%s",
        _format_table(
            ["policy", "trajectory", "steps", "infer", "MSE", "MAE", "RMSE", "error_std", "max_abs_error"],
            trajectory_table_rows,
        ),
    )

    aggregate_dimension_rows = [
        [
            row["policy"],
            row["action_index"],
            row["action_name"],
            row["mse"],
            row["mae"],
            row["rmse"],
            row["error_std"],
            row["max_abs_error"],
        ]
        for row in dimension_rows
        if row["trajectory"] == "ALL_WEIGHTED"
    ]
    logging.info(
        "Action dimension metrics table (all trajectories, step-weighted):\n%s",
        _format_table(
            ["policy", "index", "action_name", "MSE", "MAE", "RMSE", "error_std", "max_abs_error"],
            aggregate_dimension_rows,
        ),
    )
    continuity_table_rows = [
        [
            row["policy"],
            row["trajectory"],
            row["switch_count"],
            row["boundary_action_jump_mae"],
            row["boundary_velocity_error_mae"],
            row["boundary_acceleration_error_mae"],
            row["overlap_mae"],
            row["inference_mean_ms"],
            row["inference_p95_ms"],
        ]
        for row in continuity_rows
    ]
    logging.info(
        "Continuity metrics table:\n%s",
        _format_table(
            ["policy", "trajectory", "switches", "jump", "velocity_err", "accel_err", "overlap", "ms_mean", "ms_p95"],
            continuity_table_rows,
        ),
    )
    logging.info("Trajectory metrics CSV: %s", trajectory_csv_path)
    logging.info("Action dimension metrics CSV: %s", dimension_csv_path)
    logging.info("Continuity metrics CSV: %s", continuity_csv_path)
    logging.info("Metrics JSON: %s", summary_json_path)


@dataclasses.dataclass
class ArgsConfig:
    """Configuration for evaluating an OpenPI policy."""

    host: str = "127.0.0.1"
    """Policy server host. Used when --model-path is omitted."""

    port: int = 8000
    """Policy server port. Used when --model-path is omitted."""

    steps: int = 200
    """Maximum number of steps per trajectory."""

    traj_ids: list[int] = dataclasses.field(default_factory=lambda: [0])
    """Episode IDs to evaluate."""

    execution_horizon: int = 16
    """Number of control steps between inference starts."""

    rtc_enabled: bool = True
    """Run RTC alongside the same-noise baseline. RTC currently requires local JAX inference."""

    rtc_inference_delay: int = 4
    """Simulated inference delay in control steps; the old chunk executes during this interval."""

    rtc_execution_horizon: int = 10
    """Exclusive end of the RTC prefix transition region (LeRobot's execution_horizon)."""

    rtc_max_guidance_weight: float = 10.0
    """Maximum RTC denoising guidance weight."""

    rtc_prefix_attention_schedule: str = "exp"
    """RTC prefix schedule: exp, linear, ones, or zeros."""

    action_start_index: int = 7
    """First action dimension included in plots and all evaluation metrics."""

    seed: int = 42
    """Seed used to create identical flow noise for baseline and RTC."""

    dataset_path: str = ""
    """Path to a local LeRobot dataset."""

    dataset_repo_id: str = "local/open_loop_eval"
    """Synthetic repo ID used by LeRobot when loading a local dataset."""

    config_name: str = "pi05_xtrainer_custom"
    """OpenPI config matching the checkpoint model architecture and transforms."""

    model_path: str | None = None
    """Checkpoint path. Omit to use a running policy server."""

    asset_id: str | None = None
    """Norm-stats directory name under <checkpoint>/assets."""

    denoising_steps: int = 10
    """Number of flow-matching denoising steps for local inference."""

    prompt: str | None = None
    """Prompt override. By default, each episode's LeRobot task is used."""

    state_key: str = "observation.state"
    """Dataset state key."""

    action_key: str = "action"
    """Dataset ground-truth action key."""

    camera_keys: list[str] | None = None
    """Dataset camera keys. By default, all metadata camera keys are used."""

    save_plot_path: str | None = None
    """Plot path. Use {traj_id} for multiple trajectories."""

    metrics_output_dir: str = "/tmp/open_loop_eval"
    """Directory for error/continuity CSV tables, plots, and summary.json."""


def _create_policy(args: ArgsConfig) -> Any:
    if args.model_path is None:
        if args.rtc_enabled:
            raise ValueError("RTC open-loop evaluation requires --model-path for local JAX inference.")
        from openpi_client import websocket_client_policy

        if args.denoising_steps != ArgsConfig.denoising_steps:
            logging.warning(
                "--denoising-steps=%d is ignored when using a remote policy server; "
                "set it when starting the server instead.",
                args.denoising_steps,
            )
        return websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)

    train_config = _config.get_config(args.config_name)
    if args.asset_id is not None:
        assets = dataclasses.replace(train_config.data.assets, asset_id=args.asset_id)
        train_config = dataclasses.replace(
            train_config,
            data=dataclasses.replace(train_config.data, assets=assets),
        )

    if args.execution_horizon > train_config.model.action_horizon:
        raise ValueError(
            f"--execution-horizon={args.execution_horizon} exceeds the configured model "
            f"action horizon ({train_config.model.action_horizon})."
        )
    if args.rtc_enabled and args.rtc_execution_horizon > train_config.model.action_horizon:
        raise ValueError(
            f"--rtc-execution-horizon={args.rtc_execution_horizon} exceeds the configured model "
            f"action horizon ({train_config.model.action_horizon})."
        )
    required_chunk_steps = args.execution_horizon + args.rtc_inference_delay if args.rtc_enabled else 0
    if required_chunk_steps > train_config.model.action_horizon:
        raise ValueError(
            "--execution-horizon + --rtc-inference-delay must not exceed the configured model "
            f"action horizon ({required_chunk_steps} > {train_config.model.action_horizon})."
        )

    return _policy_config.create_trained_policy(
        train_config,
        args.model_path,
        sample_kwargs={"num_steps": args.denoising_steps},
    )


def main(args: ArgsConfig) -> None:
    logging.basicConfig(level=logging.INFO, force=True)

    if not args.dataset_path:
        raise ValueError("--dataset-path is required.")
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.execution_horizon <= 0:
        raise ValueError("--execution-horizon must be positive.")
    if args.action_start_index < 0:
        raise ValueError("--action-start-index must be non-negative.")
    if args.rtc_enabled:
        if args.rtc_inference_delay < 0:
            raise ValueError("--rtc-inference-delay must be non-negative.")
        if args.rtc_inference_delay >= args.execution_horizon:
            raise ValueError("--rtc-inference-delay must be smaller than --execution-horizon.")
        if args.rtc_execution_horizon <= 0:
            raise ValueError("--rtc-execution-horizon must be positive.")
        if args.rtc_inference_delay > args.rtc_execution_horizon:
            raise ValueError("--rtc-inference-delay cannot exceed --rtc-execution-horizon.")
        if args.rtc_max_guidance_weight <= 0:
            raise ValueError("--rtc-max-guidance-weight must be positive.")
        valid_schedules = {"exp", "linear", "ones", "zeros"}
        if args.rtc_prefix_attention_schedule not in valid_schedules:
            raise ValueError(
                f"--rtc-prefix-attention-schedule must be one of {sorted(valid_schedules)}, "
                f"got {args.rtc_prefix_attention_schedule!r}."
            )

    dataset_path = Path(args.dataset_path).expanduser().resolve()
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    metadata = LeRobotDatasetMetadata(args.dataset_repo_id, root=dataset_path)
    valid_traj_ids = []
    for traj_id in args.traj_ids:
        if not 0 <= traj_id < metadata.total_episodes:
            logging.warning("Trajectory ID %d is out of range. Skipping.", traj_id)
        else:
            valid_traj_ids.append(traj_id)

    logging.info("Dataset length: %d", metadata.total_episodes)
    logging.info("Running evaluation on trajectories: %s", args.traj_ids)
    if not valid_traj_ids:
        logging.info("No valid trajectories were evaluated.")
        logging.info("Done")
        return

    camera_keys = args.camera_keys or list(metadata.camera_keys)
    required_keys = {args.state_key, args.action_key, *camera_keys}
    missing_keys = sorted(required_keys - metadata.features.keys())
    if missing_keys:
        raise KeyError(f"Dataset is missing required keys: {missing_keys}")

    action_feature = metadata.features[args.action_key]
    action_shape = action_feature.get("shape")
    if not action_shape:
        raise ValueError(f"Dataset action feature does not define a shape: {action_feature}")
    total_action_dim = int(action_shape[-1])
    if args.action_start_index >= total_action_dim:
        raise ValueError(
            f"--action-start-index={args.action_start_index} must be smaller than action dimension {total_action_dim}."
        )

    policy = _create_policy(args)
    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=dataset_path,
        episodes=valid_traj_ids,
    )
    episode_indices = np.asarray(dataset.hf_dataset["episode_index"], dtype=np.int64)

    results = []
    for traj_id in valid_traj_ids:
        logging.info("Running trajectory: %d", traj_id)
        trajectory_results = evaluate_single_trajectory(
            policy,
            dataset,
            episode_indices,
            traj_id,
            state_key=args.state_key,
            action_key=args.action_key,
            camera_keys=camera_keys,
            prompt=args.prompt,
            steps=args.steps,
            execution_horizon=args.execution_horizon,
            rtc_enabled=args.rtc_enabled,
            rtc_inference_delay=args.rtc_inference_delay,
            rtc_execution_horizon=args.rtc_execution_horizon,
            rtc_max_guidance_weight=args.rtc_max_guidance_weight,
            rtc_prefix_attention_schedule=args.rtc_prefix_attention_schedule,
            action_start_index=args.action_start_index,
            seed=args.seed,
            save_plot_path=args.save_plot_path,
        )
        for result in trajectory_results:
            logging.info(
                "%s MSE for trajectory %d: %s, MAE: %s",
                result.policy_mode,
                traj_id,
                result.metrics.mse,
                result.metrics.mae,
            )
        results.extend(trajectory_results)

    if results:
        for policy_mode in dict.fromkeys(result.policy_mode for result in results):
            mode_results = [result for result in results if result.policy_mode == policy_mode]
            average_metrics = _mean_trajectory_metrics(mode_results)
            logging.info("%s average MSE across all trajs: %s", policy_mode, average_metrics.mse)
            logging.info("%s average MAE across all trajs: %s", policy_mode, average_metrics.mae)

        action_dim = results[0].errors.shape[1]
        if any(result.errors.shape[1] != action_dim for result in results):
            raise ValueError("Action dimensions differ between evaluated trajectories.")
        all_action_names = _get_action_names(action_feature, total_action_dim)
        action_indices = list(range(args.action_start_index, total_action_dim))
        action_names = [all_action_names[index] for index in action_indices]
        save_metrics(results, action_names, action_indices, args)
    else:
        logging.info("No valid trajectories were evaluated.")
    logging.info("Done")


if __name__ == "__main__":
    main(tyro.cli(ArgsConfig))
