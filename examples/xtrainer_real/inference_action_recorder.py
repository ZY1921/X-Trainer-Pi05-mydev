"""Persist and plot action chunks received by X-Trainer inference clients."""

import csv
import dataclasses
import json
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class ReceivedActionChunk:
    phase: str
    request_id: int
    episode_id: int
    request_step: int
    arrival_step: int
    actual_delay_steps: int
    rtc_enabled: bool
    installed: bool
    round_trip_ms: float
    server_infer_ms: float
    actions: np.ndarray


@dataclasses.dataclass(frozen=True)
class SelectedAction:
    episode_id: int
    step: int
    request_id: int
    chunk_index: int
    action: np.ndarray


class InferenceActionRecorder:
    """Collect complete inference chunks and the per-step actions selected from them."""

    def __init__(
        self,
        output_root: str | Path,
        *,
        mode: str,
        plot_start_index: int,
        config: dict[str, Any],
    ) -> None:
        if plot_start_index < 0:
            raise ValueError(f"plot_start_index must be non-negative, got {plot_start_index}.")

        self._mode = mode
        self._plot_start_index = plot_start_index
        self._config = config
        self._chunks: list[ReceivedActionChunk] = []
        self._selected: list[SelectedAction] = []
        self._saved = False
        self._output_dir = self._create_output_dir(Path(output_root), mode)

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    def record_received_chunk(
        self,
        actions: np.ndarray,
        *,
        phase: str,
        request_id: int,
        episode_id: int,
        request_step: int,
        arrival_step: int,
        actual_delay_steps: int,
        rtc_enabled: bool,
        installed: bool,
        round_trip_ms: float,
        server_infer_ms: float,
    ) -> None:
        chunk = np.asarray(actions, dtype=np.float64)
        if chunk.ndim != 2:
            raise ValueError(f"Expected received actions with shape (horizon, action_dim), got {chunk.shape}.")
        self._chunks.append(
            ReceivedActionChunk(
                phase=phase,
                request_id=request_id,
                episode_id=episode_id,
                request_step=request_step,
                arrival_step=arrival_step,
                actual_delay_steps=actual_delay_steps,
                rtc_enabled=rtc_enabled,
                installed=installed,
                round_trip_ms=round_trip_ms,
                server_infer_ms=server_infer_ms,
                actions=np.array(chunk, copy=True),
            )
        )

    def record_selected_action(
        self,
        action: np.ndarray,
        *,
        episode_id: int,
        step: int,
        request_id: int,
        chunk_index: int,
    ) -> None:
        vector = np.asarray(action, dtype=np.float64).reshape(-1)
        self._selected.append(
            SelectedAction(
                episode_id=episode_id,
                step=step,
                request_id=request_id,
                chunk_index=chunk_index,
                action=np.array(vector, copy=True),
            )
        )

    def save(self) -> Path:
        """Write machine-readable data, CSV tables, a summary, and an action plot."""
        if self._saved:
            return self._output_dir

        self._write_npz()
        self._write_received_csv()
        self._write_selected_csv()
        summary = self._build_summary()
        with (self._output_dir / "summary.json").open("w", encoding="utf-8") as file_obj:
            json.dump(summary, file_obj, indent=2, ensure_ascii=False, default=str)

        try:
            self._plot_actions()
        except Exception:
            logger.exception("Failed to render inference action plot; recorded data is still available")

        self._saved = True
        logger.info("Inference action data saved to %s", self._output_dir)
        return self._output_dir

    @staticmethod
    def _create_output_dir(output_root: Path, mode: str) -> Path:
        root = output_root.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = f"inference_actions_{mode}_{timestamp}"
        candidate = root / base_name
        suffix = 1
        while candidate.exists():
            candidate = root / f"{base_name}_{suffix}"
            suffix += 1
        candidate.mkdir()
        return candidate

    def _write_npz(self) -> None:
        padded_chunks, chunk_lengths, chunk_dims = self._padded_chunks()
        selected_actions = self._selected_action_array()
        np.savez_compressed(
            self._output_dir / "actions.npz",
            received_actions=padded_chunks,
            received_chunk_lengths=chunk_lengths,
            received_action_dims=chunk_dims,
            received_phase=np.asarray([chunk.phase for chunk in self._chunks]),
            received_request_id=np.asarray([chunk.request_id for chunk in self._chunks], dtype=np.int64),
            received_episode_id=np.asarray([chunk.episode_id for chunk in self._chunks], dtype=np.int64),
            received_request_step=np.asarray([chunk.request_step for chunk in self._chunks], dtype=np.int64),
            received_arrival_step=np.asarray([chunk.arrival_step for chunk in self._chunks], dtype=np.int64),
            received_actual_delay_steps=np.asarray(
                [chunk.actual_delay_steps for chunk in self._chunks], dtype=np.int64
            ),
            received_rtc_enabled=np.asarray([chunk.rtc_enabled for chunk in self._chunks], dtype=np.bool_),
            received_installed=np.asarray([chunk.installed for chunk in self._chunks], dtype=np.bool_),
            received_round_trip_ms=np.asarray([chunk.round_trip_ms for chunk in self._chunks], dtype=np.float64),
            received_server_infer_ms=np.asarray([chunk.server_infer_ms for chunk in self._chunks], dtype=np.float64),
            selected_actions=selected_actions,
            selected_episode_id=np.asarray([item.episode_id for item in self._selected], dtype=np.int64),
            selected_step=np.asarray([item.step for item in self._selected], dtype=np.int64),
            selected_request_id=np.asarray([item.request_id for item in self._selected], dtype=np.int64),
            selected_chunk_index=np.asarray([item.chunk_index for item in self._selected], dtype=np.int64),
        )

    def _write_received_csv(self) -> None:
        action_dim = max((chunk.actions.shape[1] for chunk in self._chunks), default=0)
        fieldnames = [
            "phase",
            "request_id",
            "episode_id",
            "request_step",
            "arrival_step",
            "actual_delay_steps",
            "rtc_enabled",
            "installed",
            "round_trip_ms",
            "server_infer_ms",
            "chunk_index",
            *[f"action_{index}" for index in range(action_dim)],
        ]
        with (self._output_dir / "received_chunks.csv").open("w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            for chunk in self._chunks:
                metadata = {
                    "phase": chunk.phase,
                    "request_id": chunk.request_id,
                    "episode_id": chunk.episode_id,
                    "request_step": chunk.request_step,
                    "arrival_step": chunk.arrival_step,
                    "actual_delay_steps": chunk.actual_delay_steps,
                    "rtc_enabled": chunk.rtc_enabled,
                    "installed": chunk.installed,
                    "round_trip_ms": chunk.round_trip_ms,
                    "server_infer_ms": chunk.server_infer_ms,
                }
                for chunk_index, action in enumerate(chunk.actions):
                    row = {**metadata, "chunk_index": chunk_index}
                    row.update({f"action_{index}": value for index, value in enumerate(action)})
                    writer.writerow(row)

    def _write_selected_csv(self) -> None:
        action_dim = max((item.action.shape[0] for item in self._selected), default=0)
        fieldnames = [
            "episode_id",
            "step",
            "request_id",
            "chunk_index",
            *[f"action_{index}" for index in range(action_dim)],
        ]
        with (self._output_dir / "selected_actions.csv").open("w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            for item in self._selected:
                row = {
                    "episode_id": item.episode_id,
                    "step": item.step,
                    "request_id": item.request_id,
                    "chunk_index": item.chunk_index,
                }
                row.update({f"action_{index}": value for index, value in enumerate(item.action)})
                writer.writerow(row)

    def _build_summary(self) -> dict[str, Any]:
        selected = self._selected_action_array()
        switch_indices = self._switch_indices()
        metric_selected = selected[:, self._plot_start_index :] if selected.ndim == 2 else selected
        if len(switch_indices) and metric_selected.shape[1] > 0:
            jumps = np.abs(metric_selected[switch_indices] - metric_selected[switch_indices - 1])
            boundary_jump_mae = float(np.mean(jumps))
            boundary_jump_max = float(np.max(jumps))
            boundary_jump_mae_by_action = np.mean(jumps, axis=0).tolist()
            boundary_jump_max_by_action = np.max(jumps, axis=0).tolist()
        else:
            boundary_jump_mae = 0.0
            boundary_jump_max = 0.0
            boundary_jump_mae_by_action = []
            boundary_jump_max_by_action = []

        return {
            "mode": self._mode,
            "config": self._config,
            "received_chunk_count": len(self._chunks),
            "installed_chunk_count": sum(chunk.installed for chunk in self._chunks),
            "selected_action_count": len(self._selected),
            "switch_count": len(switch_indices),
            "boundary_action_jump_mae": boundary_jump_mae,
            "boundary_action_jump_max": boundary_jump_max,
            "boundary_action_jump_mae_by_action": boundary_jump_mae_by_action,
            "boundary_action_jump_max_by_action": boundary_jump_max_by_action,
            "plot_start_index": self._plot_start_index,
            "boundary_action_indices": list(range(self._plot_start_index, selected.shape[1]))
            if selected.ndim == 2
            else [],
            "files": {
                "numpy": "actions.npz",
                "received_csv": "received_chunks.csv",
                "selected_csv": "selected_actions.csv",
                "plot": "actions.png",
            },
        }

    def _plot_actions(self) -> None:
        if not self._selected:
            logger.warning("No selected actions were recorded; skipping action plot")
            return

        import matplotlib as mpl

        mpl.use("Agg")
        from matplotlib import pyplot as plt

        selected = self._selected_action_array()
        action_dim = selected.shape[1]
        if self._plot_start_index >= action_dim:
            raise ValueError(
                f"plot_start_index={self._plot_start_index} must be smaller than action dimension {action_dim}."
            )

        action_indices = list(range(self._plot_start_index, action_dim))
        fig, axes = plt.subplots(
            nrows=len(action_indices),
            ncols=1,
            figsize=(12, max(3.0, 2.7 * len(action_indices))),
            sharex=True,
        )
        if len(action_indices) == 1:
            axes = [axes]

        selected_steps = np.asarray([item.step for item in self._selected], dtype=np.int64)
        selected_episode_ids = np.asarray([item.episode_id for item in self._selected], dtype=np.int64)
        switch_indices = self._switch_indices()

        for axis, action_index in zip(axes, action_indices, strict=True):
            for chunk in self._chunks:
                if not chunk.installed or action_index >= chunk.actions.shape[1]:
                    continue
                chunk_steps = chunk.request_step + np.arange(chunk.actions.shape[0])
                color = "tab:purple" if chunk.rtc_enabled else "tab:green"
                axis.plot(chunk_steps, chunk.actions[:, action_index], color=color, alpha=0.18, linewidth=0.9)

            for episode_id in dict.fromkeys(selected_episode_ids.tolist()):
                mask = selected_episode_ids == episode_id
                axis.plot(
                    selected_steps[mask],
                    selected[mask, action_index],
                    color="black",
                    linewidth=1.4,
                    label="selected action" if episode_id == selected_episode_ids[0] else None,
                )

            for position, selected_index in enumerate(switch_indices):
                axis.axvline(
                    selected_steps[selected_index],
                    color="tab:red",
                    alpha=0.35,
                    linewidth=0.9,
                    label="chunk switch" if position == 0 else None,
                )

            axis.set_ylabel(self._action_name(action_index, action_dim))
            axis.grid(alpha=0.2)

        if any(chunk.installed and not chunk.rtc_enabled for chunk in self._chunks):
            axes[0].plot([], [], color="tab:green", alpha=0.5, label="received baseline chunk")
        if any(chunk.installed and chunk.rtc_enabled for chunk in self._chunks):
            axes[0].plot([], [], color="tab:purple", alpha=0.5, label="received RTC chunk")
        axes[0].legend(loc="best")
        axes[-1].set_xlabel("control step")
        fig.suptitle(f"X-Trainer received and selected actions ({self._mode})")
        fig.tight_layout()
        fig.savefig(self._output_dir / "actions.png", dpi=160)
        plt.close(fig)

    def _padded_chunks(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._chunks:
            return (
                np.empty((0, 0, 0), dtype=np.float64),
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.int64),
            )
        lengths = np.asarray([chunk.actions.shape[0] for chunk in self._chunks], dtype=np.int64)
        dims = np.asarray([chunk.actions.shape[1] for chunk in self._chunks], dtype=np.int64)
        padded = np.full((len(self._chunks), int(np.max(lengths)), int(np.max(dims))), np.nan, dtype=np.float64)
        for index, chunk in enumerate(self._chunks):
            padded[index, : chunk.actions.shape[0], : chunk.actions.shape[1]] = chunk.actions
        return padded, lengths, dims

    def _selected_action_array(self) -> np.ndarray:
        if not self._selected:
            return np.empty((0, 0), dtype=np.float64)
        action_dim = max(item.action.shape[0] for item in self._selected)
        selected = np.full((len(self._selected), action_dim), np.nan, dtype=np.float64)
        for index, item in enumerate(self._selected):
            selected[index, : item.action.shape[0]] = item.action
        return selected

    def _switch_indices(self) -> np.ndarray:
        if len(self._selected) < 2:
            return np.empty((0,), dtype=np.int64)
        request_ids = np.asarray([item.request_id for item in self._selected], dtype=np.int64)
        episode_ids = np.asarray([item.episode_id for item in self._selected], dtype=np.int64)
        return np.flatnonzero((request_ids[1:] != request_ids[:-1]) & (episode_ids[1:] == episode_ids[:-1])) + 1

    @staticmethod
    def _action_name(action_index: int, action_dim: int) -> str:
        if action_dim == 14:
            names = [
                "left_joint_1",
                "left_joint_2",
                "left_joint_3",
                "left_joint_4",
                "left_joint_5",
                "left_joint_6",
                "left_gripper",
                "right_joint_1",
                "right_joint_2",
                "right_joint_3",
                "right_joint_4",
                "right_joint_5",
                "right_joint_6",
                "right_gripper",
            ]
            return names[action_index]
        return f"action_{action_index}"
