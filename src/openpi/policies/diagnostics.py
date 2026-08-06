import logging
from typing import Any

import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms

logger = logging.getLogger(__name__)


def _normalize_vector(values: np.ndarray, stats: Any, *, use_quantiles: bool) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if use_quantiles:
        q01 = np.asarray(stats.q01, dtype=np.float64).reshape(-1)[: values.shape[0]]
        q99 = np.asarray(stats.q99, dtype=np.float64).reshape(-1)[: values.shape[0]]
        return (values - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0

    mean = np.asarray(stats.mean, dtype=np.float64).reshape(-1)[: values.shape[0]]
    std = np.asarray(stats.std, dtype=np.float64).reshape(-1)[: values.shape[0]]
    return (values - mean) / (std + 1e-6)


def _vector_line(name: str, values: np.ndarray, digits: int = 3) -> str:
    vec = np.asarray(values).reshape(-1)
    return f"{name}[{vec.shape[0]}]=" + np.array2string(vec, precision=digits, suppress_small=False, max_line_width=240)


class PolicyDiagnosticsWrapper(_base_policy.BasePolicy):
    """Read-only diagnostics wrapper for policy inference."""

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        *,
        interval: int = 20,
        max_steps: int = 200,
        eps: float = 1e-3,
    ):
        self._policy = policy
        self._interval = max(int(interval), 1)
        self._max_steps = max(int(max_steps), 0)
        self._eps = max(float(eps), 0.0)
        self._step = 0

        self._state_norm_stats: Any | None = None
        self._action_norm_stats: Any | None = None
        self._use_quantiles = False
        self._delta_mask: np.ndarray | None = None
        self._available_state_norm = False
        self._available_action_norm = False

        self._delta_sign_pos: np.ndarray | None = None
        self._delta_sign_neg: np.ndarray | None = None

        self._init_from_inner_policy()

    def _init_from_inner_policy(self) -> None:
        input_transform = getattr(self._policy, "_input_transform", None)
        output_transform = getattr(self._policy, "_output_transform", None)
        input_transforms = list(getattr(input_transform, "transforms", ()))
        output_transforms = list(getattr(output_transform, "transforms", ()))

        normalize_transform = next((t for t in input_transforms if isinstance(t, _transforms.Normalize)), None)
        absolute_transform = next((t for t in output_transforms if isinstance(t, _transforms.AbsoluteActions)), None)

        if normalize_transform is not None and normalize_transform.norm_stats is not None:
            self._use_quantiles = bool(normalize_transform.use_quantiles)
            self._state_norm_stats = normalize_transform.norm_stats.get("state")
            self._action_norm_stats = normalize_transform.norm_stats.get("actions")
            self._available_state_norm = self._state_norm_stats is not None
            self._available_action_norm = self._action_norm_stats is not None

        if absolute_transform is not None and absolute_transform.mask is not None:
            self._delta_mask = np.asarray(absolute_transform.mask, dtype=bool).reshape(-1)

        logger.info(
            "Policy diagnostics enabled (interval=%s, max_steps=%s, quantile_norm=%s, delta_mask=%s)",
            self._interval,
            self._max_steps,
            self._use_quantiles,
            "yes" if self._delta_mask is not None else "no",
        )

    @override
    def infer(self, obs: dict, **kwargs: Any) -> dict:  # type: ignore[misc]
        outputs = self._policy.infer(obs, **kwargs)
        self._step += 1

        if self._max_steps > 0 and self._step > self._max_steps:
            return outputs

        if (self._step - 1) % self._interval != 0:
            return outputs

        try:
            self._log_step(obs, outputs)
        except Exception:
            logger.exception("Policy diagnostics failed at step %s", self._step)

        return outputs

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._step = 0
        self._delta_sign_pos = None
        self._delta_sign_neg = None

    @property
    def metadata(self) -> dict[str, Any]:
        metadata = getattr(self._policy, "metadata", {})
        if isinstance(metadata, dict):
            return metadata
        return {}

    @property
    def action_horizon(self) -> int:
        return self._policy.action_horizon  # type: ignore[attr-defined]

    @property
    def action_dim(self) -> int:
        return self._policy.action_dim  # type: ignore[attr-defined]

    def _log_step(self, obs: dict, outputs: dict) -> None:
        if "observation.state" not in obs or "actions" not in outputs:
            logger.info("PI_DIAG step=%s skipped (missing observation.state or actions)", self._step)
            return

        state = np.asarray(obs["observation.state"], dtype=np.float64).reshape(-1)
        actions = np.asarray(outputs["actions"], dtype=np.float64)
        action_first = actions[0].reshape(-1) if actions.ndim == 2 else actions.reshape(-1)

        dim = min(state.shape[0], action_first.shape[0])
        state = state[:dim]
        action_first = action_first[:dim]
        delta = action_first - state

        inferred_model_raw = action_first.copy()
        if self._delta_mask is not None:
            mask = self._delta_mask[:dim]
            inferred_model_raw[mask] = inferred_model_raw[mask] - state[mask]

        if self._delta_sign_pos is None or self._delta_sign_pos.shape[0] != dim:
            self._delta_sign_pos = np.zeros(dim, dtype=np.int64)
            self._delta_sign_neg = np.zeros(dim, dtype=np.int64)
        assert self._delta_sign_neg is not None

        delta_pos = delta > self._eps
        delta_neg = delta < -self._eps
        self._delta_sign_pos += delta_pos.astype(np.int64)
        self._delta_sign_neg += delta_neg.astype(np.int64)

        logger.info("PI_DIAG step=%s", self._step)
        logger.info("PI_DIAG %s", _vector_line("state_raw", state))
        logger.info("PI_DIAG %s", _vector_line("action_abs_raw", action_first))
        logger.info("PI_DIAG %s", _vector_line("delta_abs_minus_state", delta))
        logger.info("PI_DIAG %s", _vector_line("action_model_like_raw", inferred_model_raw))

        if self._available_state_norm:
            state_norm = _normalize_vector(state, self._state_norm_stats, use_quantiles=self._use_quantiles)
            logger.info("PI_DIAG %s", _vector_line("state_norm", state_norm, digits=2))
            if self._use_quantiles:
                outside = np.logical_or(state_norm < -1.0, state_norm > 1.0)
                logger.info(
                    "PI_DIAG state_norm_outside[-1,1]=%s/%s dims=%s",
                    int(np.sum(outside)),
                    int(outside.shape[0]),
                    np.where(outside)[0].tolist(),
                )
            else:
                outside = np.abs(state_norm) > 3.0
                logger.info(
                    "PI_DIAG state_norm_outside|z|>3=%s/%s dims=%s",
                    int(np.sum(outside)),
                    int(outside.shape[0]),
                    np.where(outside)[0].tolist(),
                )

        if self._available_action_norm:
            action_model_like_norm = _normalize_vector(
                inferred_model_raw,
                self._action_norm_stats,
                use_quantiles=self._use_quantiles,
            )
            logger.info("PI_DIAG %s", _vector_line("action_model_like_norm", action_model_like_norm, digits=2))
            if self._use_quantiles:
                outside = np.logical_or(action_model_like_norm < -1.0, action_model_like_norm > 1.0)
                logger.info(
                    "PI_DIAG action_model_like_norm_outside[-1,1]=%s/%s dims=%s",
                    int(np.sum(outside)),
                    int(outside.shape[0]),
                    np.where(outside)[0].tolist(),
                )
            else:
                outside = np.abs(action_model_like_norm) > 3.0
                logger.info(
                    "PI_DIAG action_model_like_norm_outside|z|>3=%s/%s dims=%s",
                    int(np.sum(outside)),
                    int(outside.shape[0]),
                    np.where(outside)[0].tolist(),
                )

        pos_counts = self._delta_sign_pos.tolist()
        neg_counts = self._delta_sign_neg.tolist()
        sign_summary = [f"d{i}:{pos_counts[i]}/-{neg_counts[i]}" for i in range(dim)]
        logger.info("PI_DIAG delta_sign_counts(+/-)=%s", ", ".join(sign_summary))
