from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import rtc as _rtc
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(
        self,
        obs: dict,
        *,
        noise: np.ndarray | None = None,
        prev_chunk_left_over: np.ndarray | None = None,
        inference_delay: int | None = None,
        execution_horizon: int | None = None,
        rtc_prefix_attention_schedule: _rtc.PrefixAttentionSchedule = "exp",
        rtc_max_guidance_weight: float = 10.0,
    ) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)

        rtc_prefix_steps = 0
        if prev_chunk_left_over is not None:
            if self._is_pytorch_model:
                raise NotImplementedError("RTC inference is currently implemented for JAX policies only.")
            if inference_delay is None:
                raise ValueError("inference_delay is required when prev_chunk_left_over is provided.")
            if execution_horizon is None:
                raise ValueError("execution_horizon is required when prev_chunk_left_over is provided.")
            if inference_delay < 0:
                raise ValueError(f"inference_delay must be non-negative, got {inference_delay}.")
            if not 0 < execution_horizon <= self.action_horizon:
                raise ValueError(f"execution_horizon must be in [1, {self.action_horizon}], got {execution_horizon}.")
            if inference_delay > execution_horizon:
                raise ValueError(
                    f"inference_delay ({inference_delay}) cannot exceed execution_horizon ({execution_horizon})."
                )
            if rtc_max_guidance_weight <= 0:
                raise ValueError(f"rtc_max_guidance_weight must be positive, got {rtc_max_guidance_weight}.")

            previous_actions = np.asarray(prev_chunk_left_over)
            if previous_actions.ndim != 2:
                raise ValueError(
                    f"prev_chunk_left_over must have shape (time, action_dim), got {previous_actions.shape}."
                )
            rtc_prefix_steps = min(len(previous_actions), execution_horizon)
            if rtc_prefix_steps == 0:
                raise ValueError("prev_chunk_left_over cannot be empty.")
            # Feeding the physical-space actions through the normal input pipeline
            # re-anchors delta actions to the current observation and applies the
            # checkpoint's normalization and action-dimension padding.
            inputs["actions"] = np.array(previous_actions[:rtc_prefix_steps], copy=True)

        inputs = self._input_transform(inputs)

        rtc_prev_chunk = None
        rtc_prefix_weights = None
        if prev_chunk_left_over is not None:
            transformed_previous_actions = np.asarray(inputs.pop("actions"))
            if transformed_previous_actions.shape[-1] != self.action_dim:
                raise ValueError(
                    "Transformed RTC prefix action dimension does not match the model: "
                    f"{transformed_previous_actions.shape[-1]} != {self.action_dim}."
                )
            rtc_prev_chunk = np.zeros((self.action_horizon, self.action_dim), dtype=np.float32)
            rtc_prev_chunk[:rtc_prefix_steps] = transformed_previous_actions
            rtc_prefix_weights = _rtc.get_prefix_weights(
                inference_delay,
                min(execution_horizon, rtc_prefix_steps),
                self.action_horizon,
                rtc_prefix_attention_schedule,
            )

        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        if rtc_prev_chunk is not None:
            sample_kwargs.update(
                rtc_prev_chunk=jnp.asarray(rtc_prev_chunk)[None, ...],
                rtc_prefix_weights=jnp.asarray(rtc_prefix_weights),
                rtc_max_guidance_weight=jnp.asarray(rtc_max_guidance_weight, dtype=jnp.float32),
            )

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        sampled_actions = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        if not self._is_pytorch_model:
            jax.block_until_ready(sampled_actions)
        outputs = {
            "state": inputs["state"],
            "actions": sampled_actions,
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @property
    def action_horizon(self) -> int:
        return self._model.action_horizon

    @property
    def action_dim(self) -> int:
        return self._model.action_dim


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict, **kwargs: Any) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs, **kwargs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results

    @property
    def metadata(self) -> dict[str, Any]:
        return self._policy.metadata  # type: ignore[attr-defined]

    @property
    def action_horizon(self) -> int:
        return self._policy.action_horizon  # type: ignore[attr-defined]

    @property
    def action_dim(self) -> int:
        return self._policy.action_dim  # type: ignore[attr-defined]
