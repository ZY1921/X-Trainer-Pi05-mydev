from collections.abc import Callable
from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp

PrefixAttentionSchedule: TypeAlias = Literal["linear", "exp", "ones", "zeros"]


def get_prefix_weights(
    start: int | jax.Array,
    end: int | jax.Array,
    total: int,
    schedule: PrefixAttentionSchedule,
) -> jax.Array:
    """Build RTC prefix-attention weights.

    ``start`` is the inference delay: actions before it are fully constrained to
    the previous chunk. ``end`` is the exclusive end of the transition region.
    """
    start = jnp.minimum(start, end)
    indices = jnp.arange(total)

    if schedule == "ones":
        weights = jnp.ones(total, dtype=jnp.float32)
    elif schedule == "zeros":
        weights = (indices < start).astype(jnp.float32)
    elif schedule in ("linear", "exp"):
        weights = jnp.clip((start - 1 - indices) / (end - start + 1) + 1, 0, 1)
        if schedule == "exp":
            weights = weights * jnp.expm1(weights) / (jnp.e - 1)
    else:
        raise ValueError(f"Invalid RTC prefix attention schedule: {schedule}")

    return jnp.where(indices >= end, 0, weights)


def guided_velocity(
    denoise_fn: Callable[[jax.Array], jax.Array],
    x_t: jax.Array,
    time: jax.Array,
    prev_chunk: jax.Array,
    prefix_weights: jax.Array,
    max_guidance_weight: float | jax.Array,
) -> jax.Array:
    """Apply RTC guidance to one reverse-time OpenPI denoising step.

    OpenPI samples from ``time=1`` (noise) to ``time=0`` (actions). This is the
    reverse of the convention in the RTC reference implementation, hence the
    clean-action estimate ``x_t - time * v_t`` and the subtraction of the VJP
    correction from the velocity.
    """

    def clean_action_estimate(current_x_t: jax.Array) -> tuple[jax.Array, jax.Array]:
        velocity = denoise_fn(current_x_t)
        return current_x_t - time * velocity, velocity

    predicted_actions, vjp_fn, velocity = jax.vjp(clean_action_estimate, x_t, has_aux=True)
    error = (prev_chunk - predicted_actions) * prefix_weights[None, :, None]
    correction = vjp_fn(error)[0]

    tau = 1 - time
    time_squared = time**2
    inv_r_squared = (time_squared + tau**2) / time_squared
    max_weight = jnp.asarray(max_guidance_weight, dtype=jnp.float32)
    c = jnp.nan_to_num(time / tau, posinf=max_weight)
    guidance_weight = jnp.minimum(jnp.nan_to_num(c * inv_r_squared, posinf=max_weight), max_weight)
    return velocity - guidance_weight * correction
