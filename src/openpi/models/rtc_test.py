import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import rtc


@pytest.mark.parametrize(
    ("schedule", "expected"),
    [
        ("zeros", [1, 1, 0, 0, 0, 0, 0, 0]),
        ("ones", [1, 1, 1, 1, 1, 1, 0, 0]),
        ("linear", [1, 1, 0.8, 0.6, 0.4, 0.2, 0, 0]),
    ],
)
def test_get_prefix_weights(schedule, expected):
    weights = rtc.get_prefix_weights(start=2, end=6, total=8, schedule=schedule)
    np.testing.assert_allclose(weights, expected)


def test_get_prefix_weights_exp_matches_reference_formula():
    weights = rtc.get_prefix_weights(start=5, end=14, total=25, schedule="exp")
    expected_middle = [0.7645, 0.5706, 0.4130, 0.2871, 0.1888, 0.1145, 0.0611, 0.0258, 0.0061]
    np.testing.assert_allclose(weights[:5], 1.0)
    np.testing.assert_allclose(weights[5:14], expected_middle, atol=1e-4)
    np.testing.assert_allclose(weights[14:], 0.0)


def test_guided_velocity_uses_denoiser_vjp():
    x_t = jnp.ones((1, 3, 1), dtype=jnp.float32)
    prev_chunk = jnp.full_like(x_t, 2.0)
    weights = jnp.ones(3, dtype=jnp.float32)
    time = jnp.asarray(0.5, dtype=jnp.float32)

    # v=3x gives x_0=x-time*v=-0.5 and d(x_0)/dx=-0.5 at time=0.5.
    # The error is 2.5 and the RTC guidance weight is 2, so the guided
    # velocity is 3 - 2 * (-1.25) = 5.5.
    result = rtc.guided_velocity(
        lambda x: 3 * x,
        x_t,
        time,
        prev_chunk,
        weights,
        max_guidance_weight=10.0,
    )

    np.testing.assert_allclose(result, 5.5)


def test_invalid_prefix_schedule():
    with pytest.raises(ValueError, match="Invalid RTC"):
        rtc.get_prefix_weights(0, 1, 2, "invalid")
