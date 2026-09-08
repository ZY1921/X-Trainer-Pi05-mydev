"""Tests for checkpoint serialization behavior."""

# ruff: noqa: SLF001

import jax.numpy as jnp
import orbax.checkpoint as ocp

from openpi.training import checkpoints


def test_synchronous_pytree_handler_round_trip(tmp_path):
    handler = checkpoints._SynchronousPyTreeCheckpointHandler()
    assert not isinstance(handler, ocp.AsyncCheckpointHandler)
    assert handler._handler._save_concurrent_bytes == 6 * 10**9
    assert handler._handler._save_device_host_concurrent_bytes == 6 * 10**9
    handler.close()

    manager = ocp.CheckpointManager(
        tmp_path,
        item_handlers={
            "train_state": checkpoints._SynchronousPyTreeCheckpointHandler(),
            "params": checkpoints._SynchronousPyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(create=True, enable_async_checkpointing=False),
    )
    items = {
        "train_state": {"step": jnp.array(7)},
        "params": {"params": {"weight": jnp.arange(4)}},
    }

    assert manager.save(7, items)
    restored = manager.restore(7, items=items)
    manager.close()

    assert restored["train_state"]["step"] == 7
    assert (restored["params"]["params"]["weight"] == jnp.arange(4)).all()
