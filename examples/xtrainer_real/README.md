# Run XTrainer (Real Robot)

This example shows how to stream actions from an `openpi` policy server to a real bimanual Dobot XTrainer setup.

This runtime is self-contained in this repository (no dependency on an external `../lerobot` source tree).

## Quick Start (Minimal)

Policy server (GPU machine):

```bash
uv run scripts/deployment/serve_policy.py --env XTRAINER
```

Robot control machine:

```bash
uv pip install -r examples/xtrainer_real/requirements.txt && \
python -m examples.xtrainer_real.main \
  --host <server_ip> \
  --camera-top-serial <top_serial> \
  --camera-left-wrist-serial <left_wrist_serial> \
  --camera-right-wrist-serial <right_wrist_serial>
```

## 1) Start policy server (GPU machine)

Use the new XTrainer env shortcut:

```bash
uv run scripts/deployment/serve_policy.py --env XTRAINER
```

This is equivalent to:

```bash
uv run scripts/deployment/serve_policy.py policy:checkpoint --policy.config=pi05_xtrainer --policy.dir=gs://openpi-assets/checkpoints/pi05_base
```

If you have an xtrainer-specific fine-tuned checkpoint with `assets/xtrainer` norm stats, use:

```bash
uv run scripts/deployment/serve_policy.py policy:checkpoint --policy.config=pi05_xtrainer_custom --policy.dir=/path/to/your/checkpoint
```

## 2) Run robot client (control machine)

Install minimal runtime dependencies in your robot environment:

```bash
cd $OPENPI_ROOT/packages/openpi-client
pip install -e .
pip install pyrealsense2 pyserial
```

Then run:

```bash
python -m examples.xtrainer_real.main \
  --host <server_ip> \
  --port 8000 \
  --prompt "pick up the block and place it in the tray" \
  --camera-top-serial <top_serial> \
  --camera-left-wrist-serial <left_wrist_serial> \
  --camera-right-wrist-serial <right_wrist_serial>
```

Useful optional arguments:
- `--left-robot-ip` / `--right-robot-ip`
- `--left-gripper-port` / `--right-gripper-port`
- `--control-hz`
- `--action-horizon`

## Notes

- The client sends observations with keys:
  - `observation.state`
  - `observation.images.top`
  - `observation.images.left_wrist`
  - `observation.images.right_wrist`
- Actions are interpreted as 14-dim bimanual absolute joint targets:
  - `left(6 joints + gripper), right(6 joints + gripper)`.
- The runtime includes interpolation and gripper-threshold logic for safer execution on real hardware.
