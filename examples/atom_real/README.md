# Run Atom (Real Robot)

This example adapts OpenPI pi0.5 to Dobot Atom upper-body + dexterous hands.

It adds:
- Atom policy transforms (28-dim state/action)
- Atom train configs (`pi05_atom*`)
- Atom real runtime client
- Atom raw collect_data -> LeRobot v2.1 converter
- Atom HDF5 -> LeRobot v2.1 converter

## 1) Convert Atom dataset to LeRobot v2.1

### Raw collect_data format

Use this for Atom raw data collected as episode folders:

```text
collect_data/
  <episode_id>/
    top_left/*.jpg
    top_right/*.jpg
    wrist_left/*.jpg
    wrist_right/*.jpg
    observation/*.pkl   # {"obs": (28,), "action": (28,)}
```

Convert directly:

```bash
uv run examples/atom_real/convert_raw_to_lerobot_2_1.py \
  --raw_root /home/dobot/gbw/dataset/classfication/collect_data \
  --output_root /path/to/lerobot_atom_v21 \
  --repo_id your_hf_username/my_atom_dataset \
  --task "pick and place" \
  --top_camera top_left \
  --overwrite_output
```

By default `top_left` is exported as `observation.images.top`. If the better overhead view is `top_right`, use:

```bash
uv run examples/atom_real/convert_raw_to_lerobot_2_1.py \
  --raw_root /home/dobot/gbw/dataset/classfication/collect_data \
  --output_root /path/to/lerobot_atom_v21 \
  --repo_id your_hf_username/my_atom_dataset \
  --task "pick and place" \
  --top_camera top_right \
  --overwrite_output
```

The converter writes the image keys expected by the Atom policy:

```text
top_left/top_right -> observation.images.top
wrist_left         -> observation.images.left_wrist
wrist_right        -> observation.images.right_wrist
obs                -> observation.state
action             -> action
```

For a quick smoke test without video encoding:

```bash
uv run examples/atom_real/convert_raw_to_lerobot_2_1.py \
  --raw_root /home/dobot/gbw/dataset/classfication/collect_data \
  --output_root /tmp/atom_lerobot_v21_smoke \
  --no_videos \
  --max_episodes 1 \
  --max_frames_per_episode 20 \
  --overwrite_output
```

### HDF5 format

```bash
uv run examples/atom_real/convert_hdf5_to_lerobot_2_1.py \
  --raw_root /path/to/atom_hdf5_dataset \
  --output_root /path/to/lerobot_atom_v21 \
  --repo_id your_hf_username/my_atom_dataset \
  --task "pick and place" \
  --overwrite_output
```

## 2) Train pi0.5 on Atom

Full fine-tuning (JAX):

```bash
uv run scripts/training/train.py pi05_atom_finetune --exp_name atom_ft_001
```

LoRA fine-tuning:

```bash
uv run scripts/training/train.py pi05_atom_lora_finetune --exp_name atom_lora_001
```

If using PyTorch trainer:

```bash
uv run scripts/training/train_pytorch.py pi05_atom_finetune --exp_name atom_ft_torch_001
```

## 3) Start policy server

Default Atom env (pi05 base + atom transforms):

```bash
uv run scripts/deployment/serve_policy.py --env ATOM --port 8000
```

Or custom checkpoint:

```bash
uv run scripts/deployment/serve_policy.py policy:checkpoint \
  --policy.config pi05_atom_finetune \
  --policy.dir /path/to/checkpoint/20000 \
  --port 8000
```

## 4) Run Atom robot client

Install runtime dependencies:

```bash
uv sync --group atom-runtime
uv pip install -r examples/atom_real/requirements.txt
```

Run:

```bash
python -m examples.atom_real.main \
  --host <server_ip> \
  --port 8000 \
  --camera-top-id <top_video_index> \
  --camera-top-fps 60 \
  --camera-left-wrist-serial <left_wrist_serial> \
  --camera-right-wrist-serial <right_wrist_serial> \
  --prompt "pick and place"
```

For an OpenCV/UVC top camera, pass the `/dev/videoX` index as `--camera-top-id X`.
For a RealSense top camera, omit `--camera-top-id` and pass `--camera-top-serial <top_serial>` instead.

## Notes

- Atom state/action order in this adaptation is fixed to:
  - `left_arm(7), left_hand(6), right_arm(7), right_hand(6), head(2)`
- Delta transform is applied only on arm joints during training.
- Runtime uses built-in Atom DDS control code under `examples/atom_real/hardware/robot_control_dds`.
- `cyclonedds` is only installed with `uv sync --group atom-runtime`, because it requires the CycloneDDS C library on the robot PC.
- Keep your Atom raw `collect_data` directory or HDF5 episodes (`.hdf5` / `.h5`) in any local directory and pass it via `--raw_root`.
