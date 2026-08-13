"""Compute normalization statistics for a config.

This training script computes the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import argparse
import sys

import numpy as np
import tqdm

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def _parse_train_config_with_overrides(config_args: list[str]) -> _config.TrainConfig:
    original_argv = sys.argv
    try:
        # Reuse the same overridable config CLI as train scripts.
        sys.argv = [original_argv[0], *config_args]
        return _config.cli()
    finally:
        sys.argv = original_argv


def _parse_args() -> tuple[_config.TrainConfig, int | None]:
    parser = argparse.ArgumentParser(
        description=(
            "Compute normalization statistics for a train config.\n\n"
            "You can pass either:\n"
            "1) --config-name <name> (legacy mode), or\n"
            "2) full train-config arguments like:\n"
            "   pi05_xtrainer_finetune --data.repo-id ... --data.assets.asset-id ...\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--config-name", type=str, default=None, help="Config name (legacy mode).")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames for stats estimation.")

    known_args, config_override_args = parser.parse_known_args()
    if known_args.config_name is None and not config_override_args:
        parser.error("No config specified. Provide --config-name <name> or pass config subcommand arguments.")

    config_args = list(config_override_args)
    if known_args.config_name is not None:
        config_args = [known_args.config_name, *config_args]

    config = _parse_train_config_with_overrides(config_args)
    return config, known_args.max_frames


def main(config: _config.TrainConfig, max_frames: int | None = None):
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_asset_id = data_config.asset_id or data_config.repo_id
    if output_asset_id is None:
        raise ValueError("Cannot determine output stats path: both data_config.asset_id and repo_id are None.")

    output_path = config.assets_dirs / output_asset_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    parsed_config, parsed_max_frames = _parse_args()
    main(parsed_config, parsed_max_frames)
