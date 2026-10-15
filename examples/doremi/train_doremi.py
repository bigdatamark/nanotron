# [MODULE DOC]
# Purpose: Entry script for DoReMi proxy training. Parses --config-file, loads a DoReMiConfig, loads one on-disk HF dataset per domain from the first data stage's dataset path joined with each domain name, builds DoReMiTrainer (which loads the reference model), builds the DoReMi dataloader and trains.
# Usage: torchrun with CUDA_DEVICE_MAX_CONNECTIONS=1, run with examples/doremi as the working directory or on the path (imports the local doremi package).
# Side effects (as script): disk reads, distributed initialization, training, checkpoint writes.
"""
DoReMi training script.

Usage:

export CUDA_DEVICE_MAX_CONNECTIONS=1 # important for some distributed operations
torchrun --nproc_per_node=4 examples/doremi/train_doremi.py --config-file examples/doremi/configs/config_280m_llama_proxy.yaml
"""
import argparse

from nanotron.config import get_config_from_file

from doremi.config import DoReMiConfig
from doremi.dataloader import get_dataloader, get_datasets
from doremi.trainer import DoReMiTrainer


def get_args():
    # [FUNCTION DOC]
    # Behavior: Parses the required --config-file argument.
    # Inputs: None. Outputs: argparse Namespace. Exceptions: SystemExit on missing argument. Side effects: reads sys.argv.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True, help="Path to the YAML or python config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config_file = args.config_file
    config: DoReMiConfig = get_config_from_file(config_file, config_class=DoReMiConfig)

    dataset_paths = [
        f"{config.data_stages[0].data.dataset.hf_dataset_or_datasets}/{name}" for name in config.doremi.domain_names
    ]
    datasets = get_datasets(dataset_paths)

    trainer = DoReMiTrainer(config_file, config_class=DoReMiConfig)
    dataloader = get_dataloader(trainer, datasets)
    trainer.train(dataloader)
