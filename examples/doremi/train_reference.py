# [MODULE DOC]
# Purpose: Entry script for training the DoReMi reference model. Parses --config-file, loads a DoReMiConfig, loads per-domain datasets from disk, picks initial domain weights (from config or proportional to dataset sizes), asserts they sum to 1, builds ReferenceTrainer, the DoReMi dataloader, and trains.
# Side effects (as script): disk reads, distributed initialization, training.
# KNOWN ISSUE: the docstring usage line points to train_doremi.py instead of this script.
# KNOWN ISSUE: reads config.data.dataset, but the config uses data_stages (train_doremi.py uses data_stages[0]), so this likely raises AttributeError.
# KNOWN ISSUE: ReferenceTrainer construction fails with TypeError because of how it builds DoReMiContext (see doremi/trainer.py).
"""
DoReMi training script.

Usage:

export CUDA_DEVICE_MAX_CONNECTIONS=1 # important for some distributed operations
torchrun --nproc_per_node=4 examples/doremi/train_doremi.py --config-file examples/doremi/configs/config_280m_llama.yaml
"""

import argparse

import torch
from doremi.config import DoReMiConfig
from doremi.dataloader import get_dataloader, get_datasets
from doremi.trainer import ReferenceTrainer
from doremi.utils import compute_domain_weights_based_on_token_count

from nanotron.config import get_config_from_file


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
    config = get_config_from_file(config_file, config_class=DoReMiConfig)

    dataset_paths = [f"{config.data.dataset.hf_dataset_or_datasets}/{name}" for name in config.doremi.domain_names]
    datasets = get_datasets(dataset_paths)

    # TODO(xrsrke): add retrieving domain weights from config
    # or calculate it in the trainer
    if config.doremi.domain_weights is None:
        initial_domain_weights = compute_domain_weights_based_on_token_count(datasets)
    else:
        initial_domain_weights = torch.tensor(config.doremi.domain_weights)

    assert torch.allclose(initial_domain_weights.sum(), torch.tensor(1.0), rtol=1e-3)

    domain_names = config.doremi.domain_names
    trainer = ReferenceTrainer(initial_domain_weights, domain_names, config_file, config_class=DoReMiConfig)
    dataloader = get_dataloader(trainer, datasets)
    trainer.train(dataloader)
