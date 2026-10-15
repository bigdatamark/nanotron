# [MODULE DOC]
# Purpose: entry-point script to train the example Mamba model with nanotron.
# Behavior when run: parses --config-file, builds a MambaTrainer with MambaModelConfig and MambaForTraining,
# obtains a dataloader from the repo's run_train.get_dataloader, and calls trainer.train.
# Import-time side effects: appends the repository root (two levels up) to sys.path so run_train is
# importable; creates a module logger. Must be run from examples/mamba (imports config, mamba, trainer as top-level modules).
import argparse
import os
import sys

from config import MambaModelConfig
from mamba import MambaForTraining
from trainer import MambaTrainer

from nanotron import logging

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from run_train import get_dataloader  # noqa

logger = logging.get_logger(__name__)


def get_args():
    # [FUNCTION DOC]
    # HELPER: command-line parsing for the script.
    # Behavior: defines a required string option --config-file (YAML or python config path) and parses sys.argv.
    # Inputs: none. Outputs: argparse Namespace with config_file. Exceptions: SystemExit on missing/invalid args.
    # Side effects: reads sys.argv; may print usage to stderr.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True, help="Path to the YAML or python config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config_file = args.config_file

    # Load trainer and data
    trainer = MambaTrainer(config_file, model_config_class=MambaModelConfig, model_class=MambaForTraining)
    dataloader = get_dataloader(trainer)

    # Train
    trainer.train(dataloader)
