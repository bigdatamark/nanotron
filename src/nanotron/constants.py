# [MODULE DOC]
# Purpose: global constants for nanotron.
# Contents: CHECKPOINT_VERSION (packaging Version 1.5, written into checkpoint metadata and compared on load),
# PY_VERSION (parsed version of the running Python interpreter), CHECKPOINT_FILE_NAME (name of the checkpoint
# metadata JSON file) and MODEL_CONFIG_FILE_NAME (name of the saved model config JSON).
# Import-time side effects: queries the platform for the Python version. No functions or classes.
import platform

from packaging.version import Version, parse

CHECKPOINT_VERSION = Version("1.5")

PY_VERSION = parse(platform.python_version())

#### FOR SERIALIZATION ####

CHECKPOINT_FILE_NAME = "checkpoint_metadata.json"
MODEL_CONFIG_FILE_NAME = "model_config.json"
