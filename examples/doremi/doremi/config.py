# [MODULE DOC]
# Purpose: Configuration dataclasses for the DoReMi example (domain reweighting with a proxy and a reference model) layered on top of nanotron's Config.
# Main contents: DoReMiArgs (DoReMi hyperparameters and domain info) and DoReMiConfig (nanotron Config extended with a doremi section).
# Side effects at import: None.
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import torch
from nanotron.config import Config


@dataclass
class DoReMiArgs:
    # [CLASS DOC]
    # Purpose: Holds DoReMi hyperparameters parsed from the YAML "doremi" section.
    # Attributes: smoothing_param (float, default 1e-3, mixing with uniform), step_size (float, default 1.0, exponentiated-gradient step), domain_names (comma-separated string normalized to a list of strings), domain_weights (optional comma-separated string or list of floats, normalized to list of floats summing to 1), ref_model_resume_checkpoint_path (path of the reference model checkpoint, normalized to Path).
    # Invariants after construction: domain_names is a list; domain_weights, if given, sums to 1 within rtol 1e-3.
    smoothing_param: float = 1e-3
    step_size: float = 1.0

    domain_names: Optional[Union[str, List[str]]] = None
    domain_weights: Optional[Union[str, List[float]]] = None

    # NOTE: the path where you want to load the
    # reference model checkpoint for proxy training
    ref_model_resume_checkpoint_path: Optional[Path] = None

    def __post_init__(self):
        # [FUNCTION DOC]
        # Behavior: Asserts domain_names and ref_model_resume_checkpoint_path are provided; splits domain_names on commas and strips; if domain_weights is a string, splits and parses floats, otherwise uses it as given; asserts weights sum to 1 (rtol 1e-3); converts the checkpoint path to Path.
        # Inputs: self. Outputs: None.
        # Exceptions: AssertionError on missing fields or weights not summing to 1; ValueError on unparsable floats.
        # Side effects: mutates self fields in place.
        # KNOWN ISSUE: domain_names is typed as string or list, but a list input fails with AttributeError since split is called unconditionally.
        assert self.domain_names is not None, "Domain names must be provided."
        assert self.ref_model_resume_checkpoint_path is not None, "Reference model checkpoint path must be provided."

        self.domain_names = [str(name.strip()) for name in self.domain_names.split(",")]

        if self.domain_weights is not None:
            if isinstance(self.domain_weights, str):
                domain_weights = [float(weight.strip()) for weight in self.domain_weights.split(",")]
            else:
                domain_weights = self.domain_weights

            assert torch.allclose(
                torch.tensor(domain_weights).sum(), torch.tensor(1.0), rtol=1e-3
            ), "Domain weights must sum to 1.0."
            self.domain_weights = domain_weights

        self.ref_model_resume_checkpoint_path = Path(self.ref_model_resume_checkpoint_path)


@dataclass(kw_only=True)  # pylint: disable=unexpected-keyword-arg
class DoReMiConfig(Config):
    # [CLASS DOC]
    # Purpose: nanotron Config subclass (keyword-only dataclass) adding the doremi field of type DoReMiArgs, used for proxy training.
    # Attributes: all Config fields plus doremi.
    """Configuration for DoReMi's Proxy Training."""

    doremi: DoReMiArgs
