# [MODULE DOC]
# Activation function library (adapted from HuggingFace transformers).
# Main contents: a set of torch.nn.Module activation classes (GELU variants, SiLU, Mish, Laplace, ReLU squared,
# identity), the ACT2CLS name-to-class table, the ACT2FN lazily-instantiating dictionary, get_activation, and
# module-level pre-built instances (gelu, gelu_new, gelu_fast, quick_gelu, silu, mish, linear_act, gelu_python).
# Fit in nanotron: model configs name their hidden activation by string (for example "silu" or "gelu") and model
# code calls ACT2FN with that string to get a fresh module instance.
# Import-time side effects: creates a module logger and instantiates the backward-compatibility activation objects.
# Exports: the classes above, ACT2CLS, ACT2FN, get_activation and the pre-built instances.
# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from collections import OrderedDict

import torch
from packaging import version
from torch import Tensor, nn

from nanotron import logging

logger = logging.get_logger(__name__)


class PytorchGELUTanh(nn.Module):
    # [CLASS DOC]
    # Tanh-approximated GELU using the native PyTorch kernel (gelu with approximate set to tanh).
    # Inheritance: torch.nn.Module with no parameters or buffers.
    # Construction inputs: none. Requires torch 1.12.0 or newer.
    """
    A fast C implementation of the tanh approximation of the GeLU activation function. See
    https://arxiv.org/abs/1606.08415.

    This implementation is equivalent to NewGELU and FastGELU but much faster. However, it is not an exact numerical
    match due to rounding errors.
    """

    def __init__(self):
        # [FUNCTION DOC]
        # Initializes the module and checks the installed torch version.
        # Inputs: none.
        # Outputs: None.
        # Exceptions: ImportError if torch is older than 1.12.0.
        # Side effects: None.
        super().__init__()
        if version.parse(torch.__version__) < version.parse("1.12.0"):
            raise ImportError(
                f"You are using torch=={torch.__version__}, but torch>=1.12.0 is required to use "
                "PytorchGELUTanh. Please upgrade torch."
            )

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC]
        # Applies the tanh-approximated GELU elementwise.
        # Inputs: input, a tensor of any shape.
        # Outputs: tensor of the same shape and dtype.
        # Exceptions: None explicitly raised.
        # Side effects: None.
        return nn.functional.gelu(input, approximate="tanh")


class NewGELUActivation(nn.Module):
    # [CLASS DOC]
    # GELU tanh approximation written out in Python (BERT/GPT variant):
    # one half times x times (1 plus tanh of sqrt(2/pi) times (x plus 0.044715 times x cubed)).
    # Inheritance: torch.nn.Module, stateless.
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC]
        # Computes the tanh-approximated GELU formula elementwise.
        # Inputs: input, tensor of any shape.
        # Outputs: tensor of the same shape.
        # Exceptions: None explicitly raised.
        # Side effects: None.
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class GELUActivation(nn.Module):
    # [CLASS DOC]
    # Exact (erf-based) GELU. Either uses torch's native gelu or a pure Python erf formula.
    # Key attributes: act, the callable applied in forward.
    # Construction inputs: use_gelu_python (bool, default False) selects the Python implementation.
    # Inheritance: torch.nn.Module, stateless.
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, use_gelu_python: bool = False):
        # [FUNCTION DOC]
        # Chooses the implementation: the bound _gelu_python method when use_gelu_python is true, otherwise torch's
        # functional gelu.
        # Inputs: use_gelu_python, bool, default False.
        # Outputs: None.
        # Exceptions: None explicitly raised.
        # Side effects: sets self.act.
        super().__init__()
        if use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = nn.functional.gelu

    def _gelu_python(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC]
        # HELPER: Python fallback implementation of exact GELU for GELUActivation.
        # Computes x times one half times (1 plus erf of x divided by sqrt 2).
        # Inputs: input, tensor of any shape.
        # Outputs: tensor of the same shape.
        # Exceptions: None explicitly raised.
        # Side effects: None.
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC]
        # Applies self.act to the input.
        # Inputs: input, tensor of any shape.
        # Outputs: tensor of the same shape.
        # Exceptions: None explicitly raised.
        # Side effects: None.
        return self.act(input)


class FastGELUActivation(nn.Module):
    # [CLASS DOC]
    # Fast tanh GELU approximation using the constant 0.7978845608 (sqrt of 2/pi) and the 0.044715 cubic term,
    # factored as x times (1 plus 0.044715 x squared). Stateless torch.nn.Module.
    """
    Applies GELU approximation that is slower than QuickGELU but more accurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC]
        # Computes one half x times (1 plus tanh of 0.7978845608 x times (1 plus 0.044715 x squared)).
        # Inputs: input, tensor. Outputs: tensor of the same shape.
        # Exceptions: None explicitly raised. Side effects: None.
        return 0.5 * input * (1.0 + torch.tanh(input * 0.7978845608 * (1.0 + 0.044715 * input * input)))


class QuickGELUActivation(nn.Module):
    # [CLASS DOC]
    # Sigmoid-based GELU approximation: x times sigmoid of 1.702 x. Stateless torch.nn.Module.
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC]
        # Computes input times sigmoid of 1.702 times input.
        # Inputs: input, tensor. Outputs: tensor of the same shape.
        # Exceptions: None explicitly raised. Side effects: None.
        return input * torch.sigmoid(1.702 * input)


class ClippedGELUActivation(nn.Module):
    # [CLASS DOC]
    # GELU whose output is clamped into the closed range from min to max (useful for quantization).
    # Key attributes: min and max (floats).
    # Construction inputs: min, max. Inheritance: torch.nn.Module.
    # Invariant: min is not greater than max.
    """
    Clip the range of possible GeLU outputs between [min, max]. This is especially useful for quantization purpose, as
    it allows mapping negatives values in the GeLU spectrum. For more information on this trick, please refer to
    https://arxiv.org/abs/2004.09602.

    Gaussian Error Linear Unit. Original Implementation of the gelu activation function in Google Bert repo when
    initially created.

    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))). See https://arxiv.org/abs/1606.08415
    """

    def __init__(self, min: float, max: float):
        # [FUNCTION DOC]
        # Validates and stores the clipping bounds.
        # Inputs: min (float), max (float).
        # Outputs: None.
        # Exceptions: ValueError when min is greater than max (raised before the Module base is initialized).
        # Side effects: sets self.min and self.max.
        if min > max:
            raise ValueError(f"min should be < max (got min: {min}, max: {max})")

        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x: Tensor) -> Tensor:
        # [FUNCTION DOC]
        # Applies the module-level gelu instance (exact GELU) and clips the result between self.min and self.max.
        # Inputs: x, tensor. Outputs: tensor of the same shape.
        # Exceptions: None explicitly raised. Side effects: None.
        return torch.clip(gelu(x), self.min, self.max)


class AccurateGELUActivation(nn.Module):
    # [CLASS DOC]
    # Tanh GELU approximation with a precomputed sqrt(2/pi) constant (from MEGA).
    # Key attributes: precomputed_constant (float). Inheritance: torch.nn.Module.
    """
    Applies GELU approximation that is faster than default and more accurate than QuickGELU. See:
    https://github.com/hendrycks/GELUs

    Implemented along with MEGA (Moving Average Equipped Gated Attention)
    """

    def __init__(self):
        # [FUNCTION DOC]
        # Stores precomputed_constant as the square root of 2 divided by pi.
        # Inputs: none. Outputs: None. Exceptions: None explicitly raised. Side effects: sets self.precomputed_constant.
        super().__init__()
        self.precomputed_constant = math.sqrt(2 / math.pi)

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC]
        # Computes one half x times (1 plus tanh of precomputed_constant times (x plus 0.044715 x cubed)).
        # Inputs: input, tensor. Outputs: tensor of the same shape.
        # Exceptions: None explicitly raised. Side effects: None.
        return 0.5 * input * (1 + torch.tanh(self.precomputed_constant * (input + 0.044715 * torch.pow(input, 3))))


class SiLUActivation(nn.Module):
    # [CLASS DOC]
    # SiLU / Swish activation (x times sigmoid x) via torch's functional silu. Stateless torch.nn.Module.
    """
    See Gaussian Error Linear Units (Hendrycks et al., https://arxiv.org/abs/1606.08415) where the SiLU (Sigmoid Linear
    Unit) was originally introduced and coined, and see Sigmoid-Weighted Linear Units for Neural Network Function
    Approximation in Reinforcement Learning (Elfwing et al., https://arxiv.org/abs/1702.03118) and Swish: a Self-Gated
    Activation Function (Ramachandran et al., https://arxiv.org/abs/1710.05941v1) where the SiLU was experimented with
    later.
    """

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC]
        # Applies torch functional silu.
        # Inputs: input, tensor. Outputs: tensor of the same shape.
        # Exceptions: None explicitly raised. Side effects: None.
        return nn.functional.silu(input)


class MishActivation(nn.Module):
    # [CLASS DOC]
    # Mish activation: x times tanh of softplus x. Uses torch's native mish when torch is 1.9.0 or newer, else a Python
    # fallback. Key attributes: act, the chosen callable. Inheritance: torch.nn.Module.
    """
    See Mish: A Self-Regularized Non-Monotonic Activation Function (Misra., https://arxiv.org/abs/1908.08681). Also
    visit the official repository for the paper: https://github.com/digantamisra98/Mish
    """

    def __init__(self):
        # [FUNCTION DOC]
        # Selects self.act: the _mish_python method for torch older than 1.9.0, otherwise torch functional mish.
        # Inputs: none. Outputs: None. Exceptions: None explicitly raised. Side effects: sets self.act.
        super().__init__()
        if version.parse(torch.__version__) < version.parse("1.9.0"):
            self.act = self._mish_python
        else:
            self.act = nn.functional.mish

    def _mish_python(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC]
        # HELPER: Python fallback for Mish used on old torch versions.
        # Computes input times tanh of softplus of input.
        # Inputs: input, tensor. Outputs: tensor of the same shape.
        # Exceptions: None explicitly raised. Side effects: None.
        return input * torch.tanh(nn.functional.softplus(input))

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC]
        # Applies self.act.
        # Inputs: input, tensor. Outputs: tensor of the same shape.
        # Exceptions: None explicitly raised. Side effects: None.
        return self.act(input)


class LinearActivation(nn.Module):
    # [CLASS DOC]
    # Identity activation; returns its input unchanged. Stateless torch.nn.Module.
    """
    Applies the linear activation function, i.e. forwarding input directly to output.
    """

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC]
        # Returns input as is (same tensor object).
        # Inputs: input, tensor. Outputs: the input tensor.
        # Exceptions: None explicitly raised. Side effects: None.
        return input


class LaplaceActivation(nn.Module):
    # [CLASS DOC]
    # Laplace-function activation from MEGA: a Gaussian CDF centered at mu with scale sigma. Stateless torch.nn.Module.
    """
    Applies elementwise activation based on Laplace function, introduced in MEGA as an attention activation. See
    https://arxiv.org/abs/2209.10655

    Inspired by squared relu, but with bounded range and gradient for better stability
    """

    def forward(self, input, mu=0.707107, sigma=0.282095):
        # [FUNCTION DOC]
        # Normalizes the input as (input minus mu) divided by (sigma times sqrt 2) and returns one half times
        # (1 plus erf of that value).
        # Inputs: input, tensor; mu, float default 0.707107; sigma, float default 0.282095.
        # Outputs: tensor of the same shape with values in the range 0 to 1.
        # Exceptions: None explicitly raised. Side effects: None.
        input = (input - mu).div(sigma * math.sqrt(2.0))
        return 0.5 * (1.0 + torch.erf(input))


class ReLUSquaredActivation(nn.Module):
    # [CLASS DOC]
    # Squared ReLU activation (relu of x, then squared). Stateless torch.nn.Module.
    """
    Applies the relu^2 activation introduced in https://arxiv.org/abs/2109.08668v2
    """

    def forward(self, input):
        # [FUNCTION DOC]
        # Applies relu then elementwise square.
        # Inputs: input, tensor. Outputs: tensor of the same shape.
        # Exceptions: None explicitly raised. Side effects: None.
        relu_applied = nn.functional.relu(input)
        squared = torch.square(relu_applied)
        return squared


class ClassInstantier(OrderedDict):
    # [CLASS DOC]
    # OrderedDict subclass whose item lookup instantiates the stored class instead of returning it.
    # Values are either a class or a pair of (class, keyword-argument dictionary). Used to build ACT2FN.
    def __getitem__(self, key):
        # [FUNCTION DOC]
        # Looks up key, unpacks a (class, kwargs) pair (or uses an empty kwargs dictionary for a bare class), and returns a
        # new instance built with those keyword arguments.
        # Inputs: key, activation name string.
        # Outputs: a freshly constructed activation module.
        # Exceptions: KeyError if key is absent; any exception from the class constructor.
        # Side effects: None.
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


ACT2CLS = {
    "gelu": GELUActivation,
    "gelu_10": (ClippedGELUActivation, {"min": -10, "max": 10}),
    "gelu_fast": FastGELUActivation,
    "gelu_new": NewGELUActivation,
    "gelu_python": (GELUActivation, {"use_gelu_python": True}),
    "gelu_pytorch_tanh": PytorchGELUTanh,
    "gelu_accurate": AccurateGELUActivation,
    "laplace": LaplaceActivation,
    "linear": LinearActivation,
    "mish": MishActivation,
    "quick_gelu": QuickGELUActivation,
    "relu": nn.ReLU,
    "relu2": ReLUSquaredActivation,
    "relu6": nn.ReLU6,
    "sigmoid": nn.Sigmoid,
    "silu": SiLUActivation,
    "swish": SiLUActivation,
    "tanh": nn.Tanh,
}
ACT2FN = ClassInstantier(ACT2CLS)


def get_activation(activation_string):
    # [FUNCTION DOC]
    # Returns a new activation module instance for the given name from ACT2FN.
    # Inputs: activation_string, name such as "gelu" or "silu".
    # Outputs: a torch.nn.Module instance.
    # Exceptions: KeyError listing available names when the name is unknown.
    # Side effects: None.
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError(f"function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}")


# For backwards compatibility with: from activations import gelu_python
gelu_python = get_activation("gelu_python")
gelu_new = get_activation("gelu_new")
gelu = get_activation("gelu")
gelu_fast = get_activation("gelu_fast")
quick_gelu = get_activation("quick_gelu")
silu = get_activation("silu")
mish = get_activation("mish")
linear_act = get_activation("linear")
