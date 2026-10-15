# [MODULE DOC]
# Normalization layers for nanotron.
# Main contents: TritonLayerNorm and TritonRMSNorm (fused Triton kernels from flash_attn supporting fused residual
# add and dropout), and LlamaRMSNorm (plain PyTorch RMSNorm).
# Import-time side effects: imports flash_attn.ops.triton.layer_norm, so flash_attn with Triton must be installed.
# Exports: TritonLayerNorm, TritonRMSNorm, LlamaRMSNorm.
import torch
from flash_attn.ops.triton.layer_norm import layer_norm_fn
from torch import nn


class TritonLayerNorm(nn.LayerNorm):
    # [CLASS DOC]
    # torch.nn.LayerNorm subclass whose forward uses flash_attn's fused Triton layer_norm_fn.
    # Key attributes inherited from LayerNorm: weight, bias, eps. Construction inputs: same as torch.nn.LayerNorm.
    def forward(
        self, input, residual=None, dropout_p=0.0, prenorm=False, residual_in_fp32=False, return_dropout_mask=False
    ):

        # [FUNCTION DOC]
        # Calls layer_norm_fn with LayerNorm mode (not RMS), self.weight, self.bias and self.eps, optionally fusing a
        # residual add and dropout before normalization.
        # Inputs: input tensor [..., hidden]; residual (optional tensor added before norm, default None); dropout_p (float,
        # default 0.0); prenorm (bool, default False, also return the pre-norm sum); residual_in_fp32 (bool, default False);
        # return_dropout_mask (bool, default False).
        # Outputs: the normalized tensor, or a tuple that additionally includes the residual and/or dropout mask depending
        # on prenorm and return_dropout_mask (as defined by layer_norm_fn).
        # Exceptions: None explicitly raised. Side effects: dropout consumes CUDA RNG when dropout_p is positive.
        return layer_norm_fn(
            input,
            self.weight,
            self.bias,
            residual=residual,
            eps=self.eps,
            dropout_p=dropout_p,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
            is_rms_norm=False,
            return_dropout_mask=return_dropout_mask,
        )


# This is equivalent to LLaMA RMSNorm
# https://github.com/huggingface/transformers/blob/28952248b19db29ca25ccf34a5eec413376494a9/src/transformers/models/llama/modeling_llama.py#L112
class TritonRMSNorm(nn.Module):
    # [CLASS DOC]
    # RMSNorm (LLaMA-equivalent) backed by flash_attn's fused Triton layer_norm_fn.
    # Key attributes: eps (float), weight (Parameter of shape [hidden_size]), bias registered as None.
    # Construction inputs: hidden_size, eps (default 1e-5), device, dtype. Inheritance: torch.nn.Module.
    def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None):
        # [FUNCTION DOC]
        # Creates an uninitialized weight of shape [hidden_size] on the given device/dtype, registers bias as None, then
        # calls reset_parameters.
        # Inputs: hidden_size (int); eps (float, default 1e-5); device and dtype (optional).
        # Outputs: None. Exceptions: None explicitly raised. Side effects: allocates the parameter.
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        # [FUNCTION DOC]
        # Fills weight with ones.
        # Inputs: none. Outputs: None. Exceptions: None explicitly raised. Side effects: mutates self.weight in place.
        nn.init.ones_(self.weight)

    def forward(
        self, input, residual=None, dropout_p=0.0, prenorm=False, residual_in_fp32=False, return_dropout_mask=False
    ):

        # [FUNCTION DOC]
        # Same as TritonLayerNorm.forward but in RMS mode with no bias.
        # Inputs: input, residual, dropout_p, prenorm, residual_in_fp32, return_dropout_mask (see TritonLayerNorm.forward).
        # Outputs: normalized tensor or tuple per layer_norm_fn.
        # Exceptions: None explicitly raised. Side effects: dropout consumes CUDA RNG when dropout_p is positive.
        return layer_norm_fn(
            input,
            self.weight,
            None,
            residual=residual,
            eps=self.eps,
            dropout_p=dropout_p,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
            is_rms_norm=True,
            return_dropout_mask=return_dropout_mask,
        )


class LlamaRMSNorm(nn.Module):
    # [CLASS DOC]
    # Pure PyTorch RMSNorm (equivalent to T5LayerNorm): normalizes by root mean square in float32 then scales.
    # Key attributes: weight (Parameter of ones, shape [hidden_size]), eps (float).
    # Construction inputs: hidden_size, eps (default 1e-6). Inheritance: torch.nn.Module.
    def __init__(self, hidden_size, eps=1e-6):
        # [FUNCTION DOC]
        # Creates weight initialized to ones and stores eps.
        # Inputs: hidden_size (int); eps (float, default 1e-6).
        # Outputs: None. Exceptions: None explicitly raised. Side effects: allocates the parameter.
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, input):
        # [FUNCTION DOC]
        # Casts input to float32, divides by the square root of (mean of squares over the last dimension plus eps), casts back
        # to the original dtype and multiplies by weight.
        # Inputs: input tensor [..., hidden_size].
        # Outputs: tensor of the same shape (dtype follows type promotion between weight and input dtype).
        # Exceptions: None explicitly raised. Side effects: None.
        input_dtype = input.dtype
        input = input.to(torch.float32)
        variance = input.pow(2).mean(-1, keepdim=True)
        input = input * torch.rsqrt(variance + self.eps)
        return self.weight * input.to(input_dtype)

    def extra_repr(self):
        # [FUNCTION DOC]
        # Returns the string shown in the module repr: weight shape tuple and eps.
        # Inputs: none. Outputs: str. Exceptions: None explicitly raised. Side effects: None.
        return f"{tuple(self.weight.shape)}, eps={self.eps}"
