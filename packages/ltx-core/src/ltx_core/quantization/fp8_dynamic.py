"""Native FP8 GEMM with per-tensor dynamic activation scaling (Hopper+).

Load ordinary BF16 weights, quantize selected linears once, and retain BF16
outputs/biases. Only activation quantization is compiled, not the transformer.
The transposed weight layout also matches the existing scaled-FP8 LoRA merger.
"""

import torch
from torch import nn

from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.sd_ops import KeyValueOperationResult, SDOps
from ltx_core.model.transformer import LTXModel

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
LINEAR_SUFFIXES = (".to_q", ".to_k", ".to_v", ".to_out.0", "ff.net.0.proj", "ff.net.2")


def quantize_per_tensor(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return E4M3 values and a FP32 dequantization scale; handle all-zero inputs."""
    values = x.float()
    scale = values.abs().amax().clamp_min(1e-12) / FP8_MAX
    quantized = (values / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return quantized, scale.reshape(())


@torch.compile(dynamic=True, fullgraph=True)
def _quantize_with_amax(x: torch.Tensor, amax: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = amax.float().clamp_min(1e-12) / FP8_MAX
    quantized = (x.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return quantized, scale.reshape(())


def quantize_activation(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Keep the global reduction separate: compiling it together with the cast
    # under dynamic shapes can serialize the entire tensor onto one CUDA block.
    # Native amax uses a parallel reduction; the compiled conversion is pointwise.
    return _quantize_with_amax(x, x.abs().amax())


class DynamicFP8Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, bias: bool, device: torch.device | str | None = None,
    ):
        super().__init__()
        if in_features % 16 or out_features % 16:
            raise ValueError("FP8 GEMM requires input/output feature sizes divisible by 16")
        self.in_features = in_features
        self.out_features = out_features
        # Column-major (K, N), without a per-forward transpose/copy of weights.
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn, device=device).t(),
            requires_grad=False,
        )
        self.weight_scale = nn.Parameter(torch.empty((), dtype=torch.float32, device=device), requires_grad=False)
        self.bias = nn.Parameter(torch.empty(out_features, device=device), requires_grad=False) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype not in (torch.bfloat16, torch.float16):
            raise TypeError(f"Dynamic FP8 expects BF16/FP16 activations, got {x.dtype}")
        flat = x.reshape(-1, self.in_features).contiguous()
        # Some GEMM backends require M aligned to 16 (audio lengths can be odd).
        rows = flat.shape[0]
        padding = (-rows) % 16
        if padding:
            flat = torch.nn.functional.pad(flat, (0, 0, 0, padding))
        qinput, input_scale = quantize_activation(flat)
        output = torch._scaled_mm(
            qinput, self.weight, scale_a=input_scale, scale_b=self.weight_scale,
            out_dtype=x.dtype, use_fast_accum=False,
        )
        output = output[:rows]
        if self.bias is not None:
            output = output + self.bias.to(x.dtype)
        return output.reshape(*x.shape[:-1], self.out_features)


def selected_linear(name: str) -> bool:
    return name.startswith("transformer_blocks.") and name.endswith(LINEAR_SUFFIXES)


def quantize_weight(key: str, value: torch.Tensor) -> list[KeyValueOperationResult]:
    if not selected_linear(key.removesuffix(".weight")):
        return [KeyValueOperationResult(key, value)]
    if value.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("fp8-dynamic requires an unquantized checkpoint")
    weight, scale = quantize_per_tensor(value)
    return [
        KeyValueOperationResult(key, weight.t()),
        KeyValueOperationResult(key.removesuffix(".weight") + ".weight_scale", scale),
    ]


def replace_linears(model: nn.Module) -> nn.Module:
    for name, layer in list(model.named_modules()):
        if isinstance(layer, nn.Linear) and selected_linear(name):
            parent, attr = name.rsplit(".", 1)
            setattr(model.get_submodule(parent), attr, DynamicFP8Linear(
                layer.in_features, layer.out_features, layer.bias is not None, device=layer.weight.device,
            ))
    return model


DYNAMIC_FP8_SD_OPS = SDOps("dynamic_fp8_weights").with_kv_operation(
    quantize_weight, key_prefix="transformer_blocks.", key_suffix=".weight",
)
DYNAMIC_FP8_MODULE_OPS = ModuleOps(
    name="dynamic_fp8_linears", matcher=lambda model: isinstance(model, LTXModel), mutator=replace_linears,
)
