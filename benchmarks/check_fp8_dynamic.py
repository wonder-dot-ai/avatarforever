"""H100 numerical and dispatch checks for the dynamic FP8 inference path."""

import json

import torch

from ltx_core.loader import SingleGPUModelBuilder  # noqa: F401 (initialize existing loader import cycle)
from ltx_core.quantization.fp8_dynamic import DynamicFP8Linear, quantize_per_tensor, quantize_weight


@torch.inference_mode()
def main():
    torch.manual_seed(42)
    weight = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16) * 0.05
    entries = dict(quantize_weight("transformer_blocks.0.attn1.to_q.weight", weight))
    layer = DynamicFP8Linear(128, 256, True, device="cuda")
    layer.weight = torch.nn.Parameter(entries["transformer_blocks.0.attn1.to_q.weight"], requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(entries["transformer_blocks.0.attn1.to_q.weight_scale"], requires_grad=False)
    layer.bias.zero_()
    weight_pointer = layer.weight.data_ptr()
    checks = []
    for shape, magnitude in [((2, 17, 128), 1.0), ((1, 32, 128), 0.0), ((1, 31, 128), 1e-8), ((1, 48, 128), 1e3)]:
        x = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * magnitude
        y = layer(x)
        qx, sx = quantize_per_tensor(x)
        reference = torch.nn.functional.linear(qx.float() * sx, layer.weight.t().float() * layer.weight_scale)
        assert y.shape == (*shape[:-1], 256) and y.dtype == x.dtype
        assert torch.isfinite(y).all()
        torch.testing.assert_close(y.float(), reference, rtol=0.02, atol=max(magnitude * 0.015, 1e-12))
        if magnitude == 0:
            assert torch.count_nonzero(y) == 0
        baseline = torch.nn.functional.linear(x, weight)
        relative_rmse = ((y.float() - baseline.float()).square().mean().sqrt() /
                         baseline.float().square().mean().sqrt().clamp_min(1e-12)).item()
        assert relative_rmse < 0.08, relative_rmse
        assert layer.weight.data_ptr() == weight_pointer and layer.weight.dtype == torch.float8_e4m3fn
        checks.append({"shape": shape, "magnitude": magnitude, "relative_rmse_vs_bf16": relative_rmse})
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        layer(x)
    names = [event.key for event in prof.key_averages()]
    assert "aten::_scaled_mm" in names, names
    print(json.dumps({"checks": checks, "native_scaled_mm_observed": True}, indent=2))


if __name__ == "__main__":
    main()
