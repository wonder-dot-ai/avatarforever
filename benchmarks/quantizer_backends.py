"""Compare only activation quantization; no transformer/GEMM capture."""
import json
import statistics
import time

import torch
from ltx_core.loader import SingleGPUModelBuilder  # noqa: F401
from ltx_core.quantization.fp8_dynamic import quantize_activation, quantize_per_tensor
from ltx_core.quantization.fp8_quantizer import CapturedQuantizer, fused_quantize


def timing(fn, x):
    for _ in range(10):
        fn(x)
    times = []
    for _ in range(5):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            fn(x)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1e6 / 100)
    return statistics.median(times)


@torch.inference_mode()
def main():
    torch.manual_seed(42)
    methods = {'previous_partial_compile': quantize_activation,
               'previous_quantizer_cudagraph': CapturedQuantizer(quantize_activation),
               'fused_triton': fused_quantize,
               'fused_triton_cudagraph': CapturedQuantizer(fused_quantize)}
    records = []
    rounding_checks = []
    for shape in [(96, 2048), (1536, 4096), (1536, 16384), (4608, 4096)]:
        for magnitude in (0, 1e-8, 1, 1000):
            # New values on every call verify replay does not use stale inputs/scales.
            x = torch.randn(shape, device='cuda', dtype=torch.bfloat16) * magnitude
            expected, scale = quantize_per_tensor(x)
            for name, fn in methods.items():
                actual, actual_scale = fn(x)
                torch.testing.assert_close(actual_scale, scale, rtol=1e-6, atol=0)
                a, b = actual.float(), expected.float()
                mismatch = (a != b).float().mean().item()
                assert mismatch < 1e-5, (name, shape, magnitude, mismatch)
                torch.testing.assert_close(a, b, rtol=0.126, atol=0.002, msg=name)
                rounding_checks.append({'backend': name, 'shape': shape, 'magnitude': magnitude,
                                        'different_fraction': mismatch})
        x = torch.randn(shape, device='cuda', dtype=torch.bfloat16)
        records.append({'shape': shape, 'quantizer_us': {name: timing(fn, x) for name, fn in methods.items()}})
    print(json.dumps({'method': 'Median of five groups of 100 calls after warmup; synchronized wall-clock. '
                      'CUDA graph timings include copying each input into static storage. '
                      'Only quantization is captured. No GEMM or transformer compilation/capture.',
                      'correctness': 'Scale tolerance 1e-6; fewer than 1e-5 FP8 values differ, by at most one FP8 step; replay with changed inputs.',
                      'rounding_checks': rounding_checks, 'results': records}, indent=2))


if __name__ == '__main__':
    main()
