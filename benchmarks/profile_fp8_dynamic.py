"""Time representative linear operations after the full inference benchmark."""
import json
import time

import torch
import torch.nn.functional as F

from ltx_core.loader import SingleGPUModelBuilder  # noqa: F401
from ltx_core.quantization.fp8_dynamic import quantize_activation, quantize_per_tensor


def measure(fn, repeats=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / repeats


@torch.inference_mode()
def main():
    torch.manual_seed(42)
    results = []
    for name, m, k, n in [('video_projection', 1536, 4096, 4096),
                           ('video_ffn_expansion', 1536, 4096, 16384),
                           ('audio_projection', 96, 2048, 2048)]:
        x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
        w = torch.randn(n, k, device='cuda', dtype=torch.bfloat16) * 0.02
        qw, sw = quantize_per_tensor(w)
        qw = qw.t()
        qx, sx = quantize_activation(x)
        def mm(a, scale):
            return torch._scaled_mm(a, qw, scale_a=scale, scale_b=sw,
                                    out_dtype=torch.bfloat16, use_fast_accum=False)
        operations = {
            'bf16_gemm': lambda: F.linear(x, w),
            'dynamic_activation_quantization': lambda: quantize_activation(x),
            'fp8_gemm_prequantized_inputs': lambda: mm(qx, sx),
            'dynamic_quantization_plus_fp8_gemm': lambda: mm(*quantize_activation(x)),
            'fp8_weight_upcast_only': lambda: qw.to(torch.bfloat16),
        }
        timings = {key: measure(fn) for key, fn in operations.items()}
        results.append({'name': name, 'M': m, 'K': k, 'N': n, 'milliseconds_per_call': timings})
    print(json.dumps({'method': 'Five warmups, 30 calls, synchronized wall-clock average. '
                      'Representative shapes; excludes bias/padding and full pipeline overhead.',
                      'results': results}, indent=2))


if __name__ == '__main__':
    main()
