"""Parallel tensorwise FP8 activation quantization and optional quantizer-only graphs."""

from collections.abc import Callable

import torch
import triton
import triton.language as tl


@triton.jit
def _partial_max(X, Partial, N: tl.constexpr, BLOCK: tl.constexpr) -> None:  # noqa: ANN001, N803
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(X + offsets, offsets < N, other=0).to(tl.float32)
    tl.store(Partial + tl.program_id(0), tl.max(tl.abs(values), 0))


@triton.jit
def _scale(Partial, Scale, COUNT: tl.constexpr, BLOCK: tl.constexpr) -> None:  # noqa: ANN001, N803
    offsets = tl.arange(0, BLOCK)
    maximum = tl.max(tl.load(Partial + offsets, offsets < COUNT, other=0), 0)
    tl.store(Scale, tl.maximum(maximum, 1e-12) * (1.0 / 448.0))


@triton.jit
def _cast(X, Scale, Q, N: tl.constexpr, BLOCK: tl.constexpr) -> None:  # noqa: ANN001, N803
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(X + offsets, offsets < N, other=0).to(tl.float32)
    scaled = tl.div_rn(values, tl.load(Scale))
    tl.store(Q + offsets, tl.minimum(tl.maximum(scaled, -448.0), 448.0), offsets < N)


def fused_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Three launches: fused abs/max partials, scalar reduction/scale, scale/clamp/cast."""
    if not x.is_cuda or not x.is_contiguous() or not x.numel():
        raise ValueError("Expected a nonempty contiguous CUDA tensor")
    count = triton.cdiv(x.numel(), 4096)
    partial = torch.empty(count, dtype=torch.float32, device=x.device)
    scale = torch.empty((), dtype=torch.float32, device=x.device)
    output = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    _partial_max[(count,)](x, partial, x.numel(), 4096)
    _scale[(1,)](partial, scale, count, triton.next_power_of_2(count))
    _cast[(count,)](x, scale, output, x.numel(), 4096)
    return output, scale


class CapturedQuantizer:
    """One graph per shape/device/dtype/stream, shared across sequential linears.

    Includes input copy cost. Returned buffers are overwritten on the next call
    of the same shape; callers must consume them on the calling CUDA stream.
    This is intended for sequential inference, not concurrent host-thread use.
    """

    def __init__(self, quantizer: Callable = fused_quantize):
        self.quantizer = quantizer
        self.entries = {}

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        current = torch.cuda.current_stream(x.device)
        key = (x.device, x.dtype, tuple(x.shape), current.cuda_stream)
        if key not in self.entries:
            static = torch.empty_like(x)
            static.copy_(x)
            stream = torch.cuda.Stream(device=x.device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.quantizer(static)
            current.wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = self.quantizer(static)
            current.wait_stream(stream)
            self.entries[key] = (static, output, graph)
        static, output, graph = self.entries[key]
        static.copy_(x)
        graph.replay()
        return output


class AdaptiveQuantizer:
    """Use measured H100 crossover: graphs for <=8M elements, direct fusion above."""

    def __init__(self):
        self.captured = CapturedQuantizer()

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.captured(x) if x.numel() <= 8_000_000 else fused_quantize(x)
