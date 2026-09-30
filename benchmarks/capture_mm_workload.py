"""Record actual linear shapes and example tensors from AR forwards 16-19.

Diagnostic capture only: CPU copies invalidate all timing fields from latency.py.
"""
import collections
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import latency

output = Path(sys.argv[sys.argv.index('--output-dir') + 1])
original_linear = F.linear
original_forward = latency.X0Model.forward
calls = 0
active = False
shapes = collections.Counter()
examples = {}


def linear(x, w, bias=None):
    if active:
        m, k, n = x.numel() // x.shape[-1], x.shape[-1], w.shape[0]
        key = f'{m}x{k}x{n}'
        shapes[key] += 1
        if key not in examples:
            row = {'M': m, 'K': k, 'N': n, 'input_stride': list(x.stride()),
                   'weight_stride': list(w.stride()), 'dtype': str(x.dtype),
                   'bias': bias is not None}
            examples[key] = row
            # Real inputs/weights for the dominant video GEMMs, not synthetic quality claims.
            if m >= 1000 and k >= 2048 and n >= 2048:
                path = output / 'tensors' / (key + '.pt')
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({'x': x.detach().reshape(m, k).cpu(), 'w': w.detach().cpu(),
                            'bias': None if bias is None else bias.detach().cpu()}, path)
                row['tensor_path'] = str(path)
    return original_linear(x, w, bias)


def forward(*args, **kwargs):
    global calls, active
    index = calls
    calls += 1
    active = 16 <= index < 20
    try:
        result = original_forward(*args, **kwargs)
    finally:
        active = False
    if index == 19:
        rows = [{**examples[k], 'key': k, 'calls_per_chunk': v,
                 'flops_per_chunk': 2 * examples[k]['M'] * examples[k]['K'] * examples[k]['N'] * v}
                for k, v in shapes.items()]
        rows.sort(key=lambda r: r['flops_per_chunk'], reverse=True)
        (output / 'workload.json').write_text(json.dumps({'forward_indices': [16,17,18,19],
            'note': 'Real single-stage FP8-storage/BF16-compute workload. Tensor examples come from the first linear of each shape. Diagnostic timings invalid.',
            'total_linear_flops': sum(r['flops_per_chunk'] for r in rows), 'shapes': rows}, indent=2)+'\n')
    return result


if __name__ == '__main__':
    F.linear = linear
    latency.X0Model.forward = forward
    latency.main()
