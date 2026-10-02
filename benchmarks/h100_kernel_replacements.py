"""Experimental inference-only adapters. Never imported by production code.

cuBLASLt plans are shared per shape and used on one stream only. These adapters
are not a multithreaded server or CUDA Graph implementation.
"""
from contextlib import contextmanager
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

LINEAR = F.linear
CHOICES = {}
PLANS = {}
EXTENSION = None
HITS = Counter()


def linear_key(x, w, bias):
    return f'{x.numel() // x.shape[-1]}x{x.shape[-1]}x{w.shape[0]}-bias{int(bias is not None)}'


@torch.library.custom_op('avatar_h100::tuned_linear', mutates_args=())
def tuned_linear(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    key = linear_key(x,w,bias)
    HITS[key] += 1
    choice = CHOICES[key]
    m,k,n = x.numel() // x.shape[-1], x.shape[-1], w.shape[0]
    y = torch.empty((*x.shape[:-1],n),device=x.device,dtype=x.dtype)
    if choice['name'] == 'cublaslt_tuned':
        if key not in PLANS:
            PLANS[key] = EXTENSION.Plan(m,k,n,0 if bias is None else bias.data_ptr(),False)
        PLANS[key].run_bias(choice['index'],w.data_ptr(),x.data_ptr(),0 if bias is None else bias.data_ptr(),
                           y.data_ptr(),torch.cuda.current_stream().cuda_stream)
    else:
        from h100_kernel_study import persistent_mm
        import triton
        bm,bn,bk,warps,stages = choice['config']
        sms = torch.cuda.get_device_properties(x.device).multi_processor_count
        persistent_mm[(min(sms,triton.cdiv(m,bm)*triton.cdiv(n,bn)),)](
            x,w,bias if bias is not None else y,y,m,n,k,bm,bn,bk,sms,bias is not None,num_warps=warps,num_stages=stages)
    return y


@tuned_linear.register_fake
def _(x,w,bias):
    return x.new_empty((*x.shape[:-1],w.shape[0]))


def choose(report, margin=.03):
    selected = {}
    for row in report['linears']:
        base = min(row['candidates'][name]['median_ms'] for name in ('compiled_bf16','torch_bf16','torch_bf16_after'))
        valid = [(name,r) for name,r in row['candidates'].items()
                 if name in ('cublaslt_tuned','triton_persistent') and 'median_ms' in r
                 and r['accuracy']['finite'] and r['accuracy']['relative_rms'] < .01
                 and r['median_ms'] < base*(1-margin)]
        if valid:
            name,best = min(valid,key=lambda t:t[1]['median_ms'])
            selected[row['key']] = {'name':name, **{k:best[k] for k in ('index','config') if k in best}}
    return selected


@contextmanager
def replacements(choices):
    global CHOICES, EXTENSION
    CHOICES = choices
    if any(c['name'] == 'cublaslt_tuned' for c in choices.values()) and EXTENSION is None:
        EXTENSION = load(name='avatar_h100_mm_lt',sources=[str(Path(__file__).with_name('mm_cublaslt.cpp'))],
            extra_include_paths=['/usr/local/cuda/include'],extra_ldflags=['-L/usr/local/cuda/lib64','-lcublasLt','-lcudart',
            '-Wl,-rpath,/usr/local/cuda/lib64'],with_cuda=False,verbose=True)
    def linear(x,w,bias=None):
        if (x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16 and x.is_contiguous()
                and w.is_contiguous() and linear_key(x,w,bias) in CHOICES):
            return tuned_linear(x,w,bias)
        return LINEAR(x,w,bias)
    F.linear = linear
    try:
        yield
    finally:
        F.linear = LINEAR
        torch.cuda.synchronize()
        PLANS.clear()
