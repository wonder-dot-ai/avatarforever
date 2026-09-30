"""Remeasure selected kernels with sustained GPU warmup and rotating order.

Includes the efficient transposed-weight INT8 layout and compiled full INT8
linear. Results are still microbenchmarks, not measured video FPS.
"""
import json
import random
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
from torch.utils.cpp_extension import load
from benchmark_mm_candidates import mm,compiled_cast_linear,quant_x,dequant_y,accuracy


@torch.compile(fullgraph=True,dynamic=False)
def compiled_int8_linear(x,qw,sw,b):
    sx=x.float().abs().amax(dim=1,keepdim=True).clamp_min(1e-12)/127
    qx=(x.float()/sx).round().clamp(-127,127).to(torch.int8)
    return (torch._int_mm(qx,qw).float()*sx*sw.T+b.float()).to(torch.bfloat16)


@torch.inference_mode()
def main():
    root=Path('outputs/a100-mm-study')
    search=json.loads((root/'microbench.json').read_text())
    module=load(name='avatar_mm_lt',sources=[str(Path(__file__).with_name('mm_cublaslt.cpp'))],
        extra_include_paths=['/usr/local/cuda/include'],extra_ldflags=['-L/usr/local/cuda/lib64',
        '-lcublasLt','-lcudart','-Wl,-rpath,/usr/local/cuda/lib64'],with_cuda=False)
    flush=torch.empty(64*1024*1024,device='cuda',dtype=torch.uint8)
    heat=torch.randn(4096,4096,device='cuda',dtype=torch.bfloat16)
    heat_out=torch.empty_like(heat)
    def measure(fn):
        for _ in range(3):fn()
        for _ in range(400):torch.mm(heat,heat,out=heat_out)
        torch.cuda.synchronize()
        starts=[torch.cuda.Event(enable_timing=True) for _ in range(21)]
        ends=[torch.cuda.Event(enable_timing=True) for _ in range(21)]
        for a,b in zip(starts,ends):
            flush.zero_();a.record();fn();b.record()
        torch.cuda.synchronize()
        times=[a.elapsed_time(b) for a,b in zip(starts,ends)]
        return statistics.median(times)
    rows=[]
    report={'method':'Three rounds, shuffled candidate order (seed 42), each with 21 cold-L2 GPU-event samples. '
            '400 4096-square BF16 GEMMs preheat GPU before each batch, after lazy compilation. '
            '64 MiB zero-fill precedes and is excluded from each timed operation. '
            'Weights already packed. Full INT8 includes dynamic activation quantization, MM, scaling and bias. '
            'Microbenchmark data, not video throughput.','results':rows}
    rng=random.Random(42)
    for row in search['results']:
        shape=row['key'];print('SHAPE',shape,flush=True)
        t=torch.load(row['tensor_path'],weights_only=True)
        x=t['x'].cuda().contiguous();wf=t['w'].cuda().to(torch.float8_e4m3fn);w=wf.to(torch.bfloat16)
        bias=t['bias'].cuda();y=torch.empty(row['M'],row['N'],device='cuda',dtype=torch.bfloat16)
        yi=torch.empty_like(y,dtype=torch.int32)
        ref=F.linear(x,w,bias)
        sw=w.float().abs().amax(dim=1,keepdim=True).clamp_min(1e-12)/127
        qw=(w.float()/sw).round().clamp(-127,127).to(torch.int8)
        qx,sx=quant_x(x)
        p=module.Plan(row['M'],row['K'],row['N'],bias.data_ptr(),False)
        pi=module.Plan(row['M'],row['K'],row['N'],0,True)
        idx=row['candidates']['cublaslt_tuned']['index']
        idxi=min(row['candidates']['cublaslt_int8_full']['search'],key=lambda r:r['median_ms'])['index']
        def lt():
            p.run(idx,w.data_ptr(),x.data_ptr(),y.data_ptr(),torch.cuda.current_stream().cuda_stream)
            return y
        def lt_cast():
            wb=wf.to(torch.bfloat16)
            p.run(idx,wb.data_ptr(),x.data_ptr(),y.data_ptr(),torch.cuda.current_stream().cuda_stream)
            return y
        def lt_int8():
            q,s=quant_x(x)
            pi.run(idxi,qw.data_ptr(),q.data_ptr(),yi.data_ptr(),torch.cuda.current_stream().cuda_stream)
            return dequant_y(yi,s,sw,bias)
        def torch_int8():
            q,s=quant_x(x)
            return dequant_y(torch._int_mm(q,qw.T),s,sw,bias)
        def triton_mm(kind):
            entry=row['candidates'][kind]
            if kind=='triton_int8_full':cfg=min((r for r in entry['search'] if 'median_ms' in r),key=lambda r:r['median_ms'])['config']
            else:cfg=entry['config']
            bm,bn,bk,warps,stages=cfg
            is_i8=kind=='triton_int8_full';is_f8=kind=='triton_fused_fp8_bf16'
            a,s=quant_x(x) if is_i8 else (x,bias)
            weight=qw if is_i8 else wf.view(torch.uint8) if is_f8 else w
            mm[(triton.cdiv(row['M'],bm),triton.cdiv(row['N'],bn))](a,weight,bias,y,s,sw,
                row['M'],row['N'],row['K'],bm,bn,bk,is_f8,is_i8,num_warps=warps,num_stages=stages)
            return y
        candidates={
            'torch_bf16':lambda:F.linear(x,w,bias),
            'compiled_fp8_cast_linear':lambda:compiled_cast_linear(x,wf,bias),
            'cublaslt_bf16':lt,'cublaslt_cast_plus_bf16':lt_cast,
            'triton_bf16':lambda:triton_mm('triton_bf16'),
            'triton_fused_fp8_bf16':lambda:triton_mm('triton_fused_fp8_bf16'),
            'torch_int8_full_transposed':torch_int8,
            'compiled_int8_full_transposed':lambda:compiled_int8_linear(x,qw.T,sw,bias),
            'cublaslt_int8_full':lt_int8,'triton_int8_full':lambda:triton_mm('triton_int8_full')}
        numerics={name:accuracy(fn(),ref) for name,fn in candidates.items()}
        results={name:[] for name in candidates}
        for _ in range(3):
            order=list(candidates);rng.shuffle(order)
            for name in order:results[name].append(measure(candidates[name]))
        final={**{k:row[k] for k in ('key','M','K','N','calls_per_chunk','flops_per_chunk','fp8_roundtrip_exact')},
               'candidates':{name:{'median_ms':statistics.median(values),'round_medians_ms':values,
                                   'accuracy':numerics[name]} for name,values in results.items()}}
        rows.append(final)
        (root/'remeasured.json').write_text(json.dumps(report,indent=2)+'\n')
        print('RESULT',shape,{k:round(v['median_ms'],4) for k,v in final['candidates'].items()},flush=True)
        del p,pi,x,w,wf,bias,y,yi,ref,qw,qx,sx,sw,t


if __name__=='__main__':main()
