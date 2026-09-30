"""A100 microbenchmarks on captured real AvatarForever linear tensors.

Not an end-to-end video benchmark. Every timed sample flushes L2 with 64 MiB
before its start event. We report median GPU-event time including the operation's
launch gap; allocation, weight packing and compilation are excluded unless stated.
"""
import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.utils.cpp_extension import load


@triton.jit
def mm(X,W,B,Y,SX,SW,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
       BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,FP8:tl.constexpr,INT8:tl.constexpr=False):
    rm=tl.program_id(0)*BM+tl.arange(0,BM)
    rn=tl.program_id(1)*BN+tl.arange(0,BN)
    rk=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.int32 if INT8 else tl.float32)
    for block in range(tl.cdiv(K,BK)):
        kk=block*BK+rk
        a=tl.load(X+rm[:,None]*K+kk[None,:],(rm[:,None]<M)&(kk[None,:]<K),0)
        w=tl.load(W+rn[None,:]*K+kk[:,None],(rn[None,:]<N)&(kk[:,None]<K),0)
        if FP8:
            # Exact E4M3FN -> BF16 software decode, including subnormals and NaNs.
            u=w.to(tl.uint16); e=(u>>3)&15; mant=u&7
            bits=((u&128)<<8)|((e+120)<<7)|(mant<<4)
            normal=bits.to(tl.bfloat16,bitcast=True)
            sub=(mant.to(tl.float32)*(1.0/512.0))*tl.where((u&128)!=0,-1.,1.)
            w=tl.where(e==0,sub,normal.to(tl.float32)).to(tl.bfloat16)
            w=tl.where((e==15)&(mant==7),float('nan'),w).to(tl.bfloat16)
        acc=tl.dot(a,w,acc)
    if INT8:
        acc=acc.to(tl.float32)*tl.load(SX+rm,rm<M,0)[:,None]*tl.load(SW+rn,rn<N,0)[None,:]
    out=acc+tl.load(B+rn,rn<N,0)[None,:]
    tl.store(Y+rm[:,None]*N+rn[None,:],out.to(tl.bfloat16),(rm[:,None]<M)&(rn[None,:]<N))


@torch.compile(fullgraph=True, dynamic=False)
def quant_x(x):
    scale=x.float().abs().amax(dim=1,keepdim=True).clamp_min(1e-12)/127
    q=(x.float()/scale).round().clamp(-127,127).to(torch.int8)
    return q,scale


@torch.compile(fullgraph=True, dynamic=False, options={"emulate_precision_casts": True})
def compiled_cast_linear(x,w,b):
    return F.linear(x,w.to(x.dtype),b)


@torch.compile(fullgraph=True, dynamic=False)
def dequant_y(y,sx,sw,b):
    return (y.float()*sx*sw.T+b.float()).to(torch.bfloat16)


def accuracy(actual,reference):
    a=actual.float();b=reference.float();diff=a-b
    return {'finite':bool(torch.isfinite(a).all()),
            'relative_rms':float(diff.square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-12)),
            'max_abs':float(diff.abs().max()),
            'cosine':float(F.cosine_similarity(a.flatten(),b.flatten(),dim=0))}


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('outputs/a100-mm-study'))
    parser.add_argument('--limit',type=int,default=8)
    args=parser.parse_args()
    output=args.root/'microbench.json'
    workload=json.loads((args.root/'capture/workload.json').read_text())
    module=load(name='avatar_mm_lt',sources=[str(Path(__file__).with_name('mm_cublaslt.cpp'))],
                extra_include_paths=['/usr/local/cuda/include'],
                extra_ldflags=['-L/usr/local/cuda/lib64','-lcublasLt','-lcudart',
                               '-Wl,-rpath,/usr/local/cuda/lib64'],with_cuda=False,verbose=True)
    flush=torch.empty(64*1024*1024,device='cuda',dtype=torch.uint8)
    def measure(fn,reps=21):
        for _ in range(3):fn()
        torch.cuda.synchronize()
        starts=[torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        ends=[torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        for start,end in zip(starts,ends):
            flush.zero_();start.record();fn();end.record()
        torch.cuda.synchronize()
        vals=[s.elapsed_time(e) for s,e in zip(starts,ends)]
        return {'median_ms':statistics.median(vals),'min_ms':min(vals),'max_ms':max(vals)}
    rows=[]
    configs=[(64,64,32,4,3),(64,128,32,4,3),(128,64,32,4,3),(128,128,32,8,3),
             (128,256,32,8,3),(64,256,32,8,3),(128,128,64,8,3),(64,128,64,4,4)]
    report={'method':__doc__,'torch':torch.__version__,'triton':triton.__version__,
            'device':torch.cuda.get_device_name(),'total_linear_flops':workload['total_linear_flops'],'results':rows}
    shapes=[r for r in workload['shapes'] if 'tensor_path' in r][:args.limit]
    for shape in shapes:
        print('SHAPE',shape['key'],flush=True)
        tensors=torch.load(shape['tensor_path'],weights_only=True)
        x=tensors['x'].cuda().contiguous();w_original=tensors['w'].cuda().contiguous()
        wf=w_original.to(torch.float8_e4m3fn);w=wf.to(torch.bfloat16)
        bias=tensors['bias'].cuda() if tensors['bias'] is not None else torch.zeros(shape['N'],device='cuda',dtype=torch.bfloat16)
        y=torch.empty((shape['M'],shape['N']),device='cuda',dtype=torch.bfloat16)
        ref=F.linear(x,w,bias)
        row={**shape,'fp8_roundtrip_exact':bool(torch.equal(w,w_original)),'candidates':{}}
        rows.append(row); candidates=row['candidates']
        def save():output.write_text(json.dumps(report,indent=2)+'\n')
        candidates['torch_bf16']=measure(lambda:F.linear(x,w,bias))
        candidates['fp8_cast_plus_torch_bf16']=measure(lambda:F.linear(x,wf.to(torch.bfloat16),bias))
        candidates['compiled_fp8_cast_linear']=measure(lambda:compiled_cast_linear(x,wf,bias))
        candidates['fp8_cast_only']=measure(lambda:wf.to(torch.bfloat16))
        plan=module.Plan(shape['M'],shape['K'],shape['N'],bias.data_ptr(),False)
        lt=[]
        for i in range(plan.count()):
            fn=lambda i=i:plan.run(i,w.data_ptr(),x.data_ptr(),y.data_ptr(),torch.cuda.current_stream().cuda_stream)
            try:
                timing=measure(fn,reps=9);fn();check=accuracy(y,ref)
                lt.append({'index':i,**timing,'accuracy':check})
            except RuntimeError as e:lt.append({'index':i,'error':str(e)})
        valid=[r for r in lt if 'median_ms' in r and r['accuracy']['finite'] and r['accuracy']['relative_rms']<.01]
        best=min(valid,key=lambda r:r['median_ms']);index=best['index']
        fn=lambda:plan.run(index,w.data_ptr(),x.data_ptr(),y.data_ptr(),torch.cuda.current_stream().cuda_stream)
        candidates['cublaslt_tuned']={**measure(fn),'index':index,'search':lt,'accuracy':best['accuracy']}
        del plan
        for fused in (False,True):
            search=[];weight=wf.view(torch.uint8) if fused else w
            def run(config):
                bm,bn,bk,warps,stages=config
                mm[(triton.cdiv(shape['M'],bm),triton.cdiv(shape['N'],bn))](x,weight,bias,y,bias,bias,
                    shape['M'],shape['N'],shape['K'],bm,bn,bk,fused,num_warps=warps,num_stages=stages)
            for config in configs:
                try:
                    timing=measure(lambda:run(config),reps=9);run(config);check=accuracy(y,ref)
                    search.append({'config':config,**timing,'accuracy':check})
                except Exception as e:search.append({'config':config,'error':str(e)[:1000]})
            valid=[r for r in search if 'median_ms' in r and r['accuracy']['finite'] and r['accuracy']['relative_rms']<.01]
            name='triton_fused_fp8_bf16' if fused else 'triton_bf16'
            if valid:
                best=min(valid,key=lambda r:r['median_ms']);config=best['config']
                candidates[name]={**measure(lambda:run(config)),'config':config,'accuracy':best['accuracy'],'search':search}
            else:candidates[name]={'error':'No valid candidates','search':search}
        sw=w.float().abs().amax(dim=1,keepdim=True).clamp_min(1e-12)/127
        qw_row=(w.float()/sw).round().clamp(-127,127).to(torch.int8)
        qw=qw_row.T.contiguous()
        qx,sx=quant_x(x)
        def int8_full():
            q,s=quant_x(x)
            return dequant_y(torch._int_mm(q,qw),s,sw,bias)
        try:
            candidates['int8_mm_only']=measure(lambda:torch._int_mm(qx,qw))
            candidates['int8_mm_transposed_weight']=measure(lambda:torch._int_mm(qx,qw_row.T))
            candidates['int8_dynamic_full']={**measure(int8_full),'accuracy':accuracy(int8_full(),ref),
                'note':'Different numerical computation: per-row dynamic INT8 activations, per-output-channel INT8 weights. Includes activation quantization and output rescaling; excludes offline weight packing.'}
        except Exception as e:candidates['int8_dynamic_full']={'error':str(e)[:1000]}
        yi=torch.empty_like(y,dtype=torch.int32)
        try:
            ip=module.Plan(shape['M'],shape['K'],shape['N'],0,True)
            searches=[]
            for i in range(ip.count()):
                f=lambda i=i:ip.run(i,qw_row.data_ptr(),qx.data_ptr(),yi.data_ptr(),torch.cuda.current_stream().cuda_stream)
                timing=measure(f,9);f()
                searches.append({'index':i,**timing,'accuracy':accuracy(dequant_y(yi,sx,sw,bias),ref)})
            best=min(searches,key=lambda r:r['median_ms']);idx=best['index']
            def lt_int8_full():
                q,s=quant_x(x)
                ip.run(idx,qw_row.data_ptr(),q.data_ptr(),yi.data_ptr(),torch.cuda.current_stream().cuda_stream)
                return dequant_y(yi,s,sw,bias)
            candidates['cublaslt_int8_full']={**measure(lt_int8_full),'search':searches,
                'accuracy':accuracy(lt_int8_full(),ref)}
            del ip
        except Exception as e:candidates['cublaslt_int8_full']={'error':str(e)[:1000]}
        searches=[]
        def run_int8(config,dynamic):
            bm,bn,bk,warps,stages=config
            q,s=quant_x(x) if dynamic else (qx,sx)
            mm[(triton.cdiv(shape['M'],bm),triton.cdiv(shape['N'],bn))](q,qw_row,bias,y,s,sw,
                shape['M'],shape['N'],shape['K'],bm,bn,bk,False,True,num_warps=warps,num_stages=stages)
            return y
        for config in configs:
            try:
                searches.append({'config':config,**measure(lambda:run_int8(config,True),9),
                                 'accuracy':accuracy(run_int8(config,True),ref)})
            except Exception as e:searches.append({'config':config,'error':str(e)[:1000]})
        valid=[r for r in searches if 'median_ms' in r and r['accuracy']['finite']]
        if valid:
            best=min(valid,key=lambda r:r['median_ms']);cfg=best['config']
            candidates['triton_int8_full']={**measure(lambda:run_int8(cfg,True)),
                'accuracy':best['accuracy'],'search':searches}
        save()
        print('RESULT',shape['key'],{k:round(v['median_ms'],4) for k,v in candidates.items() if 'median_ms' in v},flush=True)
        del x,w,w_original,wf,bias,y,ref,tensors,qw,qw_row,qx,sx,sw,yi
    report['covered_linear_flops']=sum(r['flops_per_chunk'] for r in rows)
    save()


if __name__=='__main__':main()
