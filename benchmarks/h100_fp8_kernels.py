"""Native FP8 GEMM versus BF16 on matched H100 batch-two AR inference.

Frozen-chunk timings include activation quantization. A full 257-frame BF16
control and the fastest FP8 candidate also run with measured full-clip decoding.
Attention, steps, resolution, cache and conditioning remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function
import triton
import triton.language as tl

from h100_kernel_study import accuracy
from paired_inference import (ARA2VidDistilledPipeline, SpatialTilingConfig, TemporalTilingConfig,
    TilingConfig, autoregressive_euler_denoising_loop, encode_first_frame_channel_condition,
    paired_sampler, prepare, generate, unpack, decode_video, decode_audio_from_file,
    encode_video, Audio, get_video_chunks_number)
from ltx_core.quantization.fp8_dynamic import selected_linear, quantize_per_tensor
from ltx_core.quantization.fp8_quantizer import fused_quantize

FUSED_BIAS = {'tensor': True, 'row': True}


@triton.jit
def _row_quantize(X,Q,S,ROWS:tl.constexpr,K:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0)
    columns=tl.arange(0,BLOCK)
    values=tl.load(X+row*K+columns,columns<K,0).to(tl.float32)
    scale=tl.maximum(tl.max(tl.abs(values),0),1e-12)*(1./448.)
    quantized=tl.minimum(tl.maximum(tl.div_rn(values,scale),-448.),448.)
    tl.store(Q+row*K+columns,quantized,columns<K)
    tl.store(S+row,scale)


def quantize_rows(x):
    rows,k=x.shape
    q=torch.empty_like(x,dtype=torch.float8_e4m3fn)
    scales=torch.empty((rows,1),device=x.device,dtype=torch.float32)
    _row_quantize[(rows,)](x,q,scales,rows,k,triton.next_power_of_2(k),num_warps=8 if k>=8192 else 4)
    return q,scales


class KernelFP8Linear(nn.Module):
    def __init__(self,weight,bias,scaling,fast_accum=False):
        super().__init__()
        self.in_features=weight.shape[1]
        self.out_features=weight.shape[0]
        self.scaling=scaling
        self.fast_accum=fast_accum
        self.fuse_bias=FUSED_BIAS[scaling]
        if scaling=='tensor':
            q,scale=quantize_per_tensor(weight)
        else:
            values=weight.float()
            scale=values.abs().amax(dim=1,keepdim=True).clamp_min(1e-12)/448.
            q=(values/scale).clamp(-448.,448.).to(torch.float8_e4m3fn)
            scale=scale.T.contiguous()
        self.weight=nn.Parameter(q.T,requires_grad=False)
        self.weight_scale=nn.Parameter(scale,requires_grad=False)
        self.bias=nn.Parameter(bias.clone(),requires_grad=False) if bias is not None else None

    def forward(self,x):
        flat=x.reshape(-1,self.in_features).contiguous()
        rows=flat.shape[0]
        # H100 cuBLASLt rejects the audio FF-down projection at M=16,
        # K=8192,N=2048 with row scales and fast accumulation. The steady
        # M=64 path is supported. Zero-pad short tails to that proven size;
        # trim outputs after GEMM, without changing any input/weight scales.
        padding=max(64,((rows+15)//16)*16)-rows
        if padding:
            flat=torch.nn.functional.pad(flat,(0,0,0,padding))
        q,scale=fused_quantize(flat) if self.scaling=='tensor' else quantize_rows(flat)
        output=torch._scaled_mm(q,self.weight,scale_a=scale,scale_b=self.weight_scale,
                                bias=self.bias if self.fuse_bias else None,out_dtype=x.dtype,use_fast_accum=self.fast_accum)
        if not self.fuse_bias and self.bias is not None:
            output=output+self.bias
        return output[:rows].reshape(*x.shape[:-1],self.out_features)


def install(model,originals,scaling,fast_accum):
    velocity=model.velocity_model
    if not originals:
        for name,layer in list(velocity.named_modules()):
            if isinstance(layer,nn.Linear) and selected_linear(name):
                replacement=KernelFP8Linear(layer.weight,layer.bias,scaling,fast_accum)
                originals[name]=layer.to('cpu')
                parent,attr=name.rsplit('.',1)
                setattr(velocity.get_submodule(parent),attr,replacement)
    else:
        for name,original in originals.items():
            current=velocity.get_submodule(name)
            if current.scaling==scaling:
                current.fast_accum=fast_accum
            else:
                w=original.weight.cuda()
                b=None if original.bias is None else original.bias.cuda()
                replacement=KernelFP8Linear(w,b,scaling,fast_accum)
                parent,attr=name.rsplit('.',1)
                setattr(velocity.get_submodule(parent),attr,replacement)
                del w,b,current
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return len(originals)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','gemma-root','audio'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--frames',type=int,default=257)
    p.add_argument('--runs',type=int,default=3)
    p.add_argument('--quantization',choices=['fp8-dynamic'],default='fp8-dynamic')
    p.add_argument('--fp8-modes',nargs='+',choices=['fp8_tensor','fp8_row','fp8_row_fast','fp8_tensor_fast'],
                   default=['fp8_tensor','fp8_row','fp8_row_fast','fp8_tensor_fast'])
    p.add_argument('--frozen-only',action='store_true',help='Screen additional kernels without repeating full video generation.')
    args=p.parse_args()
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    report={'protocol':__doc__,'torch':torch.__version__,'triton':triton.__version__,
            'gpu':torch.cuda.get_device_name(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'gpu_info':subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total,power.limit','--format=csv,noheader'],capture_output=True,text=True).stdout.strip(),
            'frozen':{},'full':{},'preflight':{},'target_pair_seconds':1.28,'frozen_only':args.frozen_only,
            'notes':'FP8 weights quantized once; dynamic activation quantization INCLUDED. BF16 attention/output. Fused bias when supported; preflight records any separate BF16 bias add fallback. No static activation calibration; no CUDA Graph capture.'}
    def save():
        (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    modes=[('fp8_tensor','tensor',False),('fp8_row','row',False),('fp8_row_fast','row',True),('fp8_tensor_fast','tensor',True)]
    modes=[mode for mode in modes if mode[0] in args.fp8_modes]
    torch.manual_seed(123)
    x=torch.randn(128,256,device='cuda',dtype=torch.bfloat16)
    w=torch.randn(512,256,device='cuda',dtype=torch.bfloat16)
    bias=torch.randn(512,device='cuda',dtype=torch.bfloat16)
    ref=torch.nn.functional.linear(x,w,bias)
    available=[]
    for name,scaling,fast in modes:
        try:
            layer=KernelFP8Linear(w,bias,scaling,fast)
            try:
                prediction=layer(x)
            except RuntimeError as exc:
                if 'bias' not in str(exc).lower():
                    raise
                FUSED_BIAS[scaling]=False
                layer.fuse_bias=False
                prediction=layer(x)
            check={**accuracy(prediction,ref),'fused_bias':layer.fuse_bias}
            assert check['finite'] and check['relative_rms']<.1,check
            report['preflight'][name]=check
            available.append((name,scaling,fast))
        except Exception as exc:
            report['preflight'][name]={'error':str(exc)}
    print('PREFLIGHT',json.dumps(report['preflight']),flush=True)
    save()
    if not available:
        raise RuntimeError('No supported FP8 candidates')
    del x,w,bias,ref,layer
    # Exercise the exact small audio shape that failed during full-clip warmup,
    # including strict compilation, before paying for loading the full model.
    report['small_audio_regression']={}
    for name,scaling,fast in available:
        x=torch.randn(14,8192,device='cuda',dtype=torch.bfloat16)
        w=torch.randn(2048,8192,device='cuda',dtype=torch.bfloat16)*.01
        b=torch.randn(2048,device='cuda',dtype=torch.bfloat16)*.01
        layer=KernelFP8Linear(w,b,scaling,fast)
        compiled=torch.compile(layer,fullgraph=True,dynamic=False,options={'emulate_precision_casts':True})
        check=accuracy(compiled(x),torch.nn.functional.linear(x,w,b))
        assert check['finite'] and check['relative_rms']<.1,check
        report['small_audio_regression'][name]=check
        del x,w,b,layer,compiled
    print('SMALL_AUDIO_REGRESSION',json.dumps(report['small_audio_regression']),flush=True)
    save()
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    limit='recompile_limit' if hasattr(torch._dynamo.config,'recompile_limit') else 'cache_size_limit'
    setattr(torch._dynamo.config,limit,max(getattr(torch._dynamo.config,limit),96))
    pipeline=ARA2VidDistilledPipeline(distilled_checkpoint_path=args.checkpoint,spatial_upsampler_path=None,
        gemma_root=args.gemma_root,loras=[],quantization=None,transformer_compile='regional',compile_video_decoder=True)
    tiling=TilingConfig(spatial_config=SpatialTilingConfig(tile_size_in_pixels=512,tile_overlap_in_pixels=64),
                       temporal_config=TemporalTilingConfig(tile_size_in_frames=256,tile_overlap_in_frames=8))
    reference=encode_first_frame_channel_condition(pipeline,image_path=args.reference,enabled=True,height=512,width=768,crf=0)
    requests=[prepare(pipeline,args,42+i,15.*i,reference,tiling) for i in range(2)]
    for name in ('text_encoder','embeddings_processor','audio_encoder','video_encoder'):
        getattr(pipeline._fast_modules,name).to('cpu')
    torch.cuda.empty_cache()
    paired=paired_sampler(requests)
    frozen=dict(paired)
    with torch.compiler.set_stance('force_eager'):
        v,a=autoregressive_euler_denoising_loop(**frozen,start_chunk_idx=0,end_chunk_idx=4)
    frozen.update(video_state=v,audio_state=a)
    def chunk():
        state=dict(frozen,video_state=frozen['video_state'].clone(),audio_state=frozen['audio_state'].clone())
        return autoregressive_euler_denoising_loop(**state,start_chunk_idx=4,end_chunk_idx=5)
    def measure_frozen(name,baseline=None):
        print('WARM_FROZEN',name,flush=True)
        chunk()
        chunk()
        times=[]
        torch.cuda.reset_peak_memory_stats()
        for _ in range(5):
            torch.cuda.synchronize()
            before=time.perf_counter()
            result=chunk()
            torch.cuda.synchronize()
            times.append((time.perf_counter()-before)*1000)
        row={'chunk_ms':times,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
        if baseline is not None:
            changed=(baseline!=frozen['video_state'].latent).any(dim=-1)
            row['current_chunk_accuracy']=accuracy(result[0].latent[changed],baseline[changed])
            assert row['current_chunk_accuracy']['finite']
        report['frozen'][name]=row
        save()
        print('FROZEN_RESULT',name,json.dumps(row),flush=True)
        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
            with record_function('FROZEN_'+name):
                chunk()
                torch.cuda.synchronize()
        prof.export_chrome_trace(str(out/(name+'-trace.json')))
        return result[0].latent.detach().clone()
    def full(name):
        print('WARM_FULL',name,flush=True)
        warm,_=generate([paired])
        warm_latents=unpack(warm,pipeline,args.frames)
        print('WARM_DECODER',name,flush=True)
        for latent in warm_latents:
            for _ in decode_video(latent,pipeline._fast_modules.video_decoder,tiling):
                pass
        del warm,warm_latents,latent
        records=[]
        for repeat in range(args.runs):
            states,row=generate([paired])
            latents=unpack(states,pipeline,args.frames)
            decode=[]
            for latent in latents:
                torch.cuda.synchronize()
                before=time.perf_counter()
                count=0
                for frames in decode_video(latent,pipeline._fast_modules.video_decoder,tiling):
                    count+=frames.shape[0]
                torch.cuda.synchronize()
                assert count==args.frames
                decode.append(time.perf_counter()-before)
            row.update(repeat=repeat,decode_seconds=decode,finite=all(bool(torch.isfinite(x).all()) for x in latents))
            assert row['finite']
            row['steady_pair_with_average_decode_seconds']=row['steady_pair_chunk_median_seconds']+sum(decode)*32/args.frames
            row['per_request_fps_estimate']=32/row['steady_pair_with_average_decode_seconds']
            records.append(row)
            print('FULL_RESULT',name,json.dumps(row),flush=True)
            with (out/'results.jsonl').open('a') as file:
                file.write(json.dumps({'mode':name,**row})+'\n')
        report['full'][name]=records
        save()
        cpu_latents=[x.cpu() for x in latents]
        for i,latent in enumerate(cpu_latents):
            torch.save(latent,out/f'{name}-{i}.pt')
            audio=decode_audio_from_file(args.audio,pipeline.device,15.*i,args.frames/25)
            audio=Audio(waveform=audio.waveform.squeeze(0),sampling_rate=audio.sampling_rate)
            encode_video(video=decode_video(latent.cuda(),pipeline._fast_modules.video_decoder,tiling),fps=25,
                audio=audio,output_path=str(out/f'{name}-{i}.mp4'),
                video_chunks_number=get_video_chunks_number(args.frames,tiling),crf=12,preset='fast')
        return cpu_latents
    baseline=measure_frozen('bf16')
    baseline_full=None if args.frozen_only else full('bf16')
    originals={}
    model=requests[0]['transformer']
    for name,scaling,fast in available:
        print('INSTALL_FP8',name,flush=True)
        torch._dynamo.reset()
        report['converted_linears']=install(model,originals,scaling,fast)
        measure_frozen(name,baseline)
    best=min(available,key=lambda mode:statistics.median(report['frozen'][mode[0]]['chunk_ms']))
    report['selected_fp8']=best[0]
    report['full_latent_comparison']=[]
    if not args.frozen_only:
        torch._dynamo.reset()
        install(model,originals,best[1],best[2])
        winner=full(best[0])
        for before,after in zip(baseline_full,winner):
            row=accuracy(after,before)
            row['per_latent_frame_relative_rms']=[accuracy(after[:,:,i],before[:,:,i])['relative_rms'] for i in range(after.shape[2])]
            report['full_latent_comparison'].append(row)
    # Restore BF16 weights and remeasure the fixed chunk to expose time/clock drift.
    torch._dynamo.reset()
    for name,original in originals.items():
        parent,attr=name.rsplit('.',1)
        setattr(model.velocity_model.get_submodule(parent),attr,original.cuda())
    torch.cuda.empty_cache()
    measure_frozen('bf16_after',baseline)
    report['dynamo']={k:dict(torch._dynamo.utils.counters[k]) for k in ('stats','graph_break','unimplemented')}
    report['complete']=True
    save()
    if args.frozen_only:
        (out/'results.jsonl').write_text(json.dumps({'complete':True,'frozen_only':True})+'\n')


if __name__=='__main__':
    main()
