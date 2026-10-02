"""BF16 kernel study on a frozen, real batch-two AR chunk (no quality shortcuts).

Capture is untimed. Warmed full-chunk controls and a Chrome GPU trace establish
the denominator; isolated kernel times are estimates, not serving throughput.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.cpp_extension import load

from paired_inference import (ARA2VidDistilledPipeline, SpatialTilingConfig,
    TemporalTilingConfig, TilingConfig, autoregressive_euler_denoising_loop,
    encode_first_frame_channel_condition, paired_sampler, prepare)


@triton.jit
def persistent_mm(X, W, B, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  SMS: tl.constexpr, HAS_BIAS: tl.constexpr):
    # Persistent SM scheduling, grouped rows, BF16 inputs / FP32 accumulation.
    nm, nn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    for tile in range(tl.program_id(0), nm * nn, SMS):
        group = tile // (8 * nn)
        first = group * 8
        gm = tl.minimum(nm - first, 8)
        mi = first + (tile % (8 * nn)) % gm
        ni = (tile % (8 * nn)) // gm
        rm, rn, rk = mi * BM + tl.arange(0, BM), ni * BN + tl.arange(0, BN), tl.arange(0, BK)
        acc = tl.zeros((BM, BN), tl.float32)
        for block in range(tl.cdiv(K, BK)):
            kk = block * BK + rk
            x = tl.load(X + rm[:, None] * K + kk[None, :], (rm[:, None] < M) & (kk[None, :] < K), 0)
            w = tl.load(W + rn[None, :] * K + kk[:, None], (rn[None, :] < N) & (kk[:, None] < K), 0)
            acc = tl.dot(x, w, acc)
        if HAS_BIAS:
            acc += tl.load(B + rn, rn < N, 0)[None, :]
        tl.store(Y + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), (rm[:, None] < M) & (rn[None, :] < N))


def accuracy(a, b):
    a, b = a.float(), b.float()
    return dict(finite=bool(torch.isfinite(a).all()),
                relative_rms=float((a-b).square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-12)),
                max_abs=float((a-b).abs().max()))


def copy_strided(x, device):
    if x is None:
        return None
    y = torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=device)
    y.copy_(x)
    return y


def apply_study(args, chunk, frozen, report, save):
    """Measure real replacements in the compiled sampler, with a final control."""
    import h100_kernel_replacements as replacement
    source = json.loads(args.candidate_from.read_text())
    assert source.get('complete') and source['gpu'] == torch.cuda.get_device_name()
    choices = replacement.choose(source)
    report['candidate_source'] = str(args.candidate_from)
    report['selected_mm_kernels'] = choices
    report['integrated'] = {}
    def measure():
        torch.cuda.synchronize()
        before = time.perf_counter()
        result = chunk()
        torch.cuda.synchronize()
        return (time.perf_counter()-before)*1000, result
    print('Warm baseline for integrated comparison',flush=True)
    chunk()
    chunk()
    times=[]
    for _ in range(5):
        ms, control = measure()
        times.append(ms)
    report['baseline_chunk_ms'] = times
    control_latent = control[0].latent.detach().clone()
    changed = (control_latent != frozen['video_state'].latent).any(dim=-1)
    assert changed.any()
    print('INTEGRATED_BASELINE',times,flush=True)
    save()
    # All captured attention shapes already select cuDNN by default; forcing
    # Flash SDPA was slower in the source study, so preserve default attention.
    for name in ('tuned_mm',):
        print('Warm integrated replacement',name,flush=True)
        torch._dynamo.reset()
        with replacement.replacements(choices):
            chunk()
            chunk()
            replacement.HITS.clear()
            times=[]
            for _ in range(5):
                ms, result = measure()
                times.append(ms)
            row = dict(chunk_ms=times, calls_across_five_chunks=dict(replacement.HITS),
                       changed_video_tokens=int(changed.sum()),
                       current_chunk_accuracy=accuracy(result[0].latent[changed],control_latent[changed]),
                       all_video_accuracy=accuracy(result[0].latent,control_latent))
            report['integrated'][name] = row
            print('INTEGRATED_RESULT',name,json.dumps(row),flush=True)
            save()
            with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
                with record_function('FROZEN_BATCH_TWO_REPLACEMENT_CHUNK'):
                    chunk()
                    torch.cuda.synchronize()
            prof.export_chrome_trace(str(args.output_dir / (name+'-trace.json')))
    torch._dynamo.reset()
    chunk()
    chunk()
    report['baseline_after_ms'] = [measure()[0] for _ in range(5)]
    print('INTEGRATED_BASELINE_AFTER',report['baseline_after_ms'],flush=True)
    report['dynamo'] = {k:dict(torch._dynamo.utils.counters[k]) for k in ('stats','graph_break','unimplemented')}
    report['complete'] = True
    save()
    (args.output_dir / 'results.jsonl').write_text(json.dumps({'complete':True})+'\n')


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'gemma-root', 'audio'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--frames', type=int, default=257)
    p.add_argument('--runs', type=int, default=5)
    p.add_argument('--quantization', choices=['none'], default='none')
    p.add_argument('--candidate-from', type=Path)
    args = p.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    report = {'protocol': __doc__, 'torch': torch.__version__, 'triton': triton.__version__,
              'gpu': torch.cuda.get_device_name(), 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'chunk_index': 4, 'batch_size': 2, 'video_frames_per_request': 32,
              'deadline_ms': 1280, 'linears': [], 'attention': []}
    def save():
        (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    pipeline = ARA2VidDistilledPipeline(distilled_checkpoint_path=args.checkpoint,
        spatial_upsampler_path=None, gemma_root=args.gemma_root, loras=[],
        quantization=None, transformer_compile='regional', compile_video_decoder=False)
    tiling = TilingConfig(spatial_config=SpatialTilingConfig(tile_size_in_pixels=512, tile_overlap_in_pixels=64),
                         temporal_config=TemporalTilingConfig(tile_size_in_frames=256, tile_overlap_in_frames=8))
    ref = encode_first_frame_channel_condition(pipeline, image_path=args.reference, enabled=True, height=512, width=768, crf=0)
    requests = [prepare(pipeline, args, 42+i, 15.0*i, ref, tiling) for i in range(2)]
    for name in ('text_encoder', 'embeddings_processor', 'audio_encoder', 'video_encoder'):
        getattr(pipeline._fast_modules, name).to('cpu')
    torch.cuda.empty_cache()
    frozen = paired_sampler(requests)
    print('Prepare history eagerly (untimed)', flush=True)
    with torch.compiler.set_stance('force_eager'):
        v, a = autoregressive_euler_denoising_loop(**frozen, start_chunk_idx=0, end_chunk_idx=4)
    frozen.update(video_state=v, audio_state=a)
    def chunk():
        state = dict(frozen, video_state=frozen['video_state'].clone(), audio_state=frozen['audio_state'].clone())
        return autoregressive_euler_denoising_loop(**state, start_chunk_idx=4, end_chunk_idx=5)

    if args.candidate_from is not None:
        apply_study(args,chunk,frozen,report,save)
        return

    # Real operator signatures and one representative nonzero-weight example.
    linear_original, attention_original = F.linear, F.scaled_dot_product_attention
    linear_examples, attention_examples = {}, {}
    linear_counts, attention_counts = Counter(), Counter()
    def linear(x, w, bias=None):
        m, k, n = x.numel() // x.shape[-1], x.shape[-1], w.shape[0]
        key = f'{m}x{k}x{n}-bias{int(bias is not None)}'
        linear_counts[key] += 1
        if key not in linear_examples or linear_examples[key]['zero_weight']:
            zero = not bool(torch.count_nonzero(w))
            entry = dict(M=m, K=k, N=n, input_stride=list(x.stride()), weight_stride=list(w.stride()), zero_weight=zero)
            if min(k, n) >= 1024:
                entry['tensors'] = (x.reshape(m, k).cpu(), w.cpu(), None if bias is None else bias.cpu())
            linear_examples[key] = entry
        return linear_original(x, w, bias)
    def attention(q, k, v, *pos, **kw):
        assert not pos, 'Expected keyword SDPA options'
        mask = kw.get('attn_mask')
        key = str((tuple(q.shape), tuple(k.shape), tuple(q.stride()), tuple(k.stride()), tuple(v.stride()),
                   None if mask is None else (tuple(mask.shape), tuple(mask.stride()), str(mask.dtype))))
        attention_counts[key] += 1
        if key not in attention_examples:
            attention_examples[key] = {'tensors': tuple(copy_strided(t, 'cpu') for t in (q,k,v,mask)),
                                       'kwargs': {k:v for k,v in kw.items() if k != 'attn_mask'}}
        return attention_original(q,k,v,**kw)
    print('Capture real operators (untimed)', flush=True)
    F.linear, F.scaled_dot_product_attention = linear, attention
    try:
        with torch.compiler.set_stance('force_eager'):
            captured = chunk()
        del captured
    finally:
        F.linear, F.scaled_dot_product_attention = linear_original, attention_original
    report['workload'] = sorted([dict(key=k, calls=linear_counts[k],
        flops=2*r['M']*r['K']*r['N']*linear_counts[k], **{a:b for a,b in r.items() if a != 'tensors'})
        for k,r in linear_examples.items()], key=lambda r:r['flops'], reverse=True)
    save()
    print('Warm compiled fixed chunk', flush=True)
    chunk()
    chunk()
    def measure_chunk():
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = chunk()
        torch.cuda.synchronize()
        return (time.perf_counter()-start)*1000, result
    controls = []
    for _ in range(args.runs):
        ms, control = measure_chunk()
        controls.append(ms)
    report['baseline_chunk_ms'] = controls
    print('BASELINE_MS', controls, flush=True)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with record_function('FROZEN_BATCH_TWO_AR_CHUNK'):
            chunk()
            torch.cuda.synchronize()
    prof.export_chrome_trace(str(out / 'trace.json'))
    (out / 'operator-table.txt').write_text(prof.key_averages().table(sort_by='self_device_time_total', row_limit=80))
    save()

    extension = load(name='avatar_h100_mm_lt', sources=[str(Path(__file__).with_name('mm_cublaslt.cpp'))],
        extra_include_paths=['/usr/local/cuda/include'], extra_ldflags=['-L/usr/local/cuda/lib64','-lcublasLt','-lcudart',
        '-Wl,-rpath,/usr/local/cuda/lib64'], with_cuda=False, verbose=True)
    flush = torch.empty(64*1024*1024, dtype=torch.uint8, device='cuda')
    def measure(fn, reps=15):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(reps)]
        for s,e in pairs:
            flush.zero_()
            s.record()
            fn()
            e.record()
        torch.cuda.synchronize()
        samples = [s.elapsed_time(e) for s,e in pairs]
        return dict(median_ms=statistics.median(samples), min_ms=min(samples), max_ms=max(samples))
    # Dominant shape families cover video and audio linears; all remain BF16.
    selected = [r for r in report['workload'] if 'tensors' in linear_examples[r['key']]][:12]
    for shape in selected:
        print('MM', shape['key'], flush=True)
        x,w,b = (None if t is None else t.cuda().contiguous() for t in linear_examples[shape['key']]['tensors'])
        m,k,n = shape['M'],shape['K'],shape['N']
        y = torch.empty((m,n), device='cuda', dtype=torch.bfloat16)
        reference = F.linear(x,w,b)
        row = dict(key=shape['key'], calls=shape['calls'], flops=shape['flops'], candidates={})
        report['linears'].append(row)
        candidates = row['candidates']
        base = lambda:F.linear(x,w,b)
        candidates['torch_bf16'] = measure(base)
        compiled = torch.compile(lambda x,w,b:F.linear(x,w,b), fullgraph=True, dynamic=False,
                                 options={'emulate_precision_casts': True})
        candidates['compiled_bf16'] = dict(**measure(lambda:compiled(x,w,b)), accuracy=accuracy(compiled(x,w,b),reference))
        plan = extension.Plan(m,k,n,0 if b is None else b.data_ptr(),False)
        search = []
        for idx in range(plan.count()):
            fn = lambda idx=idx:plan.run(idx,w.data_ptr(),x.data_ptr(),y.data_ptr(),torch.cuda.current_stream().cuda_stream)
            try:
                timing = measure(fn,7)
                fn()
                search.append(dict(index=idx,**timing,accuracy=accuracy(y,reference)))
            except RuntimeError as exc:
                search.append(dict(index=idx,error=str(exc)[:500]))
        valid = [r for r in search if 'median_ms' in r and r['accuracy']['finite'] and r['accuracy']['relative_rms'] < .01]
        if valid:
            best = min(valid,key=lambda r:r['median_ms'])
            fn = lambda:plan.run(best['index'],w.data_ptr(),x.data_ptr(),y.data_ptr(),torch.cuda.current_stream().cuda_stream)
            candidates['cublaslt_tuned'] = dict(**measure(fn), index=best['index'],accuracy=best['accuracy'],search=search)
        del plan
        configs = [(128,128,64,8,3),(128,256,64,8,3),(64,128,64,4,3),(128,128,32,4,4)]
        searches = []
        sms = torch.cuda.get_device_properties(0).multi_processor_count
        def run(cfg):
            bm,bn,bk,warps,stages = cfg
            persistent_mm[(min(sms,triton.cdiv(m,bm)*triton.cdiv(n,bn)),)](
                x,w,b if b is not None else y,y,m,n,k,bm,bn,bk,sms,b is not None,num_warps=warps,num_stages=stages)
        for cfg in configs:
            try:
                timing = measure(lambda:run(cfg),7)
                run(cfg)
                searches.append(dict(config=cfg,**timing,accuracy=accuracy(y,reference)))
            except Exception as exc:
                searches.append(dict(config=cfg,error=str(exc)[:1000]))
        valid = [r for r in searches if 'median_ms' in r and r['accuracy']['finite'] and r['accuracy']['relative_rms'] < .01]
        if valid:
            best = min(valid,key=lambda r:r['median_ms'])
            candidates['triton_persistent'] = dict(**measure(lambda:run(best['config'])),config=best['config'],accuracy=best['accuracy'],search=searches)
        else:
            candidates['triton_persistent'] = dict(error='No valid candidate',search=searches)
        # Remeasure the control after searches to expose clock/order drift.
        candidates['torch_bf16_after'] = measure(base)
        print('MM_RESULT',shape['key'],{k:round(v['median_ms'],4) for k,v in candidates.items() if 'median_ms' in v},flush=True)
        save()
        del x,w,b,y,reference
    for key, example in attention_examples.items():
        q,k,v,mask = (copy_strided(t,'cuda') for t in example['tensors'])
        fn = lambda:F.scaled_dot_product_attention(q,k,v,attn_mask=mask,**example['kwargs'])
        reference = fn()
        row = dict(key=key,calls=attention_counts[key],candidates={'default':measure(fn)})
        report['attention'].append(row)
        for backend in (SDPBackend.FLASH_ATTENTION,SDPBackend.CUDNN_ATTENTION,SDPBackend.EFFICIENT_ATTENTION):
            try:
                with sdpa_kernel(backend):
                    row['candidates'][backend.name] = dict(**measure(fn),accuracy=accuracy(fn(),reference))
            except Exception as exc:
                row['candidates'][backend.name] = dict(error=str(exc)[:1200])
        save()
        print('ATTENTION_RESULT',key,{k:round(v['median_ms'],4) for k,v in row['candidates'].items() if 'median_ms' in v},flush=True)
        del q,k,v,mask,reference
    report['dynamo'] = {k:dict(torch._dynamo.utils.counters[k]) for k in ('stats','graph_break','unimplemented')}
    report['complete'] = True
    save()
    (out / 'results.jsonl').write_text(json.dumps({'complete':True})+'\n')


if __name__ == '__main__':
    main()
