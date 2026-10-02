"""Build the native FP8 latency/quality comparison from completed Modal results."""
import argparse
import gzip
import html
import json
import os
from pathlib import Path
import shutil
import statistics

from build_h100_kernel_report import analyze_trace, table

LABELS={'bf16':'BF16','bf16_after':'BF16, restored','fp8_tensor':'FP8 · tensor scaling',
        'fp8_row':'FP8 · row scaling','fp8_row_fast':'FP8 · row scaling + fast accumulation',
        'fp8_tensor_fast':'FP8 · tensor scaling + fast accumulation'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--screening-run',type=Path)
    p.add_argument('--sweep-run',type=Path,help='Preserve fixed-chunk results from the initial incomplete full-clip run.')
    p.add_argument('--output',type=Path,default=Path('outputs/h100-fp8-kernels'))
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    s=json.loads((args.run/'summary.json').read_text())
    assert s['complete']
    best=s['selected_fp8']
    pooled=statistics.median(s['frozen']['bf16']['chunk_ms']+s['frozen']['bf16_after']['chunk_ms'])
    frozen={name:{'median_ms':statistics.median(r['chunk_ms']),'max_ms':max(r['chunk_ms']),
                  'reduction_percent':100*(1-statistics.median(r['chunk_ms'])/pooled),
                  'peak_allocated_gib':r['peak_allocated_gib'],
                  'current_chunk_accuracy':r.get('current_chunk_accuracy')}
            for name,r in s['frozen'].items()}
    full={name:{key:statistics.median(r[key] for r in rows) for key in
          ('steady_pair_chunk_median_seconds','steady_pair_with_average_decode_seconds',
           'per_request_fps_estimate','peak_allocated_gib')} for name,rows in s['full'].items()}
    base=full['bf16']['steady_pair_chunk_median_seconds']
    generation=full[best]['steady_pair_chunk_median_seconds']
    total=full[best]['steady_pair_with_average_decode_seconds']
    reduction=100*(1-generation/base)
    analysis={'run':args.run.name,'selected_fp8':best,'frozen':frozen,'full':full,
              'full_generation_reduction_percent':reduction,'pair_deadline_seconds':1.28,
              'remaining_after_amortized_decode_seconds':1.28-total,
              'frozen_pooled_bf16_ms':pooled,'full_latent_comparison':s['full_latent_comparison'],
              'scope':'Two synchronized requests, 768x512, 25 FPS, four steps, regional compile, ForeverCache; activation quantization included. Full-clip VAE average, not streaming decode; excludes audio encoding and transport.'}
    screen=None
    sweep=None
    if args.sweep_run:
        sweep=json.loads((args.sweep_run/'summary.json').read_text())
        control=statistics.median(sweep['frozen']['bf16']['chunk_ms'])
        analysis['initial_sweep']={'run':args.sweep_run.name,'complete':bool(sweep.get('complete')),
            'bf16_control_ms':control,'candidates':{n:{'median_ms':statistics.median(r['chunk_ms']),
                'reduction_percent':100*(1-statistics.median(r['chunk_ms'])/control),
                'current_chunk_accuracy':r['current_chunk_accuracy']}
                for n,r in sweep['frozen'].items() if n.startswith('fp8_')},
            'limitation':'Fixed chunks completed; full FP8 warmup failed at a small audio GEMM (M=16,K=8192,N=2048). The repaired full run pads short inputs to 64 rows, trims outputs and validates the exact shape under compilation.'}
    if args.screening_run:
        screen=json.loads((args.screening_run/'summary.json').read_text())
        assert screen['complete'] and screen['frozen_only']
        control=statistics.median(screen['frozen']['bf16']['chunk_ms']+screen['frozen']['bf16_after']['chunk_ms'])
        analysis['additional_screening']={'run':args.screening_run.name,'pooled_bf16_ms':control,
            'candidates':{n:{'median_ms':statistics.median(r['chunk_ms']),
                'reduction_percent':100*(1-statistics.median(r['chunk_ms'])/control),
                'current_chunk_accuracy':r['current_chunk_accuracy']}
                for n,r in screen['frozen'].items() if n.startswith('fp8_')}}
    (args.output/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
    def relative(path):return html.escape(os.path.relpath(path,args.output),quote=True)
    body=f'''<header><div class="tag">AVATARFOREVER · H100 · TWO REQUESTS · NATIVE FP8</div>
<h1>FP8: latency and output quality</h1><p class="lead">Full-generation time reduction: <strong>{reduction:.1f}%</strong></p>
<p>Full-clip candidate: <b>{LABELS[best]}</b>. FP8 weights stay resident; activations are dynamically quantized for every multiplication.
Activation scaling and casts are included in all timings. Attention and output activations remain BF16.</p></header>'''
    body+='<section><h2>Complete inference comparison</h2>'+table(
        ['Mode','Pair generation / chunk','Pair + average VAE','Estimated FPS / request','Generation peak'],
        [[LABELS[name],f"{r['steady_pair_chunk_median_seconds']:.3f} s",f"{r['steady_pair_with_average_decode_seconds']:.3f} s",
          f"{r['per_request_fps_estimate']:.2f}",f"{r['peak_allocated_gib']:.2f} GiB"] for name,r in full.items()])
    body+=f'<p>Both requests produce 32 frames per regular chunk, sharing a <b>1.280 s</b> playback deadline. '
    body+=f'A 25–30% reduction from this BF16 control means <b>{base*.70:.3f}–{base*.75:.3f} s</b> for generation. '
    body+=f'The measured FP8 generation plus amortized VAE leaves <b>{(1.28-total)*1000:+.1f} ms</b> against that deadline.</p>'
    body+='<p class="warn">Three measured 257-frame runs after warmup. VAE timing decodes each complete clip; it does not validate streaming decode or first-frame latency. Audio encoding, preparation, transport and jitter are excluded. These short clips do not establish long-session quality.</p></section>'
    body+='<section><h2>Compare the generated clips</h2><p>Each row has identical portrait, prompt, seed and audio across modes. '
    body+='Request 1 uses seed 42/audio offset 0; request 2 uses seed 43/audio offset 15 seconds. Playback buttons synchronize each pair; only the BF16 player supplies audio.</p>'
    for i in range(2):
        body+=f'<h3>Request {i+1}</h3><button class="play" data-row="{i}">Play / pause pair</button> <button class="reset" data-row="{i}">Restart pair</button><div class="videos" data-group="{i}">'
        for name in ('bf16',best):
            body+=f'<figure><figcaption>{LABELS[name]}</figcaption><video controls preload="metadata" playsinline {"muted" if name!= "bf16" else ""} src="{relative(args.run/f"{name}-{i}.mp4")}"></video></figure>'
        body+='</div>'
    body+='</section><section><h2>Same-input kernel comparison</h2><p>Five warmed measurements of chunk index 4, replayed from the same BF16 history for each candidate. '
    body+='FP8 conversion and compilation occur before timing. The restored BF16 control checks time drift.</p>'
    rows=[]
    for name,r in frozen.items():
        q=r['current_chunk_accuracy']
        rows.append([LABELS[name],f"{r['median_ms']:.2f} ms",f"{r['max_ms']:.2f} ms",f"{r['reduction_percent']:+.2f}%",
                     'reference' if q is None else f"{100*q['relative_rms']:.3f}%"])
    body+=table(['Mode','Median','Maximum','Time reduction vs pooled BF16','Current-chunk latent RMS difference'],rows)
    body+='<p>Tensor scaling uses three Triton launches for reduction, scale and cast. Row scaling computes the per-row maximum, scale and cast in one Triton launch. '
    body+='Row scaling also uses per-output-channel weight scales. Fast accumulation changes the FP8 GEMM accumulation strategy. '
    body+='These are numerical changes, so speed and quality need separate assessment.</p></section>'
    if sweep:
        extra=analysis['initial_sweep']
        body+='<section><h2>Initial fixed-chunk sweep</h2>'
        body+=table(['Mode','Median chunk','Reduction vs initial BF16','Chunk latent RMS difference'],[
            [LABELS[n],f"{r['median_ms']:.2f} ms",f"{r['reduction_percent']:.2f}%",f"{100*r['current_chunk_accuracy']['relative_rms']:.3f}%"]
            for n,r in extra['candidates'].items()])
        body+='<p class="warn">'+html.escape(extra['limitation'])+'</p>'
        body+=f'<p>These fixed-chunk timings are valid, but this initial run has no measured full-video FP8 result. '
        body+=f'<a href="{relative(args.sweep_run/"summary.json")}">Initial measurements</a></p></section>'
    if screen:
        extra=analysis['additional_screening']
        body+='<section><h2>Additional kernel screening</h2><p>Separate H100 run, with its own before/after BF16 controls. '
        body+=f"Pooled BF16 reference: <b>{extra['pooled_bf16_ms']:.2f} ms</b>. This run measures fixed chunks only.</p>"
        body+=table(['Mode','Median chunk','Reduction vs its own BF16','Chunk latent RMS difference'],[
            [LABELS[n],f"{r['median_ms']:.2f} ms",f"{r['reduction_percent']:.2f}%",f"{100*r['current_chunk_accuracy']['relative_rms']:.3f}%"]
            for n,r in extra['candidates'].items()])
        body+=f'<p><a href="{relative(args.screening_run/"summary.json")}">Screening measurements and controls</a></p></section>'
    body+='<section><h2>Numerical drift across each clip</h2>'+table(['Request','Final latent RMS difference','Maximum absolute difference','Finite'],[
        [i+1,f"{100*r['relative_rms']:.2f}%",f"{r['max_abs']:.4f}",r['finite']] for i,r in enumerate(s['full_latent_comparison'])])
    body+='<p>Latent RMS differences are not perceptual quality scores. Autoregressive differences can grow over successive chunks; inspect the clips above. '
    body+='The shared conditioned first latent and all subsequent frame-wise differences are recorded in the raw JSON.</p></section>'
    body+='<section><h2>Profiler traces and raw evidence</h2><p>'
    profiles={}
    for name in s['frozen']:
        path=args.run/(name+'-trace.json')
        profiles[name]=analyze_trace(path)
        target=args.output/(name+'-trace.json.gz')
        with path.open('rb') as src,gzip.open(target,'wb') as dest:shutil.copyfileobj(src,dest)
        body+=f'<a href="{target.name}" download>{LABELS[name]} trace</a> · '
    if screen:
        for name in analysis['additional_screening']['candidates']:
            path=args.screening_run/(name+'-trace.json')
            profiles['screening-'+name]=analyze_trace(path)
            target=args.output/('screening-'+name+'-trace.json.gz')
            with path.open('rb') as src,gzip.open(target,'wb') as dest:shutil.copyfileobj(src,dest)
            body+=f'<a href="{target.name}" download>Screening: {LABELS[name]} trace</a> · '
    body+=f'<a href="{relative(args.run/"summary.json")}">Raw measurements</a> · <a href="analysis.json">Analysis JSON</a></p>'
    body+='<p>Open downloaded traces in <a href="https://ui.perfetto.dev/">Perfetto</a>. Profiled timings are diagnostic; the tables above use unprofiled measurements.</p></section>'
    (args.output/'profiles.json').write_text(json.dumps(profiles,indent=2)+'\n')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>H100 native FP8 comparison</title><style>
:root{color-scheme:dark;font:16px system-ui;background:#0c1421;color:#e7edf5}body{max-width:1250px;margin:auto;padding:30px}h1{font-size:38px}h2{font-size:23px}p{line-height:1.7;color:#bfcddd}.lead{font-size:26px;color:#fff}.tag{color:#66ddc2;letter-spacing:2px;font-size:12px}section{padding:24px;background:#132033;border:1px solid #2c405a;border-radius:12px;margin:22px 0}a{color:#89c6ff}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:10px;border-bottom:1px solid #2c405a;text-align:right}th:first-child,td:first-child{text-align:left}.warn{border-left:3px solid #eab968;padding-left:14px}.videos{display:grid;grid-template-columns:1fr 1fr;gap:16px}figure{margin:14px 0}figcaption{padding:8px 0}video{width:100%;background:#000;border-radius:8px}button{padding:9px 16px;background:#243c55;border:1px solid #587999;color:#fff;border-radius:6px;cursor:pointer}@media(max-width:700px){.videos{grid-template-columns:1fr}body{padding:12px}section{padding:14px}}
</style>'''+body+'''<script>
const vids=row=>[...document.querySelectorAll('[data-group="'+row+'"] video')];
document.querySelectorAll('.play').forEach(b=>b.onclick=()=>{const v=vids(b.dataset.row),start=v[0].paused;v[1].currentTime=v[0].currentTime;v.forEach(x=>start?x.play().catch(()=>{}):x.pause())});
document.querySelectorAll('.reset').forEach(b=>b.onclick=()=>vids(b.dataset.row).forEach(x=>{x.pause();x.currentTime=0}));
</script></html>'''
    (args.output/'index.html').write_text(page)
    print(json.dumps({k:v for k,v in analysis.items() if k not in ('full_latent_comparison',)},indent=2))


if __name__=='__main__':main()
