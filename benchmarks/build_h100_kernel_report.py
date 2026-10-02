"""Build a local, auditable report from the H100 kernel study artifacts."""
import argparse
from collections import Counter
import gzip
import html
import json
import os
from pathlib import Path
import shutil
import statistics


def analyze_trace(path):
    trace = json.loads(path.read_text())
    gpu = [e for e in trace['traceEvents'] if e.get('ph') == 'X' and e.get('dur',0) > 0
           and e.get('cat') in ('kernel','gpu_memcpy','gpu_memset')]
    merged=[]
    for a,b in sorted((e['ts'],e['ts']+e['dur']) for e in gpu):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1],b)
        else:
            merged.append([a,b])
    span = merged[-1][1]-merged[0][0]
    active = sum(b-a for a,b in merged)
    categories, durations, calls = Counter(), Counter(), Counter()
    for e in gpu:
        name = e['name'].lower()
        if e['cat'] != 'kernel':
            category = 'Copy / memset'
        elif name.startswith('cudnn_generated'):
            category = 'Attention'
        elif any(x in name for x in ('nvjet','gemm','cutlass','xmma')):
            category = 'Matrix multiplication'
        else:
            category = 'Other kernels'
        categories[category] += e['dur']/1000
        if e['cat'] == 'kernel':
            durations[e['name']] += e['dur']/1000
            calls[e['name']] += 1
    return dict(span_ms=span/1000,active_union_ms=active/1000,idle_ms=(span-active)/1000,
                idle_percent=100*(span-active)/span,categories_ms=dict(categories),
                kernel_count=sum(calls.values()),top_kernels=[dict(name=n,ms=v,calls=calls[n]) for n,v in durations.most_common(50)])


def table(headers,rows):
    return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in headers)+'</tr>'+''.join(
        '<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table></div>'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--integrated',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('outputs/h100-kernels'))
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    study=json.loads((a.study/'summary.json').read_text())
    applied=json.loads((a.integrated/'summary.json').read_text())
    assert study['complete'] and applied['complete']
    profile=analyze_trace(a.study/'trace.json')
    before=statistics.median(applied['baseline_chunk_ms'])
    after=statistics.median(applied['baseline_after_ms'])
    tuned=statistics.median(applied['integrated']['tuned_mm']['chunk_ms'])
    baseline=statistics.median(applied['baseline_chunk_ms']+applied['baseline_after_ms'])
    reduction=1-tuned/baseline
    def candidate_total(kind,name):
        return sum(r['calls']*r['candidates'][name]['median_ms'] for r in study[kind] if 'median_ms' in r['candidates'].get(name,{}))
    facts=dict(study_run=a.study.name,integrated_run=a.integrated.name,baseline_before_ms=before,
               baseline_after_ms=after,baseline_pooled_ms=baseline,tuned_ms=tuned,reduction_percent=100*reduction,
               target_25_percent_ms=.75*baseline,target_30_percent_ms=.70*baseline,
               linear_flop_coverage=sum(r['flops'] for r in study['linears'])/sum(r['flops'] for r in study['workload']),
               profile=profile,attention_weighted_ms={n:candidate_total('attention',n)
                   for n in ('default','FLASH_ATTENTION','CUDNN_ATTENTION','EFFICIENT_ATTENTION')},
               quality=applied['integrated']['tuned_mm']['current_chunk_accuracy'],
               selected_mm=applied['selected_mm_kernels'],
               note='Warmed frozen AR chunk, batch two, four steps, 768x512, original BF16, regional compilation. No end-to-end or long-video quality claim.')
    (a.output/'analysis.json').write_text(json.dumps(facts,indent=2)+'\n')
    for name,path in [('baseline',a.study/'trace.json'),('tuned',a.integrated/'tuned_mm-trace.json')]:
        with path.open('rb') as src,gzip.open(a.output/(name+'-trace.json.gz'),'wb') as dst:
            shutil.copyfileobj(src,dst)
    def link(path,label):
        return '<a href="'+html.escape(os.path.relpath(path,a.output),quote=True)+'">'+html.escape(label)+'</a>'
    body=f'''<div class="tag">AVATARFOREVER · MODAL H100 · BF16 · TWO REQUESTS</div>
<h1>Do kernel replacements buy 25–30%?</h1>
<p class="lead">Measured chunk-time change: <strong>{100*reduction:+.2f}% reduction</strong>.</p>
<p>Each chunk produces 32 video frames per request at 25 FPS. Both requests must finish within <b>1,280 ms</b>.
Your stricter generation target leaves room for audio encoding and video decoding.</p>'''
    body+='<section><h2>Actual compiled sampler</h2>'+table(['Configuration','Median chunk time'],[
        ['Baseline, before replacements',f'{before:.2f} ms'],['Tuned kernel replacements',f'{tuned:.2f} ms'],
        ['Baseline, after restoring original kernels',f'{after:.2f} ms'],
        ['25% / 30% reduction targets',f'{.75*baseline:.2f} / {.70*baseline:.2f} ms']])
    body+='<p>Five warmed measurements per condition. Same frozen input and AR history; synchronization only at chunk boundaries. '
    body+='The final baseline checks for drift during tuning. The percentage uses the pooled median of both baseline groups. '
    body+='Compilation and profiling are outside these measurements. Audio encoding and VAE decoding are excluded.</p></section>'
    body+='<section><h2>Where the baseline spends GPU time</h2>'+table(['Recorded activity','Duration'],[
        *[[k,f'{v:.2f} ms'] for k,v in profile['categories_ms'].items()],
        ['Gaps between GPU activities',f"{profile['idle_ms']:.2f} ms ({profile['idle_percent']:.2f}%)"]])
    body+='<p>Profiler durations diagnose the bottleneck; use unprofiled measurements above for speed. GPU gaps include causes other than launch overhead. '
    body+='Activity durations are summed, while gaps use the union across streams.</p></section>'
    body+='<section><h2>Matrix multiplication candidates</h2><p>Captured shapes cover '
    body+=f"{100*facts['linear_flop_coverage']:.2f}% of linear FLOPs. Inputs and weights are original BF16. "
    body+='Each microbenchmark uses CUDA events, a 64 MiB L2 flush before timing, and an independent retest of the winning configuration. '
    body+='These are isolated operation times, not full inference latency. Changes in clocks, layout and surrounding fusion can affect transfer to the model.</p>'
    rows=[]
    for r in study['linears']:
        cs=r['candidates']
        row=[r['key'],r['calls']]
        for name in ('compiled_bf16','cublaslt_tuned','triton_persistent'):
            row.append(f"{cs[name]['median_ms']:.4f} ms" if 'median_ms' in cs.get(name,{}) else 'unsupported')
        rows.append(row)
    body+=table(['M × K × N','Calls','Compiled BF16','Tuned cuBLASLt','Persistent Triton'],rows)
    body+='<p>Only shape families at least 3% faster than every control measurement were installed in the full sampler. '
    body+='The experimental adapters preserve BF16 operands, biases, dense attention, four denoising steps and ForeverCache.</p></section>'
    body+='<section><h2>Attention backends</h2>'+table(['Backend','Calls-weighted microbenchmark sum'],[
        [n,f'{v:.2f} ms'] for n,v in facts['attention_weighted_ms'].items()])
    body+='<p>The default already selects cuDNN Hopper kernels in this environment, confirmed by the trace. '
    body+='Forcing PyTorch’s Flash SDPA backend is slower. This does not benchmark the separately installed FlashAttention-3 package.</p></section>'
    q=facts['quality']
    body+='<section><h2>Numerical check and scope</h2>'
    body+=f"<p>Changed video tokens in the frozen chunk: relative RMS difference <b>{100*q['relative_rms']:.6f}%</b>, maximum absolute difference <b>{q['max_abs']:.6g}</b>; finite: <b>{q['finite']}</b>. "
    body+='This is a single-chunk numerical check, not a long-video perceptual evaluation. No streaming decoder, audio overlap, native FP8 or new denoising schedule was tested in this pass.</p></section>'
    body+='<section><h2>Inspect the telemetry</h2><p><a href="baseline-trace.json.gz" download>Baseline trace</a> · '
    body+='<a href="tuned-trace.json.gz" download>Tuned trace</a> · <a href="analysis.json">Analysis JSON</a> · '
    body+=link(a.study/'summary.json','Kernel searches and numerical checks')+' · '+link(a.integrated/'summary.json','Full chunk measurements')+'</p>'
    body+='<p>Open the traces in <a href="https://ui.perfetto.dev/">Perfetto</a>. The table below lists baseline GPU kernels.</p>'
    body+='<input id="filter" type="search" placeholder="Filter kernel names" aria-label="Filter kernel names">'
    body+='<div id="kernels">'+table(['Kernel','Calls','Total ms'],[[r['name'],r['calls'],f"{r['ms']:.3f}"] for r in profile['top_kernels']])+'</div></section>'
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>H100 kernel replacement study</title><style>
:root{color-scheme:dark;font:16px system-ui;background:#0c1421;color:#e7edf5}body{max-width:1250px;margin:auto;padding:32px}h1{font-size:38px}h2{font-size:22px}p{line-height:1.7;color:#bfcddd}.lead{font-size:25px;color:#fff}.tag{color:#66ddc2;letter-spacing:2px;font-size:12px}section{padding:24px;background:#132033;border:1px solid #2c405a;border-radius:12px;margin:22px 0}a{color:#89c6ff}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:10px;border-bottom:1px solid #2c405a;text-align:right}th:first-child,td:first-child{text-align:left}#kernels td:first-child{font:12px ui-monospace,monospace;max-width:800px;overflow-wrap:anywhere}input{width:100%;padding:12px;background:#0c1421;color:inherit;border:1px solid #526981;border-radius:6px;box-sizing:border-box}
</style>'''+body+'''<script>document.querySelector('#filter').oninput=e=>{const q=e.target.value.toLowerCase();document.querySelectorAll('#kernels tr').forEach((r,i)=>{if(i)r.hidden=!r.textContent.toLowerCase().includes(q)})}</script></html>'''
    (a.output/'index.html').write_text(page)
    print(json.dumps({k:v for k,v in facts.items() if k!='profile'},indent=2))


if __name__=='__main__':
    main()
