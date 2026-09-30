"""Summarize Kineto traces and build a dependency-free, local timeline viewer."""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import statistics
from pathlib import Path


def union(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def analyze(directory):
    trace = json.loads((directory / "trace.json").read_text())
    events = [e for e in trace["traceEvents"] if e.get("ph") == "X" and e.get("dur", 0) > 0]
    gpu = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    kernels = [e for e in gpu if e.get("cat") == "kernel"]
    if not gpu:
        raise ValueError(f"No CUDA activities in {directory}")
    start = min(e["ts"] for e in gpu)
    end = max(e["ts"] + e["dur"] for e in gpu)
    launches = [e for e in events if e.get("cat") in ("cuda_runtime", "cuda_driver")
                and "launch" in e["name"].lower()]
    apis = [e for e in events if e.get("cat") in ("cuda_runtime", "cuda_driver")]
    markers = [e for e in events if e.get("cat") == "user_annotation" and e["name"].startswith("DIT_STEP_")]
    gpu_markers = [e for e in events if e.get("cat") == "gpu_user_annotation" and e["name"].startswith("DIT_STEP_")]
    stream_syncs = [e for e in apis if e["name"] == "cudaStreamSynchronize"]
    active = union((e["ts"], e["ts"] + e["dur"]) for e in gpu)
    gaps = [[a[1], b[0]] for a, b in zip(active, active[1:])]
    busy = sum(b - a for a, b in active)
    span = end - start
    # Runtime and driver APIs can be nested: union, rather than sum, for exposure.
    launch_union = union((e["ts"], e["ts"] + e["dur"]) for e in launches)
    launch_duration = sum(b - a for a, b in launch_union)
    groups = collections.defaultdict(lambda: {"calls": 0, "total_us": 0.0})
    for event in kernels:
        row = groups[event["name"]]
        row["calls"] += 1
        row["total_us"] += event["dur"]
    top = sorted(({"name": name, **row} for name, row in groups.items()),
                 key=lambda row: row["total_us"], reverse=True)
    window_record = json.loads((directory / "window-results.json").read_text())
    windows = window_record["windows"]
    control = [r["window_cuda_event_ms"] for r in windows if not r["warmup"] and not r["profiled"]]
    profiled = [r["window_cuda_event_ms"] for r in windows if r["profiled"]]
    assert len(control) == 3 and len(profiled) == 1
    manifest = json.loads((directory / "manifest.json").read_text())
    results = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]
    assert len(results) == 5 and all(r.get("final_latent_finite") for r in results)
    assert all(r.get("frames") == 257 for r in results)
    assert all(r["dynamo_counters"] == results[0]["dynamo_counters"] for r in results), "Compilation changed after warmup"
    assert not any("graphlaunch" in e["name"].lower() for e in launches), "Unexpected CUDA graph replay"
    with (directory / "trace.json").open("rb") as trace_file:
        trace_sha256 = hashlib.file_digest(trace_file, "sha256").hexdigest()
    summary = {
        "mode": directory.name, "control_window_ms": control,
        "control_median_ms": statistics.median(control),
        "profiled_window_ms": profiled[0],
        "profiler_slowdown": profiled[0] / statistics.median(control),
        "gpu_span_ms": span / 1000, "gpu_active_union_ms": busy / 1000,
        "gpu_idle_gaps_ms": (span - busy) / 1000,
        "gpu_idle_gap_percent": 100 * (span - busy) / span,
        "gpu_gaps_over_10us": sum(b - a > 10 for a, b in gaps),
        "kernel_count": len(kernels), "launch_api_count": len(launches),
        "launch_api_names": dict(collections.Counter(e["name"] for e in launches)),
        "host_launch_api_union_ms": launch_duration / 1000,
        "host_launch_api_median_us": statistics.median(e["dur"] for e in launches) if launches else None,
        "stream_sync_count": len(stream_syncs),
        "stream_sync_host_wait_ms": sum(e["dur"] for e in stream_syncs) / 1000,
        "trace_sha256": trace_sha256,
        "profile_harness_sha256": window_record["source_sha256"],
        "gpu_activity_categories": dict(collections.Counter(e.get("cat") for e in gpu)),
        "top_kernels": top[:30], "all_outputs_finite": True,
        "last_dynamo_counters": results[-1]["dynamo_counters"],
        "arguments": manifest["arguments"],
        "source_sha256": manifest["source_sha256"],
        "gpu": manifest["gpu"], "versions": manifest["versions"],
        "audio_sha256": manifest["audio_sha256"], "reference_sha256": manifest["reference_sha256"],
        "trace_categories": dict(collections.Counter(e.get("cat") for e in events)),
    }
    names, name_ids = [], {}
    def name_id(value):
        if value not in name_ids:
            name_ids[value] = len(names)
            names.append(value)
        return name_ids[value]
    # GPU streams are separate lanes; CPU operators remain in the original trace.
    stream_ids = sorted(set(str(e.get("args", {}).get("stream", "unknown")) for e in gpu))
    lanes = ["CPU: denoising steps", "CPU: CUDA launch APIs", "CPU: copies / sync / other", "GPU: denoising steps"]
    lanes += ["GPU stream " + stream for stream in stream_ids]
    packed = []
    for event in markers + gpu_markers + apis + gpu:
        if event.get("cat") == "user_annotation":
            lane = 0
        elif event.get("cat") == "gpu_user_annotation":
            lane = 3
        elif event.get("cat") in ("cuda_runtime", "cuda_driver"):
            lane = 1 if "launch" in event["name"].lower() else 2
        else:
            lane = 4 + stream_ids.index(str(event.get("args", {}).get("stream", "unknown")))
        category = "kernel" if event.get("cat") == "kernel" else event.get("cat", "annotation")
        if "Synchronize" in event["name"]:
            category = "cuda_sync"
        packed.append([round((event["ts"] - start) / 1000, 6), round(event["dur"] / 1000, 6),
                       lane, name_id(event["name"]), category,
                       event.get("args", {}).get("correlation")])
    return summary, {"names": names, "events": sorted(packed), "lanes": lanes,
                     "span": span / 1000, "gaps": [[(a-start)/1000, (b-start)/1000] for a,b in gaps]}


HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>A100 · Kernel launch telemetry</title>
<style>
:root{color-scheme:dark;font:15px system-ui;background:#0c111b;color:#e6edf5}*{box-sizing:border-box}
body{margin:0 auto;max-width:1500px;padding:30px}h1{font-size:30px;margin:6px 0 12px}h2{font-size:20px;margin:0 0 14px}
p{line-height:1.6;color:#aebfd2}a{color:#86c5ff}header{margin-bottom:24px}.tag{color:#73dbc9;font-size:12px;letter-spacing:2px}
.panel{border:1px solid #2d3b50;background:#121c2a;border-radius:12px;padding:22px;margin:18px 0}
.warn{border-left:3px solid #ffcf7b;padding-left:14px}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{text-align:right;border-bottom:1px solid #2c3747;padding:10px}th:first-child,td:first-child{text-align:left}
.controls{display:flex;align-items:center;flex-wrap:wrap;gap:10px;margin:14px 0}button,select,input{background:#1c2b40;color:#e6edf5;border:1px solid #415675;border-radius:5px;padding:8px}button{cursor:pointer}button:hover{background:#30465f}
input[type=range]{flex:1;min-width:150px}canvas{display:block;width:100%;border:1px solid #34445b;border-radius:6px;touch-action:none;cursor:crosshair}
.hint{font-size:13px}.tooltip{min-height:58px;font:12px ui-monospace,monospace;overflow-wrap:anywhere;line-height:1.7;padding:12px;background:#0b131f}
.kernel{max-width:780px;word-break:break-all;font:12px ui-monospace,monospace}.legend{display:flex;gap:20px;font-size:12px;color:#afbed1;margin:10px 0}.legend span:before{content:'■';color:var(--c);margin-right:6px}
details summary{cursor:pointer;padding:10px 0}.scroll{overflow-x:auto}footer{margin:30px 0;color:#9aabc1;font-size:13px}
</style><header><div class="tag">AVATARFOREVER / A100 40 GB / MEASURED TELEMETRY</div>
<h1>Where does time go between kernels?</h1>
<p>Matched eager and torch.compile runs · FP8 weight storage / BF16 compute · 768 × 512 · ForeverCache · four denoising steps of one steady AR chunk.</p><a href="#timelines">Inspect the timelines ↓</a></header>
<section class="panel"><h2>Timing & launch activity</h2><div class="scroll"><table id="metrics"></table></div>
<p class="warn">GPU gaps are observed inactivity between recorded GPU operations, <b>not a direct measurement of kernel launch overhead</b>. CPU dispatch, launch APIs, dependencies, and profiling itself can contribute. Host launch time overlaps GPU work and must not be added to GPU time. Active GPU intervals do not measure SM utilization or peak FLOP/bandwidth efficiency.</p>
<p id="interpretation"></p></section>
<section class="panel" id="timelines"><h2>Explore the timelines</h2>
<p class="hint">Both views share the same time scale. Time zero is each trace’s first GPU activity. CPU step markers show host submission, not GPU completion. Drag to pan; scroll to zoom around the pointer; hover over a bar for the exact kernel or API.</p>
<div class="controls"><button id="reset">Full window</button><button id="zin">Zoom in ×4</button><button id="zout">Zoom out ×4</button><button id="gap">Largest eager gap</button><label>Start <input id="position" type="range" min="0" max="1000" value="0"></label><span id="range"></span></div>
<div class="legend"><span style="--c:#8d9dff">CPU steps</span><span style="--c:#ffcb7c">Launch APIs</span><span style="--c:#ee8b91">CPU synchronization</span><span style="--c:#62dac4">GPU kernels</span><span style="--c:#f293d0">GPU copies / memset</span></div>
<h3>Eager</h3><canvas id="eager"></canvas><div class="tooltip" id="eager-tip">Hover over a bar to inspect.</div>
<h3>torch.compile · regional</h3><canvas id="compiled"></canvas><div class="tooltip" id="compiled-tip">Hover over a bar to inspect.</div>
<p class="hint">Zoom into microsecond-scale gaps to see individual launches. Black areas in GPU lanes contain no recorded GPU activity. Other CUDA API lanes include the intentionally marked synchronization at the end of the window.</p></section>
<section class="panel"><h2>Raw telemetry</h2>
<p>These Chrome trace files contain the complete recorded CPU operator hierarchy, runtime calls, CUDA kernels, and CPU-to-GPU correlation flows. Stream synchronization can wait for queued GPU work; its host duration is not GPU idle time. Open <a href="https://ui.perfetto.dev/" target="_blank" rel="noopener">Perfetto</a>, choose <b>Open trace file</b>, and select a downloaded trace. The local viewer does not upload traces.</p>
<div class="controls"><a href="eager/trace.json.gz" download>Eager trace (.json.gz)</a><a href="compiled/trace.json.gz" download>Compiled trace (.json.gz)</a><a href="comparison.json">Analysis JSON</a><a href="eager/operator-table.txt">Eager operator table</a><a href="compiled/operator-table.txt">Compiled operator table</a></div>
<details><summary>Protocol and interpretation limits</summary><p>One complete 257-frame warmup, three unprofiled control requests, then one profiled request, in a fresh process per mode. The measured window covers transformer calls 16–19 (zero based): cache population followed by three reuse steps, including the AR updates and preparation before the next chunk’s first forward. All internal per-forward synchronization from the old latency benchmark is removed. Harness-added synchronization remains immediately before and after the selected window. Existing implicit synchronization inside model/sampler operations is preserved and visible in the trace. Compilation, model loading, and VAE decoding are outside the capture. Ordinary regional compilation is compared with eager; autotuning and CUDA graphs are disabled in both. Profiling records CPU and CUDA activity without stack, shape, or memory tracking.</p><p>Unprofiled control time uses CUDA events around the complete window, including idle gaps. GPU span runs from first GPU activity to last activity; active time is the union across streams, and gaps are its complement. Host launch API time is the union of runtime/driver launch intervals to avoid nested double counting. CUDA API call duration is not the same as exposed latency: much of it can overlap GPU execution. Compilation can reduce both kernel count and kernel execution time; this experiment does not assign its entire speedup to launch overhead. No CUDA graph speedup is claimed.</p></details>
</section><section class="panel"><h2>Top GPU kernels by total duration</h2><div class="controls"><select id="topmode"><option value="eager">Eager</option><option value="compiled">Compiled</option></select></div><div class="scroll"><table id="top"></table></div></section>
<footer>Generated from actual CUDA profiler traces. Source: benchmarks/profile_launch_overhead.py and benchmarks/build_launch_profile_viewer.py. Timing numbers from profiled traces are diagnostic; use unprofiled controls for performance.</footer>
<script>const DATA=__DATA__;
const $=id=>document.getElementById(id), fmt=(x,n=2)=>Number(x).toLocaleString(undefined,{maximumFractionDigits:n,minimumFractionDigits:n});
const rows=[['Unprofiled chunk, median of 3','control_median_ms','ms'],['Profiled chunk','profiled_window_ms','ms'],['Profiler slowdown','profiler_slowdown','×'],['GPU activity span','gpu_span_ms','ms'],['GPU active time (union)','gpu_active_union_ms','ms'],['GPU idle gaps','gpu_idle_gaps_ms','ms'],['GPU idle fraction','gpu_idle_gap_percent','%'],['Kernel count','kernel_count','count'],['Host launch API count','launch_api_count','count'],['Host launch API time (union)','host_launch_api_union_ms','ms'],['Median launch API duration','host_launch_api_median_us','µs'],['GPU gaps larger than 10 µs','gpu_gaps_over_10us','count'],['Model CUDA stream synchronizations','stream_sync_count','count'],['Host stream-sync wait (overlaps GPU work)','stream_sync_host_wait_ms','ms']];
$('metrics').innerHTML='<tr><th>Measurement</th><th>Eager</th><th>Compiled</th></tr>'+rows.map(([label,key,unit])=>'<tr><td>'+label+'</td>'+['eager','compiled'].map(m=>'<td>'+fmt(DATA[m].summary[key],unit==='count'?0:2)+' '+(unit==='count'?'':unit)+'</td>').join('')+'</tr>').join('');
const speed=DATA.eager.summary.control_median_ms/DATA.compiled.summary.control_median_ms;
$('interpretation').textContent='Compilation gives '+fmt(speed)+'× faster unprofiled chunk execution in this experiment. Recorded GPU active time falls by '+fmt(DATA.eager.summary.gpu_active_union_ms-DATA.compiled.summary.gpu_active_union_ms)+' ms, while gaps fall by '+fmt(DATA.eager.summary.gpu_idle_gaps_ms-DATA.compiled.summary.gpu_idle_gaps_ms)+' ms. Most of the observed timeline reduction is in GPU activity. Profiler inflation is shown above; these observations do not establish a CUDA graph speedup.';
let maxSpan=Math.max(DATA.eager.timeline.span,DATA.compiled.timeline.span), viewStart=0,viewWidth=maxSpan;
const labelWidth=185,laneHeight=36,topHeight=30;
function color(e){return e[2]===0||e[2]===3?'#8d9dff':e[2]===1?'#ffcb7c':e[2]===2?(e[4]==='cuda_sync'?'#ee8b91':'#a1abc0'):e[4]==='kernel'?'#62dac4':'#f293d0'}
function render(mode){const canvas=$(mode),t=DATA[mode].timeline,w=canvas.clientWidth,h=topHeight+t.lanes.length*laneHeight+12,dpr=devicePixelRatio||1;canvas.width=w*dpr;canvas.height=h*dpr;canvas.style.height=h+'px';const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);ctx.fillStyle='#0b131f';ctx.fillRect(0,0,w,h);const plotW=w-labelWidth;
ctx.font='11px system-ui';for(let i=0;i<=5;i++){let x=labelWidth+plotW*i/5;ctx.strokeStyle='#253348';ctx.beginPath();ctx.moveTo(x,topHeight);ctx.lineTo(x,h);ctx.stroke();ctx.fillStyle='#aebfd2';ctx.fillText(fmt(viewStart+viewWidth*i/5,viewWidth<1?4:2)+' ms',Math.min(x,w-74),18)}
t.lanes.forEach((name,i)=>{ctx.fillStyle='#aebfd2';ctx.fillText(name,8,topHeight+i*laneHeight+22)});
ctx.save();ctx.beginPath();ctx.rect(labelWidth,topHeight,plotW,h-topHeight);ctx.clip();for(const e of t.events){if(e[0]+e[1]<viewStart||e[0]>viewStart+viewWidth)continue;let x=labelWidth+(e[0]-viewStart)/viewWidth*plotW,bw=Math.max(.7,e[1]/viewWidth*plotW);ctx.fillStyle=color(e);ctx.fillRect(x,topHeight+e[2]*laneHeight+6,bw,23)}ctx.restore();}
function draw(){viewWidth=Math.min(maxSpan,Math.max(.02,viewWidth));viewStart=Math.max(0,Math.min(maxSpan-viewWidth,viewStart));['eager','compiled'].forEach(render);$('range').textContent=fmt(viewStart)+'–'+fmt(viewStart+viewWidth)+' ms';$('position').value=maxSpan>viewWidth?1000*viewStart/(maxSpan-viewWidth):0;}
function zoom(factor,anchor=.5){const point=viewStart+viewWidth*anchor;viewWidth*=factor;viewStart=point-viewWidth*anchor;draw()}
$('reset').onclick=()=>{viewStart=0;viewWidth=maxSpan;draw()};$('zin').onclick=()=>zoom(.25);$('zout').onclick=()=>zoom(4);$('position').oninput=e=>{viewStart=(maxSpan-viewWidth)*e.target.value/1000;draw()};$('gap').onclick=()=>{const gap=DATA.eager.timeline.gaps.reduce((a,b)=>b[1]-b[0]>a[1]-a[0]?b:a);viewWidth=Math.max(.1,(gap[1]-gap[0])*3);viewStart=gap[0]-viewWidth/3;draw()};
for(const mode of ['eager','compiled']){const c=$(mode);let drag=null;c.onpointerdown=e=>{drag={x:e.clientX,start:viewStart};c.setPointerCapture(e.pointerId)};c.onpointerup=()=>drag=null;c.onpointercancel=()=>drag=null;c.onpointermove=e=>{const rect=c.getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top;if(drag){viewStart=drag.start-(e.clientX-drag.x)/(rect.width-labelWidth)*viewWidth;draw();return}let lane=Math.floor((y-topHeight)/laneHeight),time=viewStart+(x-labelWidth)/(rect.width-labelWidth)*viewWidth,t=DATA[mode].timeline;const epsilon=viewWidth/(rect.width-labelWidth);let matches=t.events.filter(v=>v[2]===lane&&v[0]<=time+epsilon&&v[0]+v[1]>=time-epsilon);let ev=matches.sort((a,b)=>a[1]-b[1])[0];$(mode+'-tip').textContent=ev?t.names[ev[3]]+' | start '+fmt(ev[0],6)+' ms | duration '+fmt(ev[1]*1000,3)+' µs | correlation '+ev[5]:'Time '+fmt(time,6)+' ms · no event under cursor';};c.addEventListener('wheel',e=>{e.preventDefault();const r=c.getBoundingClientRect();zoom(e.deltaY>0?1.4:1/1.4,Math.max(0,Math.min(1,(e.clientX-r.left-labelWidth)/(r.width-labelWidth))))},{passive:false});}
function topTable(){const mode=$('topmode').value,table=$('top');table.replaceChildren();let head=document.createElement('tr');for(const name of ['Kernel','Calls','Total ms','Mean µs']){let th=document.createElement('th');th.textContent=name;head.append(th)}table.append(head);for(const r of DATA[mode].summary.top_kernels){let tr=document.createElement('tr');[r.name,r.calls,fmt(r.total_us/1000),fmt(r.total_us/r.calls)].forEach((v,i)=>{let td=document.createElement('td');td.textContent=v;if(!i)td.className='kernel';tr.append(td)});table.append(tr)}}$('topmode').onchange=topTable;topTable();window.addEventListener('resize',draw);draw();
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/a100-profile"))
    args = parser.parse_args()
    data, summary = {}, {}
    for mode in ("eager", "compiled"):
        directory = args.root / mode
        stats, timeline = analyze(directory)
        data[mode] = {"summary": stats, "timeline": timeline}
        summary[mode] = stats
        with (directory / "trace.json").open("rb") as src, gzip.open(directory / "trace.json.gz", "wb") as dst:
            import shutil
            shutil.copyfileobj(src, dst)
    ignored_args = {"compile_transformer", "compile_video_decoder", "output_dir"}
    assert ({k: v for k, v in summary["eager"]["arguments"].items() if k not in ignored_args}
            == {k: v for k, v in summary["compiled"]["arguments"].items() if k not in ignored_args})
    for key in ("source_sha256", "profile_harness_sha256", "audio_sha256", "reference_sha256", "versions", "gpu"):
        assert summary["eager"][key] == summary["compiled"][key], key
    summary["interpretation"] = (
        "GPU idle gaps are not exclusively launch overhead. Profiler overhead is quantified "
        "against three unprofiled CUDA-event windows. Host API time overlaps GPU execution. "
        "No CUDA graph capture or claim of its expected speedup. "
        "The original window-results protocol says synchronization only at boundaries: "
        "this describes harness-added barriers; implicit model synchronization was retained."
    )
    (args.root / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    payload = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")
    (args.root / "index.html").write_text(HTML.replace("__DATA__", payload))
    print(json.dumps({mode: {k: v for k, v in summary[mode].items()
                           if k not in ("top_kernels", "arguments")} for mode in data}, indent=2))


if __name__ == "__main__":
    main()
