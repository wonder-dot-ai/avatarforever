"""Build a local eager-versus-compiled video comparison from benchmark outputs."""

import argparse
import html as html_module
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/fp8-compile"))
    parser.add_argument("--baseline-dir", default="eager")
    parser.add_argument("--candidate-dir", default="regional-precision")
    parser.add_argument("--candidate-label", default="Compiled · BF16 rounding preserved")
    parser.add_argument("--gpu", default="H100")
    parser.add_argument("--quality-dir", default="quality-precision")
    parser.add_argument("--record", default="../../benchmarks/FP8_COMPILE.md")
    parser.add_argument("--reload-models", action="store_true")
    args = parser.parse_args()
    root = args.root
    names = (args.baseline_dir, args.candidate_dir)
    labels = ("Eager", html_module.escape(args.candidate_label))
    summaries = [json.loads((root / name / "summary.json").read_text())[0] for name in names]
    def median(index, key):
        return summaries[index][key]["median"]
    reduction = 100 * (1 - median(1, "end_to_end_seconds") / median(0, "end_to_end_seconds"))
    rows = "".join(
        f"<tr><th>{label}</th><td>{median(0, key):.2f} {unit}</td><td>{median(1, key):.2f} {unit}</td></tr>"
        for label, key, unit in (
            ("Whole request", "end_to_end_seconds", "s"),
            ("AR sampling", "ar_sampling_seconds", "s"),
            ("Video decoding", "vae_decode_seconds", "s"),
            ("Sampling + decoding throughput", "ar_plus_vae_fps", "FPS"),
            ("Peak allocated VRAM", "peak_allocated_gib", "GiB"),
        )
    )
    panels = "".join(
        f'<article><h2>{label}</h2><video id="v{index}" src="{name}/cache-on-measured-1.mp4" '
        f'preload="auto" playsinline {"muted" if index else ""} aria-label="{label} video"></video>'
        f'<p><a href="{name}/cache-on-measured-1.mp4">Open video</a> · '
        f'<a href="{name}/manifest.json">Run settings</a></p></article>'
        for index, (name, label) in enumerate(zip(names, labels, strict=True))
    )
    html = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AvatarForever · FP8 storage, compiled BF16 compute</title><style>
:root{color-scheme:dark;font:15px/1.55 system-ui;background:#10141b;color:#ebeff7}*{box-sizing:border-box}
body{max-width:1400px;padding:32px 24px;margin:auto}h1{font-size:clamp(26px,3vw,42px);letter-spacing:-.03em;margin:8px 0}
h2{font-size:18px;margin:14px 18px}p{color:#abb8cc}.eyebrow{font-size:12px;letter-spacing:.14em;color:#99caff;text-transform:uppercase}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}article,.controls,.measurements{background:#1a2230;border:1px solid #344158;border-radius:12px;overflow:hidden}
video{width:100%;display:block;aspect-ratio:3/2;background:black;cursor:pointer}article p{margin:12px 18px}a{color:#a4cdff}
.controls{padding:18px;margin:18px 0}.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
button,select{font:inherit;padding:8px 14px;border:1px solid #546c8b;border-radius:7px;background:#2b3e58;color:#fff;cursor:pointer}
#play{background:#b3d5ff;color:#102035;font-weight:650}input{width:100%;accent-color:#b3d5ff;margin-top:18px}
#position{margin-left:auto;font-variant-numeric:tabular-nums}#status{font-size:13px;color:#e9be87;min-height:20px}
.measurements{padding:20px}table{border-collapse:collapse;width:100%;max-width:800px}th,td{text-align:left;padding:9px;border-bottom:1px solid #344158}
.note{max-width:1050px}strong{color:#f1f5ff}@media(max-width:750px){.grid{grid-template-columns:1fr}body{padding:20px 12px}}
</style></head><body><div class="eyebrow">AvatarForever · GPU_NAME compilation experiment</div>
<h1>FP8 storage, compiled BF16 compute</h1><p>Same speech, reference, prompt and seed · one-stage AR · ForeverCache · 768 × 512 · 25 FPS · 10.28 seconds.</p>
<main class="grid">PANELS</main><section class="controls" aria-label="Shared video controls"><div class="row">
<button id="play" disabled>Play both</button><button id="restart" disabled>Restart</button><button id="back" disabled>−1 frame</button><button id="next" disabled>+1 frame</button>
<label>Audio <select id="audio"><option value="0">Eager</option><option value="1">Compiled</option><option value="none">Muted</option></select></label><output id="position">Loading…</output></div>
<input id="seek" type="range" min="0" max="256" step="1" value="0" disabled aria-label="Frame"><div id="status" role="status">Loading videos…</div></section>
<section class="measurements"><h2>PERFORMANCE_TITLE</h2><p>Median of three measured runs after one warmup. Whole request includes MP4 encoding. LOADING_NOTE</p>
<table><thead><tr><th>Measurement</th><th>Eager</th><th>Compiled</th></tr></thead><tbody>ROWS</tbody></table></section>
<p class="note"><strong>Compiled scope:</strong> all 48 transformer blocks, input preparation, output projections and each video VAE tile. FP8 weights are still cast for BF16 computation. Strict fullgraph regions; intermediate BF16 rounding is preserved.</p>
<p class="note"><strong>Quality:</strong> outputs are not bit-identical. This is a short comparison, not a new long-duration drift test. Pixel differences alone do not establish quality or lip-sync accuracy. <a href="QUALITY_DIR/sampled-comparison.png">Sampled frames</a> · <a href="QUALITY_DIR/quality-check.json">Pixel comparison</a> · <a href="RECORD_PATH">Experiment record</a></p>
<script>
const videos=[document.getElementById('v0'),document.getElementById('v1')],main=videos[0];
const play=document.getElementById('play'),seek=document.getElementById('seek'),position=document.getElementById('position'),status=document.getElementById('status');
let ready=false;
function display(){seek.value=Math.min(256,Math.round(main.currentTime*25));position.textContent=`${main.currentTime.toFixed(2)} / 10.28 s · frame ${Number(seek.value)+1} / 257`;}
function pause(){videos.forEach(v=>v.pause());play.textContent='Play both';}
function go(time){pause();videos.forEach(v=>v.currentTime=Math.max(0,Math.min(256/25,time)));display();}
async function toggle(){if(!ready)return;if(!main.paused){pause();return;}if(main.ended||main.currentTime>=256/25)go(0);videos[1].currentTime=main.currentTime;try{await Promise.all(videos.map(v=>v.play()));play.textContent='Pause both';status.textContent='';}catch(e){pause();status.textContent=`Playback failed: ${e.message}. Use Open video.`;}}
function loaded(){if(videos.every(v=>v.readyState>=2)){ready=true;document.querySelectorAll('button,input').forEach(v=>v.disabled=false);status.textContent='';display();}}
videos.forEach(v=>{v.addEventListener('click',toggle);v.addEventListener('loadeddata',loaded);v.addEventListener('error',()=>status.textContent='Could not load a local video. Use the Open video links.');});
play.onclick=toggle;main.addEventListener('ended',pause);main.addEventListener('seeked',display);
document.getElementById('restart').onclick=()=>go(0);document.getElementById('back').onclick=()=>go((Math.round(main.currentTime*25)-1)/25);document.getElementById('next').onclick=()=>go((Math.round(main.currentTime*25)+1)/25);
seek.oninput=()=>go(Number(seek.value)/25);document.getElementById('audio').onchange=e=>videos.forEach((v,i)=>v.muted=String(i)!==e.target.value);
function tick(){if(!main.paused){if(Math.abs(videos[1].currentTime-main.currentTime)>.12&&!videos[1].seeking)videos[1].currentTime=main.currentTime;display();}requestAnimationFrame(tick);}loaded();tick();
</script></body></html>'''
    loading_note = (
        "Compilation warmup is excluded, but model loading is included in each whole request. "
        "Sampling + decoding FPS excludes model loading, prompt/audio processing and MP4 encoding; "
        "it is not complete-request or streaming throughput."
        if args.reload_models else
        "Compilation and initial model loading are excluded from these warm figures."
    )
    replacements = {
        "PANELS": panels, "ROWS": rows,
        "PERFORMANCE_TITLE": (
            f"{100 * (median(1, 'ar_plus_vae_fps') / median(0, 'ar_plus_vae_fps') - 1):.1f}% "
            "higher sampling + decoding throughput"
            if args.reload_models else f"{reduction:.1f}% less time per warm request"
        ),
        "GPU_NAME": html_module.escape(args.gpu), "LOADING_NOTE": loading_note,
        "QUALITY_DIR": html_module.escape(args.quality_dir, quote=True),
        "RECORD_PATH": html_module.escape(args.record, quote=True),
    }
    for key, value in replacements.items():
        html = html.replace(key, value)
    (root / "index.html").write_text(html)


if __name__ == "__main__":
    main()
