"""Build a standalone viewer for the three continuous 21-minute outputs."""
import argparse
import json
from pathlib import Path

MODES = ('none','fp8-cast','fp8-dynamic')
LABELS = ('BF16 baseline','FP8 storage / BF16 compute','FP8 matrix multiplication')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    summaries={m:json.loads((args.root/'long'/m/'summary.json').read_text())[0] for m in MODES}
    cards=''.join(f'''<article><h2>{label}</h2><video id="v{i}" src="long/{mode}/cache-on-measured-1.mp4" poster="quality/{mode}-0010s.png" preload="metadata" playsinline {'muted' if i else ''}></video><p><a href="long/{mode}/cache-on-measured-1.mp4">Open full video</a> · <a href="long/{mode}/manifest.json">Settings</a></p></article>''' for i,(mode,label) in enumerate(zip(MODES,LABELS)))
    rows=''.join('<tr><th>'+label+'</th>'+''.join(f'<td>{summaries[m][key]["median"]:.2f} {unit}</td>' for m in MODES)+'</tr>' for label,key,unit in [('Full request','end_to_end_seconds','s'),('AR sampling','ar_sampling_seconds','s'),('Peak live VRAM','peak_allocated_gib','GiB')])
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AvatarForever · 21-minute FP8 comparison</title><style>
:root{color-scheme:dark;font:15px/1.55 system-ui;background:#101216;color:#eef1f8}body{max-width:1680px;margin:auto;padding:28px}h1{font-size:34px;margin-bottom:8px}h2{font-size:18px;padding:0 14px}.muted{color:#acb7cb}a{color:#a9c2ff}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}article,.controls{background:#1a202a;border:1px solid #354055;border-radius:12px;overflow:hidden}article p{padding:0 14px}video{width:100%;aspect-ratio:3/2;background:black}.controls{padding:18px;margin-top:20px}button,select{font:inherit;border:1px solid #4e5d77;color:inherit;background:#2a3547;border-radius:6px;padding:7px 12px;margin:4px}input[type=range]{width:100%;accent-color:#abc1ff}output{margin-left:12px;font-variant-numeric:tabular-nums}table{border-collapse:collapse;margin:24px 0;width:100%}th,td{text-align:left;border-bottom:1px solid #354055;padding:10px}figure{margin:20px 0}figure img{max-width:100%;width:1152px}.status{color:#e9bb86}@media(max-width:800px){.grid{grid-template-columns:1fr}body{padding:12px}}
</style><body><p class="muted">AvatarForever · H100 · continuous autoregressive generation</p><h1>21 minutes, three precision modes</h1><p class="muted">31,505 frames · 768 × 512 · 25 FPS · seed 42 · identical reference, prompt and audio · one-stage, four denoising steps · ForeverCache on.</p><main class="grid">CARDS</main>
<section class="controls"><button id="play">Play all</button><button id="restart">Restart</button><button id="back">−1 frame</button><button id="next">+1 frame</button><label>Audio <select id="audio"><option value="0">BF16</option><option value="1">FP8 storage</option><option value="2">FP8 compute</option><option value="-1">Muted</option></select></label><label>Speed <select id="speed"><option>1</option><option>0.5</option><option>2</option></select>×</label><output id="position"></output><input id="seek" type="range" min="0" max="31504" step="1" value="0" aria-label="Video frame"><div id="jumps"></div><p id="status" class="status">Load the videos, then use shared playback or the time buttons.</p></section>
<p>Each video is one continuous AR generation; history is never reset or restarted. The FP8 compute mode uses fused activation quantization, with CUDA graph capture only for smaller activation tensors. GEMM, attention and the transformer remain uncaptured. All modes use the same overlapping-window audio latents.</p><p>The long runs unload models between stages to keep memory bounded. Their peak VRAM and total times are not directly comparable to the earlier all-models-resident short tests. Each long mode has one measured run following a 257-frame warm-up.</p><table><thead><tr><th>Long run</th><th>BF16</th><th>FP8 storage</th><th>FP8 compute</th></tr></thead><tbody>ROWS</tbody></table>
<p>Speech: first 21 minutes of <a href="https://commons.wikimedia.org/wiki/File:Jfk_American_University_4654_06-10-63.ogg">JFK’s American University address, June 10, 1963</a> (public-domain government recording). The images are generated model outputs. <a href="quality/quality.json">Frame/audio checks and statistics</a> · <a href="comparison-summary.json">Measurements</a> · <a href="../../benchmarks/FP8_LONG.md">Report</a></p><h2>Visual observations</h2><p>All three modes show clothing and background drift, including BF16. The inspected native FP8 samples remain coherent through 20:55, with no obvious late collapse unique to that mode. Drift is not monotonic: some later frames return closer to the initial appearance. This is one seed and recording; sampled stills do not establish equal perceptual quality or validate lip synchronization.</p><h2>Samples across the full duration</h2><p>Columns: BF16, FP8 storage, FP8 compute. Pose changes alone do not establish quality loss. Inspect identity, detail, temporal consistency and lip synchronization in the full videos.</p><figure><img src="quality/contact-1.png" alt="Samples from 0 to 6 minutes"></figure><figure><img src="quality/contact-2.png" alt="Samples from 8 to 14 minutes"></figure><figure><img src="quality/contact-3.png" alt="Samples from 16 to nearly 21 minutes"></figure>
<script>
const videos=[0,1,2].map(i=>document.getElementById('v'+i)), master=videos[0],fps=25,last=31504;
const play=document.getElementById('play'),seek=document.getElementById('seek'),position=document.getElementById('position'),status=document.getElementById('status');
function stamp(t){return Math.floor(t/60)+':'+(t%60).toFixed(2).padStart(5,'0')}
function display(){const f=Math.min(last,Math.round(master.currentTime*fps));seek.value=f;position.textContent=stamp(master.currentTime)+' / 21:00.20 · frame '+(f+1)}
function pause(){videos.forEach(v=>v.pause());play.textContent='Play all'}
function go(t){pause();videos.forEach(v=>{v.currentTime=Math.max(0,Math.min(t,last/fps))});display()}
async function toggle(){if(!master.paused){pause();return}if(master.ended)go(0);videos.slice(1).forEach(v=>v.currentTime=master.currentTime);try{await Promise.all(videos.map(v=>v.play()));play.textContent='Pause all';status.textContent=''}catch(e){pause();status.textContent=e.message}}
play.onclick=toggle;document.getElementById('restart').onclick=()=>go(0);document.getElementById('back').onclick=()=>go((Math.round(master.currentTime*fps)-1)/fps);document.getElementById('next').onclick=()=>go((Math.round(master.currentTime*fps)+1)/fps);seek.oninput=()=>go(Number(seek.value)/fps);
document.getElementById('audio').onchange=e=>videos.forEach((v,i)=>v.muted=i!==Number(e.target.value));document.getElementById('speed').onchange=e=>videos.forEach(v=>v.playbackRate=Number(e.target.value));
for(const t of [0,60,300,600,900,1200,1250]){const b=document.createElement('button');b.textContent=stamp(t);b.onclick=()=>go(t);document.getElementById('jumps').appendChild(b)}
videos.forEach(v=>{v.onclick=toggle;v.addEventListener('error',()=>status.textContent='Unable to load a video; keep the HTML beside the long/ and quality/ folders.')});master.addEventListener('ended',pause);master.addEventListener('seeked',display);
function tick(){if(!master.paused){videos.slice(1).forEach(v=>{if(Math.abs(v.currentTime-master.currentTime)>.12&&!v.seeking)v.currentTime=master.currentTime});display()}requestAnimationFrame(tick)}display();tick();
</script></body></html>'''
    (args.root/'index.html').write_text(html.replace('CARDS',cards).replace('ROWS',rows))
    (args.root/'comparison-summary.json').write_text(json.dumps(summaries,indent=2)+'\n')


if __name__=='__main__':
    main()
