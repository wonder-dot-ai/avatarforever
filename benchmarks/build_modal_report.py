"""Build a local review page from completed Modal inference artifacts."""
import html
import json
from pathlib import Path
import statistics


root = Path('outputs/modal')
baseline_rows, pair_rows, media, records = [], [], [], []
for path in sorted(root.glob('h100-*/summary.json')):
    folder = path.parent
    summary = json.loads(path.read_text())
    manifest = json.loads((folder / 'manifest.json').read_text())
    quantization = manifest['arguments']['quantization']
    if manifest['arguments'].get('preexpand_fp8'):
        quantization = 'FP8 values preexpanded to BF16'
    label = html.escape(folder.name)
    link = f'<a href="{folder.name}/summary.json">{label}</a>'
    if isinstance(summary, list):
        row = summary[0]
        baseline_rows.append(f'<tr><td>{link}</td><td>{row["ar_plus_vae_fps"]["median"]:.2f}</td>'
                             f'<td>{row["vae_decode_seconds"]["median"]:.3f} s</td></tr>')
        media.append(f'<section><h2>Single-request FP8 storage / BF16 compute</h2>'
                     f'<video controls preload="metadata" src="{folder.name}/cache-on-measured-1.mp4"></video>'
                     '<p>257 frames (10.28 seconds). Cold compilation is excluded from the performance table.</p></section>')
    else:
        for mode in ('alternating', 'batched'):
            selected = [r for r in summary['records'] if r['mode'] == mode]
            steady = [t for r in selected for t in r['pair_chunk_seconds'][2:-1]]
            row = {'run': folder.name, 'quantization': quantization, 'mode': mode,
                   'steady_generation_median_seconds': statistics.median(steady),
                   'steady_generation_max_seconds': max(steady),
                   'steady_pair_with_average_decode_seconds': statistics.median(
                       r['steady_pair_with_average_decode_seconds'] for r in selected),
                   'peak_allocated_gib': max(r['peak_allocated_gib'] for r in selected),
                   'all_finite': all(r['finite'] for r in selected)}
            records.append(row)
            pair_rows.append(f'<tr><td>{html.escape(quantization)} / {mode}<small>{link}</small></td>'
                             f'<td>{row["steady_generation_median_seconds"]:.3f}</td>'
                             f'<td>{row["steady_generation_max_seconds"]:.3f}</td>'
                             f'<td>{row["steady_pair_with_average_decode_seconds"]:.3f}</td>'
                             f'<td>{row["peak_allocated_gib"]:.2f}</td></tr>')
        errors = summary['batched_vs_alternating_latent_error']
        errors_text = ', '.join(f'request {i}: {r["relative_rms"]*100:.3f}% RMS' for i, r in enumerate(errors))
        comparisons = ''.join(
            f'<li><a href="{folder.name}/{p.name}">{html.escape(p.stem)}</a></li>'
            for p in sorted(folder.glob('comparison-with-*.json')))
        videos = ''.join(f'<figure><video controls preload="metadata" src="{folder.name}/{mode}-{i}.mp4"></video>'
                         f'<figcaption>{mode}, request {i}</figcaption></figure>'
                         for mode in ('alternating', 'batched') for i in range(2))
        media.append(f'<section><h2>{html.escape(quantization)}</h2><p>Batching latent difference: {errors_text}. '
                     'This is a numerical comparison, not a perceptual quality score.</p><div class="videos">'
                     f'{videos}</div><ul>{comparisons}</ul></section>')
if not baseline_rows and not pair_rows:
    raise SystemExit('No completed inference measurements found; refusing to create a results page.')
(root / 'comparison.json').write_text(json.dumps({'paired': records, 'deadline_seconds': 1.28,
    'decoder_caveat': 'Full-clip decoder throughput, not a streaming decoder deadline test.'}, indent=2) + '\n')
page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AvatarForever on Modal H100</title><style>
body{font:16px system-ui;background:#0c1420;color:#e6edf5;max-width:1250px;margin:40px auto;padding:0 24px}
p{line-height:1.6;color:#c1cede}a{color:#8bccff}section{padding:22px;border:1px solid #354961;border-radius:12px;margin:24px 0}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:12px;text-align:right;border-bottom:1px solid #354961}
td:first-child,th:first-child{text-align:left}small{display:block;font-size:10px;max-width:360px;overflow-wrap:anywhere}
.scroll{overflow-x:auto}.videos{display:grid;grid-template-columns:1fr 1fr;gap:20px}figure{margin:0}video{width:100%}figcaption{padding:10px}
@media(max-width:700px){.videos{grid-template-columns:1fr}}
</style><h1>AvatarForever on Modal H100</h1><p>768×512 · 25 FPS per request · Four denoising steps · ForeverCache · dev environment</p>
<section><h2>Two requests: 1.28-second deadline</h2><p>Each regular AR chunk contains four latent frames / 32 video frames per request.
The two requests must both produce their next chunk in 1.28 seconds. Timings below are measured on synchronized requests with distinct
audio offsets and seeds. The decoder contribution is allocated from full-clip decoding; these measurements do not establish streaming
first-pixel latency or service-level deadline guarantees.</p><div class="scroll"><table><tr><th>Mode</th><th>Generation median (s)</th>
<th>Generation max (s)</th><th>Median + average decode (s)</th><th>Generation peak VRAM (GiB)</th></tr>'''
page += ''.join(pair_rows) + '</table></div></section>'
if baseline_rows:
    page += '<section><h2>Single-request control</h2><table><tr><th>Run</th><th>AR + VAE FPS</th><th>Full-clip decoder</th></tr>'
    page += ''.join(baseline_rows) + '</table></section>'
page += ''.join(media) + '<p><a href="comparison.json">Machine-readable comparison</a></p></html>'
(root / 'index.html').write_text(page)
print(root / 'index.html')
