"""Build the A100 MM study report from measured data; no inferred FPS claims."""
import html
import json
from pathlib import Path


ROOT=Path('outputs/a100-mm-study')
workload=json.loads((ROOT/'capture/workload.json').read_text())
initial=json.loads((ROOT/'remeasured.json').read_text())
corrected=json.loads((ROOT/'nonzero/remeasured.json').read_text())
correction_manifest=json.loads((ROOT/'nonzero/manifest.json').read_text())
rows=initial['results'].copy()
replacement=corrected['results'][0]
rows=[replacement if r['key']==replacement['key'] else r for r in rows]
assert len(rows)==8
assert all(c['accuracy']['finite'] for r in rows for c in r['candidates'].values())
names=list(rows[0]['candidates'])
totals={name:sum(r['candidates'][name]['median_ms']*r['calls_per_chunk'] for r in rows) for name in names}
base=totals['compiled_fp8_cast_linear']
profile=json.loads(Path('benchmarks/results/a100-profile-2026-09-30/comparison.json').read_text())
compile_record=json.loads(Path('benchmarks/results/a100-compile-2026-09-30/comparison.json').read_text())
generation=profile['compiled']['control_median_ms']
decode=compile_record['modes']['regional-offloaded']['median']['vae_decode_seconds']/257*32*1000
coverage=sum(r['flops_per_chunk'] for r in rows)/workload['total_linear_flops']
sources=[
 ('NVIDIA A100 specifications','https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf'),
 ('cuBLASLt algorithm selection','https://docs.nvidia.com/cuda/cublas/index.html'),
 ('CUTLASS architecture / datatype support','https://docs.nvidia.com/cutlass/latest/media/docs/cpp/functionality.html'),
 ('Marlin original kernel and intended workload','https://github.com/IST-DASLab/marlin'),
 ('vLLM FP8 Marlin on Ampere','https://github.com/vllm-project/vllm/blob/main/docs/features/quantization/llm_compressor/fp8.md'),
 ('GemLite kernels and supported precisions','https://github.com/dropbox/gemlite'),
 ('TorchAO quantization workflows','https://github.com/pytorch/ao/blob/main/docs/source/workflows/inference.md'),
 ('Transformer Engine precision support','https://docs.nvidia.com/deeplearning/transformer-engine/index.html'),
 ('DeepGEMM hardware requirements','https://github.com/deepseek-ai/DeepGEMM'),
]
summary={
 'date':'2026-09-30','benchmark_code_commit':'aa3bf36; subsequent capture correction replaces a zero example with a nonzero projection',
 'scope':'Single A100-SXM4-40GB, AvatarForever, 768x512, 25 FPS, four latent frames / 32 output frames per steady chunk, four denoising steps, ForeverCache, compiled FP8 storage / BF16 compute.',
 'decision':'Stop pursuing 25 FPS at the current settings through MM-kernel replacement alone. Some kernels help individual layers, but none tested closes the gap. This is not a proof that every model/precision/algorithm/resolution change must fail.',
 'target_chunk_ms':1280,'generation_control_ms':generation,'decoder_average_per_chunk_ms':decode,
 'decoder_note':'Allocated from the measured full-clip average, not a separately measured streaming-chunk decode.',
 'total_linear_flops':workload['total_linear_flops'],'dense_bf16_peak_tflops':312,
 'ideal_dense_bf16_linear_ms':workload['total_linear_flops']/312e12*1000,
 'dense_int8_peak_tops':624,'ideal_dense_int8_linear_ms':workload['total_linear_flops']/624e12*1000,
 'tested_linear_flop_fraction':coverage,'weighted_tested_linear_ms':totals,
 'weighted_speedup_vs_compiled_cast':{k:base/v for k,v in totals.items()},
 'illustrative_chunk_ms_not_measured':{k:generation+decode-base+v for k,v in totals.items()},
 'extrapolation_caveat':'Subtract the measured-subset microbenchmark cost from the actual generation-plus-average-decode budget, then add candidate cost. Different benchmark contexts and weights mean this is an illustrative estimate, not observed full-pipeline latency. Untested shapes and all other work held unchanged.',
 'numerical_caveat':'One real nonzero tensor sample per tested shape. All candidates use a shared FP8-rounded weight reference. Seven samples were already FP8-representable; the replacement sample was rounded to FP8 for the storage-path benchmark, so this is not a quality comparison against its original unrounded weights. INT8 requantizes these weights and dynamically quantizes activations; relative RMS errors are per-layer observations, not an output video quality score. No full INT8 video or 20-minute quality test.',
 'correction_manifest':correction_manifest,
 'representative_sample_correction':'4608x4096x4096 initially captured zero conditioning weights/activations. Its final results are replaced with a nonzero real projection. Original files are preserved. Timings for the other seven shapes are from the original warmed remeasurement.',
 'measured_candidates':['PyTorch BF16','compiled FP8 conversion + BF16','cuBLASLt BF16: up to 32 heuristic algorithms','Triton BF16: eight configurations','custom Triton FP8 software conversion + BF16 GEMM: eight configurations','PyTorch / compiled PyTorch INT8 with transposed weights','cuBLASLt INT8: up to 32 algorithms','Triton INT8: eight configurations'],
 'research_only':['Standalone CUTLASS','Marlin / vLLM FP8 Marlin','GemLite','Transformer Engine / DeepGEMM'],
 'sources':[{'title':t,'url':u} for t,u in sources], 'rows':rows,
}
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
labels={
 'torch_bf16':('PyTorch BF16','Expanded weights; no conversion cost'),
 'compiled_fp8_cast_linear':('Current compiled FP8 → BF16 linear','Reference'),
 'cublaslt_bf16':('Tuned cuBLASLt BF16','Optimistic: expanded weights; conversion excluded'),
 'cublaslt_cast_plus_bf16':('Tuned cuBLASLt + eager FP8 conversion','Includes conversion; not fused'),
 'triton_bf16':('Tuned Triton BF16','Expanded weights; conversion excluded'),
 'triton_fused_fp8_bf16':('Custom fused FP8 conversion + BF16','Eight tested tile configurations; not an exhaustive search'),
 'torch_int8_full_transposed':('PyTorch INT8 full linear','Includes dynamic quantization and rescaling'),
 'compiled_int8_full_transposed':('Compiled PyTorch INT8 full linear','Includes dynamic quantization and rescaling'),
 'cublaslt_int8_full':('Tuned cuBLASLt INT8 full linear','Includes dynamic quantization and rescaling'),
 'triton_int8_full':('Tuned Triton INT8 full linear','Includes dynamic quantization and fused rescaling'),
}
def table_row(name):
 label,note=labels[name]
 err=max(r['candidates'][name]['accuracy']['relative_rms'] for r in rows)
 return f'<tr><td>{label}<small>{note}</small></td><td>{totals[name]:.1f} ms</td><td>{base/totals[name]:.2f}×</td><td>{err*100:.3f}%</td></tr>'
table=''.join(table_row(name) for name in names)
shape_table=''.join(f'<tr><td>{r["M"]} × {r["K"]} × {r["N"]}</td><td>{r["calls_per_chunk"]}</td>'+''.join(f'<td>{r["candidates"][k]["median_ms"]:.4f}</td>' for k in ('compiled_fp8_cast_linear','cublaslt_bf16','compiled_int8_full_transposed'))+'</tr>' for r in rows)
links=''.join(f'<li><a href="{u}">{html.escape(t)}</a></li>' for t,u in sources)
bars=''
for label,value,note in [('Realtime budget',1280,'Requirement'),('Current generation + decoding average',generation+decode,'Generation measured; decoder averaged'),('Tuned BF16, conversions free',summary['illustrative_chunk_ms_not_measured']['cublaslt_bf16'],'Optimistic estimate, not measured video'),('Compiled INT8 replacement',summary['illustrative_chunk_ms_not_measured']['compiled_int8_full_transposed'],'Illustrative estimate; quality unvalidated')]:
 bars+=f'<div class="barrow"><div>{label}<small>{note}</small></div><div class="bar" style="width:{value/25:.1f}%">{value/1000:.3f} s</div></div>'
page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>A100 matrix multiplication study</title><style>
:root{{font:15px system-ui;color-scheme:dark;background:#0c121c;color:#e7edf7}}body{{max-width:1200px;margin:40px auto;padding:0 24px}}h1{{font-size:32px}}h2{{font-size:21px}}p,li{{line-height:1.65;color:#c0cada}}a{{color:#86c5ff}}section{{border:1px solid #33435b;border-radius:12px;background:#141f2e;padding:24px;margin:22px 0}}.verdict{{border-left:5px solid #ffcb7c}}small{{display:block;color:#92a5be;font-size:12px;margin-top:5px}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}td,th{{text-align:right;border-bottom:1px solid #33435b;padding:12px}}td:first-child,th:first-child{{text-align:left}}.scroll{{overflow-x:auto}}.barrow{{margin:18px 0}}.bar{{margin-top:8px;background:#356f9b;border-radius:4px;padding:9px;min-width:100px;box-sizing:border-box}}code{{font-size:13px;color:#9eddd3}}
</style><p>AVATARFOREVER · A100 SXM4 40 GB · 2026-09-30</p><h1>Can a different MM kernel make this realtime?</h1>
<section class="verdict"><h2>Decision: stop the MM-only push at the current settings</h2><p>Several kernels improve individual layers, but the tested gains do not close the gap to <b>1.28 seconds per chunk</b>. This conclusion applies to 768×512, four denoising steps and the current model/cache behavior. Lower resolution, less computation, or a different quantized model would be separate experiments.</p></section>
<section><h2>The deadline versus the measured workload</h2>{bars}<p>The last two bars are <b>microbenchmark-based estimates, not full-video measurements</b>. They replace only the tested linear shapes and leave other work unchanged. Decoding is about {decode:.0f} ms per 32 frames when allocated from the full-video average; actual streaming decoding was not measured.</p>
<p>The captured chunk contains <b>{workload['total_linear_flops']/1e12:.2f} trillion linear FLOPs</b>. A100’s dense BF16 peak is 312 TFLOP/s, giving an ideal linear-only floor of <b>{summary['ideal_dense_bf16_linear_ms']:.0f} ms</b>. Adding the currently measured FlashAttention kernel time (~235 ms) and decoder average (~206 ms) already reaches ~1.43 seconds—even granting zero time to casts, normalization, other kernels, and dispatch. This is a conditional budget argument, not proof against changing attention or the model algorithm.</p></section>
<section><h2>Measured candidate comparison</h2><p>Eight real shapes cover <b>{coverage*100:.1f}% of linear FLOPs</b>. Each time below is a sum of isolated per-call medians multiplied by the captured call counts. These are weighted costs for that subset, not complete pipeline timings. “Faster” compares with the current compiled FP8-storage/BF16-compute linear.</p><div class="scroll"><table><tr><th>Implementation</th><th>Weighted subset cost</th><th>Faster</th><th>Max layer relative RMS error</th></tr>{table}</table></div><p>BF16 tuning gives modest headroom. Weight-only FP8 fusion was slower in the custom kernel tested. INT8 benefits depend strongly on layout and include quantization overhead; its measured errors require separate calibration and video-quality validation.</p></section>
<section><h2>Per-shape details</h2><p>Dimensions are M × K × N. All times are milliseconds per call. The tuned BF16 column excludes weight conversion; INT8 includes activation quantization and output scaling.</p><div class="scroll"><table><tr><th>Shape</th><th>Calls/chunk</th><th>Compiled FP8/BF16</th><th>Tuned BF16 MM</th><th>Compiled INT8 full</th></tr>{shape_table}</table></div></section>
<section><h2>Kernel families researched</h2><ul>
<li><b>cuBLASLt:</b> directly benchmarked up to 32 heuristic algorithms per shape, using 64 MiB workspace, for BF16 and INT8. Actual shape/layout selection matters; it does not create a new hardware throughput tier.</li>
<li><b>Triton:</b> directly tested eight tile configurations each for BF16, software FP8-to-BF16 fusion, and INT8. The custom fused FP8 kernel redoes conversion within GEMM tiles and was slower than the existing separate conversion plus library GEMM.</li>
<li><b>CUTLASS SM80:</b> supports A100 BF16 and integer Tensor Core GEMMs. Standalone CUTLASS was researched, not separately benchmarked. A BF16 implementation remains subject to the same dense compute ceiling.</li>
<li><b>Marlin / vLLM FP8 Marlin:</b> supports weight-only low-precision inference on Ampere. The original Marlin’s near-4× result targets roughly 16–32 token batches and memory-bound work. Our major matrices have 1,024–4,608 rows. Those headline gains do not transfer directly, and FP8 weight-only execution on A100 still computes using higher precision. Researched, not benchmarked here.</li>
<li><b>GemLite / TorchAO:</b> real A100-capable low-bit options, including integer weight/activation quantization. Their precision formats and workload conditions matter. They were researched, not installed or benchmarked; our tests exercise the same broad W8A8 versus weight-only tradeoff, not those exact implementations.</li>
<li><b>DeepGEMM / Transformer Engine native FP8:</b> not an A100-native FP8 route. Hopper/Blackwell-oriented FP8 throughput claims cannot be assigned to Ampere.</li>
<li><b>2:4 sparsity or more aggressive quantization:</b> requires a suitably compressed model and quality validation. The advertised sparse peak is not available merely by selecting a different kernel for this dense checkpoint.</li></ul></section>
<section><h2>Method and limits</h2><p>Actual tensors and call counts were captured from steady AR forwards 16–19. Search results were followed by three remeasurement rounds in shuffled order, each using 21 samples. Before each batch, 400 large BF16 GEMMs preheated the GPU. A 64 MiB zero-fill flushed L2 before each sample, outside its CUDA-event timing. Compilation and offline weight packing were excluded. Full INT8 operations include dynamic per-row activation quantization, per-output-channel weight scaling, GEMM, bias, and BF16 output conversion.</p>
<p>The first 4608×4096×4096 example was from a zero-initialized conditioning path. It was replaced with a nonzero real projection and retested; original records are retained. The replacement weights were not exactly FP8-representable, so all candidates compare against the same FP8-rounded reference for this storage-path benchmark. That reference change is not included in the table’s error numbers; on this sample alone it changes the original layer output by {correction_manifest["output_fp8_roundtrip_relative_rms"]*100:.2f}% relative RMS. The other seven samples were already exactly FP8-representable. Other shapes use their original nonzero samples. Accuracy is checked on one captured example per shape, not on generated video. Different GEMM reduction orders are not bit-identical. Microbenchmarks omit model-level interactions, and the extrapolation is not a promised FPS result. No inference/model source changes were deployed.</p>
<p><a href="summary.json">Summary and full remeasured rows</a> · <a href="microbench.json">Original algorithm/configuration search</a> · <a href="remeasured.json">Initial warmed remeasurement</a> · <a href="nonzero/remeasured.json">Nonzero-sample correction</a> · <a href="capture/workload.json">Full 40-shape census</a></p></section>
<section><h2>Primary sources</h2><ul>{links}</ul></section></html>'''
(ROOT/'index.html').write_text(page)
print(json.dumps({k:v for k,v in summary.items() if k not in ('rows','sources')},indent=2))
