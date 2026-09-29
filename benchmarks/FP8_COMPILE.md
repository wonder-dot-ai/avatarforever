# Compiling FP8 storage / BF16 compute on H100

Measured 2026-09-29. The pre-compilation rollback point is **`1e6a25f`**.
Compilation is opt-in; omitting the new flags preserves eager inference. This
change does not select dynamic FP8 GEMMs or change the checkpoint/conditioning.

## Matched performance results

H100 80 GB, 768 × 512, 257 frames at 25 FPS (10.28 seconds), one-stage AR,
four denoising steps, seed 42, ForeverCache on, persistent models, identical
reference/prompt and the first 10.28 seconds of the long JFK recording. Both
paths reuse the same precomputed audio latents. Each median is from three warm
requests after one complete warmup.

| Measurement | Eager FP8 storage | Final compiled FP8 storage |
| --- | ---: | ---: |
| Request to completed MP4 | 12.592 s | **9.048 s** |
| Request range | 12.585–12.600 s | 9.032–9.061 s |
| AR sampling | 9.763 s | **6.419 s** |
| Transformer forwards | 9.721 s | 6.377 s |
| Video VAE decode | 0.837 s | **0.662 s** |
| First decoded pixel chunk | 10.746 s | 7.229 s |
| Complete-request throughput | 20.41 FPS | 28.40 FPS |
| Peak PyTorch allocation | 54.110 GiB | 54.103 GiB |

Whole-request latency fell **28.1%** (1.39× throughput); AR sampling fell **34.2%**.
The memory difference is negligible. These are completed-file requests: the
entry point still samples all latents before decoding, so this is **not** a
low-latency streaming implementation. Stage timings overlap; do not add them to
the whole-request total. Prior BF16 and dynamic-FP8 results use other experiment
settings and are not a fresh three-way comparison here.

The first final-config request took **377.58 seconds (6.29 minutes)** with a new
Inductor cache directory, including model loading and compilation. Eager's first
request took 31.76 seconds. These times exclude Python imports, downloads and
environment setup. The 51 captured graphs and 16,589 captured calls were unchanged
across warmup and all three measured requests; graph-break and unimplemented
counters were empty. First use of new shapes can still trigger compilation.

Exploratory results before preserving intermediate BF16 rounding:
transformer-only regional compilation took 9.205 seconds / 56.707 GiB;
transformer plus decoder took 8.978 seconds / 53.445 GiB. The implemented
configuration trades about 0.07 seconds versus the latter for much closer
fixed-input numerics. These older results are retained in the evidence folder,
but are not the final settings.

Raw manifests, JSONL measurements and numerical checks are in
[`results/h100-2026-09-29/fp8-compile`](results/h100-2026-09-29/fp8-compile).
The local video viewer is `outputs/fp8-compile/index.html`; regenerate it with
`python benchmarks/build_compile_viewer.py` once its referenced media and JSON
files are present. Generated media is intentionally excluded from Git.

## What is compiled

`--compile-transformer regional` captures all 48 audio/video transformer blocks,
including the existing FP8-to-BF16 weight casts, attention, feed-forward networks
and normalization. It additionally captures both modality input preprocessors
(including positional/timestep preparation and reference conditioning) and output
normalization/projections. Python block/cache dispatch, the AR sampling loop and
the final velocity-to-X0 conversion remain eager.

`--compile-video-decoder` captures each video VAE tile's complete forward pass.
The tiled decoder invokes `self.forward` directly, so the implementation compiles
that method rather than only installing a compiled `Module.__call__`.

Gemma/text processing, input video/audio encoding, audio decoding, media IO and
Python tiling/stitching remain eager. With `--fast-infer`, model instances persist
across requests, which is necessary to amortize compilation and loading costs.

Every selected region uses `fullgraph=True, dynamic=False` and
`options={"emulate_precision_casts": True}` to preserve intermediate BF16 rounding.
Unrestricted fusion showed substantial numerical differences and is not the final default. A graph break raises
an error; this change does not turn on `suppress_errors` or silently fall back to
eager. Regional compilation follows the approach described in the
[PyTorch recipe](https://docs.pytorch.org/tutorials/recipes/regional_compilation.html):
compile a repeated block and reuse its graphs across layers rather than trace
all 48 layers into each large graph.

The FP8 linear policy is unchanged. Selected parameter storage is FP8 E4M3FN;
linear operations still request BF16 operands/results. Compilation captures the
casts and surrounding operations, but does **not** make this an FP8 matrix
multiplication path. The model storage audit reports 18,526,765,056 FP8 bytes,
949,295,616 BF16 bytes and 18,923,520 FP32 bytes before inference. Peak allocator
measurements include additional compiler/runtime temporaries.

## Compile obstacles and changes

- A strict whole-transformer trace with `backend=eager` completed its first
  forward in 114.9 seconds. Cache reuse then required another large trace. We
  interrupted this probe; it was not a completed Inductor benchmark. The `full`
  option remains experimental and is not the recommended setting.
- The first static block attempt exceeded Dynamo's default eight variants for
  the same code object. AR population/reuse, history lengths, audio lengths and
  strides require distinct guards. This was a specialization-limit failure,
  not an unsupported operator or graph break.
- Opting into transformer or decoder compilation now raises the process-wide per-code
  recompile limit to at least 32 (using the version-appropriate config name).
  This allowed the tested shape family to complete under strict fullgraph.
  Other resolutions, durations or cache settings can compile new variants and
  can still reach this finite limit.
- A `dynamic=True` diagnostic still specialized on cache slices and shape/stride
  equalities and had expensive traces. It was interrupted, not benchmarked to
  completion. The implemented default is static regional compilation.
- No attention mathematics, cache eviction rules, reference-conditioning
  weights or quantization policy were rewritten to make compilation pass.

## Reproduction

Use the existing CUDA environment; do not recreate it with a package sync.
The tested environment is PyTorch 2.14 + CUDA 13.0 on an H100 80 GB.

```sh
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false \
TORCHINDUCTOR_COMPILE_THREADS=8 \
TORCHINDUCTOR_CACHE_DIR=outputs/fp8-compile/compiler-cache-precision-final \
TORCH_LOGS=graph_breaks,recompiles \
.venv/bin/python benchmarks/latency.py \
  --checkpoint checkpoints/avatarforever-ltx-2.3-22b.safetensors \
  --gemma-root checkpoints/gemma-3-12b-it-qat-q4_0-unquantized \
  --audio data/jfk-american-university.ogg \
  --audio-latents outputs/fp8-long-comparison/audio-latents.pt \
  --reference outputs/stage-comparison/reference.png \
  --quantization fp8-cast \
  --compile-transformer regional --compile-video-decoder \
  --frames 257 --runs 3 --warmup-runs 1 --save-latents \
  --cache on --fast-infer \
  --output-dir outputs/fp8-compile/regional-precision
```

For eager comparison omit both compile flags. For transformer-only compilation,
omit `--compile-video-decoder`. For ordinary `inference.py` use the same two
compile flags alongside `--quantization fp8-cast --fast-infer` and the existing
inference arguments. Warm the exact resolution/cache/shape family before serving
requests. A short warmup may not exercise final chunks, saturated history or all
VAE tile sizes. Each benchmark output directory should be new because the raw
JSONL file appends records.

The benchmark records source hashes, arguments, versions, cumulative Dynamo
counters, allocated/reserved peaks and per-stage synchronized timings. Source
hashes are authoritative for this experiment. After the timed run the helper
was reformatted and its variant-budget setup was shared with decoder-only
compilation; the measured combined configuration still uses the same budget
and compile options: the remote checkout's Git HEAD
predates locally synced changes. Saved latents are written outside the timed
interval. No concurrent GPU workload was used during performance measurements.

## Quality and scope limitations

Same-seed outputs are not bit-identical. Small numerical changes can accumulate
through denoising and autoregressive feedback. Pixel MAE/PSNR measure agreement,
not perceptual quality or lip synchronization. Fixed-input transformer and VAE
checks separately test numerical differences without allowing prior output
trajectories to diverge.

This is a short latency/functional experiment, not another 21-minute stability
study. The checkpoint's inactive extra first-frame conditioning and the earlier
long-video drift investigation remain unchanged; see [DRIFT_AUDIT.md](DRIFT_AUDIT.md).

## Fixed-input numerical checks

`check_compiled_transformer.py` clones each selected forward's complete argument
tree together, preserving shared cache aliases, then evaluates eager and compiled
forwards independently. It covers denoising calls 0, 1, 4, 5, 8, 9, 32 and 33:
initial history, cache population/reuse, longer history and the final chunk.
These are velocity-model outputs, not a perceptual quality score. Diagnostic
timings are excluded from performance results.

| Compilation arithmetic | Video relative RMS error, eight calls |
| --- | ---: |
| Default fusion (exploratory) | 8.28–18.33% |
| Preserve intermediate BF16 rounding (implemented) | 0.62–1.63% |

The relevant local PyTorch source is `torch/_inductor/config.py`,
`emulate_precision_casts`. Its comments explain that fusion normally removes
intermediate downcast/upcast pairs; this option preserves those truncations.
The reduction in fixed-input error after changing only that setting supports
rounding as a major contributor. It does not prove every remaining difference
has the same cause or guarantee bitwise equivalence. The initial comparison used
the equivalent `TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1` environment setting; the
final implementation passes the option explicitly to each compiled region.

Three warm requests produced identical final latent tensors within each tested
process (eager, both initial compiled benchmarks and the final preserved-rounding
configuration). Outputs from independently
compiled exploratory processes differed; the report does not attribute that
difference to decoder compilation alone. The final viewer compares eager against
the implemented rounding-preserving configuration.

The fixed-input decoder probe uses one real latent tile and resets the CUDA
generator before the compiled call. With preserved rounding, relative RMS error
was **0.0397%**, maximum absolute difference **0.0078125**, all values were finite,
and generator states after eager and compiled calls were identical. This is one
tile check; it does not establish pixel equality for complete generated videos.

Complete-video comparison: both outputs decode to 257 frames at 768 × 512,
25 FPS, with identical decoded audio. Eager versus final compiled output has
RGB MAE **13.08/255** and PSNR **18.22 dB**. Inspection of frames 32, 96, 160 and
224 found coherent faces and backgrounds without obvious numerical corruption,
but visible differences in pose/expression. This is not a formal quality or
lip-sync evaluation and does not establish long-duration equivalence.

## Additional validation

A **513-frame / 20.52-second** functional run completed with ForeverCache off,
then on in the same process, using regional transformer plus video-decoder
compilation. Both results had finite latents, exactly 513 decoded frames and
empty graph-break/unimplemented counters. This checks additional AR and VAE tail
shapes and switching cache mode. These requests included new compilations and
are not warm latency measurements. This is still not a 20-minute stability test.

Validation also included Python syntax checks, Ruff's full rules for the new
compiler helper, fatal/undefined-name checks on edited runtime files, unused-name
checks on the new diagnostic scripts, and `git diff --check`. The generated
viewer's JavaScript parses and all local media/report links resolve; sampled
output frames were inspected. No model checkpoint or default eager setting was
changed. To disable the optimization, omit both compile flags. To revert the
entire change, revert the dedicated compilation commit; the preceding checkpoint
is `1e6a25f`.
