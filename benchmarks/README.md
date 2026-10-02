# H100 latency benchmark

## Modal results — 2026-10-02

Original BF16 fits on an H100 80GB after offloading preparation models, but the
completed configurations do **not** sustain two 25 FPS streams at 768×512.
Each regular chunk produces 32 frames per request, giving both requests a
shared 1.28-second deadline. Three warmed runs per mode, each 257 frames:

| Weights / compute | Scheduling | Pair generation median / max | Pair + amortized VAE | Estimated FPS / request | Generation peak |
| --- | --- | ---: | ---: | ---: | ---: |
| Original BF16 / BF16 | Alternating | 1.459 / 1.465 s | 1.625 s | 19.69 | 40.51 GiB |
| Original BF16 / BF16 | Batch 2 | 1.388 / 1.399 s | 1.554 s | **20.60** | 44.66 GiB |
| FP8 storage / BF16 | Alternating | 1.621 / 1.625 s | 1.785 s | 17.92 | 23.33 GiB |
| FP8 storage / BF16 | Batch 2 | 1.459 / 1.468 s | 1.624 s | 19.70 | 27.43 GiB |

Keeping original BF16 weights improves batched throughput by about 4.5% over
FP8 storage. Generation alone still exceeds the deadline. VAE figures amortize
full-clip decoding; preparation, encoding, delivery and streaming decoder latency
are not included. These short clips do not establish long-session stability.

**Quality equivalence is not established.** BF16 batching changes motion and
the second request partly exits the frame near 5.12 seconds. Batched versus
alternating latent relative RMS differences are 54.2% and 82.7%. The alternating
FP8 harness also differs from the original single-request control (43.6% relative
RMS). The conditioned first latent matches, with differences growing later;
the cause has not been isolated. These are numerical differences, not perceptual
quality scores. Four CPU regression tests pass for timestep broadcasting,
strict graph capture, gated attention and cached-history independence across
both RoPE layouts. Both completed GPU comparisons report zero graph breaks and
finite latents; all eight videos were verified as 257 frames, 25 FPS, 768×512,
with an audio stream.

The additional native FP8 matrix multiplication trial was canceled during
decoder warmup after the local session interruption. It has no measured result
and is excluded from the table. All GPU apps are stopped. Conservative total
accounting includes failed and interrupted trials: **2.151 H100-hours**, below
the authorized five-hour cap; this is an app-lifetime upper bound, not billing.

Videos and an HTML viewer are local in `outputs/modal/index.html`. Committed
evidence is in [the result directory](results/modal-h100-2026-10-02/), including
[status and limitations](results/modal-h100-2026-10-02/experiment-status.json),
per-run raw timings, environment manifests, latent comparisons and budget evidence.

## Modal experiments

`modal_benchmark.py` runs on-demand jobs; it does not deploy a serving endpoint.
The experiment uses the `explore` workspace (`ac-8M7zh69xwqIgmGTbCOOKl7`),
`dev` environment, `huggingface-secret` with `HF_TOKEN`, and the `avatarforever-benchmarks` Volume.
The token must have access to the gated Gemma checkpoint. Never commit tokens.

```bash
uv venv --python 3.11 .venv-modal
uv pip install --python .venv-modal/bin/python modal==1.6.0
.venv-modal/bin/modal token new
.venv-modal/bin/modal run --env dev benchmarks/modal_benchmark.py --action check-access
.venv-modal/bin/modal run --env dev benchmarks/modal_benchmark.py --action check-runtime
.venv-modal/bin/modal run --env dev benchmarks/modal_benchmark.py --action prepare
.venv-modal/bin/modal run --detach --env dev benchmarks/modal_benchmark.py --action baseline --quantization fp8-cast
.venv-modal/bin/modal run --detach --env dev benchmarks/modal_benchmark.py --action paired --quantization none
```

Access checks, import checks, and weight downloads allocate no GPU. Inference
uses one H100 with no retries, a 30-minute execution timeout, and scale-down
after two idle seconds. Each GPU call conservatively reserves startup plus
execution timeout in `outputs/modal/h100-budget.json`; the entrypoint refuses
reservations above the authorized five GPU-hours. This ledger does not account
for unrelated team jobs or invocations that bypass this entrypoint. Do not
reset it merely to bypass the experiment budget.
Stopped reservations can be reconciled against recorded Modal app lifecycle
timestamps, using the smaller of the original hard cap and app lifetime plus a
60-second margin. This includes failed runs and all startup time; it is a
conservative GPU-time bound, not a billing-meter reading.

Results, logs, manifests, latents, and videos persist under `runs/<run-id>` in
the Volume and are downloaded to `outputs/modal/<run-id>` after a successful
run. Failed-run artifacts remain in the Volume. Model download revisions and
the container's installed package versions are recorded. Container builds are
separate from inference timing.

The paired experiment prepares two requests with seeds 42/43 and audio offsets
0/15 seconds, using the same portrait and prompt. It compares alternating
single-request chunks with a batch of two independent states, keeping one copy
of the transformer and decoder resident and offloading the preparation models.
It warms both modes, alternates measurement order, compares final latents, and
decodes/saves both videos. It currently tests equal-length, synchronized
requests and full-clip decoding; it is not a general concurrent serving API or
a validation of streaming-decoder latency. The `--quantization` option also
accepts `none` and `fp8-dynamic` for subsequent controlled experiments.
For the paired benchmark, `--quantization fp8-preexpanded` loads the same
FP8-rounded weights as `fp8-cast`, then expands them to BF16 once after moving
the preparation models to CPU. This preserves those weight values while
trading VRAM for elimination of per-forward weight conversion. The one-time
conversion duration and number of affected linears are recorded separately.

Batching requires a scalar diffusion sigma to be expanded to the `(B,)` contract
before prompt/cross-modal timestep preparation. `check-runtime` includes a CPU
regression using a small real audio/video transformer, comparing batched results
with independent samples and checking strict graph capture. The paired benchmark
also runs a full-size eager batch smoke check before warming compiled paths.
Its 96-variant allowance accommodates both batch sizes and AR/audio/cache shapes;
`fullgraph=True` stays enabled, with no eager fallback. The ordinary single-request
compiler setting remains unchanged.

After runs complete, use `--action compare --reference-run <id> --candidate-run <id>`
for CPU-only latent comparisons, then `python benchmarks/build_modal_report.py`
to build the local `outputs/modal/index.html` video and timing viewer.

## Earlier dedicated H100 measurements

Avatar-Forever was cloned from `https://github.com/wonder-dot-ai/avatarforever.git`
and run on the supplied H100 on 2026-09-28. The remote checkout and its environment
are at `/home/ubuntu/work/avatarforever`. The checkpoint and Gemma weights are in
that checkout's `checkpoints/` directory.

For the released checkpoint's inactive first-frame channel, remaining conditioning
paths, author discussion and unresolved drift questions, see the
[conditioning and drift report](DRIFT_AUDIT.md) (researched 2026-09-29).

For the opt-in FP8-storage/BF16-compute compiler optimization, startup cost,
numerical checks and matched measurements, see [FP8_COMPILE.md](FP8_COMPILE.md).

## Measured results

One H100 80GB HBM3, 768 × 512, 257 frames at 25 FPS (10.28 seconds of video).
One-stage inference, four Euler steps, AR chunk size 4, one history chunk,
first-chunk sink and relative positions enabled. Default prompt, seed 42,
generated first-frame channel conditioning, no reference image.

These are medians of **three measured requests per mode**, following one warm-up
request per mode. `--fast-infer` retains models on the GPU between requests.

| Measurement | ForeverCache off | ForeverCache on |
| --- | ---: | ---: |
| Request to completed MP4 | 14.287 s | **10.742 s** |
| Request-to-MP4 range across 3 runs | 14.275–14.292 s | 10.738–10.783 s |
| Complete-request throughput | 17.99 FPS | **23.93 FPS** |
| Request to first decoded pixel chunk | 12.398 s | **8.833 s** |
| AR sampling, including transformer calls | 11.424 s | **7.856 s** |
| VAE decoding | 0.845 s | 0.845 s |
| AR sampling + VAE throughput | 20.95 FPS | **29.53 FPS** |
| Encoding / transfer / mux and other decode-iterator overhead | 1.888 s | 1.917 s |
| Peak PyTorch GPU allocation | 71.36 GiB | 71.36 GiB |

Cache reduced complete-request latency by **24.8%** and AR sampling time by
**31.2%**. These are measurements for this input, duration, hardware and software
configuration. Cache reuse is approximate and was not evaluated for visual
quality equivalence.

The compute portion with cache is faster than 25 FPS playback, but the full MP4
request remains slightly slower than the 10.28-second clip. The entry point
finishes sampling all requested latents before VAE decoding. Therefore the
8.83-second first-pixel latency is not a low-latency streaming result.

For interior full-size chunks with sink + history + current present, the median
sum of four transformer forward calls was 1.443 s without cache and 0.932 s with
cache. Those chunks represent 32 output frames (1.28 seconds at 25 FPS). This
figure excludes VAE decoding, sampler bookkeeping and encoding. It is not the
latency to deliver a playable chunk.

The first 257-frame request in the resident-model process, with cache off, took
**33.483 s** including loading and initialization. It was excluded from the warm
medians. Model factory timings totaled approximately 16.88 s. An earlier
65-frame smoke test with `fast_infer=False` completed in 27.719 s, using 39.81 GiB
peak PyTorch allocation. These are first-request timings with weights already
downloaded; they do not include installation, downloads, Python imports, or
fresh-machine boot time.

## Environment and inputs

- GPU: NVIDIA H100 80GB HBM3, driver 580.126.20, 700 W power limit.
- Python 3.11.16; PyTorch 2.14.0 + CUDA 13.0; Transformers 4.55.4.
- Repository attention mode: `default`, using PyTorch SDPA; no external
  xFormers or FlashAttention 3 package installed. PyTorch's flash, efficient and
  math SDPA backends were enabled; their per-operation dispatch was not profiled.
- bfloat16 inference; no quantization or `torch.compile` was added.
- Eight CPU threads (`OMP_NUM_THREADS=8`, `MKL_NUM_THREADS=8`).
- Base commit: `4dfc42b0e2dbbded4d148387d186219bd7601279`.
- Main checkpoint: `LetsThink/AvatarForever`,
  `avatarforever-ltx-2.3-22b.safetensors`, 46,183,958,894 bytes.
- Text encoder: `google/gemma-3-12b-it-qat-q4_0-unquantized`.
- Speech fixture: [OpenAI Whisper's JFK test audio](https://github.com/openai/whisper/blob/main/tests/jfk.flac),
  11 seconds, cropped to the requested video duration by the pipeline.
- Fixture SHA-256:
  `63a4b1e4c1dc655ac70961ffbf518acd249df237e5a0152faae9a4a836949715`.
- Spatial VAE tiles: 512 px, overlap 64 px. Temporal tiles: 256 frames,
  overlap 8 frames. Output: H.264 CRF 12, preset `fast`, source audio as AAC.

The full environment is recorded in
[`results/h100-2026-09-28/environment.txt`](results/h100-2026-09-28/environment.txt).

## Reproduce on the configured H100

```bash
cd /home/ubuntu/work/avatarforever
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false

.venv/bin/python benchmarks/latency.py \
  --checkpoint checkpoints/avatarforever-ltx-2.3-22b.safetensors \
  --gemma-root checkpoints/gemma-3-12b-it-qat-q4_0-unquantized \
  --audio data/jfk.flac \
  --frames 257 --runs 3 --warmup-runs 1 \
  --cache both --fast-infer \
  --output-dir outputs/benchmark-repeat-257
```

Use a new output directory for each experiment: `results.jsonl` appends each
completed request and any failure record. `summary.json` describes the current
invocation. Omit `--fast-infer` to measure the default module-loading behavior.
Use `--cache off` or `--cache on` to measure only one mode.

To set up a fresh clone, install `uv`, then run
`uv sync --all-packages --python 3.11`, followed by
`uv pip install transformers==4.55.4` to reproduce the measured version.
The workspace resolver selects 4.53.1 because of the optional TensorRT
dependency; its Gemma layout was checked, but the timings above use 4.55.4.
Use `.venv/bin/python` as shown to avoid `uv run` resynchronizing the environment
to the workspace lock. Authenticate to Hugging Face and download
the two models described in the root README. The existing server is already set
up and authenticated.

## Changes needed to run

1. `inference.py` now supplies the required `loras=[]` constructor argument.
2. `ltx-core` bounds Transformers to 4.52–4.55.4; this benchmark used 4.55.4.
   The range also permits the optional TensorRT extra's 4.53.1 requirement.
   The original unbounded dependency selected
   5.17.0; its changed Gemma/SigLIP layout failed with
   `AttributeError: 'SiglipVisionModel' object has no attribute 'vision_model'`.
   The upper bound retains the API used by this repository's loader.
3. `benchmarks/latency.py` instruments existing model factories, transformer
   calls, AR sampling and lazy VAE iteration. It does not replace model inference.

The dependency pin was first applied to the environment for measurement, then
recorded in package metadata. The raw manifest captures the working tree at
measurement time.

## Timing definitions and verification

All GPU timing boundaries call `torch.cuda.synchronize()`. The measured request
starts immediately before the pipeline call and ends after MP4 encoding and
muxing complete. Prompt/audio processing is repeated per request. Model loading
is included when it occurs; resident warm requests reuse previously loaded
modules. Extra synchronization can impose a small instrumentation overhead.

The decoder is lazy. Its timing covers actual `next()` calls on the video
iterator, including conversion to uint8. First-pixel latency records delivery
of the first decoded tensor to the encoder, not browser playback. The encoding
remainder includes CPU transfer, H.264/AAC work, muxing and residual iterator
overhead. Nested measurements overlap: transformer time is inside AR time, and
model-loading time can be inside prompt-processing time. Do not sum overlapping
fields. Component medians need not sum exactly to the median total.

Both final cache-off and cache-on MP4s were decoded and verified to contain
257 frames, 768 × 512 resolution, 10.28-second duration, H.264 video and AAC audio.
Representative generated frames were visually inspected. This is a latency
benchmark, not a lip-sync accuracy or long-horizon quality evaluation.

Raw evidence:

- [Warm request records](results/h100-2026-09-28/warm-257/results.jsonl)
- [Warm summary](results/h100-2026-09-28/warm-257/summary.json)
- [Environment and configuration manifest](results/h100-2026-09-28/warm-257/manifest.json)
- [Smoke records, including the initial compatibility failure](results/h100-2026-09-28/smoke-65/results.jsonl)

Server outputs are in `outputs/benchmark-warm-257/`, and logs are in
`benchmarks/logs/`. A cache-on sample has also been copied to the local
`outputs/h100-benchmark/cache-on-measured-3.mp4`. Generated media and model
weights remain ignored by Git.
