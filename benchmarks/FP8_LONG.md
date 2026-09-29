# Quantizer optimization and continuous long-video comparison

The prior work is preserved in commit `a5504bc` on branch
`codex/fp8-inference-comparison`. All changes in this experiment are subsequent
changes. Generated videos remain in ignored `outputs/` directories.

## What is compiled or captured

The BF16 and FP8-storage baselines do not compile or capture the transformer.
The previous dynamic FP8 implementation compiled scale/clamp/cast arithmetic,
with the maximum reduction outside that compiled function. It did not compile
the whole model.

This experiment adds three parallel Triton kernels: fused absolute-value/partial
maximum, final reduction/scale, and scale/clamp/FP8 conversion. CUDA graph capture
is limited to those quantization operations. Graph measurements include copying
the current activation into static input storage on every call. Weights, GEMM,
attention and transformer forward remain outside capture.

The H100 quantizer-only benchmark uses five groups of 100 calls following warm-up.
Values are median synchronized wall-clock microseconds per quantization, including
Python/dispatch overhead. The input tensors change during correctness checks to
verify that replay updates both activations and scales.

| Activation shape | Previous partial compile | Previous quantizer captured | Fused Triton | Fused Triton captured |
| --- | ---: | ---: | ---: | ---: |
| 96 × 2048 | 87.93 | 15.15 | 51.77 | 12.87 |
| 1536 × 4096 | 88.05 | 49.11 | 52.05 | 20.47 |
| 1536 × 16384 | 130.14 | 166.54 | 53.21 | 82.73 |
| 4608 × 4096 | 99.97 | 127.00 | 52.90 | 60.38 |

The optional `--fp8-activation-backend auto` selects quantizer-only graphs for
inputs with at most 8 million elements, direct fused kernels above that size.
This is a heuristic derived from the supplied H100, not a universal crossover.
Graphs are shared by sequential linear layers using the same shape and stream,
so we do not allocate a separate set of buffers for every transformer layer.
Returned graph outputs must be consumed on that CUDA stream before reuse.

The existing default `compiled` backend is preserved. `triton` and `cudagraph`
force the respective experimental alternatives. These options apply only to
`--quantization fp8-dynamic`.

## Matched short-video results

Same speech, precomputed audio conditioning, portrait, prompt, seed 42, 257 frames,
768×512/25 FPS, four steps, one-stage AR and ForeverCache. Each mode ran in its own
process with one warm-up followed by three measured requests. All models remained
resident (`--fast-infer`). Medians:

| Measurement | BF16 | FP8 storage / BF16 compute | FP8 matrix multiplication, auto quantizer |
| --- | ---: | ---: | ---: |
| Complete request, seconds | 10.648 | 12.675 | 14.187 |
| AR sampling, seconds | 7.822 | 9.778 | 11.358 |
| Peak live VRAM, GiB | 71.364 | 54.110 | 54.230 |

Despite faster quantization, native FP8 remains slower for this workload. These
short-run measurements include dynamic scaling and graph input copies; one-time
compilation and graph construction occur in warm-up.

## Continuous long-video protocol

Use the first 1260.2 seconds of the 26:47 recording of
[JFK's American University address, June 10, 1963](https://commons.wikimedia.org/wiki/File:Jfk_American_University_4654_06-10-63.ogg).
The source identifies it as a public-domain US federal government recording.
All three modes receive the same recording; it is not a repeated short clip.

Each output has 31,505 frames at 25 FPS: **21 minutes and 0.20 seconds**. Generate
one continuous autoregressive sequence per mode with no resets or independently
generated segments. Use the same portrait, prompt, seed, resolution, four-step
schedule, history chunk count, sink and ForeverCache configuration as the short
comparison. Model unloading between stages is enabled in all long modes to make
room for the long latent sequence. Long-run total times include model loading;
long-run peaks should not be compared directly with the resident-model short test.

The audio VAE uses quadratic attention. `prepare_long_audio.py` encodes the speech
in 30-second windows, with two seconds of neighboring context on either side,
then trims overlap at the 25 Hz latent grid. This changes the audio encoder's
context relative to an impractical full-length encode. The resulting audio
latents are computed once and loaded unchanged for all modes. Video AR history
is still uninterrupted across the entire 21 minutes. The original waveform is
used for the final soundtrack.

Every long mode has a 257-frame warm-up and one full-length measured request.
Record full source hashes, audio and latent hashes, per-transformer-call timings,
peak VRAM and latent finiteness. The analysis checks every decoded frame count
and audio hash, samples images throughout the duration, and summarizes brightness,
spatial gradients and one-second frame changes per minute. These statistics are
not perceptual quality, identity or lip-sync scores. Visual inspection is needed.

## Continuous 21-minute measurements

| Measurement | BF16 | FP8 storage / BF16 compute | FP8 matrix multiplication, auto quantizer |
| --- | ---: | ---: | ---: |
| Complete request, seconds | 1256.831 | 1490.416 | 1551.209 |
| Complete request, mm:ss | 20:57 | 24:50 | 25:51 |
| AR sampling, seconds | 923.880 | 1156.809 | 1212.236 |
| Transformer forward, seconds | 918.397 | 1151.345 | 1206.771 |
| VAE decode, seconds | 99.644 | 98.659 | 99.363 |
| Peak live VRAM, GiB | 42.312 | 25.085 | 25.157 |

Native FP8 takes 23.4% longer than BF16 and 4.1% longer than FP8 storage for the
complete request. Its AR sampling takes 31.2% longer than BF16. Both FP8 paths
reduce peak live memory by about 17 GiB in this staged-loading configuration.
These are single long-run measurements, not confidence intervals. Subsequent
frame/audio analysis and transfer of each completed file are excluded from its
request timing.

This harness completes AR sampling before decoding pixels. Average throughput
must not be interpreted as streaming latency: the first decoded pixels arrived
after 944.2, 1177.2 and 1234.4 seconds respectively. Audio conditioning was
precomputed identically for all modes and is excluded from request times.

## Visual review and validation

**Subsequent checkpoint audit:** the released checkpoint's additional first-frame
projection and gate tensors are all zero, making that conditioning channel
inactive. Initial-image and first-chunk sink conditioning remain active. Intent
and the channel's contribution to drift are unresolved; upstream issue #5 already
tracks the discrepancy. See [the conditioning report](DRIFT_AUDIT.md). These runs remain matched precision
comparisons, but should not be treated as a verified reproduction of the paper's
long-horizon stability result.

All three MP4s decoded to exactly 31,505 frames at 25 FPS; decoded audio hashes
match. All warm-up and measured requests have finite final latents. Runtime
source hashes, original audio, precomputed audio latents, reference and sampling
settings match across modes except for the intended precision policy.

Reviewed 12 matching timestamps per mode: 0:10, 2:00, 4:00, 6:00, 8:00, 10:00,
12:00, 14:00, 16:00, 18:00, 20:00 and 20:55. The full outputs and contact sheets
are in [the local comparison viewer](../outputs/fp8-long-comparison/index.html).

- **BF16 already drifts.** The initially light, plain jacket gains strong checks,
  changes material/brightness, and later changes again. Background geometry and
  texture also develop artifacts, particularly visible around 6:00 and 16:00.
- **FP8 storage shows similar drift.** Checked jackets, darkened clothing and
  occasional background artifacts appear in the middle of the sequence. The
  final sampled jacket is closer to its initial light appearance.
- **Native FP8 remains coherent in the inspected late samples.** Clothing and
  background changes still occur, including strongly checked clothing around
  8–10 minutes. The 20:00 and 20:55 samples show a coherent face and light jacket;
  there is no obvious catastrophic late collapse unique to this mode.

These observations do not establish that FP8 has equal perceptual quality or
that drift increases monotonically with elapsed time. This is one portrait,
prompt, recording and seed. Small precision changes produce different AR
trajectories, so pixel differences are not themselves a quality score. The
inspection used sampled stills, not a complete manual review of every frame or a
lip-sync evaluation. Use synchronized video playback for temporal inspection.
The per-minute brightness, gradient and frame-change statistics are diagnostics,
not validated identity or perceptual quality metrics.

Raw short/long manifests, request timings and quality checks are archived under
`benchmarks/results/h100-2026-09-29/fp8-long/`. The remote checkout's recorded Git
HEAD predates the local rollback snapshot; the per-file SHA-256 records identify
the actual runtime files used and match across all three long runs.

## Reproduce on the supplied H100

```bash
cd /home/ubuntu/work/avatarforever
mkdir -p data
curl -L --fail \
  https://upload.wikimedia.org/wikipedia/commons/b/b5/Jfk_American_University_4654_06-10-63.ogg \
  -o data/jfk-american-university.ogg
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 .venv/bin/python benchmarks/quantizer_backends.py
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 .venv/bin/python benchmarks/prepare_long_audio.py \
  --checkpoint checkpoints/avatarforever-ltx-2.3-22b.safetensors \
  --audio data/jfk-american-university.ogg \
  --output outputs/fp8-long-comparison/audio-latents.pt
bash benchmarks/run_long_comparison.sh
.venv/bin/python benchmarks/build_long_viewer.py --root outputs/fp8-long-comparison
```

Use fresh output directories for repeated benchmarks; request records append.
The long-run script writes `driver-status.txt` and separate per-mode logs. The
viewer provides synchronized full-video playback, a shared frame slider, time
jumps, audio selection and samples from early through late generation.
