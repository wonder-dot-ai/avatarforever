# FP8 weight-storage experiment

Branch: `codex/fp8-inference-comparison`. Tested on the supplied H100 on
2026-09-28. The original checkpoint is unchanged.

The experiment uses the existing `QuantizationPolicy.fp8_cast()` policy. Selected
attention and feed-forward weights and biases are stored as FP8 E4M3 and cast to
the input dtype for each linear operation. This reduces persistent weight memory;
it does not use native FP8 matrix multiplication or INT8. Activations and
ForeverCache are not quantized. Gemma, prompt connectors and VAEs are unchanged.

## Matched results

One-stage AR, 768 × 512, 257 frames, 25 FPS, four denoising steps, seed 42,
default prompt, JFK speech, the shared reference image from the earlier comparison,
ForeverCache enabled, AR chunk size 4 and one history chunk plus sink.
Fast inference retains all models on the GPU. Each mode ran in a separate
process with one warm-up followed by three measured requests. Times are medians.

| Measurement | Bfloat16 baseline | FP8 cast |
| --- | ---: | ---: |
| Transformer weight storage, decimal GB | 38.012 | 19.495 |
| Peak PyTorch live allocation, GiB | 71.364 | 54.110 |
| Complete request including MP4 encoding, seconds | 10.710 | 12.605 |
| AR sampling, seconds | 7.898 | 9.802 |

FP8 reduces transformer storage by approximately 48.7% and overall live peak by
24.2%. Complete requests are approximately 17.7% slower in this implementation.
Per-operation upcasts are part of this path; these results should not be used
to predict performance of a native FP8 kernel implementation. The instrumentation
synchronizes GPU timing boundaries, identically for both modes.

Actual quantized transformer storage by dtype:

| Dtype | Bytes |
| --- | ---: |
| FP8 E4M3 | 18,526,765,056 |
| Bfloat16 | 949,295,616 |
| Float32 | 18,923,520 |
| **Total** | **19,494,984,192** |

The policy preserves some parameters in checkpoint precision. The baseline
transformer has 38,012,287,488 bytes of bfloat16 parameters. Unique storage
addresses are counted to avoid double-counting tied tensors.

## Output checks

All eight benchmark requests completed and their final video latents were finite.
The selected BF16 and FP8 clips each decode to 257 H.264 frames at 768 × 512,
25 FPS, with matching presentation times and identical decoded audio.
Both use the same reference image, input audio, prompt and seed.

Pixel comparison gives RGB MAE 9.420 on a 0–255 scale and PSNR 20.375 dB.
These measure output difference, not perceptual quality or lip-sync accuracy.
Small numerical differences can lead diffusion sampling along a different valid
motion trajectory, so pixel identity is not expected.

Visual inspection of the four sampled frame pairs found preserved subject
identity, clothing, scene structure and facial detail, with no obvious gross
corruption. Head pose, blinking and mouth expression diverge, especially later
in the clip. These sampled stills do not establish temporal smoothness or lip-sync
equivalence; the synchronized videos are provided for that comparison.

Local review files are in `outputs/fp8-comparison/`: `index.html` offers synchronized
playback and frame stepping; `bf16.mp4` and `fp8.mp4` are the selected third measured
requests. `sampled-comparison.png` shows frames 32, 96, 160 and 224, and full-size
PNGs are provided for each. This is one 10.28-second input, not a broad quality
or lip-sync benchmark.

## Reproduce

On `/home/ubuntu/work/avatarforever`, use the existing environment without
resynchronizing its tested package versions:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false \
.venv/bin/python benchmarks/latency.py \
  --checkpoint checkpoints/avatarforever-ltx-2.3-22b.safetensors \
  --gemma-root checkpoints/gemma-3-12b-it-qat-q4_0-unquantized \
  --audio data/jfk.flac \
  --reference outputs/stage-comparison/reference.png \
  --quantization fp8-cast \
  --frames 257 --runs 3 --warmup-runs 1 \
  --cache on --fast-infer \
  --output-dir outputs/fp8-comparison-repeat/fp8-cast
```

Run again in a fresh process with `--quantization none` and a different output
directory for the baseline. Use a fresh directory each time because request
records append. Reference-image channel encoding happens once before timed
requests in this harness; ordinary image conditioning inside the pipeline call
is included. See the root README for the simpler `inference.py` command.

Raw manifests and per-request results are in
[`results/h100-2026-09-28/fp8`](results/h100-2026-09-28/fp8).
