# Dynamic activation FP8 on H100

Tested on 2026-09-28 on the supplied H100 80GB, on branch
`codex/fp8-inference-comparison`. Dynamic FP8 works and saves memory, but this
implementation is slower than BF16 and the previous FP8 weight-storage path.

## Results

Fresh separate processes for each mode, one warm-up and three measured requests.
Same 768×512, 257-frame / 25 FPS clip, reference image, JFK audio, prompt, seed 42,
four denoising steps, one-stage AR, ForeverCache enabled, chunk size 4, one history
chunk plus sink, relative positions and fast inference. Values are medians.

| Measurement | BF16 | FP8 cast | Dynamic FP8 |
| --- | ---: | ---: | ---: |
| Complete request including MP4 encoding, seconds | 10.661 | 12.613 | 19.312 |
| AR sampling, seconds | 7.872 | 9.781 | 16.489 |
| Peak PyTorch live allocation, GiB | 71.364 | 54.110 | 54.116 |
| Transformer storage, decimal GB | 38.012 | 19.495 | 19.500 |

Dynamic FP8 saves about 17.25 GiB of peak live VRAM (24.2%), but complete requests
are 81.1% slower than BF16 and 53.1% slower than FP8 cast. These times include
activation scaling/conversion on every call. Initial loading and compilation
are excluded from the medians. The dynamic warm-up request took 39.99 seconds,
including model loading and first-use compilation.

## Implementation

`--quantization fp8-dynamic` is available in `inference.py`, its two-stage entry
point, and `benchmarks/latency.py`. The default remains `none`.

- Quantize the same attention projections and feed-forward weight matrices as
  the previous cast policy, across all 48 blocks, including the audio stream.
- Quantize each weight tensor once while loading the original BF16 checkpoint:
  E4M3 values plus one FP32 dequantization scale. Store a column-major `(K, N)`
  view compatible with scaled GEMM and the existing scaled-FP8 LoRA merger.
- For every linear call, compute `amax(abs(x))` across the current activation
  tensor, clamp the maximum to at least `1e-12`, divide by FP8's maximum, and use
  that scale to convert the activations. No offline calibration is used.
- Use PyTorch `torch._scaled_mm` with FP8 inputs, BF16 output for this pipeline,
  and `use_fast_accum=False`. Preserve bias precision and add bias after GEMM.
- Pad the token dimension to a multiple of 16 where needed, then remove padding
  from the output. Padding does not change the activation maximum.
- Run the maximum reduction with native PyTorch. Compile only the remaining
  scale/conversion arithmetic using `torch.compile(dynamic=True)`.
- Attention, normalization, RoPE and history caches keep their existing paths;
  Gemma, prompt connectors and VAEs are unchanged. No TensorRT-LLM is required.

Biases are not quantized in this policy, unlike FP8 cast. Together with the FP32
weight scales, this explains the small storage increase over FP8 cast. Parameters
outside the selected matrices retain checkpoint precision, as in the cast policy.
The source checkpoint is not rewritten. Only the no-LoRA one-stage path was
exercised in the full benchmark; LoRA compatibility and two-stage quality were
not evaluated here.

## Why it is slower

A separate probe used five warm-ups and 30 repeated calls for representative
matrix sizes, measuring synchronized wall-clock average milliseconds. These
include Python/dispatch overhead, and exclude bias/padding and the full pipeline.
Component timings are separate experiments and should not be added together.

| Shape `(M, K, N)` | BF16 GEMM | FP8 GEMM with prequantized input | Dynamic quantization alone | Dynamic quantization + FP8 GEMM |
| --- | ---: | ---: | ---: | ---: |
| Video projection `(1536,4096,4096)` | 0.0704 | 0.0409 | 0.0938 | 0.1188 |
| Video FFN expansion `(1536,4096,16384)` | 0.2638 | 0.1526 | 0.0922 | 0.1971 |
| Audio projection `(96,2048,2048)` | 0.0117 | 0.0138 | 0.0883 | 0.1146 |

The larger FP8 GEMMs are faster. Activation reductions/conversion and dispatch
cost outweigh those savings in many layers, particularly the small audio
matrices. This probe supports that explanation; it is not a full per-kernel
attribution of the model's total time. Faster conversion kernels, fused
operations, selective layer quantization or graph capture would be separate
optimization experiments.

An initial implementation compiled the entire reduction and conversion together.
Inspection of generated code showed a single CUDA block scanning the full
activation tensor, giving approximately 105 seconds of sampling. That run was
interrupted and kept in `outputs/fp8-dynamic-comparison/initial-serial-reduction`.
The implementation above fixes that issue. Those diagnostic timings are excluded
from the comparison table.

## Output and correctness checks

All 12 final benchmark requests completed with finite final video latents.
The numerical test checked zero activations, magnitudes from `1e-8` to `1e3`,
odd token counts, output shape/dtype, stable FP8 weight storage, agreement with
dequantized reference multiplication, and actual `aten::_scaled_mm` dispatch.
Random-layer relative RMSE against BF16 was approximately 3.7%, and exactly zero
for zero activations. This layer test is not a video-quality metric.

The selected BF16 and dynamic clips each decode to 257 aligned frames at
768×512/25 FPS and have identical decoded audio. Sampled pairs at frames 32, 96,
160 and 224 preserve identity, clothing, scene structure and facial detail with
no obvious gross corruption. Pose and expression differ. Still-frame inspection
does not establish temporal smoothness or lip-sync equivalence.

RGB MAE is 12.405/255 and PSNR is 18.864 dB. These quantify difference from BF16,
not perceptual quality; numerical changes alter diffusion motion trajectories.
This is one clip, not a broad quality evaluation.

Open `outputs/fp8-dynamic-comparison/index.html` for synchronized three-way video
playback. Raw records, correctness checks and linear timings are archived in
[`results/h100-2026-09-28/fp8-dynamic`](results/h100-2026-09-28/fp8-dynamic).

## Reproduce

On `/home/ubuntu/work/avatarforever`, using the existing tested environment:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false \
.venv/bin/python benchmarks/latency.py \
  --checkpoint checkpoints/avatarforever-ltx-2.3-22b.safetensors \
  --gemma-root checkpoints/gemma-3-12b-it-qat-q4_0-unquantized \
  --audio data/jfk.flac \
  --reference outputs/stage-comparison/reference.png \
  --quantization fp8-dynamic --frames 257 --runs 3 --warmup-runs 1 \
  --cache on --fast-infer \
  --output-dir outputs/fp8-dynamic-repeat
```

Use a fresh output directory for each execution because records append. Run
`benchmarks/check_fp8_dynamic.py` for numerical checks and
`benchmarks/profile_fp8_dynamic.py` for the isolated timing probe. All GPU runs
should be separate from each other to avoid timing interference.
