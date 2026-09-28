<div align="center">

<img src="assets/avatar-forever-logo.png" alt="Avatar-Forever" width="720" />

### Decoupled Parallel Training for High-Quality Real-Time Infinite Avatars

**Real-time · Long-horizon · Audio-driven · 27.2 FPS at 768 × 512 on one H100**

Ruibin Li<sup>1,†</sup> · Tao Yang<sup>2</sup> · Zhiyuan Ma<sup>1</sup> · Fangzhou Ai<sup>3</sup> · Shilei Wen<sup>2</sup> · Lei Zhang<sup>1,*</sup>

<sup>1</sup> The Hong Kong Polytechnic University · <sup>2</sup> ByteDance · <sup>3</sup> AMD

<p>
  <a href="https://leeruibin.github.io/avatarforever-project-page/"><strong>Project Page</strong></a>
  ·
  <a href="https://github.com/leeruibin/avatarforever"><strong>Code Repository</strong></a>
  ·
  <a href="https://huggingface.co/LetsThink/AvatarForever"><strong>Model</strong></a>
  ·
  <a href="https://arxiv.org/abs/2608.12107"><strong>Paper</strong></a>
</p>

</div>

<sub><sup>†</sup> Work done during an internship at ByteDance. <sup>*</sup> Corresponding author.</sub>

<!-- > **Research preview.** The paper, code, models, and demos are being prepared for public release. -->

## Release Status

- [x] Method overview
- [x] Paper and supplementary material
- [x] Inference code
- [x] Model checkpoints
- [ ] Training code
- [ ] Training data
- [ ] Interactive demo

## Highlights

| Capability | Result |
|---|---:|
| Video resolution | 768 × 512 |
| End-to-end throughput | **27.2 FPS** |
| Hardware | **1× NVIDIA H100** |
| Backbone | 22B video foundation model |
| ForeverCache speedup | **23%** |
| Generation horizon | Effectively unbounded streaming |

End-to-end throughput includes both DiT inference and VAE decoding.

## Overview

Avatar-Forever is a framework for high-quality, real-time, and effectively unbounded audio-driven avatar generation. It addresses a central limitation of existing streaming video systems: sequential distillation pipelines entangle few-step efficiency with long-horizon robustness, so distribution shifts and optimization failures introduced early in training propagate to later stages.

Our key insight is to learn these capabilities independently and compose them only at deployment:

- **Efficiency branch:** full-parameter distribution matching distillation trains a high-quality few-step generator.
- **Robustness branch:** a lightweight long-horizon adapter is trained with **Recovery-oriented Rollout Training (RRT)** under accumulated autoregressive errors.
- **Streaming inference:** **ForeverCache** reuses stable historical features across denoising steps to avoid redundant context computation.

![Avatar-Forever framework](assets/overview.png)

## Inference

### Installation

The released inference code is organized as a `uv` workspace and requires Python 3.11 and a CUDA-capable GPU. From the repository root, install all workspace packages with:

```bash
uv sync --all-packages
```

Optional optimized attention backends depend on the target GPU and CUDA environment.

### Model checkpoints

Model weights are not stored in Git. Download both the Avatar-Forever checkpoint and the Gemma text encoder into `checkpoints/` so the repository has this layout:

```text
checkpoints/
├── avatarforever-ltx-2.3-22b.safetensors
└── gemma-3-12b-it-qat-q4_0-unquantized/
    ├── config.json
    ├── tokenizer.json
    └── ...
```

Install the Hugging Face CLI and authenticate first. Gemma is a gated model, so accept its license on [Google's Gemma 3 12B QAT model page](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized) before downloading:

```bash
pip install -U huggingface_hub
hf auth login

hf download LetsThink/AvatarForever \
  avatarforever-ltx-2.3-22b.safetensors \
  --local-dir checkpoints

hf download google/gemma-3-12b-it-qat-q4_0-unquantized \
  --local-dir checkpoints/gemma-3-12b-it-qat-q4_0-unquantized
```

The `checkpoints/` directory is ignored by Git; do not commit model weights.

### Run inference

```bash
uv run python inference.py \
  --distilled-checkpoint-path checkpoints/avatarforever-ltx-2.3-22b.safetensors \
  --gemma-root checkpoints/gemma-3-12b-it-qat-q4_0-unquantized \
  --audio-path /path/to/input.wav \
  --output-path outputs/result.mp4
```

The default configuration uses one-stage distilled inference at 768 × 512 and 25 FPS, generates 2001 frames, uses AR chunk size 4 with one history chunk, derives first-frame channel conditioning from the first generated chunk, and saves H.264 video at CRF 12. Frame counts must be positive and follow `8n+1`, such as 161, 2001, or 8001.

Use `python inference.py --help` to see all controls, including video length, autoregressive history, first-frame conditioning, VAE tiling, x264 CRF, and encoding preset.

### 8-bit transformer weights

Add `--quantization fp8-cast` to store the transformer's attention projections and
feed-forward weights/biases in FP8 E4M3. This uses the repository's existing
quantization policy: each linear operation upcasts its weights to the input
dtype, normally bfloat16. Activations and ForeverCache are not quantized; Gemma,
prompt connectors and VAEs are unchanged. This is FP8 weight storage, not INT8 or
native FP8 matrix multiplication. Some transformer parameters retain bfloat16 or
float32 checkpoint precision.

```bash
.venv/bin/python inference.py \
  --distilled-checkpoint-path checkpoints/avatarforever-ltx-2.3-22b.safetensors \
  --gemma-root checkpoints/gemma-3-12b-it-qat-q4_0-unquantized \
  --audio-path /path/to/input.wav \
  --first-frame-condition-image-path /path/to/reference.png \
  --quantization fp8-cast \
  --num-frames 257 --seed 42 --height 512 --width 768 \
  --ar-history-feature-cache --fast-infer \
  --output-path outputs/fp8.mp4
```

For a bfloat16 baseline, use the same command with `--quantization none` and a
different output path. Quantization happens while loading; it does not rewrite
the source checkpoint. The two-stage entry point also accepts this flag.

Use `--quantization fp8-dynamic` on Hopper or newer CUDA GPUs to run these
projections with native FP8 matrix multiplication instead. This experimental
policy quantizes weights with per-tensor scales once during loading, calculates
a fresh per-tensor activation scale for each call, and returns BF16 outputs.
Biases retain checkpoint precision. It uses PyTorch `torch._scaled_mm`, with
`torch.compile` applied only to the activation quantizer; the first request
includes compilation. No static calibration file or TensorRT-LLM is needed.
Attention, ForeverCache, Gemma and VAEs retain their existing computation paths.

To try the optimized activation quantizer on H100, add
`--fp8-activation-backend auto` alongside `--quantization fp8-dynamic`. It uses
parallel fused Triton kernels and captures only the quantizer for smaller
activation tensors; GEMM and the transformer are not captured. The original
`compiled` backend remains the default. See [the long-video comparison report](benchmarks/FP8_LONG.md)
for quantizer timings, the matched three-mode results, and the continuous
21-minute speech test. The long benchmark uses precomputed overlapping-window
audio latents shared by all modes; it never restarts video AR history.

### Experimental two-stage comparison

`inference_two_stage.py` runs the existing AR sampler at half resolution, upsamples
the completed latent sequence by 2×, then refines it with a second AR pass at the
output resolution. This is an experiment with the Avatar-Forever checkpoint;
the published training and evaluation do not establish two-stage quality.
It does not stream chunks between stages.

Download the [LTX-2.3 spatial ×2 upscaler](https://huggingface.co/Lightricks/LTX-2.3/blob/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors):

```bash
hf download Lightricks/LTX-2.3 ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --local-dir checkpoints

.venv/bin/python inference_two_stage.py \
  --distilled-checkpoint-path checkpoints/avatarforever-ltx-2.3-22b.safetensors \
  --spatial-upsampler-path checkpoints/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --gemma-root checkpoints/gemma-3-12b-it-qat-q4_0-unquantized \
  --audio-path /path/to/input.wav \
  --first-frame-condition-image-path /path/to/reference.png \
  --num-frames 257 --seed 42 --height 512 --width 768 \
  --ar-history-feature-cache --fast-infer \
  --output-path outputs/two-stage.mp4
```

For a matched one-stage run, use `inference.py` with the same arguments, remove
`--spatial-upsampler-path`, and set `--output-path outputs/one-stage.mp4`.
The reference image is optional; use the same image in both modes to control
identity. Matching seeds do not produce identical noise tensors across resolutions.

The two-stage script defaults to 257 frames and requires output dimensions divisible
by 64. Stage 1 uses the same four-step schedule as `inference.py`; stage 2 defaults
to three steps (`--stage2-sigmas 0.909375 0.725 0.421875 0`). The reference image is
encoded separately at each stage's resolution. Both scripts save a JSON sidecar
with arguments and cold request time, including model loading and MP4 encoding.
These times are not warmed throughput benchmarks.

### Prompt guidance

The following general-purpose prompt is the default for audio-driven generation:

```text
Natural audio-driven speaking motion with accurate lip synchronization, smooth and continuous facial animation, subtle head movement, natural blinking, gentle breathing, and relaxed upper-body motion. Expressions and gestures should respond naturally to the rhythm, tone, and emotion of the speech while remaining restrained and realistic.

Maintain strong temporal consistency across all frames. Keep facial appearance, identity, pose, body structure, clothing details, lighting, and background stable throughout the video. Avoid sudden motion changes, excessive gestures, unnatural expression shifts, frame-to-frame appearance variation, flickering, jitter, ghosting, texture instability, temporal artifacts, or deformation. All motion should be coherent, fluid, stable, and naturally driven by the audio.
```

For longer videos, prepend a concrete scene description covering the subject, identity and appearance, framing, pose, clothing, lighting, background, and camera position. For T2V scene generation, describe the visual content and scene details explicitly, then append the motion and temporal-stability prompt above. When the composition should remain stable, also specify a single continuous shot with a fixed camera and no cuts, transitions, angle changes, or viewpoint changes.

## Why Avatar-Forever?

Streaming avatar generation must satisfy two objectives that operate on different temporal scales:

1. **Few-step efficiency** preserves visual quality while reducing the number of denoising steps.
2. **Long-horizon robustness** prevents identity drift, motion degradation, and error accumulation over recursive rollouts.

Optimizing both objectives inside one sequential distillation pipeline creates stage-wise dependence and objective interference. Avatar-Forever instead trains them in parallel, making the training process simpler to optimize, diagnose, and scale.

## Method

### Decoupled Parallel Training

Starting from a 22B video foundation model, Avatar-Forever separates efficient generation from long-horizon adaptation. The resulting robustness adapter is merged with the distilled few-step generator at deployment.

### Recovery-oriented Rollout Training

RRT targets the error-propagation pattern encountered during streaming inference. It perturbs an early historical context, rolls the degradation forward through multiple autoregressive chunks, and applies standard flow-matching supervision after errors have accumulated. The model therefore learns to recover under long-horizon inference conditions rather than only reconstruct locally corrupted inputs.

### ForeverCache

ForeverCache is a chunk-wise history feature cache for streaming diffusion inference. It populates historical context features on the first denoising step of each chunk, then reuses those stable features while forwarding only the current chunk tokens in subsequent steps.

## Citation
If you find the method useful, please cite
```
@article{li2026avatarForever,
  title={Avatar-Forever: Decoupled Parallel Training for High-Quality Real-Time Infinite Avatars},
  author={Li, Ruibin and Yang, tao and Ma, Zhiyuan and Ai, Fangzhou and Wen, shilei and Zhang, Lei},
  journal={arXiv preprint arXiv:2608.12107},
  year={2026}
}
```
