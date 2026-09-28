# One-stage H100 VRAM breakdown

Measured on 2026-09-28 with `benchmarks/vram.py`, using the same environment as
the latency benchmark. One-stage, bfloat16, `fast_infer=True`, ForeverCache on,
768 × 512, 257 frames, 25 FPS, four denoising steps, AR chunk size 4 and one
history chunk plus the first-chunk sink. No quantization or upsampler.

GiB means bytes / 2^30; decimal GB means bytes / 10^9. This distinction matters
when comparing PyTorch, nvidia-smi and advertised GPU capacity.

## Resident models

Actual CUDA storage of parameters and registered buffers, deduplicated by storage
address within each model to avoid double-counting tied weights and views:

| Component | GiB | Decimal GB |
| --- | ---: | ---: |
| Diffusion transformer, including channel conditioning | 35.402 | 38.012 |
| Gemma text encoder, including its vision tower | 22.701 | 24.375 |
| Prompt feature extractor and audio/video connectors | 5.909 | 6.344 |
| Video VAE encoder | 0.594 | 0.638 |
| Video VAE decoder | 0.758 | 0.814 |
| Audio VAE encoder | 0.040 | 0.043 |
| **Total** | **65.403** | **70.226** |

The prompt-processing modules comprise 2.153 GiB of feature extraction,
3.004 GiB of video connector and 0.752 GiB of audio connector weights.
Gemma parameters are bfloat16 despite the QAT label in the downloaded model name.
The fast path retains all six modules on the GPU for subsequent requests.

## Peak live allocations

The following are whole-process PyTorch allocated-memory peaks during each
phase of the warm request without a reference image. They include all resident
models. These rows must not be added together.

| Phase | Peak allocated GiB |
| --- | ---: |
| Prompt encoding | 67.357 |
| Audio encoding | 65.690 |
| AR denoising | 69.947 |
| VAE decoding | **71.364** |

The overall live peak reproduces the original benchmark's 71.36360693 GiB.
At that peak, model storage accounts for 65.403 GiB, leaving 5.960 GiB for
additional live tensors, decoder activations/work buffers and allocation rounding.
This is a remainder, not a tensor-by-tensor classification of decoder internals.
ForeverCache stores up to **3.463 GiB** during AR sampling, already included in
the AR peak. It is not an additional amount to add to the later decoder peak.
The warm reference-image run peaked at 71.36369848 GiB, essentially the same.

## Why the device reports approximately 79 GB

At the end of the reference-image request, sampled nvidia-smi usage was
**79.368 GB (73.917 GiB)**. At that same instant, PyTorch had 65.563 GiB live
and 73.217 GiB reserved. The difference includes reusable allocator blocks and
fragmented blocks; reserved memory is not all live model or activation data.
About 0.700 GiB of device memory was outside PyTorch's reserved pool.

Reservation depends on allocation history. This profiling run reached a
reserved high-water mark of 78.264 GiB during reference-run decoding, with a
successful allocator retry. There were no unrecovered PyTorch OOMs. Device usage
was sampled at phase boundaries, so this report does not claim an exact
simultaneous nvidia-smi peak during that transient spike.

After all inference, `empty_cache()` reduced device usage to 67.753 GiB while
live allocations stayed at 65.563 GiB. Moving Gemma and prompt-processing modules
to CPU and emptying the cache reduced live allocations to 36.848 GiB and device
usage to 39.497 GiB. Their deduplicated weights total 28.609 GiB. This is a
post-inference offload measurement, not an end-to-end benchmark of an offloaded
pipeline. A persistent stream could retain prompt embeddings and release these
modules after prompt processing; handling changed prompts would need to be planned.

## Reproduce

On the configured H100, from `/home/ubuntu/work/avatarforever`:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false \
.venv/bin/python benchmarks/vram.py \
  --checkpoint checkpoints/avatarforever-ltx-2.3-22b.safetensors \
  --gemma-root checkpoints/gemma-3-12b-it-qat-q4_0-unquantized \
  --audio data/jfk.flac \
  --reference outputs/stage-comparison/reference.png \
  --output-dir outputs/vram-profile
```

The reference argument is optional. The profiler synchronizes CUDA and resets
peak counters between non-nested measured phases; its timings are not latency
results. The optional reference preprocessing factory is measured, but its
encoder forward is outside the phase measurements. Raw bytes, model dtypes,
all phase boundaries and cache samples are saved in
[vram.json](results/h100-2026-09-28/vram/vram.json).

## Inside the diffusion transformer

The transformer contains 19,006,143,744 parameters stored in bfloat16:
38,012,287,488 bytes (38.012 GB / 35.402 GiB). It has 48 blocks, with a
4096-wide video stream and a 2048-wide audio stream. This count excludes
Gemma, the separately loaded prompt connectors and all VAEs.

| Component, summed across 48 blocks | Decimal GB |
| --- | ---: |
| Video feed-forward layers | 12.887 |
| Video self-attention | 6.457 |
| Video-to-text attention | 6.457 |
| Audio feed-forward layers | 3.222 |
| Audio self-attention | 1.618 |
| Audio-to-text attention | 1.618 |
| Audio-to-video cross-attention | 2.430 |
| Video-to-audio cross-attention | 2.423 |
| Remaining input/output projections, timestep/conditioning modules and modulation tables | 0.899 |
| **Total** | **38.012** |

Runtime hooks on every module owning parameters found no uncalled parameter
owners in the tested 65-frame one-stage AR run. Calls alone do not prove every
computation affects the final video; the following conclusions also use source
dataflow and an ablation test.

Clean supplied audio has a zero denoising mask, but the audio modality remains
enabled. Audio self-attention, text attention and feed-forward layers produce
hidden features used by audio-to-video attention. Earlier blocks' video-to-audio
updates affect the audio features used by subsequent blocks. Removing the whole
audio branch or every video-to-audio module would therefore alter video generation.

There is a small output-only tail for this video-only sampling path:

| Candidate | Bytes | Decimal MB |
| --- | ---: | ---: |
| Final block audio feed-forward | 67,129,344 | 67.129 |
| Final block video-to-audio attention | 50,487,360 | 50.487 |
| Final audio output projection | 524,544 | 0.525 |
| **Total** | **118,141,248** | **118.141** |

These computations occur after the final audio-to-video update, so they only
affect the audio prediction. The sampler clamps audio to its supplied clean
latents and returns the original source audio. This tail is about 0.311% of
transformer weight storage; it is not a multi-GB savings opportunity.

`benchmarks/transformer_components.py` temporarily substitutes zero-output modules
for these three computations, then restores the originals. It produced
bit-identical video latents (maximum absolute difference 0) for 65 frames at
768 × 512, seed 42, JFK audio and generated first-chunk conditioning, both with
ForeverCache enabled and disabled. This is a bounded equivalence test; it does
not establish behavior for other inference tasks such as joint audio generation.
The test retains original weights for restoration and does not measure a VRAM
reduction. No production modules or checkpoint weights were removed.

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false \
.venv/bin/python benchmarks/transformer_components.py \
  --checkpoint checkpoints/avatarforever-ltx-2.3-22b.safetensors \
  --gemma-root checkpoints/gemma-3-12b-it-qat-q4_0-unquantized \
  --audio data/jfk.flac \
  --output outputs/vram-profile/transformer-components.json
```

Raw counts, calls and equivalence results:
[transformer-components.json](results/h100-2026-09-28/vram/transformer-components.json).

## ForeverCache enabled versus disabled

Measured in two fresh processes, one per mode, each with a cold request followed
by a warm request. Configuration: 257 frames, 768 × 512, 25 FPS, four denoising
steps, seed 42, default prompt, JFK audio, generated first-chunk conditioning,
AR chunk size 4, one history chunk plus the first-chunk sink, resident bfloat16
models (`fast_infer=True`). The table uses the warm request in each process.

| Measurement | Cache off, GiB | Cache on, GiB |
| --- | ---: | ---: |
| Maximum persistent history-cache storage | 0 | 3.463249 |
| Peak live allocations during AR sampling | 66.538019 | 69.947345 |
| Peak live allocations for the complete request | 71.363607 | 71.363607 |
| Peak PyTorch reservation during the warm request | 73.220703 | 73.925781 |

Thus enabling the cache increases the AR live peak by 3.409326 GiB, but does not
increase the overall live peak in this workload because VAE decoding dominates.
Reserved memory includes reusable/free allocator blocks and is not the cache's
tensor size. The exact reservation depends on allocation history.

ForeverCache stores per-block history features rather than projected attention
keys and values. `Attention.forward()` recomputes `to_k(context)` and
`to_v(context)` on every call. Their temporary allocations are included in the
AR peak, even with caching disabled. There is no separate persistent projected
KV cache in this diffusion-transformer path.

At the largest observed cache, deduplicated CUDA backing storage was:

| Stored data | GiB |
| --- | ---: |
| Video self-attention history features | 1.687500 |
| Video cross-attention history features | 1.687500 |
| Audio self- and cross-attention history features | 0.034790 |
| Video/audio positional data | 0.053459 |
| **Total** | **3.463249** |

The total is 3.718636 decimal GB. No attention-mask tensors were retained in the
cache for this configuration. Shared storage and tensor views are deduplicated;
simply adding tensor element counts would incorrectly count shared positional
data multiple times. Each cache field's storage and logical tensor size are
recorded separately in the raw results.

There is also a potential storage optimization: cached video history tensors
contain 2.25 GiB of logical elements but retain 3.375 GiB of backing storage.
`_history_along_dim()` returns a narrow view when history is one contiguous
prefix, and `.detach()` does not release the current-chunk portion of its backing
allocation. Compact history copies could reduce retained storage, but that
change and its effect on actual peak allocation have not been implemented or
benchmarked here.

Observed cache storage after population for the nine chunks was approximately
0, 2.309, 3.463, 3.463, 3.463, 3.463, 3.463, 3.463 and 2.597 GiB. The last chunk
is shorter. The cache is recreated for each AR chunk and reused for the remaining
denoising steps within that chunk; with this fixed history window, it does not
accumulate all earlier chunks. Other full-sequence buffers can still grow with
video length.

Reproduce using the earlier VRAM command without `--reference`, once with
`--cache off --output-dir outputs/cache-vram-off`, then in a new process with
`--cache on --output-dir outputs/cache-vram-on`.

Raw results: [off](results/h100-2026-09-28/cache-vram/off.json),
[on](results/h100-2026-09-28/cache-vram/on.json),
[summary](results/h100-2026-09-28/cache-vram/summary.json).
