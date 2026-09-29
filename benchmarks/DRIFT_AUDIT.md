# AvatarForever conditioning and drift: evidence report

**Last researched:** 2026-09-29 (Asia/Seoul). **Status:** unresolved upstream discrepancy.

The published checkpoint contains the first-frame channel-conditioning tensors,
but all three are zero. In the released forward code this makes that additional
reference pathway inactive. Our full-file checksum matches the published model,
so the zeros were not introduced by our quantization or loading changes.

**This does not establish an author mistake, intentional removal, or the cause of
our video drift.** Ordinary image conditioning, first-chunk attention and recent
history remain active. We have no comparison against trained, nonzero channel
weights. Earlier descriptions calling this a confirmed export mistake were too
strong and are superseded by this report.

There is already an [upstream report, issue #5](https://github.com/leeruibin/avatarforever/issues/5),
with an author acknowledgement and independent confirmation. It remains open
without a published explanation in the comments checked for this report.

## 1. What the paper says

[Section 3.2, Global Reference Conditioning](https://arxiv.org/html/2608.12107v1#S3.SS2)
describes encoding the first frame into a fixed reference latent and injecting
it into denoising tokens through a gated channel-conditioning module. Figure 2
identifies that pathway and the video-side LoRA as trainable. This is more
specific than generic image-to-video conditioning.

[Section 5.1](https://arxiv.org/html/2608.12107v1#S5.SS1) describes zero initialization
and use on target denoising tokens. Zero initialization describes the starting
state; it does not establish that a released, trained module should remain zero.
The paper's quantitative splits are 5 and 30 seconds.

[Section 5.2 / Figure 7](https://arxiv.org/html/2608.12107v1#S5.SS2) additionally
reports one continuous 11+ minute example without progressive drift. That is
relevant evidence of stability, not a matched reproduction of our 21-minute
recording or a guarantee for every reference, prompt and seed.

## 2. Published checkpoint and direct measurements

| Item | Audited value |
| --- | --- |
| Model | `LetsThink/AvatarForever` |
| File | `avatarforever-ltx-2.3-22b.safetensors` |
| HF revision | `622ea82482c09e7fb2d8e29cb7ff1d009c820f05` |
| Size | 46,183,958,894 bytes |
| Published and independently computed SHA-256 | `08dea67ed3d35a75da49b056ce2ca59a0e9ce538b429143fb9022c9bd5c3b145` |
| Upstream source revision checked | `4dfc42b0e2dbbded4d148387d186219bd7601279` |

The hash is listed on the [versioned model page](https://huggingface.co/LetsThink/AvatarForever/blob/622ea82482c09e7fb2d8e29cb7ff1d009c820f05/avatarforever-ltx-2.3-22b.safetensors).
Our independent `sha256sum` read the complete downloaded file. Tensor inspection
read the file directly with `safetensors.safe_open`, before constructing a model.

All names below have prefix `model.diffusion_model.`:

| Tensor | Shape | Nonzero elements |
| --- | --- | ---: |
| `video_channel_condition_proj.weight` | 4096 × 128 | 0 / 524,288 |
| `video_channel_condition_gate.weight` | 4096 × 4096 | 0 / 16,777,216 |
| `video_channel_condition_gate.bias` | 4096 | 0 / 4,096 |

The [released forward code](https://github.com/leeruibin/avatarforever/blob/4dfc42b0e2dbbded4d148387d186219bd7601279/packages/ltx-core/src/ltx_core/model/transformer/transformer_args.py#L163-L176)
computes:

```text
c = P(reference)                 # P has no bias
multiplier = 2 × sigmoid(G(c))
video_features += c × multiplier
```

For the audited weights, `c = 0`, `multiplier = 1`, and the added reference
features are zero for any finite reference input. A CPU check using three random
128-dimensional inputs confirmed this result. **The zero projection is decisive;
a zero gate alone would give a multiplier of one, not disable the pathway.**

The [loader](https://github.com/leeruibin/avatarforever/blob/4dfc42b0e2dbbded4d148387d186219bd7601279/packages/ltx-core/src/ltx_core/loader/single_gpu_model_builder.py#L95-L128)
initializes compatible missing parameters, but skips names already in the state
dict. These tensors are present and zero on disk. Changing `zero` to `xavier`
in an inference initialization flag cannot recover trained values; random
replacement is not a faithful fix.

Relevant checkpoint metadata:

- Base: `ltx-2.3-22b-distilled-1.1.safetensors`.
- Adapter: `lora_weights_step_03000.safetensors`; merge strength: `0.8`.
- `avatarforever_complete_channel_condition_weights=true`.

This records an adapter merge and the presence of conditioning tensors. It does
not independently verify the merge, establish that the tensors were trained,
or explain why they are zero. The absence of separate LoRA keys is compatible
with merged weights; it does not show that all robustness adaptation is absent.

## 3. Why reasonable consistency is still possible

| Path in our run | Effect | Status |
| --- | --- | --- |
| Initial image latent | Inserts the reference at frame zero with strength 1; its tokens are protected from denoising | Active |
| First-chunk sink | Retains the initial chunk, including reference-frame tokens, in later attention contexts | Active |
| Previous generated chunk | Supplies recent pose, appearance and motion | Active |
| Audio and text | Guide speech-related motion and requested behavior | Active |
| Additional first-frame channel | Adds projected reference features directly to current video-token features | Inactive |

With one history chunk and the sink enabled, a later chunk sees the first chunk,
the previous chunk and the current noisy chunk. The selection is explicit in
[the original AR implementation](https://github.com/leeruibin/avatarforever/blob/4dfc42b0e2dbbded4d148387d186219bd7601279/packages/ltx-pipelines/src/ltx_pipelines/utils/autoregressive.py#L60-L83).
The extra prefix-conditioning option was off; the first-chunk sink was on.

Thus, “first-frame conditioning is entirely off” is incorrect. The model can
preserve broad appearance through attention and history even with the additional
channel inactive. This explains the available mechanism, not a measured causal
contribution from each path. Attention also does not hard-constrain clothing or
background details to match the reference indefinitely.

## 4. What upstream discussions establish

| Date (UTC) | Source | Finding and evidentiary limit |
| --- | --- | --- |
| 2026-08-27 | [Issue #5](https://github.com/leeruibin/avatarforever/issues/5) | A user reports drift in a 20-second run and zero channel-conditioning weights. Their suspected connection is not a causal demonstration. |
| 2026-08-28 | [Author reply on #5](https://github.com/leeruibin/avatarforever/issues/5#issuecomment-5449644201) | The repository owner asks for the prompt/image and says they will check channel conditioning. This acknowledges the question; it does not confirm an export error or intentional disabling. |
| 2026-09-02 | [Independent follow-up on #5](https://github.com/leeruibin/avatarforever/issues/5#issuecomment-5511165310) | Another user reports the same three zero tensors using the public checkpoint and commit `4dfc42b`, while also noting variable generation outcomes. They request exact demo reproduction inputs. |
| 2026-08-28 | [Author reply on #6](https://github.com/leeruibin/avatarforever/issues/6#issuecomment-5449678700) | The author states an intention to release further components, starting with the trainable RRT LoRA. This does not mean the merged checkpoint requires an additional adapter. |
| 2026-09-23 | [Issue #9](https://github.com/leeruibin/avatarforever/issues/9) | A user reports drift and lip-motion problems with a stylized portrait, worse with cache enabled, but says a realistic front-facing reference works. Different inputs and settings prevent direct comparison with our run. |

As checked on September 29, #5 has two comments and no later author resolution.
The GitHub default-branch history still ends at `4dfc42b`. The HF repository's
main history contains its initial commit and the August 23 checkpoint upload;
its file tree contains the checkpoint and `.gitattributes`, with no model card
or standalone adapter. The HF discussions endpoint returned zero discussions.
These are observations about the checked repository endpoints, not a claim that
no explanation or artifact could exist elsewhere.

A future contact should follow up on #5 rather than open a duplicate. No issue
or comment was posted as part of this research.

## 5. Our experiment and remaining uncertainty

[The precision comparison](FP8_LONG.md) used three continuous 21-minute sequences
with identical audio, reference, prompt and seed. Sampled frames showed clothing
and background changes in BF16 as well as FP8. This was not a complete manual
review or a validated perceptual/identity/lip-sync evaluation. There was no
working-channel control, so the benefit of that channel was not measured.

Compared with original source, our BF16 transformer, attention, AR sampler,
relative positions and cache implementation are unchanged. Our long-run changes
were overlapping audio encoding (30-second windows, two seconds of surrounding
context) and precomputed audio-latent injection. ForeverCache was enabled, while
the original CLI defaults it off. BF16 never enters our FP8 conversion or kernel
paths. Audio preprocessing and cache behavior remain possible contributors to
output differences. Our checkpoint's inactive channel is another confound when
comparing with the paper's demonstration.

| Explanation | Current status | What would distinguish it |
| --- | --- | --- |
| Authors intentionally disabled the additional channel for this release | Possible, not documented in the checked material | Author confirmation and the released/evaluated inference recipe |
| Trained channel weights were omitted or replaced during export | Possible, not confirmed | Compare pre-export training state with released tensors |
| The channel never updated during training or the release uses a different training variant | Possible, no direct training evidence | Training configuration, gradients/checkpoint history and author clarification |
| Zero channel is the sole cause of our drift | Not established | Matched runs differing only in verified trained channel weights |
| Our FP8 changes created the zero weights | Ruled out for the audited artifact | File bytes match the published SHA-256 before model loading |

## 6. Questions and next checks

1. Are the all-zero channel tensors intentional in revision `622ea824`?
2. Does this revision correspond to the model used for Figure 7 and the website
   long-duration demo? If so, which conditioning paths were active?
3. If trained channel weights exist, can they be released with the standalone
   adapter, including any full-rank parameters outside LoRA?
4. Can the authors provide one complete demo recipe: reference, original audio,
   prompt, seed, checkpoint revision, cache/sink/history settings and audio
   preprocessing?

If trained weights become available, preserve the audited file and run an A/B
comparison changing only that pathway. Independently, a manageable-duration BF16
comparison can isolate full versus windowed audio encoding and cache off/on.
Longer evaluation should follow matched short controls and use multiple inputs
and seeds. None of these proposed experiments has been performed in this audit.

## 7. Reproduce the tensor check

This reads only the three conditioning tensors on CPU; no GPU generation is needed:

```bash
.venv/bin/python - <<'PY'
import torch
from safetensors import safe_open

path = "checkpoints/avatarforever-ltx-2.3-22b.safetensors"
names = ("video_channel_condition_proj.weight",
         "video_channel_condition_gate.weight",
         "video_channel_condition_gate.bias")
with safe_open(path, framework="pt", device="cpu") as f:
    tensors = [f.get_tensor("model.diffusion_model." + n).float() for n in names]
for name, t in zip(names, tensors):
    print(name, tuple(t.shape), "nonzero:", t.count_nonzero().item(), "/", t.numel())
x = torch.randn(3, 128)
c = torch.nn.functional.linear(x, tensors[0])
g = 2 * torch.sigmoid(torch.nn.functional.linear(c, tensors[1], tensors[2]))
print("maximum added conditioning:", (c * g).abs().max().item())
PY
sha256sum checkpoints/avatarforever-ltx-2.3-22b.safetensors
```

[Research source index](results/conditioning-audit-2026-09-29/source-index.json)
records retrieval time, public endpoint URLs, revisions and discussion metadata.
The local measured evidence predates this additional source review and is
summarized above; the research index does not imply the full checksum was rerun.
