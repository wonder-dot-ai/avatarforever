# ltx-pipelines: AvatarForever inference subset

This package is intentionally limited to the distilled autoregressive
audio-to-video inference path used by AvatarForever.

## Exported pipeline

- `ARA2VidDistilledPipeline`

## Included implementation

- `a2vid_distilled.py`: distilled audio-to-video model execution.
- `ar_a2vid_distilled_pipeline.py`: chunk-wise autoregressive wrapper.
- `utils/`: model loading, media I/O, sampling, conditioning, and AR helpers.

Native A2V, training, API, interactive, evaluation, and unrelated generation
pipelines are intentionally excluded from this release subset.
