from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import SpatialTilingConfig, TemporalTilingConfig, TilingConfig
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.media_io import load_image_conditioning

if TYPE_CHECKING:
    from ltx_core.model.video_vae import VideoEncoder
    from ltx_pipelines import ARA2VidDistilledPipeline


def optional_int(value: str) -> int | None:
    if value.lower() in {"none", "null", ""}:
        return None
    return int(value)


def valid_num_frames(value: str) -> int:
    frames = int(value)
    if frames <= 0 or (frames - 1) % 8 != 0:
        raise argparse.ArgumentTypeError(
            f"num_frames must be positive and follow 8n+1 (for example 161, 2001, or 8001), got {frames}."
        )
    return frames


def build_loras(lora_path: Path | None, strength: float) -> list[LoraPathStrengthAndSDOps]:
    if lora_path is None:
        return []
    return [
        LoraPathStrengthAndSDOps(
            str(lora_path),
            strength,
            LTXV_LORA_COMFY_RENAMING_MAP,
        )
    ]


def build_first_frame_images(
    image_path: Path | None,
    *,
    strength: float,
    crf: int,
) -> list[ImageConditioningInput]:
    if image_path is None:
        return []
    return [
        ImageConditioningInput(
            path=str(image_path),
            frame_idx=0,
            strength=strength,
            crf=crf,
        )
    ]


def _video_encoder_for_pipeline(pipeline: ARA2VidDistilledPipeline) -> VideoEncoder:
    if hasattr(pipeline, "model_ledger"):
        return pipeline.model_ledger.video_encoder()
    return pipeline.stage_1_model_ledger.video_encoder()


def _clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def encode_first_frame_channel_condition(
    pipeline: ARA2VidDistilledPipeline,
    *,
    image_path: Path | None,
    enabled: bool,
    height: int,
    width: int,
    crf: int,
) -> torch.Tensor | None:
    if not enabled or image_path is None:
        return None

    video_encoder = _video_encoder_for_pipeline(pipeline)
    image = load_image_conditioning(
        image_path=str(image_path),
        height=height,
        width=width,
        dtype=pipeline.dtype,
        device=pipeline.device,
        crf=crf,
    )
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=pipeline.dtype)
        if pipeline.device.type == "cuda"
        else contextlib.nullcontext()
    )
    with torch.inference_mode(), autocast_context:
        latent = video_encoder(image)

    latent = latent[:1].detach().contiguous()
    del image, video_encoder
    _clear_cuda_cache()
    return latent


def build_tiling_config(args: argparse.Namespace) -> TilingConfig:
    return TilingConfig(
        spatial_config=SpatialTilingConfig(
            tile_size_in_pixels=args.spatial_tile_size,
            tile_overlap_in_pixels=args.spatial_tile_overlap,
        ),
        temporal_config=TemporalTilingConfig(
            tile_size_in_frames=args.temporal_tile_size,
            tile_overlap_in_frames=args.temporal_tile_overlap,
        ),
    )


def resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output_path is not None:
        output_path = args.output_path
    else:
        history_count = "none" if args.ar_history_chunk_count is None else str(args.ar_history_chunk_count)
        output_path = args.output_dir / (
            f"f{args.num_frames}_chunk{args.ar_video_chunk_size}_hist{history_count}.mp4"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path
