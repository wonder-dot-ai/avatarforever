"""Profile resident CUDA model storage and phase peaks for one-stage AR inference.

Measurements synchronize CUDA and reset peak counters per phase. They are memory
diagnostics, not latency measurements. No tensors or modules are retained by the
profiler. CUDA storage is deduplicated to account for tied weights and views.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

import ltx_pipelines.a2vid_distilled as a2v
from inference import DEFAULT_PROMPT
from ltx_core.model.transformer.model import LTXModel
from ltx_core.model.video_vae import get_video_chunks_number
from ltx_pipelines import ARA2VidDistilledPipeline
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.model_ledger import ModelLedger
from util import build_first_frame_images, build_tiling_config, encode_first_frame_channel_condition


def storage_bytes(tensors):
    storages = {}
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor) and tensor.device.type == "cuda":
            storage = tensor.untyped_storage()
            storages[storage.data_ptr()] = storage.nbytes()
    return sum(storages.values())


def tensors_in(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            yield from tensors_in(getattr(value, field.name))
    elif isinstance(value, dict):
        for child in value.values():
            yield from tensors_in(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from tensors_in(child)


def snapshot():
    torch.cuda.synchronize()
    stats = torch.cuda.memory_stats()
    device_used_mib = int(subprocess.check_output(
        ["nvidia-smi", "--id=0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True,
    ).strip())
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "device_used_bytes": device_used_mib * 2**20,
        "inactive_split_bytes": stats.get("inactive_split_bytes.all.current", 0),
        "allocation_retries": stats.get("num_alloc_retries", 0),
        "oom_count": stats.get("num_ooms", 0),
    }


class MemoryRecorder:
    def __init__(self):
        self.run = "setup"
        self.phases = []
        self.models = {}
        self.caches = []

    def measure(self, name, fn, *args, **kwargs):
        before = snapshot()
        torch.cuda.reset_peak_memory_stats()
        result = fn(*args, **kwargs)
        after = snapshot()
        self.phases.append({
            "run": self.run, "phase": name, "before": before, "after": after,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        })
        return result

    def wrap(self, owner, method, label, model=False):
        original = getattr(owner, method)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            result = self.measure(label, original, *args, **kwargs)
            if model:
                parameters = list(result.parameters())
                buffers = list(result.buffers())
                self.models[method] = {
                    "parameter_count": sum(t.numel() for t in parameters),
                    "cuda_storage_bytes": storage_bytes(parameters + buffers),
                    "parameter_dtypes": sorted({str(t.dtype) for t in parameters}),
                    "children": {
                        name: storage_bytes(list(child.parameters()) + list(child.buffers()))
                        for name, child in result.named_children()
                    },
                }
            return result

        setattr(owner, method, wrapped)

    def install(self):
        # With fast_infer=True factories run before these inference phases;
        # measured phases do not nest and therefore do not reset each other's peaks.
        for name in ("text_encoder", "gemma_embeddings_processor", "audio_encoder",
                     "video_encoder", "transformer", "video_decoder"):
            self.wrap(ModelLedger, name, "load_" + name, model=True)
        self.wrap(a2v.A2VidDistilledPipeline, "_encode_guidance_contexts", "prompt_encoding")
        self.wrap(a2v, "vae_encode_audio", "audio_encoding")
        self.wrap(a2v.A2VidDistilledPipeline, "_build_video_conditionings", "image_conditioning")
        self.wrap(a2v.A2VidDistilledPipeline, "_denoise_video_only_ar", "ar_denoising")
        original = LTXModel._process_transformer_blocks_with_ar_feature_cache

        @functools.wraps(original)
        def cached_blocks(*args, **kwargs):
            mode = "reuse" if kwargs["ar_feature_cache"].populated else "populate"
            result = original(*args, **kwargs)
            cache = kwargs["ar_feature_cache"]
            # Assign each unique backing storage to its first field. This makes
            # the breakdown additive even when tensors are shared or are views.
            seen = set()
            by_field = {}
            logical_by_field = {}
            for block in cache.block_caches:
                for field in dataclasses.fields(block):
                    for tensor in tensors_in(getattr(block, field.name)):
                        if tensor.device.type != "cuda":
                            continue
                        logical_by_field[field.name] = (
                            logical_by_field.get(field.name, 0) + tensor.numel() * tensor.element_size()
                        )
                        storage = tensor.untyped_storage()
                        if storage.data_ptr() not in seen:
                            seen.add(storage.data_ptr())
                            by_field[field.name] = by_field.get(field.name, 0) + storage.nbytes()
            self.caches.append({
                "run": self.run,
                "mode": mode,
                "cuda_storage_bytes": storage_bytes(tensors_in(cache)),
                "unique_storage_bytes_by_field": by_field,
                "logical_bytes_by_field": logical_by_field,
                "allocated_after_blocks_bytes": torch.cuda.memory_allocated(),
            })
            return result

        LTXModel._process_transformer_blocks_with_ar_feature_cache = cached_blocks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gemma-root", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--cache", choices=("on", "off"), default="on")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    recorder = MemoryRecorder()
    recorder.install()
    pipeline = ARA2VidDistilledPipeline(
        distilled_checkpoint_path=args.checkpoint, gemma_root=args.gemma_root,
        spatial_upsampler_path=None, loras=[],
    )
    tiling = build_tiling_config(argparse.Namespace(
        spatial_tile_size=512, spatial_tile_overlap=64,
        temporal_tile_size=256, temporal_tile_overlap=8,
    ))
    requests = [("cold-no-reference", None), ("warm-no-reference", None)]
    if args.reference:
        requests.append(("warm-reference", args.reference))
    ends = {}
    with torch.inference_mode():
        for label, reference in requests:
            recorder.run = label
            image_latent = encode_first_frame_channel_condition(
                pipeline, image_path=reference, enabled=True, height=512, width=768, crf=0,
            )
            video, audio = pipeline(
                prompt=DEFAULT_PROMPT, seed=42, height=512, width=768,
                num_frames=257, frame_rate=25,
                images=build_first_frame_images(reference, strength=1.0, crf=0),
                audio_path=args.audio, audio_start_time=0.0, audio_max_duration=257 / 25,
                tiling_config=tiling, enhance_prompt=False, stage_mode="one-stage",
                stage1_sigmas=[1.0, 0.98125, 0.909375, 0.421875, 0.0],
                ar_video_chunk_size=4, ar_history_chunk_count=1,
                ar_sink_first_chunk=True, ar_relative_positions=True,
                ar_history_feature_cache=args.cache == "on",
                first_frame_channel_condition_latent=image_latent,
                first_frame_channel_condition_mode="gated", first_frame_channel_condition_init="zero",
                ar_first_frame_channel_condition_from_first_chunk=reference is None,
                ar_channel_condition_current_chunk_only=True, fast_infer=True,
            )

            def measured_video():
                iterator = iter(video)
                index = 0
                while True:
                    chunk = recorder.measure(f"vae_decode_chunk_{index}", next, iterator, None)
                    if chunk is None:
                        return
                    yield chunk
                    index += 1

            encode_video(
                video=measured_video(), fps=25, audio=audio,
                output_path=str(args.output_dir / f"{label}.mp4"),
                video_chunks_number=get_video_chunks_number(257, tiling), crf=12, preset="fast",
            )
            del video, audio, image_latent
            ends[label] = snapshot()
        torch.cuda.empty_cache()
        empty_cache_snapshot = snapshot()
        pipeline._fast_modules.text_encoder.to("cpu")
        pipeline._fast_modules.embeddings_processor.to("cpu")
        torch.cuda.empty_cache()
        offloaded_text_snapshot = snapshot()
    report = {
        "config": vars(args), "models": recorder.models, "phases": recorder.phases,
        "cache_samples": recorder.caches, "request_end": ends,
        "after_empty_cache": empty_cache_snapshot,
        "after_text_and_connectors_offloaded": offloaded_text_snapshot,
        "device_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "notes": "Bytes, not GB. Model storage deduplicates CUDA storage including buffers. "
        "Phase peaks are absolute process allocations, NOT additive. Phase resets are sequential. "
        "Cache storage is included in AR phase allocations. Device usage is sampled at boundaries, not a peak. "
        "Reference preprocessing's factory load is measured, but its encoder forward is outside phase peaks. "
        "CPU-offload snapshot is after all inference and does not benchmark offloaded inference.",
    }
    (args.output_dir / "vram.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    for name, value in recorder.models.items():
        print(f"MODEL {name}: {value['cuda_storage_bytes'] / 2**30:.4f} GiB")
    for phase in recorder.phases:
        print(f"PHASE {phase['run']} {phase['phase']}: {phase['peak_allocated_bytes'] / 2**30:.4f} GiB")


if __name__ == "__main__":
    main()
