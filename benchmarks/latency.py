"""Measure real A2V inference, including lazy VAE decode and MP4 encoding.

Run with the repository's Python environment. Every timed GPU boundary is
synchronized. Nested timings overlap and must not be added together.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.metadata
import json
import logging
import platform
import statistics
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

import ltx_pipelines.a2vid_distilled as a2v
from inference import DEFAULT_PROMPT, build_quantization_policy
from ltx_core.model.transformer import X0Model
from ltx_core.model.video_vae import SpatialTilingConfig, TemporalTilingConfig, TilingConfig, get_video_chunks_number
from ltx_pipelines import ARA2VidDistilledPipeline
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.model_ledger import ModelLedger
from util import build_first_frame_images, encode_first_frame_channel_condition


def sync():
    torch.cuda.synchronize()


class Recorder:
    def __init__(self):
        self.samples = defaultdict(list)
        self.model_info = None

    def wrap(self, owner, name, label):
        original = getattr(owner, name)

        @functools.wraps(original)
        def timed(*args, **kwargs):
            sync()
            start = time.perf_counter()
            result = original(*args, **kwargs)
            sync()
            self.samples[label].append(time.perf_counter() - start)
            if label == "load_transformer" and self.model_info is None:
                model = result.velocity_model
                storages = {}
                bytes_by_dtype = defaultdict(int)
                for tensor in list(model.parameters()) + list(model.buffers()):
                    if tensor.device.type != "cuda":
                        continue
                    storage = tensor.untyped_storage()
                    if storage.data_ptr() not in storages:
                        storages[storage.data_ptr()] = storage.nbytes()
                        bytes_by_dtype[str(tensor.dtype)] += storage.nbytes()
                self.model_info = {
                    "class": type(model).__name__,
                    "parameter_count": sum(p.numel() for p in model.parameters()),
                    "blocks": len(model.transformer_blocks),
                    "video_hidden_dim": model.inner_dim,
                    "audio_hidden_dim": model.audio_inner_dim,
                    "dtype": str(next(model.parameters()).dtype),
                    "cuda_storage_bytes": sum(storages.values()),
                    "cuda_storage_bytes_by_dtype": dict(bytes_by_dtype),
                    "attention": str(model.transformer_blocks[0].attn1.attention_function),
                }
            return result

        setattr(owner, name, timed)

    def install(self):
        for name in ("text_encoder", "gemma_embeddings_processor", "audio_encoder", "video_encoder", "transformer", "video_decoder"):
            self.wrap(ModelLedger, name, "load_" + name)
        self.wrap(a2v.A2VidDistilledPipeline, "_encode_guidance_contexts", "prompt_with_loading")
        self.wrap(a2v.A2VidDistilledPipeline, "_denoise_video_only_ar", "ar_sampling")
        self.wrap(a2v, "vae_encode_audio", "audio_encoding")
        self.wrap(X0Model, "forward", "dit_forward")


def command_output(args):
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gemma-root", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--quantization", choices=("none", "fp8-cast", "fp8-dynamic"), default="none")
    parser.add_argument("--fp8-activation-backend", choices=("compiled", "triton", "cudagraph", "auto"), default="compiled")
    parser.add_argument("--audio-latents", type=Path)
    parser.add_argument("--warmup-frames", type=int)
    parser.add_argument("--frames", type=int, default=257)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--fps", type=float, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--cache", choices=("off", "on", "both"), default="both")
    parser.add_argument("--fast-infer", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.frames <= 0 or (args.frames - 1) % 8:
        parser.error("--frames must be positive and 8n+1")
    if args.warmup_frames is not None and (args.warmup_frames <= 0 or (args.warmup_frames - 1) % 8):
        parser.error("--warmup-frames must be positive and 8n+1")
    if args.runs < 1 or args.warmup_runs < 0:
        parser.error("--runs must be positive and --warmup-runs nonnegative")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("inference.py", "benchmarks/latency.py",
                         "packages/ltx-core/src/ltx_core/quantization/fp8_dynamic.py",
                         "packages/ltx-core/src/ltx_core/quantization/fp8_quantizer.py",
                         "packages/ltx-pipelines/src/ltx_pipelines/a2vid_distilled.py",
                         "packages/ltx-pipelines/src/ltx_pipelines/ar_a2vid_distilled_pipeline.py")
        },
        "commit": command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "working_tree": command_output(["git", "-C", str(ROOT), "status", "--short"]),
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {k: importlib.metadata.version(k) for k in ("torch", "torchaudio", "transformers", "safetensors", "av")},
        "cuda": torch.version.cuda,
        "torch_cpu_threads": torch.get_num_threads(),
        "sdpa_backends_enabled": {
            "flash": torch.backends.cuda.flash_sdp_enabled(),
            "memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math": torch.backends.cuda.math_sdp_enabled(),
        },
        "checkpoint_bytes": Path(args.checkpoint).stat().st_size,
        "audio_sha256": hashlib.sha256(Path(args.audio).read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest() if args.reference else None,
        "gpu": command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,power.limit", "--format=csv,noheader"]),
        "timing_method": "Synchronized wall clock. End-to-end starts before pipeline call and ends after MP4 mux. Imports and downloads excluded. Model loading included unless already resident. VAE timing measures generator next() calls. Nested timings overlap. No simulated inference.",
        "sigmas": [1.0, 0.98125, 0.909375, 0.421875, 0.0],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    precomputed_audio = torch.load(args.audio_latents, map_location="cpu", weights_only=True) if args.audio_latents else None
    if precomputed_audio is not None:
        assert precomputed_audio["audio_sha256"] == manifest["audio_sha256"], "Audio latent/source mismatch"
        manifest["audio_latents_sha256"] = hashlib.sha256(args.audio_latents.read_bytes()).hexdigest()
        manifest["audio_preprocessing"] = precomputed_audio["recipe"]
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    recorder = Recorder()
    recorder.install()
    pipeline = ARA2VidDistilledPipeline(
        distilled_checkpoint_path=args.checkpoint,
        spatial_upsampler_path=None,
        gemma_root=args.gemma_root,
        loras=[],
        quantization=build_quantization_policy(args),
    )
    images = build_first_frame_images(args.reference, strength=1.0, crf=0)
    reference_latent = encode_first_frame_channel_condition(
        pipeline, image_path=args.reference, enabled=True,
        height=args.height, width=args.width, crf=0,
    )
    tiling = TilingConfig(
        spatial_config=SpatialTilingConfig(tile_size_in_pixels=512, tile_overlap_in_pixels=64),
        temporal_config=TemporalTilingConfig(tile_size_in_frames=256, tile_overlap_in_frames=8),
    )
    results = []
    modes = [False, True] if args.cache == "both" else [args.cache == "on"]
    for cached in modes:
        for run in range(args.warmup_runs + args.runs):
            warmup = run < args.warmup_runs
            request_frames = args.warmup_frames if warmup and args.warmup_frames else args.frames
            label = f"cache-{'on' if cached else 'off'}-{'warmup' if warmup else 'measured'}-{run}"
            recorder.samples.clear()
            torch.cuda.reset_peak_memory_stats()
            logging.info("BENCHMARK START %s", label)
            sync()
            started = time.perf_counter()
            try:
                with torch.inference_mode():
                    video, audio = pipeline(
                        prompt=DEFAULT_PROMPT,
                        seed=args.seed,
                        height=args.height,
                        width=args.width,
                        num_frames=request_frames,
                        frame_rate=args.fps,
                        images=images,
                        audio_path=args.audio,
                        audio_start_time=0.0,
                        audio_max_duration=request_frames / args.fps,
                        precomputed_audio_latent=precomputed_audio["latent"] if precomputed_audio else None,
                        tiling_config=tiling,
                        enhance_prompt=False,
                        stage_mode="one-stage",
                        stage1_sigmas=manifest["sigmas"],
                        ar_video_chunk_size=4,
                        ar_history_chunk_count=1,
                        ar_sink_first_chunk=True,
                        ar_relative_positions=True,
                        ar_history_feature_cache=cached,
                        first_frame_channel_condition_mode="gated",
                        first_frame_channel_condition_init="zero",
                        first_frame_channel_condition_latent=reference_latent,
                        ar_first_frame_channel_condition_from_first_chunk=args.reference is None,
                        ar_channel_condition_current_chunk_only=True,
                        fast_infer=args.fast_infer,
                    )
                    sync()
                    pipeline_seconds = time.perf_counter() - started
                    decode_times = []
                    decoded_frames = 0
                    first_pixels_seconds = None

                    def timed_video():
                        nonlocal decoded_frames, first_pixels_seconds
                        iterator = iter(video)
                        while True:
                            sync()
                            before = time.perf_counter()
                            try:
                                chunk = next(iterator)
                            except StopIteration:
                                return
                            sync()
                            decode_times.append(time.perf_counter() - before)
                            decoded_frames += int(chunk.shape[0])
                            if first_pixels_seconds is None:
                                first_pixels_seconds = time.perf_counter() - started
                            yield chunk

                    encode_started = time.perf_counter()
                    encode_video(
                        video=timed_video(), fps=args.fps, audio=audio,
                        output_path=str(args.output_dir / f"{label}.mp4"),
                        video_chunks_number=get_video_chunks_number(request_frames, tiling),
                        crf=12, preset="fast",
                    )
                    sync()
                    total_seconds = time.perf_counter() - started
                    encode_and_decode_seconds = time.perf_counter() - encode_started
                samples = dict(recorder.samples)
                dit_seconds = sum(samples.get("dit_forward", []))
                ar_seconds = sum(samples.get("ar_sampling", []))
                vae_seconds = sum(decode_times)
                result = {
                    "label": label, "warmup": warmup, "cache": cached,
                    "fast_infer": args.fast_infer, "frames": decoded_frames,
                    "pipeline_seconds": pipeline_seconds,
                    "end_to_end_seconds": total_seconds,
                    "end_to_end_fps": decoded_frames / total_seconds,
                    "first_pixels_seconds": first_pixels_seconds,
                    "dit_forward_seconds": dit_seconds,
                    "ar_sampling_seconds": ar_seconds,
                    "vae_decode_seconds": vae_seconds,
                    "ar_plus_vae_seconds": ar_seconds + vae_seconds,
                    "ar_plus_vae_fps": decoded_frames / (ar_seconds + vae_seconds),
                    "encode_other_seconds": encode_and_decode_seconds - vae_seconds,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                    "dit_calls": len(samples.get("dit_forward", [])),
                    "dit_chunk_seconds": [
                        sum(samples["dit_forward"][i:i + 4])
                        for i in range(0, len(samples.get("dit_forward", [])), 4)
                    ],
                    "samples_seconds": samples,
                    "vae_chunks_seconds": decode_times,
                    "model": recorder.model_info,
                    "final_latent_finite": bool(torch.isfinite(pipeline.last_final_video_latent).all()),
                }
                assert decoded_frames == request_frames, (decoded_frames, request_frames)
            except Exception as exc:
                result = {"label": label, "error": str(exc), "traceback": traceback.format_exc()}
                with (args.output_dir / "results.jsonl").open("a") as f:
                    f.write(json.dumps(result) + "\n")
                raise
            results.append(result)
            with (args.output_dir / "results.jsonl").open("a") as f:
                f.write(json.dumps(result) + "\n")
            logging.info("BENCHMARK RESULT %s", json.dumps({k: v for k, v in result.items() if k not in ("samples_seconds", "model", "vae_chunks_seconds")}))
    summaries = []
    for cached in modes:
        measured = [r for r in results if r["cache"] == cached and not r["warmup"]]
        summary = {"cache": cached, "runs": len(measured), "fast_infer": args.fast_infer}
        for key in ("end_to_end_seconds", "end_to_end_fps", "first_pixels_seconds", "dit_forward_seconds", "ar_sampling_seconds", "vae_decode_seconds", "ar_plus_vae_fps", "encode_other_seconds", "peak_allocated_gib"):
            values = [r[key] for r in measured]
            summary[key] = {"median": statistics.median(values), "min": min(values), "max": max(values)}
        summaries.append(summary)
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
