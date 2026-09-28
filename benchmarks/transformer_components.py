"""Audit transformer weights and executed modules; test audio-tail ablation in memory.

Does not change the checkpoint or production implementation. The ablation checks
video-latent equality for 65 frames with ForeverCache on and off.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from inference import DEFAULT_PROMPT
from ltx_pipelines import ARA2VidDistilledPipeline
from ltx_pipelines.utils.model_ledger import ModelLedger
from util import build_tiling_config


class ZeroOutput(torch.nn.Module):
    def __init__(self, width=None):
        super().__init__()
        self.width = width

    def forward(self, x, *args, **kwargs):
        return torch.zeros((*x.shape[:-1], self.width or x.shape[-1]), device=x.device, dtype=x.dtype)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gemma-root", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = collections.Counter()
    top = collections.Counter()
    owners = collections.Counter()
    calls = collections.Counter()
    audio_inputs = []
    handles = []
    original_factory = ModelLedger.transformer

    def factory(*factory_args, **factory_kwargs):
        model = original_factory(*factory_args, **factory_kwargs)
        core = model.velocity_model
        seen = set()
        for name, tensor in list(core.named_parameters()) + list(core.named_buffers()):
            storage = tensor.untyped_storage()
            if tensor.device.type != "cuda" or storage.data_ptr() in seen:
                continue
            seen.add(storage.data_ptr())
            nbytes = storage.nbytes()
            parts = name.split(".")
            group = "blocks." + parts[2] if parts[0] == "transformer_blocks" else "outside_blocks"
            groups[group] += nbytes
            top[parts[0]] += nbytes
            owners[name.rpartition(".")[0]] += nbytes
        for name, module in core.named_modules():
            if name in owners:
                def count(module, inputs, output, name=name):
                    calls[name] += 1
                handles.append(module.register_forward_hook(count))

        def inspect_inputs(module, inputs, kwargs):
            audio = kwargs.get("audio", inputs[1] if len(inputs) > 1 else None)
            audio_inputs.append({
                "enabled": audio.enabled,
                "nonzero_timesteps": int(torch.count_nonzero(audio.timesteps)),
            })
        handles.append(model.register_forward_pre_hook(inspect_inputs, with_kwargs=True))
        return model

    ModelLedger.transformer = factory
    pipeline = ARA2VidDistilledPipeline(
        distilled_checkpoint_path=args.checkpoint, gemma_root=args.gemma_root,
        spatial_upsampler_path=None, loras=[],
    )
    tiling = build_tiling_config(argparse.Namespace(
        spatial_tile_size=512, spatial_tile_overlap=64,
        temporal_tile_size=256, temporal_tile_overlap=8,
    ))

    def generate(cache):
        video, audio = pipeline(
            prompt=DEFAULT_PROMPT, seed=42, height=512, width=768,
            num_frames=65, frame_rate=25, images=[], audio_path=args.audio,
            audio_start_time=0.0, audio_max_duration=65 / 25,
            tiling_config=tiling, enhance_prompt=False, stage_mode="one-stage",
            stage1_sigmas=[1.0, 0.98125, 0.909375, 0.421875, 0.0],
            ar_video_chunk_size=4, ar_history_chunk_count=1,
            ar_sink_first_chunk=True, ar_relative_positions=True, ar_history_feature_cache=cache,
            first_frame_channel_condition_mode="gated", first_frame_channel_condition_init="zero",
            ar_first_frame_channel_condition_from_first_chunk=True,
            ar_channel_condition_current_chunk_only=True, fast_infer=True,
        )
        del video, audio
        torch.cuda.synchronize()
        return pipeline.last_final_video_latent.clone()

    tests = []
    with torch.inference_mode():
        for cache in (True, False):
            baseline = generate(cache)
            if cache:
                uncalled = {name: size for name, size in owners.items() if not calls[name]}
                baseline_calls = dict(calls)
                baseline_audio_inputs = list(audio_inputs)
                for handle in handles:
                    handle.remove()
            core = pipeline._fast_modules.transformer.velocity_model
            final_block = core.transformer_blocks[-1]
            original_ff = final_block.audio_ff
            original_v2a = final_block.video_to_audio_attn
            original_head = core.audio_proj_out
            candidates = {
                "last_block.audio_ff": sum(p.numel() * p.element_size() for p in original_ff.parameters()),
                "last_block.video_to_audio_attn": sum(p.numel() * p.element_size() for p in original_v2a.parameters()),
                "audio_proj_out": sum(p.numel() * p.element_size() for p in original_head.parameters()),
            }
            try:
                final_block.audio_ff = ZeroOutput()
                final_block.video_to_audio_attn = ZeroOutput()
                core.audio_proj_out = ZeroOutput(original_head.out_features)
                ablated = generate(cache)
            finally:
                final_block.audio_ff = original_ff
                final_block.video_to_audio_attn = original_v2a
                core.audio_proj_out = original_head
            tests.append({
                "forevercache": cache,
                "equal": torch.equal(baseline, ablated),
                "max_abs_difference": float((baseline.float() - ablated.float()).abs().max()),
                "latent_shape": list(baseline.shape),
            })
    report = {
        "config": vars(args), "group_bytes": dict(groups), "top_level_bytes": dict(top),
        "total_bytes": sum(groups.values()), "baseline_parameter_owner_calls": baseline_calls,
        "uncalled_parameter_owners": uncalled, "audio_inputs": baseline_audio_inputs,
        "ablation_candidate_bytes": candidates, "ablation_tests": tests,
        "notes": "Model storage is deduplicated. Call hooks do not prove parameter relevance; "
        "interpret with source dataflow. Ablation replaces three computations by zeros while "
        "keeping original modules referenced for restoration. It measures equivalence, NOT memory savings. "
        "No checkpoint or production pipeline changes. 65 frames, 768x512, default prompt, seed 42, "
        "JFK audio, generated first-chunk conditioning; VAE decode is not run.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({k: report[k] for k in (
        "group_bytes", "top_level_bytes", "total_bytes", "uncalled_parameter_owners",
        "ablation_candidate_bytes", "ablation_tests",
    )}, indent=2))


if __name__ == "__main__":
    main()
