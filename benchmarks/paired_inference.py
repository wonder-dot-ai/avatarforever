"""Actual two-request AR inference: alternating chunks versus batch size two.

Preparation uses the existing pipeline unchanged. A scoped diagnostic hook
captures its initialized sampler state before generation. Both modes then run
the same original AR sampler, with separate batch rows for each request.
This tests synchronized, equal-length requests, not a deployed serving API.
VAE runs on each complete output: decode throughput is measured, streaming
first-pixel latency is NOT established by this benchmark.
"""
from __future__ import annotations

import argparse
from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import ltx_pipelines.a2vid_distilled as a2v
from inference import DEFAULT_PROMPT, build_quantization_policy
from ltx_core.components.patchifiers import VideoLatentPatchifier
from ltx_core.model.video_vae import decode_video, get_video_chunks_number, SpatialTilingConfig, TemporalTilingConfig, TilingConfig
from ltx_core.types import Audio, LatentState, VideoLatentShape, VideoPixelShape
from ltx_pipelines import ARA2VidDistilledPipeline
from ltx_pipelines.utils.autoregressive import autoregressive_euler_denoising_loop, build_progressive_chunk_ranges
from ltx_pipelines.utils.helpers import simple_denoising_func
from ltx_pipelines.utils.media_io import decode_audio_from_file, encode_video
from util import build_first_frame_images, encode_first_frame_channel_condition


def stack_states(states: list[LatentState]) -> LatentState:
    values = {}
    for field in fields(states[0]):
        items = [getattr(state, field.name) for state in states]
        if isinstance(items[0], torch.Tensor):
            if any(not isinstance(x, torch.Tensor) or x.shape[0] != 1 or x.shape[1:] != items[0].shape[1:] for x in items):
                raise ValueError(f'Incompatible batch field {field.name}')
            values[field.name] = torch.cat(items, dim=0)
        else:
            if any(x != items[0] for x in items):
                raise ValueError(f'Incompatible scalar field {field.name}')
            values[field.name] = items[0]
    return LatentState(**values)


class Prepared(Exception):
    pass


def prepare(pipeline, args, seed, audio_start, reference_latent, tiling):
    captured = {}
    original_loop = a2v.autoregressive_euler_denoising_loop
    original_builder = a2v.simple_denoising_func

    def context_builder(video_context, audio_context, transformer):
        captured['video_context'] = video_context
        captured['audio_context'] = audio_context
        captured['transformer'] = transformer
        return original_builder(video_context, audio_context, transformer)

    def capture_loop(**kwargs):
        kwargs['denoise_fn_builder']()
        captured['sampler'] = kwargs
        raise Prepared()

    a2v.simple_denoising_func = context_builder
    a2v.autoregressive_euler_denoising_loop = capture_loop
    try:
        pipeline(prompt=DEFAULT_PROMPT, seed=seed, height=512, width=768,
                 num_frames=args.frames, frame_rate=25,
                 images=build_first_frame_images(args.reference, strength=1.0, crf=0),
                 audio_path=args.audio, audio_start_time=audio_start,
                 audio_max_duration=args.frames / 25, tiling_config=tiling,
                 enhance_prompt=False, stage_mode='one-stage',
                 stage1_sigmas=[1.0, .98125, .909375, .421875, 0.0],
                 ar_video_chunk_size=4, ar_history_chunk_count=1,
                 ar_sink_first_chunk=True, ar_relative_positions=True, ar_history_feature_cache=True,
                 first_frame_channel_condition_mode='gated', first_frame_channel_condition_init='zero',
                 first_frame_channel_condition_latent=reference_latent,
                 ar_first_frame_channel_condition_from_first_chunk=False,
                 ar_channel_condition_current_chunk_only=True, fast_infer=True)
    except Prepared:
        pass
    finally:
        a2v.simple_denoising_func = original_builder
        a2v.autoregressive_euler_denoising_loop = original_loop
    if 'sampler' not in captured:
        raise RuntimeError('Pipeline did not reach the AR preparation hook')
    captured['seed'] = seed
    captured['audio_start'] = audio_start
    return captured


def paired_sampler(requests):
    first = requests[0]['sampler']
    result = dict(first)
    for name in ('video_state', 'audio_state'):
        result[name] = stack_states([r['sampler'][name] for r in requests])
    video_context = torch.cat([r['video_context'] for r in requests])
    audio_context = torch.cat([r['audio_context'] for r in requests])
    transformer = requests[0]['transformer']
    assert all(r['transformer'] is transformer for r in requests)
    result['denoise_fn_builder'] = lambda: simple_denoising_func(video_context, audio_context, transformer)
    return result


def generate(samplers):
    states = [dict(s, video_state=s['video_state'].clone(), audio_state=s['audio_state'].clone()) for s in samplers]
    first = states[0]
    ranges = build_progressive_chunk_ranges(first['total_video_latent_frames'], first['video_chunk_size'])
    pair_times = []
    torch.cuda.reset_peak_memory_stats()
    for index in range(len(ranges)):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for state in states:
            video, audio = autoregressive_euler_denoising_loop(**state, start_chunk_idx=index, end_chunk_idx=index + 1)
            state['video_state'], state['audio_state'] = video, audio
        torch.cuda.synchronize()
        pair_times.append(time.perf_counter() - start)
    return states, {'pair_chunk_seconds': pair_times,
                    'steady_pair_chunk_median_seconds': statistics.median(pair_times[2:-1]),
                    'steady_pair_chunk_max_seconds': max(pair_times[2:-1]),
                    'generation_seconds': sum(pair_times),
                    'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                    'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30}


def unpack(states, pipeline, frames):
    shape = VideoLatentShape.from_pixel_shape(
        VideoPixelShape(batch=1, frames=frames, height=512, width=768, fps=25),
        latent_channels=pipeline.pipeline_components.video_latent_channels,
        scale_factors=pipeline.pipeline_components.video_scale_factors)
    patchifier = VideoLatentPatchifier(1)
    count = patchifier.get_token_count(shape)
    outputs = []
    for item in states:
        state = item['video_state']
        begin = state.conditioning_prefix_token_count
        for row in state.latent:
            outputs.append(patchifier.unpatchify(row[None, begin:begin + count], shape))
    return outputs


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--gemma-root', required=True)
    parser.add_argument('--audio', required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--quantization', choices=['none', 'fp8-cast', 'fp8-dynamic'], default='fp8-cast')
    parser.add_argument('--fp8-activation-backend', default='compiled')
    parser.add_argument('--frames', type=int, default=257)
    parser.add_argument('--runs', type=int, default=3)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = ARA2VidDistilledPipeline(distilled_checkpoint_path=args.checkpoint,
        spatial_upsampler_path=None, gemma_root=args.gemma_root, loras=[],
        quantization=build_quantization_policy(args), transformer_compile='regional', compile_video_decoder=True)
    tiling = TilingConfig(spatial_config=SpatialTilingConfig(tile_size_in_pixels=512, tile_overlap_in_pixels=64),
                         temporal_config=TemporalTilingConfig(tile_size_in_frames=256, tile_overlap_in_frames=8))
    ref = encode_first_frame_channel_condition(pipeline, image_path=args.reference, enabled=True,
                                               height=512, width=768, crf=0)
    requests = [prepare(pipeline, args, 42 + i, 15.0 * i, ref, tiling) for i in range(2)]
    # These modules are no longer needed during generation. Keep transformer and
    # decoder resident, sharing one copy across both request states.
    for name in ('text_encoder', 'embeddings_processor', 'audio_encoder', 'video_encoder'):
        getattr(pipeline._fast_modules, name).to('cpu')
    torch.cuda.empty_cache()
    paired = paired_sampler(requests)
    configs = {'alternating': [r['sampler'] for r in requests], 'batched': [paired]}
    records = []
    latents = {}
    output = args.output_dir
    (output / 'manifest.json').write_text(json.dumps({
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'requests': [{'seed': r['seed'], 'audio_start_seconds': r['audio_start']} for r in requests],
        'scope': 'Two independent AR states, same reference/prompt, different seeds/audio. Equal lengths and synchronized arrivals. Full-clip VAE; no claim of streaming decode.'}, indent=2) + '\n')
    # Warm every shape in each mode before collecting any timed comparison.
    for mode, samplers in configs.items():
        print('WARMUP', mode, flush=True)
        states, _ = generate(samplers)
        for latent in unpack(states, pipeline, args.frames):
            for _ in decode_video(latent, pipeline._fast_modules.video_decoder, tiling):
                pass
        del states
    for repeat in range(args.runs):
        for mode in (('alternating', 'batched') if repeat % 2 == 0 else ('batched', 'alternating')):
            states, record = generate(configs[mode])
            latent_outputs = unpack(states, pipeline, args.frames)
            decode_seconds = []
            for latent in latent_outputs:
                torch.cuda.synchronize()
                before = time.perf_counter()
                decoded_count = 0
                for chunk in decode_video(latent, pipeline._fast_modules.video_decoder, tiling):
                    decoded_count += chunk.shape[0]
                torch.cuda.synchronize()
                decode_seconds.append(time.perf_counter() - before)
                assert decoded_count == args.frames
            record.update(mode=mode, repeat=repeat, decode_seconds=decode_seconds,
                          finite=all(bool(torch.isfinite(x).all()) for x in latent_outputs))
            record['steady_pair_with_average_decode_seconds'] = record['steady_pair_chunk_median_seconds'] + sum(decode_seconds) * 32 / args.frames
            record['per_request_fps_estimate'] = 32 / record['steady_pair_with_average_decode_seconds']
            records.append(record)
            with (output / 'results.jsonl').open('a') as file:
                file.write(json.dumps(record) + '\n')
            print('MEASURED', json.dumps(record), flush=True)
            latents[mode] = [x.cpu() for x in latent_outputs]
            del states, latent_outputs
    numerics = []
    for a, b in zip(latents['alternating'], latents['batched']):
        a, b = a.float(), b.float()
        numerics.append({'relative_rms': float((a-b).square().mean().sqrt()/a.square().mean().sqrt()),
                         'max_abs': float((a-b).abs().max())})
    for mode, items in latents.items():
        for index, latent in enumerate(items):
            torch.save(latent, output / f'{mode}-{index}.pt')
            audio = decode_audio_from_file(args.audio, pipeline.device, 15.0 * index, args.frames / 25)
            audio = Audio(waveform=audio.waveform.squeeze(0), sampling_rate=audio.sampling_rate)
            encode_video(video=decode_video(latent.cuda(), pipeline._fast_modules.video_decoder, tiling),
                         fps=25, audio=audio, output_path=str(output / f'{mode}-{index}.mp4'),
                         video_chunks_number=get_video_chunks_number(args.frames, tiling),
                         crf=12, preset='fast')
    summary = {'records': records, 'batched_vs_alternating_latent_error': numerics,
               'target_pair_seconds': 1.28, 'decoder_note': 'Full-clip decoding average; streaming not measured.',
               'dynamo': {k: dict(torch._dynamo.utils.counters[k]) for k in ('stats', 'graph_break', 'unimplemented')}}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
