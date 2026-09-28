"""Encode a long speech once in overlapping windows for identical conditioning."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from ltx_core.model.audio_vae import encode_audio
from ltx_core.types import Audio, AudioLatentShape
from ltx_pipelines.utils.media_io import decode_audio_from_file, ensure_stereo_audio
from ltx_pipelines.utils.model_ledger import ModelLedger


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--duration', type=float, default=1260.2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audio = ensure_stereo_audio(decode_audio_from_file(str(args.audio), torch.device('cpu'), 0, args.duration))
    assert audio.waveform.shape[-1] / audio.sampling_rate >= args.duration - .05
    encoder = ModelLedger(dtype=torch.bfloat16, device=torch.device('cuda'), checkpoint_path=args.checkpoint).audio_encoder()
    target = AudioLatentShape.from_duration(batch=1, duration=args.duration, channels=8, mel_bins=16).frames
    rate, window, context = 25, 750, 50
    pieces = []
    for start in range(0, target, window):
        stop = min(start + window, target)
        left, right = max(0, start-context), min(target, stop+context)
        sample_start = round(left / rate * audio.sampling_rate)
        sample_stop = round(right / rate * audio.sampling_rate)
        chunk = Audio(waveform=audio.waveform[..., sample_start:sample_stop].cuda(), sampling_rate=audio.sampling_rate)
        latent = encode_audio(chunk, encoder)
        selected = latent[:, :, start-left:stop-left].cpu().contiguous()
        assert selected.shape[2] == stop-start, (start, selected.shape, latent.shape)
        pieces.append(selected)
        print(f'Encoded {stop/rate:.2f} / {target/rate:.2f} seconds', flush=True)
    latent = torch.cat(pieces, dim=2)
    assert torch.isfinite(latent).all() and latent.shape[2] == target
    recipe = {'window_seconds':30, 'context_each_side_seconds':2, 'latent_rate':rate,
              'duration_seconds':args.duration, 'latent_shape':list(latent.shape),
              'note':'Overlapping-window audio encoding, shared unchanged by all modes. Video AR history never resets.',
              'source_url':'https://commons.wikimedia.org/wiki/File:Jfk_American_University_4654_06-10-63.ogg',
              'license':'Public domain, US federal government recording',
              'speech':'JFK American University address, June 10 1963; first 21 minutes'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'latent':latent, 'audio_sha256':hashlib.sha256(args.audio.read_bytes()).hexdigest(), 'recipe':recipe}, args.output)
    args.output.with_suffix('.json').write_text(json.dumps(recipe,indent=2)+'\n')


if __name__ == '__main__':
    main()
