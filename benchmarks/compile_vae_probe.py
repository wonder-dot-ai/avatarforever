"""Check strict video-decoder compilation with identical latents and RNG state."""

import argparse
import json
import time

import torch
from ltx_pipelines.utils.model_ledger import ModelLedger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--latent", required=True)
    args = parser.parse_args()
    with torch.inference_mode():
        decoder = ModelLedger(
            dtype=torch.bfloat16, device=torch.device("cuda"), checkpoint_path=args.checkpoint
        ).video_decoder()
        latent = torch.load(args.latent, map_location="cuda", weights_only=True)[:, :, :4, :16, :16].contiguous()
        generator = torch.Generator(device="cuda").manual_seed(42)
        state = generator.get_state()
        expected = decoder(latent, generator=generator)
        eager_rng_state = generator.get_state()
        torch.cuda.synchronize()
        compiled = torch.compile(
            decoder.forward, fullgraph=True, dynamic=False, options={"emulate_precision_casts": True}
        )
        generator.set_state(state)
        start = time.perf_counter()
        actual = compiled(latent, generator=generator)
        torch.cuda.synchronize()
        compiled_seconds = time.perf_counter() - start
        error = actual.float() - expected.float()
        print(json.dumps({
            "first_compiled_seconds": compiled_seconds,
            "max_abs_difference": float(error.abs().max()),
            "relative_rms_error": float(error.square().mean().sqrt() / expected.float().square().mean().sqrt()),
            "rng_state_identical": bool(torch.equal(eager_rng_state, generator.get_state())),
            "finite": bool(torch.isfinite(actual).all()),
        }))


if __name__ == "__main__":
    main()
