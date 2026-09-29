"""Opt-in strict compilation of diffusion-transformer compute after weight loading."""

import logging

import torch

logger = logging.getLogger(__name__)


def configure_compile_variant_limit() -> None:
    """Allow AR/cache and VAE tile variants without enabling eager fallback."""
    limit_key = "recompile_limit" if hasattr(torch._dynamo.config, "recompile_limit") else "cache_size_limit"
    setattr(torch._dynamo.config, limit_key, max(getattr(torch._dynamo.config, limit_key), 32))


def compile_transformer(model: torch.nn.Module, scope: str = "none") -> torch.nn.Module:
    """Keep FP8 storage while compiling casts, attention, feed-forward and norms.

    Regional mode also compiles input preparation and output projections. The
    Python block/cache dispatcher stays eager; every compute region is fullgraph.
    Static AR shapes/strides need more than Dynamo's default eight variants, so
    opting in raises the process-wide per-code recompile limit to at least 32.
    This does not suppress errors or enable eager fallback on graph breaks.
    Preserve intermediate low-precision rounding: unrestricted fusion changed
    this model's BF16 outputs substantially in fixed-input comparisons.
    """
    if scope == "none":
        return model
    if scope not in ("full", "blocks", "regional"):
        raise ValueError(f"Unknown transformer compile scope: {scope}")
    configure_compile_variant_limit()
    velocity = model.velocity_model
    options = {"emulate_precision_casts": True}
    if scope == "full":
        velocity.compile(fullgraph=True, dynamic=False, options=options)
    else:
        for block in velocity.transformer_blocks:
            block.compile(fullgraph=True, dynamic=False, options=options)
        if scope == "regional":
            for preprocessor in (velocity.video_args_preprocessor, velocity.audio_args_preprocessor):
                preprocessor.prepare = torch.compile(
                    preprocessor.prepare, fullgraph=True, dynamic=False, options=options
                )
            velocity._process_output = torch.compile(
                velocity._process_output, fullgraph=True, dynamic=False, options=options
            )
    logger.info("Enabled strict torch.compile for transformer (%s)", scope)
    return model
