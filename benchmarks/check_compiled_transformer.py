"""Compare regional compilation against eager on identical real AR inputs.

Pass the usual latency.py arguments, including --compile-transformer regional.
The timings of this diagnostic are not performance measurements.
"""

import copy
import json
import os
from pathlib import Path

import torch
from latency import main
from ltx_pipelines.utils.model_ledger import ModelLedger

original_build = ModelLedger.transformer


def metrics(actual, expected):
    if actual is None:
        return None
    a, b = actual.float(), expected.float()
    error = a - b
    return {
        "shape": list(a.shape),
        "finite": bool(torch.isfinite(a).all()),
        "max_abs_difference": float(error.abs().max()),
        "relative_rms_error": float(error.square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-12)),
        "cosine_similarity": float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)),
    }


def build(self, *args, **kwargs):
    model = original_build(self, *args, **kwargs)
    if self.transformer_compile != "regional":
        raise ValueError("This diagnostic requires --compile-transformer regional")
    velocity = model.velocity_model
    forward = velocity.forward
    blocks = list(velocity.transformer_blocks)
    compiled_blocks = [block._compiled_call_impl for block in blocks]
    preprocessors = [velocity.video_args_preprocessor, velocity.audio_args_preprocessor]
    prepared = [preprocessor.prepare for preprocessor in preprocessors]
    output = velocity._process_output
    records = []
    calls = 0
    destination = Path(os.environ.get("AVATAR_NUMERICS_OUTPUT", "compiled-transformer-numerics.json"))

    def checked(*forward_args, **forward_kwargs):
        nonlocal calls
        index = calls
        calls += 1
        if index not in (0, 1, 4, 5, 8, 9, 32, 33):
            return forward(*forward_args, **forward_kwargs)
        # Clone the whole argument tree together to preserve shared cache aliases.
        cloned_args, cloned_kwargs = copy.deepcopy((forward_args, forward_kwargs))
        try:
            for block in blocks:
                block._compiled_call_impl = None
            for preprocessor, compiled in zip(preprocessors, prepared, strict=True):
                preprocessor.prepare = compiled._torchdynamo_orig_callable
            velocity._process_output = output._torchdynamo_orig_callable
            expected = forward(*cloned_args, **cloned_kwargs)
        finally:
            for block, compiled in zip(blocks, compiled_blocks, strict=True):
                block._compiled_call_impl = compiled
            for preprocessor, compiled in zip(preprocessors, prepared, strict=True):
                preprocessor.prepare = compiled
            velocity._process_output = output
        actual = forward(*forward_args, **forward_kwargs)
        record = {"call": index, "video": metrics(actual[0], expected[0]), "audio": metrics(actual[1], expected[1])}
        records.append(record)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(records, indent=2) + "\n")
        print("FIXED_INPUT_CHECK " + json.dumps(record), flush=True)
        return actual

    velocity.forward = checked
    return model


if __name__ == "__main__":
    ModelLedger.transformer = build
    main()
