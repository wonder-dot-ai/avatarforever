"""Diagnostic launcher for strict transformer compilation; benchmark args pass through."""
import os

from latency import main
from ltx_pipelines.utils.model_ledger import ModelLedger

original = ModelLedger.transformer


def build(self, *args, **kwargs):
    model = original(self, *args, **kwargs)
    backend = os.environ.get("AVATAR_COMPILE_BACKEND", "inductor")
    scope = os.environ.get("AVATAR_COMPILE_SCOPE", "full")
    if self.transformer_compile != "none":
        raise ValueError("This diagnostic installs its own compiler; omit --compile-transformer")
    if scope == "full":
        model.velocity_model.compile(backend=backend, fullgraph=True, dynamic=False)
    elif scope == "blocks":
        for block in model.velocity_model.transformer_blocks:
            block.compile(backend=backend, fullgraph=True, dynamic=False)
    else:
        raise ValueError(f"Unknown diagnostic scope: {scope}")
    return model


if __name__ == "__main__":
    ModelLedger.transformer = build
    main()
