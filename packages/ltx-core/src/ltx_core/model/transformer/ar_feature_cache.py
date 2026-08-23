from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class ARBlockFeatureCache:
    """Approximate per-block history features for experimental AR inference caching.

    These tensors intentionally cache intermediate history features rather than
    exact K/V states. This is an approximation because bidirectional attention
    would normally let history tokens change when the current chunk changes.
    """

    video_self_history: torch.Tensor | None = None
    video_self_history_pe: tuple[torch.Tensor, torch.Tensor] | None = None
    video_self_current_context_mask: torch.Tensor | None = None
    video_cross_history: torch.Tensor | None = None
    video_cross_history_pe: tuple[torch.Tensor, torch.Tensor] | None = None

    audio_self_history: torch.Tensor | None = None
    audio_self_history_pe: tuple[torch.Tensor, torch.Tensor] | None = None
    audio_self_current_context_mask: torch.Tensor | None = None
    audio_cross_history: torch.Tensor | None = None
    audio_cross_history_pe: tuple[torch.Tensor, torch.Tensor] | None = None


@dataclass
class ARFeatureCache:
    """Chunk-local cache populated by the first denoising step and reused later."""

    block_caches: list[ARBlockFeatureCache] = field(default_factory=list)
    populated: bool = False
