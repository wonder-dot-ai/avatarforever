import torch

from ltx_core.conditioning.exceptions import ConditioningError
from ltx_core.conditioning.item import ConditioningItem
from ltx_core.tools import VideoLatentTools
from ltx_core.types import LatentState


class VideoConditionByGlobalLatentPrefix(ConditioningItem):
    """
    Conditions video generation by prepending clean reference latent tokens.

    Unlike standard image conditioning, this does not replace noisy target tokens.
    The reference tokens are inserted before the generated video sequence, use
    zero positions by default, and are removed by ``LatentTools.clear_conditioning``
    before unpatchifying/decoding the generated video.
    """

    def __init__(
        self,
        latent: torch.Tensor,
        strength: float = 1.0,
        *,
        repeat_to_target_frames: bool = False,
        position_mode: str = "zero",
    ):
        self.latent = latent
        self.strength = strength
        self.repeat_to_target_frames = repeat_to_target_frames
        self.position_mode = position_mode

    def apply_to(
        self,
        latent_state: LatentState,
        latent_tools: VideoLatentTools,
    ) -> LatentState:
        cond_batch, cond_channels, cond_frames, cond_height, cond_width = self.latent.shape
        tgt_batch, tgt_channels, tgt_frames, tgt_height, tgt_width = latent_tools.target_shape.to_torch_shape()

        if (cond_batch, cond_channels, cond_height, cond_width) != (tgt_batch, tgt_channels, tgt_height, tgt_width):
            raise ConditioningError(
                f"Can't apply global prefix conditioning item to latent with shape {latent_tools.target_shape}, "
                f"expected shape is ({tgt_batch}, {tgt_channels}, F, {tgt_height}, {tgt_width}). Make sure the "
                "reference image/video latent and target latent have the same batch, channel, and spatial shape."
            )

        latent = self._expand_latent_to_target_frames(
            target_frames=tgt_frames,
            cond_frames=cond_frames,
        )
        tokens = latent_tools.patchifier.patchify(latent)
        denoise_mask = torch.full(
            size=(*tokens.shape[:2], 1),
            fill_value=1.0 - self.strength,
            device=latent.device,
            dtype=latent_state.denoise_mask.dtype,
        )
        positions = self._build_positions(
            latent_state=latent_state,
            latent_tools=latent_tools,
            num_tokens=tokens.shape[1],
        )

        return LatentState(
            latent=torch.cat([tokens, latent_state.latent], dim=1),
            denoise_mask=torch.cat([denoise_mask, latent_state.denoise_mask], dim=1),
            positions=torch.cat([positions, latent_state.positions], dim=2),
            clean_latent=torch.cat([tokens, latent_state.clean_latent], dim=1),
            attention_mask=self._prepend_attention_mask(
                latent_state=latent_state,
                num_prefix_tokens=tokens.shape[1],
                batch_size=tokens.shape[0],
                device=latent.device,
                dtype=latent_state.denoise_mask.dtype,
            ),
            conditioning_prefix_token_count=latent_state.conditioning_prefix_token_count + tokens.shape[1],
        )

    def _expand_latent_to_target_frames(self, *, target_frames: int, cond_frames: int) -> torch.Tensor:
        if not self.repeat_to_target_frames:
            return self.latent
        if cond_frames == target_frames:
            return self.latent
        if cond_frames != 1:
            raise ConditioningError(
                "repeat_to_target_frames=True expects a single-frame global latent or a latent that already matches "
                f"target frames. Got {cond_frames} conditioning frames for target {target_frames} frames."
            )
        return self.latent.repeat(1, 1, target_frames, 1, 1)

    def _build_positions(
        self,
        *,
        latent_state: LatentState,
        latent_tools: VideoLatentTools,
        num_tokens: int,
    ) -> torch.Tensor:
        if self.position_mode == "zero":
            return latent_state.positions.new_zeros(
                (
                    latent_state.latent.shape[0],
                    latent_state.positions.shape[1],
                    num_tokens,
                    latent_state.positions.shape[-1],
                )
            )
        if self.position_mode == "match_target":
            start_token = latent_state.conditioning_prefix_token_count
            stop_token = start_token + latent_tools.target_shape.token_count()
            target_positions = latent_state.positions[:, :, start_token:stop_token]
            if target_positions.shape[2] != num_tokens:
                raise ConditioningError(
                    "Global prefix position_mode='match_target' requires conditioning token count to match target "
                    f"token count. Got {num_tokens} conditioning tokens and {target_positions.shape[2]} target tokens."
                )
            return target_positions.clone()
        raise ConditioningError(f"Unsupported global prefix position_mode: {self.position_mode!r}")

    @staticmethod
    def _prepend_attention_mask(
        *,
        latent_state: LatentState,
        num_prefix_tokens: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if latent_state.attention_mask is None:
            return None

        existing_tokens = latent_state.latent.shape[1]
        total_tokens = num_prefix_tokens + existing_tokens
        attention_mask = torch.ones((batch_size, total_tokens, total_tokens), device=device, dtype=dtype)
        attention_mask[:, num_prefix_tokens:, num_prefix_tokens:] = latent_state.attention_mask
        return attention_mask
