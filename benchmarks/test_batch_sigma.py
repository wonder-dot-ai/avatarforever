"""CPU regression: batched prompt/cross-modal timesteps match independent requests."""
from dataclasses import replace
import unittest

import torch

from ltx_core.model.transformer.model import LTXModel
from ltx_core.model.transformer.ar_feature_cache import ARFeatureCache
from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.types import LatentState
from ltx_pipelines.utils.helpers import modality_from_latent_state


class BatchSigmaTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = LTXModel(
            num_attention_heads=2, attention_head_dim=32, in_channels=8, out_channels=8,
            num_layers=1, cross_attention_dim=64, audio_num_attention_heads=2,
            audio_attention_head_dim=32, audio_in_channels=8, audio_out_channels=8,
            audio_cross_attention_dim=64, cross_attention_adaln=True,
            apply_gated_attention=True,
        ).eval()
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.uniform_(-0.1, 0.1)

    def modality(self, axes, sigma):
        latent = torch.randn(2, 3, 8)
        mask = torch.tensor([[[0.], [1.], [1.]], [[1.], [0.], [1.]]])
        starts = torch.arange(3).float().reshape(1, 1, 3).expand(2, axes, 3)
        state = LatentState(latent, mask, torch.stack((starts, starts + 1), -1), latent.clone())
        return modality_from_latent_state(state, torch.randn(2, 4, 64), sigma)

    @torch.inference_mode()
    def check_sigma(self, sigma):
        video, audio = self.modality(3, sigma), self.modality(1, sigma)
        expected = sigma.reshape(-1).expand(2)
        torch.testing.assert_close(video.sigma, expected)
        torch.testing.assert_close(video.timesteps[:, 2, 0], expected)
        self.assertEqual(video.timesteps[0, 0, 0], 0)
        self.assertEqual(video.timesteps[1, 1, 0], 0)
        together = self.model(video, audio, None)
        for index in range(2):
            def row(modality):
                return replace(modality, **{name: getattr(modality, name)[index:index+1]
                    for name in ('latent', 'sigma', 'timesteps', 'positions', 'context')})
            separate = self.model(row(video), row(audio), None)
            for batched, single in zip(together, separate):
                torch.testing.assert_close(batched[index:index+1], single, atol=1e-5, rtol=1e-4)
        # Strict graph capture of the two preprocessing paths that previously failed.
        for prep, own, other in ((self.model.video_args_preprocessor, video, audio),
                                 (self.model.audio_args_preprocessor, audio, video)):
            compiled = torch.compile(prep.prepare, backend='eager', fullgraph=True)
            torch.testing.assert_close(compiled(own, other).x, prep.prepare(own, other).x)

    def test_shared_scalar(self):
        self.check_sigma(torch.tensor(0.75))

    def test_per_request_values(self):
        self.check_sigma(torch.tensor([0.25, 0.75]))

    def test_reject_mismatched_batch(self):
        with self.assertRaisesRegex(ValueError, 'one value per sample'):
            self.modality(3, torch.tensor([0.25, 0.5, 0.75]))

    @torch.inference_mode()
    def test_cached_history_matches_independent_requests(self):
        # Exercise both positional-embedding layouts and populate/reuse, which
        # ordinary uncached batch tests do not cover.
        for rope_type in LTXRopeType:
            self.model.rope_type = rope_type
            for prep in (self.model.video_args_preprocessor, self.model.audio_args_preprocessor):
                prep.simple_preprocessor.rope_type = rope_type
            for block in self.model.transformer_blocks:
                for module in block.modules():
                    if hasattr(module, 'rope_type'):
                        module.rope_type = rope_type
            video, audio = self.modality(3, torch.tensor(.75)), self.modality(1, torch.tensor(.75))
            caches = [ARFeatureCache() for _ in range(3)]
            for phase in ('populate', 'reuse'):
                def attach(modality, cache, row=None):
                    values = {name: getattr(modality, name)[row:row+1] for name in
                              ('latent', 'sigma', 'timesteps', 'positions', 'context')} if row is not None else {}
                    return replace(modality, **values, ar_feature_cache=cache, ar_current_slice=slice(1, 3))
                together = self.model(attach(video, caches[0]), attach(audio, caches[0]), None)
                for index in range(2):
                    separate = self.model(attach(video, caches[index+1], index),
                                          attach(audio, caches[index+1], index), None)
                    for batched, single in zip(together, separate):
                        torch.testing.assert_close(batched[index:index+1], single, atol=1e-5, rtol=1e-4,
                                                   msg=f'{rope_type} {phase} request {index}')


if __name__ == '__main__':
    unittest.main()
