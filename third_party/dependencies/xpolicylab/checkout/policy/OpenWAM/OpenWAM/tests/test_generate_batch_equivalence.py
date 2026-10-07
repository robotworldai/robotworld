"""generate_batch correctness: B=1 parity with generate, and batch-row independence.

The batch path exists for multi-env eval (RoboDojo ``eval_batch``): N envs'
observations are collated into one batched denoising loop. Its contract is
that sample i of a batched call matches a B=1 ``generate`` call with the same
conditions. These tests pin that contract on a deterministic CPU stub video
backbone driving the REAL dual_system joint self-attention path (real
ActionDiT, real MoT driver, real collation and denoising loop).
"""

import numpy as np
import pytest
import torch
from torch import nn


def _prompt_seed(prompt: str) -> int:
    # Process-stable deterministic seed from the prompt (str hash is salted).
    return sum(ord(c) for c in prompt) % (2**31 - 1)


class _DeterministicStubVideoBackbone(nn.Module):
    """Deterministic video backbone stub with a deploy preprocess surface.

    ``preprocess_input_for_inference`` builds per-sample conditions purely from
    (prompt, seed): latent noise from ``seed`` (mirroring Wan's
    ``build_deploy_noise``) and text context from a prompt-derived seed
    (mirroring the prompt-keyed embed cache). ``prepare``/attention hooks are
    fixed linear maps so outputs are reproducible and batch rows stay
    independent (attention mixes only the sequence axis).
    """

    num_layers = 2

    def __init__(self, dim=32, num_heads=4, text_dim=16, latent_channels=4):
        super().__init__()
        from tests.test_openwam_trainer import _MockScheduler

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.text_dim = text_dim
        self.latent_channels = latent_channels
        self.scheduler = _MockScheduler()
        self._device = torch.device("cpu")
        self._dtype = torch.float32
        self.submodule_names = []

        g = torch.Generator().manual_seed(0)
        self.register_buffer("_in_proj", torch.randn(latent_channels, dim, generator=g) * 0.2)
        self.register_buffer("_ctx_proj", torch.randn(text_dim, dim, generator=g) * 0.2)
        self.register_buffer("_out_proj", torch.randn(dim, latent_channels, generator=g) * 0.2)

    @property
    def video_attention_mask_mode(self):
        return "bidirectional"

    @video_attention_mask_mode.setter
    def video_attention_mask_mode(self, mode):
        del mode

    def set_dtype_device(self, dtype, device):
        self._dtype = dtype
        self._device = device

    def build_video_to_video_mask(self, video_seq_len, video_tokens_per_frame, device):
        del video_tokens_per_frame
        return torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)

    # --- deploy preprocess surface (consumed by generate / generate_batch) ---

    def preprocess_input_for_inference(
        self,
        *,
        prompt,
        vace_video=None,
        first_frame_image=None,
        num_frames=5,
        height=8,
        width=8,
        seed=42,
        num_inference_steps=4,
        shift=5.0,
        tiled=True,
        vace_cache=None,
        prompt_embed_cache=None,
        **kw,
    ) -> dict:
        del vace_video, first_frame_image, vace_cache, kw
        g_noise = torch.Generator().manual_seed(int(seed))
        noise = torch.randn(1, self.latent_channels, 2, 2, 2, generator=g_noise)

        if prompt_embed_cache is not None and prompt in prompt_embed_cache:
            context, seq_lens = prompt_embed_cache[prompt]
        else:
            g_ctx = torch.Generator().manual_seed(_prompt_seed(prompt))
            context = torch.randn(1, 4, self.text_dim, generator=g_ctx)
            seq_lens = torch.tensor([4], dtype=torch.long)
            if prompt_embed_cache is not None:
                prompt_embed_cache[prompt] = (context, seq_lens)

        return {
            "latents": noise,
            "noise": noise,
            "context": context,
            "seq_lens": seq_lens,
            "first_frame_latents": None,
            "prompt": prompt,
            "seed": seed,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "tiled": tiled,
        }

    # --- block-loop surface (consumed by dual_system forward + MoT driver) ---

    def prepare(self, **kw):
        from openwam.model.video_backbone.base import BlockLoopState

        latents = kw["latents"]
        timestep = kw["timestep"]
        context = kw["context"]
        context_mask = kw.get("context_mask")
        B, _, f, h, w = latents.shape
        s = f * h * w
        x = latents.reshape(B, self.latent_channels, s).transpose(1, 2) @ self._in_proj
        t_mod = (timestep.view(B, 1, 1, 1) * 1e-3).expand(B, s, 6, self.dim).contiguous()
        freqs = torch.polar(
            torch.ones(s, 1, self.head_dim // 2),
            torch.zeros(s, 1, self.head_dim // 2),
        )
        vctx = context @ self._ctx_proj
        if context_mask is None:
            context_mask = torch.ones(B, context.shape[1], dtype=torch.bool)
        return BlockLoopState(
            hidden_states=x,
            time_mod=t_mod,
            rope_freqs=freqs,
            context=vctx,
            context_mask=context_mask,
            grid_frames=f,
            grid_height=h,
            grid_width=w,
        )

    def pre_attn_at_layer(self, layer_id, state):
        del layer_id
        return state.hidden_states, state.hidden_states, state.hidden_states, {"residual": state.hidden_states}

    def post_attn_at_layer(self, layer_id, state, attn_out, post_state):
        del layer_id
        state.hidden_states = post_state["residual"] + attn_out
        return state

    def run_block(self, block_id, state):
        del block_id
        state.hidden_states = state.hidden_states + 1
        return state

    def finalize(self, state):
        B = state.hidden_states.shape[0]
        out = state.hidden_states @ self._out_proj
        return out.transpose(1, 2).reshape(
            B, self.latent_channels, state.grid_frames, state.grid_height, state.grid_width
        )

    def decode_video(self, latents, *, tiled=True):
        del latents, tiled
        return None


STATE_DIM = 9
ACTION_DIM = 7


def _make_arch():
    from openwam.model import build_architecture

    cfg = {
        "framework": "dual_system",
        "variant": "joint_self_attn",
        "action_dim": ACTION_DIM,
        "dim": 32,
        "ffn_dim": 64,
        "num_heads": 4,
        "video_dim": 32,
        "text_dim": 16,
        "bridge_layers": (0, 1),
        "use_proprioception": True,
        "state_dim": STATE_DIM,
    }
    arch = build_architecture("dual_system_self_attn", cfg)
    arch.video_backbone = _DeterministicStubVideoBackbone(dim=32, num_heads=4, text_dim=16)
    arch.build_mot_driver()
    arch._device = torch.device("cpu")
    arch._dtype = torch.float32
    arch.eval()
    return arch


_SCHEDULE = [(1000.0, 1000.0), (600.0, 600.0), (250.0, 250.0), (0.0, 0.0)]
_GEN_KW = dict(num_frames=5, action_num_frames=5, height=8, width=8, tiled=False, decode_video=False)


def _proprio(sample_seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(1000 + sample_seed)
    return torch.randn(STATE_DIM, generator=g)


def _run_single(arch, *, prompt: str, seed: int) -> np.ndarray:
    result = arch.generate(
        _SCHEDULE,
        prompt,
        first_frame_image=None,
        seed=seed,
        proprio=_proprio(seed),
        **_GEN_KW,
    )
    return np.asarray(result["actions"])


def _run_batch(arch, samples: list) -> np.ndarray:
    proprio = torch.stack([_proprio(s["seed"]) for s in samples], dim=0)
    result = arch.generate_batch(
        _SCHEDULE,
        samples,
        proprio=proprio,
        **_GEN_KW,
    )
    return np.asarray(result["actions"])


def test_generate_batch_b1_matches_generate():
    arch = _make_arch()
    single = _run_single(arch, prompt="stack the bowls", seed=3)
    batched = _run_batch(arch, [{"prompt": "stack the bowls", "seed": 3}])
    assert batched.shape == (1,) + single.shape
    np.testing.assert_allclose(batched[0], single, rtol=1e-5, atol=1e-6)


def test_generate_batch_rows_match_independent_runs():
    arch = _make_arch()
    samples = [
        {"prompt": "stack the bowls", "seed": 1},
        {"prompt": "stack the bowls", "seed": 2},
        {"prompt": "pick up the block", "seed": 3},
    ]
    batched = _run_batch(arch, samples)
    assert batched.shape[0] == len(samples)
    for i, sample in enumerate(samples):
        single = _run_single(arch, prompt=sample["prompt"], seed=sample["seed"])
        np.testing.assert_allclose(
            batched[i],
            single,
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"batched sample {i} diverged from its independent B=1 run",
        )


def test_generate_batch_uses_prompt_embed_cache_once_per_prompt():
    arch = _make_arch()
    cache: dict = {}
    samples = [
        {"prompt": "stack the bowls", "seed": 1},
        {"prompt": "stack the bowls", "seed": 2},
    ]
    proprio = torch.stack([_proprio(s["seed"]) for s in samples], dim=0)
    arch.generate_batch(_SCHEDULE, samples, proprio=proprio, prompt_embed_cache=cache, **_GEN_KW)
    assert list(cache.keys()) == ["stack the bowls"]


def test_generate_batch_rejects_video_decode_and_empty_batch():
    arch = _make_arch()
    with pytest.raises(NotImplementedError):
        arch.generate_batch(_SCHEDULE, [{"prompt": "x", "seed": 1}], decode_video=True)
    with pytest.raises(ValueError):
        arch.generate_batch(_SCHEDULE, [], **_GEN_KW)


def test_collate_rejects_mixed_conditioning_presence():
    from openwam.model.architectures.base import _collate_batch_inputs_shared

    a = {"latents": torch.zeros(1, 2), "first_frame_latents": None, "height": 8}
    b = {"latents": torch.zeros(1, 2), "first_frame_latents": torch.zeros(1, 2), "height": 8}
    with pytest.raises(ValueError, match="uniform"):
        _collate_batch_inputs_shared([a, b])

    c = {"latents": torch.zeros(1, 2), "first_frame_latents": None, "height": 16}
    with pytest.raises(ValueError, match="differs across samples"):
        _collate_batch_inputs_shared([a, c])
