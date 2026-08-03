"""Config-driven modality registry (design/models/vision.md).

A modality is one input/output stream with a trunk (encoder: obs -> token(s)) and head (decoder:
token(s) -> obs), both in the backbone width `d`. The world model builds its fuser streams, per-head
recon losses, metrics and viz by ITERATING the registry, so the modality NAME flows straight to the logs
(`loss/<name>`, `metric/<name>/*`) and enabling/disabling a stream is a config edit. Today: `proprio`
(kind=vector, 1 token) + `image` (kind=image, num_tokens via the ViT AE). Later: image1, image2, ... with
no code change. Every modality maps to a FIXED number of tokens so the token bag has a known width.

Encode/decode accept arbitrary leading dims (B,) or (B,T) — they flatten, run, and restore."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .flow import FlowField, ImageFlowHead, ImageUNetFlowHead
from .vision import ConvImageEncoder, ImageAutoencoder, VisionAEConfig


def _mlp(i: int, o: int, h: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


@dataclass
class ModalitySpec:
    name: str
    kind: str           # "vector" | "image"
    weight: float = 1.0     # per-head reconstruction-loss weight
    noise_std: float = 0.0  # per-stream input noise sigma (training only; the variations design's per-stream sigma)
    decode_kind: str = "mse"  # "mse" (deterministic decode, bit-identical to before) | "flow" (generative
    #                           decode head — a TransportHead denoising the obs from the predicted tokens)
    decode_shortcut: bool = False  # flow decode (param=v): opt-in shortcut self-consistency -> K=1 sampling
    #                                (like the dynamics `diffusion.shortcut`; never default-on). Off -> plain flow, decode_steps.
    decode_steps: int = 6     # flow decode sampling steps (K). v: ODE steps (shortcut->1). x0: consistency refine steps (1 = direct)
    decode_param: str = "v"   # flow decode parameterization: "v" (velocity, integrate ODE — imprecise for images)
    #                           | "x0" (predict the clean obs directly — precise + in-range; use for image decode). See flow.py.
    decode_arch: str = "vit"  # IMAGE decoder architecture, ORTHOGONAL to decode_kind: "vit" (all-attention, patch
    #                           grid) | "unet" (conv U-Net, no patch grid -> smoother fields). Composes with both
    #                           mse and flow (4 combos). Ignored by vector modalities. See vision.ConditionalUNet.
    encode_arch: str = "vit"  # IMAGE encoder architecture: "vit" (ViT/Perceiver, ImageAutoencoder.encode) | "conv"
    #                           (ConvImageEncoder, mirrors the U-Net down-path). Pair conv<->unet for a symmetric
    #                           conv enc/dec. Both emit num_tokens tokens (same interface). Ignored by vectors.
    decode_base: int = 32     # decode_arch=unet ONLY: U-Net base channel width (chs = [base, 2b, 4b, ...]). Bigger
    #                           -> a stronger denoiser for flow decode (sharper images). Ignored by vit/mlp.
    encode_base: int = 32     # encode_arch=conv ONLY: conv-encoder base channel width. Smaller -> less activation
    #                           memory (bigger batch), at some capacity cost. Ignored by vit.
    # vector
    dim: int = 6
    # image
    img_size: int | tuple[int, int] = 128   # int -> square; (H, W) -> non-square (e.g. recorded 112x192)
    patch: int = 16
    num_tokens: int = 8
    ae_depth: int = 4
    channels: int = 3


class Modality(nn.Module):
    """Base: holds a name, the per-step token count, and the recon weight. encode/decode flatten leading dims.
    UNIFIED decode: every modality has ONE `decode_head` (a TransportHead over the chosen net — MLP/ViT/UNet).
    `decode_kind` is just how that net is trained/used: "flow" = the noise-curriculum flow head; "mse" = the
    DEGENERATE no-noise head (deterministic x0 from x=0, L2 loss) — same net, optimized differently."""
    name: str
    n_tokens: int
    weight: float
    decode_kind: str = "mse"

    def _encode(self, obs: Tensor) -> Tensor:    # (M, ...) -> (M, n_tokens, d)
        raise NotImplementedError

    def _decode_cond(self, flat_tok: Tensor) -> Tensor:   # (M, n_tokens, d) -> conditioning for the decode head
        raise NotImplementedError

    def encode(self, obs: Tensor) -> Tensor:
        """obs (B,[T,]*obs_shape) -> tokens (B,[T,]n_tokens,d)."""
        lead = obs.shape[: obs.ndim - self._obs_ndim]
        flat = obs.reshape(-1, *obs.shape[len(lead):])
        tok = self._encode(flat)
        return tok.reshape(*lead, self.n_tokens, tok.shape[-1])

    def decode(self, tok: Tensor) -> Tensor:
        """tokens (B,[T,]n_tokens,d) -> obs (B,[T,]*obs_shape), a committed (deterministic) decode. mse/x0 ->
        1 step; v+shortcut -> K=1; v plain -> decode_steps. Same output shape for every kind/arch."""
        lead = tok.shape[:-2]
        flat = tok.reshape(-1, tok.shape[-2], tok.shape[-1])
        steps = 1 if (self.decode_head.shortcut or self.decode_head.param == "x0") else self.decode_steps
        obs = self.decode_head.sample(self._decode_cond(flat), steps=steps, deterministic=True)
        return obs.reshape(*lead, *obs.shape[1:])

    def decode_loss(self, tok: Tensor, target: Tensor):
        """Per-head decode loss via the unified head: mse (no_noise) -> (L2, None); flow -> (flow-matching, shortcut)."""
        lead = tok.shape[:-2]
        flat = tok.reshape(-1, tok.shape[-2], tok.shape[-1])
        tgt = target.reshape(-1, *target.shape[len(lead):])
        return self.decode_head.loss(self._decode_cond(flat), tgt)


class VectorModality(Modality):
    """A low-dim vector stream (e.g. proprio [p; ṗ]). 1 token; MLP trunk + MLP head."""
    _obs_ndim = 1

    def __init__(self, spec: ModalitySpec, d: int, hidden: int = 64):
        super().__init__()
        self.name, self.n_tokens, self.weight = spec.name, 1, spec.weight
        self.noise_std = float(spec.noise_std)
        self.decode_kind = spec.decode_kind
        self.dim = spec.dim
        self.enc = _mlp(spec.dim, d, hidden)
        self.decode_steps = int(spec.decode_steps)
        no_noise = self.decode_kind == "mse"      # mse = the DEGENERATE no-noise FlowField (unified net; cond = the token)
        self.decode_head = FlowField(dz=spec.dim, h_dim=d, hidden=hidden,
                                     param=("x0" if no_noise else spec.decode_param),
                                     shortcut=(spec.decode_shortcut and not no_noise), no_noise=no_noise)

    def _encode(self, obs):                      # (M, dim) -> (M, 1, d)
        return self.enc(obs).unsqueeze(1)

    def _decode_cond(self, flat_tok):             # (M, 1, d) -> (M, d)
        return flat_tok[:, 0]


class ImageModality(Modality):
    """An image stream (egocentric FPV). num_tokens tokens via the 100%-ViT autoencoder (models/vision.py)."""
    _obs_ndim = 3                                 # (H, W, C)

    def __init__(self, spec: ModalitySpec, d: int):
        super().__init__()
        self.name, self.n_tokens, self.weight = spec.name, spec.num_tokens, spec.weight
        self.noise_std = float(spec.noise_std)
        self.decode_kind = spec.decode_kind
        self.decode_arch = getattr(spec, "decode_arch", "vit")
        self.encode_arch = getattr(spec, "encode_arch", "vit")
        ae_cfg = VisionAEConfig(
            img_size=spec.img_size, patch=spec.patch, d=d, enc_depth=spec.ae_depth,
            dec_depth=spec.ae_depth, num_tokens=spec.num_tokens, channels=spec.channels,
            build_decoder=False)                   # encoder-only; the unified decode_head IS the decoder
        # `self.ae` is the encoder AND the cfg-holder the decode head reads (both variants expose .cfg + .encode()).
        self.ae = (ConvImageEncoder(ae_cfg, base=int(getattr(spec, "encode_base", 32)))
                   if self.encode_arch == "conv" else ImageAutoencoder(ae_cfg))
        self.decode_steps = int(spec.decode_steps)
        no_noise = self.decode_kind == "mse"       # mse = the DEGENERATE no-noise head (unified net; cond = latent tokens)
        param, sc = ("x0" if no_noise else spec.decode_param), (spec.decode_shortcut and not no_noise)
        if self.decode_arch == "unet":
            self.decode_head = ImageUNetFlowHead(self.ae.cfg, base=int(getattr(spec, "decode_base", 32)),
                                                 param=param, shortcut=sc, no_noise=no_noise)
        else:
            self.decode_head = ImageFlowHead(self.ae.cfg, depth=spec.ae_depth, param=param, shortcut=sc, no_noise=no_noise)

    def _encode(self, obs):                       # (M, H, W, C) [0,1] -> (M, num_tokens, d)
        return self.ae.encode(obs)

    def _decode_cond(self, flat_tok):             # (M, num_tokens, d) -> (M, num_tokens, d) (the latent tokens)
        return flat_tok


def make_modality(spec: ModalitySpec, d: int) -> Modality:
    if spec.kind == "vector":
        return VectorModality(spec, d)
    if spec.kind == "image":
        return ImageModality(spec, d)
    raise ValueError(f"unknown modality kind: {spec.kind!r}")


def build_modalities(specs: list[ModalitySpec], d: int) -> nn.ModuleDict:
    """Ordered name -> Modality. Iteration order = registry order = token-bag layout = log-key order."""
    return nn.ModuleDict({s.name: make_modality(s, d) for s in specs})
