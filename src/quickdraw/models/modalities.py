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
from torch import Tensor

from .vision import ImageAutoencoder, VisionAEConfig


def _mlp(i: int, o: int, h: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


@dataclass
class ModalitySpec:
    name: str
    kind: str           # "vector" | "image"
    weight: float = 1.0     # per-head reconstruction-loss weight
    # vector
    dim: int = 6
    # image
    img_size: int = 128
    patch: int = 16
    num_tokens: int = 8
    ae_depth: int = 4
    channels: int = 3


class Modality(nn.Module):
    """Base: holds a name, the per-step token count, and the recon weight. encode/decode flatten leading dims."""
    name: str
    n_tokens: int
    weight: float

    def _encode(self, obs: Tensor) -> Tensor:    # (M, ...) -> (M, n_tokens, d)
        raise NotImplementedError

    def _decode(self, tok: Tensor) -> Tensor:    # (M, n_tokens, d) -> (M, ...)
        raise NotImplementedError

    def encode(self, obs: Tensor) -> Tensor:
        """obs (B,[T,]*obs_shape) -> tokens (B,[T,]n_tokens,d)."""
        lead = obs.shape[: obs.ndim - self._obs_ndim]
        flat = obs.reshape(-1, *obs.shape[len(lead):])
        tok = self._encode(flat)
        return tok.reshape(*lead, self.n_tokens, tok.shape[-1])

    def decode(self, tok: Tensor) -> Tensor:
        """tokens (B,[T,]n_tokens,d) -> obs (B,[T,]*obs_shape)."""
        lead = tok.shape[:-2]
        flat = tok.reshape(-1, tok.shape[-2], tok.shape[-1])
        obs = self._decode(flat)
        return obs.reshape(*lead, *obs.shape[1:])


class VectorModality(Modality):
    """A low-dim vector stream (e.g. proprio [p; ṗ]). 1 token; MLP trunk + MLP head."""
    _obs_ndim = 1

    def __init__(self, spec: ModalitySpec, d: int, hidden: int = 64):
        super().__init__()
        self.name, self.n_tokens, self.weight = spec.name, 1, spec.weight
        self.dim = spec.dim
        self.enc = _mlp(spec.dim, d, hidden)
        self.dec = _mlp(d, spec.dim, hidden)

    def _encode(self, obs):                      # (M, dim) -> (M, 1, d)
        return self.enc(obs).unsqueeze(1)

    def _decode(self, tok):                       # (M, 1, d) -> (M, dim)
        return self.dec(tok[:, 0])


class ImageModality(Modality):
    """An image stream (egocentric FPV). num_tokens tokens via the 100%-ViT autoencoder (models/vision.py)."""
    _obs_ndim = 3                                 # (H, W, C)

    def __init__(self, spec: ModalitySpec, d: int):
        super().__init__()
        self.name, self.n_tokens, self.weight = spec.name, spec.num_tokens, spec.weight
        self.ae = ImageAutoencoder(VisionAEConfig(
            img_size=spec.img_size, patch=spec.patch, d=d, enc_depth=spec.ae_depth,
            dec_depth=spec.ae_depth, num_tokens=spec.num_tokens, channels=spec.channels))

    def _encode(self, obs):                       # (M, H, W, C) [0,1] -> (M, num_tokens, d)
        return self.ae.encode(obs)

    def _decode(self, tok):                        # (M, num_tokens, d) -> (M, H, W, C)
        return self.ae.decode(tok)


def make_modality(spec: ModalitySpec, d: int) -> Modality:
    if spec.kind == "vector":
        return VectorModality(spec, d)
    if spec.kind == "image":
        return ImageModality(spec, d)
    raise ValueError(f"unknown modality kind: {spec.kind!r}")


def build_modalities(specs: list[ModalitySpec], d: int) -> nn.ModuleDict:
    """Ordered name -> Modality. Iteration order = registry order = token-bag layout = log-key order."""
    return nn.ModuleDict({s.name: make_modality(s, d) for s in specs})
