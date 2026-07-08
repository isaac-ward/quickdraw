"""Load a trained language reward head (train_reward.py -> reward_head.pt) and score latents against a
text request, decode-free, for MPPI: `R(latent, text) = cos(f_z(latent), f_t(text))`.

Known-vocabulary requests reuse the precomputed text prototypes (no text encoder needed at plan time).
Open-vocabulary (a novel phrase) would need MiniLM live — deferred (see design/language_steering.md)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _mlp(d_in, d_hidden, d_out):                         # must match train_reward.py's head
    return torch.nn.Sequential(torch.nn.Linear(d_in, d_hidden), torch.nn.GELU(), torch.nn.Linear(d_hidden, d_out))


class LanguageReward:
    def __init__(self, path: str, device="cpu"):
        ck = torch.load(path, map_location=device)
        self.buckets = list(ck["buckets"])
        self.protos = ck["text_prototypes"].to(device)                 # (K, 384) MiniLM vocab prototypes
        self.latent_dim = int(ck["latent_dim"])
        self.f_z = _mlp(self.latent_dim, ck["hidden"], ck["embed_dim"]).to(device).eval()
        self.f_t = _mlp(384, ck["hidden"], ck["embed_dim"]).to(device).eval()
        self.f_z.load_state_dict(ck["f_z"]); self.f_t.load_state_dict(ck["f_t"])
        self.device = device

    @torch.no_grad()
    def text_embedding(self, request: str) -> torch.Tensor:
        """Request -> normalized text embedding t_e (embed_dim,). Known bucket -> its precomputed prototype."""
        if request not in self.buckets:
            raise ValueError(f"request {request!r} not in vocab {self.buckets}; open-vocab needs live MiniLM (deferred)")
        proto = self.protos[self.buckets.index(request)]
        return F.normalize(self.f_t(proto[None]), dim=-1)[0]

    @torch.no_grad()
    def score(self, latent: torch.Tensor, t_e: torch.Tensor) -> torch.Tensor:
        """latent (..., latent_dim) -> (...) cosine reward against t_e. Flatten a token bag's (n_state,d) first."""
        z = F.normalize(self.f_z(latent.to(self.device)), dim=-1)
        return z @ t_e
