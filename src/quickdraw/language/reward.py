"""Load a trained language reward head (train_reward.py -> reward_head.pt) and score latents against a
text request, decode-free, for MPPI: `R(latent, text) = cos(f_z(latent), f_t(text))`.

Known-vocabulary requests reuse the precomputed text prototypes (no text encoder needed at plan time).
Open-vocabulary (a novel phrase) would need MiniLM live — deferred (see design/language_steering.md)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _mlp(d_in, d_hidden, d_out, dropout=0.0):            # must match train_reward.py's head (incl. dropout structure)
    layers = [torch.nn.Linear(d_in, d_hidden), torch.nn.GELU()]
    if dropout > 0:
        layers.append(torch.nn.Dropout(dropout))
    layers.append(torch.nn.Linear(d_hidden, d_out))
    return torch.nn.Sequential(*layers)


class LanguageReward:
    def __init__(self, path: str, device="cpu"):
        ck = torch.load(path, map_location=device)
        # bucket order MUST match the prototype rows: `factors` (new, multi-factor) concatenated in order, else
        # the flat `buckets` list (old single-factor heads).
        self.flat = ([b for f in ck["factors"] for b in ck["factors"][f]] if "factors" in ck else list(ck["buckets"]))
        self.buckets = list(self.flat)                                 # every known word (all factors)
        self.protos = ck["text_prototypes"].to(device)                 # (K_total, 384) MiniLM vocab prototypes
        self.latent_dim = int(ck["latent_dim"])
        drop = float(ck.get("dropout", 0.0))                           # match the trained module's structure (Dropout is
        self.f_z = _mlp(self.latent_dim, ck["hidden"], ck["embed_dim"], drop).to(device).eval()   # inert at eval, but
        self.f_t = _mlp(384, ck["hidden"], ck["embed_dim"], drop).to(device).eval()               # shifts state_dict keys
        self.f_z.load_state_dict(ck["f_z"]); self.f_t.load_state_dict(ck["f_t"])
        self.text_model = ck["text_model"]
        self._tok = self._lm = None                                    # MiniLM, lazy-loaded on first query
        self.device = device

    def _embed(self, text: str) -> torch.Tensor:
        """MiniLM sentence embedding (mean-pooled, L2-normalized), lazy-loaded. This is the SAME frozen encoder
        train_reward used, so its pretraining carries: free-form phrasings ('upper red', 'the red area at the
        top') land near the bucket paraphrases f_t was trained on."""
        if self._tok is None:
            from transformers import AutoModel, AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.text_model)
            self._lm = AutoModel.from_pretrained(self.text_model).eval().to(self.device)
        enc = self._tok([text], padding=True, truncation=True, return_tensors="pt").to(self.device)
        out = self._lm(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        return F.normalize((out * m).sum(1) / m.sum(1).clamp(min=1e-9), dim=-1)   # (1, 384)

    @torch.no_grad()
    def text_embedding(self, request: str) -> torch.Tensor:
        """Request -> normalized target direction t_e (embed_dim,). The WHOLE free-form phrase is embedded by
        MiniLM (frozen, pretrained) and mapped through f_t — so 'top red', 'red top', 'upper red', 'the red area
        at the top', 'the lower blue area' all map to the right region, not just the exact bucket words.
        Compound targets fall out for free because MiniLM blends the concepts and f_t was trained per-factor."""
        return F.normalize(self.f_t(self._embed(str(request))), dim=-1)[0]

    @torch.no_grad()
    def score(self, latent: torch.Tensor, t_e: torch.Tensor) -> torch.Tensor:
        """latent (..., latent_dim) -> (...) cosine reward against t_e. Flatten a token bag's (n_state,d) first.
        t_e (embed,) -> one target for all (broadcast). t_e (L, embed) -> a PER-leading-index target (multi-query:
        latent's first dim = episode, each episode scored against its own request)."""
        z = F.normalize(self.f_z(latent.to(self.device)), dim=-1)
        if t_e.dim() == 1:
            return z @ t_e
        return torch.einsum("l...e,le->l...", z, t_e.to(z.device))
