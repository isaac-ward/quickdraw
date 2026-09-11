"""Transport heads — rectified-flow generative heads, reused for the dynamics AND the modality decoders.

`TransportHead` owns the shared algorithm (rectified flow-matching loss, K-step Euler sampling, optional
shortcut self-consistency for K=1 — Frans et al. 2024) over an INJECTED velocity net. Concrete heads
supply `velocity`:
  - `FlowField`     — a small MLP velocity (the dynamics head over the token bag; also the proprio decoder).
  - `ImageFlowHead` — a ViT velocity denoiser over an image, conditioned on the predicted latent tokens
                      (the generative image decoder — sharp, samples a mode, vs the MSE decoder's blur).

`event_dims` = how many TRAILING dims form one sample that shares a noise level (leading dims each get an
independent tau): 1 for the per-token dynamics/proprio heads, 3 (H,W,C) for the image head. eps=0 (the
noise mean) at sample time gives the deterministic, reproducible committed prediction used for metrics.
See design/models/flow_heads.md (the reuse refactor) and design/models/diffusion.md (the dynamics head).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _time_features(t: Tensor, freqs: Tensor) -> Tensor:
    """Scalar t in [0,1] (any leading shape, last dim 1) -> [sin, cos] Fourier features (..., 2*F).

    Thin alias over models.features.fourier_features so the expansion has ONE implementation. squash=None
    because tau is already in [0,1]; bit-identical to the previous inline `cat([(t*freqs).sin(), .cos()])`."""
    from .features import fourier_features
    return fourier_features(t, freqs, squash=None)


class TransportHead(nn.Module):
    """Shared rectified-flow algorithm over an injected `velocity(x, temb, cond, demb)` net. Subclass and
    implement `velocity`. The base owns the time (and shortcut step-size) embeddings + loss/sample/consistency."""

    def __init__(self, *, param: str = "v", shortcut: bool = False, event_dims: int = 1,
                 n_freq: int = 16, time_dim: int = 32, no_noise: bool = False, chunk: int = 0):
        super().__init__()
        # chunk: split the velocity forward over the leading (M = B*T) dim and CHECKPOINT each piece, so the
        # decoder's intermediate activations are recomputed during backward instead of all being retained.
        # 0 = OFF (default, bit-identical). See design/decode_memory.md: the decoder is ~78% of per-sample
        # memory, split across TWO passes (this loss, and the roundtrip anchor via _sample), so both routes
        # below go through _chunked_velocity. OPT-IN PER HEAD, never a global default: FlowField is also the
        # DYNAMICS head (multimodal.py:925) and the ACTION head (:962), whose memory profile is different.
        self.chunk = int(chunk or 0)
        # no_noise: the DEGENERATE case = a deterministic decoder (decode_kind="mse"). SAME network as the flow
        # head, trained/used WITHOUT the noise curriculum: predict the clean target from x=0 (tau=1), L2 loss, one
        # step. This unifies mse + flow onto one net per arch — mse is just "flow with no noise". Forces param="x0".
        self.no_noise = no_noise
        if no_noise:
            param, shortcut = "x0", False
        # param = what the net predicts / how we sample:
        #   "v"  (velocity, rectified flow): net predicts the velocity u = eps - x0; sample INTEGRATES the ODE
        #        noise->data over K Euler steps. Good for low-dim targets (latent tokens, proprio); for
        #        high-dim images the integration accumulates error + overshoots range -> poor reconstruction.
        #   "x0" (data / consistency-style): net predicts the CLEAN target x0 DIRECTLY from a noised input.
        #        sample = predict x0 (1 step from noise) -> precise + in-range; K>1 refines by renoising
        #        (consistency multi-step). Deterministic (eps=0) x0 = the conditional mean (~MSE); stochastic
        #        eps + multi-step gives sharp samples. This is the IWS decode approach; use it for image decode.
        assert param in ("v", "x0")
        if param == "x0":
            shortcut = False                          # x0 is a 1-step predictor; shortcut self-consistency is a v-only trick
        self.param, self.shortcut, self.event_dims, self.time_dim = param, shortcut, event_dims, time_dim
        # fixed log-spaced frequencies (deterministic -> reproducible embeddings / golden-testable)
        self.register_buffer("freqs", 2.0 * math.pi * torch.logspace(0.0, 2.0, n_freq), persistent=False)
        self.tau_mlp = nn.Sequential(nn.Linear(2 * n_freq, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim))
        self.d_mlp = (nn.Sequential(nn.Linear(2 * n_freq, time_dim), nn.GELU(),
                                    nn.Linear(time_dim, time_dim)) if shortcut else None)

    def _temb(self, tau: Tensor) -> Tensor:
        return self.tau_mlp(_time_features(tau, self.freqs))

    def _demb(self, d: Tensor) -> Tensor:
        return self.d_mlp(_time_features(d, self.freqs))

    def velocity(self, x: Tensor, temb: Tensor, cond: Tensor, demb: Tensor | None = None) -> Tensor:
        raise NotImplementedError

    def forward(self, x: Tensor, temb: Tensor, cond: Tensor, demb: Tensor | None = None) -> Tensor:
        return self.velocity(x, temb, cond, demb)   # so torch.func.functional_call (frozen-decoder probes) works

    # ---- shapes: leading dims get independent taus; trailing `event_dims` share one ----
    def _tau_shape(self, target: Tensor) -> tuple[int, ...]:
        return target.shape[: target.ndim - self.event_dims] + (1,) * self.event_dims

    def _sample_time(self, shape, device, dtype, time_sampling: str) -> Tensor:
        if time_sampling == "logit_normal":          # SD3-style: weight mid-noise levels more
            return torch.sigmoid(torch.randn(shape, device=device, dtype=dtype))
        return torch.rand(shape, device=device, dtype=dtype)   # uniform (default)

    def _chunked_velocity(self, x: Tensor, temb: Tensor, cond: Tensor, demb: Tensor | None) -> Tensor:
        """`velocity` over the leading dim in chunks, each checkpointed. Exact: velocity is per-element on
        dim 0 (the arch mixes only over dim -2), so cat-of-chunks reproduces the whole-batch result to fp
        tolerance. Falls back to a single call when off, when the batch already fits, or under no_grad --
        checkpointing without a backward is pure overhead, which is what `sample()` does at eval.
        preserve_rng_state=False is safe: nothing inside any velocity path draws randomness (tau/eps are
        sampled OUTSIDE, in `loss`), so the backward recompute is deterministic."""
        c = self.chunk
        if not c or x.shape[0] <= c or not torch.is_grad_enabled():
            return self.velocity(x, temb, cond, demb)
        outs = []
        for i in range(0, x.shape[0], c):
            sl = slice(i, i + c)
            outs.append(torch.utils.checkpoint.checkpoint(
                self.velocity, x[sl], temb[sl], cond[sl], None if demb is None else demb[sl],
                use_reentrant=False, preserve_rng_state=False))
        return torch.cat(outs, 0)

    # ---- training ----
    def predict(self, cond: Tensor, target: Tensor) -> Tensor:
        """The CLEAN prediction this head's reconstruction loss scores. `no_noise` heads ONLY.

        WHY THIS IS PUBLIC (2026-09-11). The prediction used to exist only inside `loss()`, so a second
        term could not score it without a second decoder forward -- and the image decoder is ~78% of
        per-sample memory (design/decode_memory.md), so that is not affordable. Exposing it lets
        `Modality.decode_loss` decode ONCE and hand the same tensor to both the reconstruction term and
        the derivative term (design/derivative_loss.md §5.0). The roundtrip anchor already works this way
        (`to_obs` then `recon_loss`), so this makes the decode site consistent rather than special.

        This is a pure extraction: `loss()`'s no_noise branch below and `_sample()`'s no_noise branch
        computed this IDENTICAL expression in two places. Both now route here, so there is one copy.
        `_chunked_velocity`, not `velocity`, so `decode_chunk_train`'s checkpointed chunking is unchanged.

        RAISES for a noised parameterisation, deliberately. Such a head denoises from `x_tau` at a `tau`
        sampled PER ELEMENT, so it has no single clean prediction, and differencing its predictions across
        time measures the tau draw rather than the motion -- measured at 62% tau noise on a trained
        proprio head (design/derivative_loss.md §5.1). To make a head eligible, set `decode_kind: mse`."""
        if not self.no_noise:
            raise ValueError(
                f"predict() requires a no_noise decoder; this head is param={self.param!r} with a noise "
                f"curriculum. A noised parameterisation has no clean single-pass prediction. Set "
                f"decode_kind='mse' on the modality to make it eligible (see design/derivative_loss.md)")
        return self._chunked_velocity(torch.zeros_like(target),
                                      self._temb(target.new_ones(self._tau_shape(target))), cond, None)

    def loss(self, cond: Tensor, target: Tensor, *, time_sampling: str = "uniform",
             recon_loss=None) -> tuple[Tensor, Tensor | None]:
        """param="v": rectified flow-matching ||net - (eps-target)||^2 (+ shortcut self-consistency).
        param="x0": ||net(x_tau,tau) - target||^2 — predict the clean target directly. Returns (L_main, L_shortcut|None).

        `recon_loss(pred, target) -> Tensor` REPLACES the default `F.mse_loss` on the two parameterisations
        that have a clean prediction to score: no_noise (mse) and param="x0". It exists so `VisualLoss` can own
        the whole pixel mix (L2 + L1 + LPIPS) at BOTH image loss sites, with no second decoder forward -- the
        prediction is already computed here, and re-deriving it outside would double the cost of what is
        ~78% of per-sample memory.

        REPLACES rather than ADDS deliberately. The previous `aux` hook was summed ON TOP of an `F.mse_loss`
        computed here, so a VisualLoss owning an L2 term would have double-counted it.

        For param="v" the net predicts a VELOCITY, not an image, so there is nothing for a pixel loss to score
        and `recon_loss` is ignored -- deliberately silent rather than raising, since the dynamics FlowField
        shares this method and never passes one."""
        rl = recon_loss if recon_loss is not None else F.mse_loss
        ts = self._tau_shape(target)
        if self.no_noise:                             # mse decode: deterministic cond->target, no noise curriculum
            return rl(self.predict(cond, target), target), None   # ONE copy of the formula -- see predict()
        tau = self._sample_time(ts, target.device, target.dtype, time_sampling)
        eps = torch.randn_like(target)
        x_tau = (1.0 - tau) * target + tau * eps      # straight (rectified) path
        if self.param == "x0":                        # net predicts the CLEAN target directly
            x0_hat = self._chunked_velocity(x_tau, self._temb(tau), cond, None)
            return rl(x0_hat, target), None
        u = eps - target                              # velocity along the straight path (regression target)
        demb = self._demb(torch.zeros_like(tau)) if self.shortcut else None   # flow-matching = the d->0 field
        v = self._chunked_velocity(x_tau, self._temb(tau), cond, demb)
        l_flow = F.mse_loss(v, u)
        return (l_flow, self._consistency(cond, target)) if self.shortcut else (l_flow, None)

    def _consistency(self, cond: Tensor, target: Tensor) -> Tensor:
        """Shortcut self-consistency: one step of size 2d must equal two chained steps of size d (bootstrap
        target stop-gradded) -> accurate large/single steps, so K=1 sampling works."""
        ts = self._tau_shape(target)
        k = torch.randint(1, 4, ts, device=target.device)                   # 1..3
        d = (0.5 ** k.float())                                              # 1/2, 1/4, 1/8
        tau = 2.0 * d + (1.0 - 2.0 * d) * torch.rand(ts, device=target.device, dtype=target.dtype)
        eps = torch.randn_like(target)
        x = (1.0 - tau) * target + tau * eps
        with torch.no_grad():                                              # bootstrap target (stop-grad)
            v1 = self.velocity(x, self._temb(tau), cond, self._demb(d))
            x2 = x - v1 * d
            v2 = self.velocity(x2, self._temb(tau - d), cond, self._demb(d))
            s_target = 0.5 * (v1 + v2)
        v_2d = self._chunked_velocity(x, self._temb(tau), cond, self._demb(2.0 * d))  # the large step, with grad
        return F.mse_loss(v_2d, s_target)

    # ---- inference: integrate the ODE ----
    def _sample(self, cond: Tensor, *, event_shape, lead, steps: int, deterministic: bool,
                eps: Tensor | None = None, record_path: bool = False):
        """Euler-integrate dx/dtau = v from tau=1 (x=eps) -> tau=0. eps=0 (deterministic) -> reproducible
        committed prediction. `event_shape`/`lead` let heads with different target shapes reuse this."""
        ts = tuple(lead) + (1,) * self.event_dims
        if self.no_noise:                             # mse decode: one deterministic cond->target prediction (x=0, tau=1)
            # Routed through predict() so the formula lives in ONE place. `predict` only uses its second
            # argument for shape/dtype/device (zeros_like + a ones tau of the matching shape), so handing it
            # the zeros tensor itself is exactly equivalent to the inline version this replaced.
            out = self.predict(cond, cond.new_zeros(tuple(lead) + tuple(event_shape)))
            return (out, [out]) if record_path else out
        if eps is None:
            shp = tuple(lead) + tuple(event_shape)
            eps = cond.new_zeros(shp) if deterministic else torch.randn(shp, device=cond.device, dtype=cond.dtype)
        x = eps
        path = [x]
        if self.param == "x0":                        # consistency-style: predict x0, optionally renoise + refine
            for k in range(steps):
                tau = cond.new_full(ts, 1.0 - k / steps)
                x0_hat = self._chunked_velocity(x, self._temb(tau), cond, None)   # direct clean-target prediction
                if record_path:
                    path.append(x0_hat)
                if k < steps - 1:                     # renoise to a lower level and refine (0 noise if deterministic)
                    tn = 1.0 - (k + 1) / steps
                    noise = torch.zeros_like(x0_hat) if deterministic else torch.randn_like(x0_hat)
                    x = (1.0 - tn) * x0_hat + tn * noise
            return (x0_hat, path) if record_path else x0_hat
        d = cond.new_full(ts, 1.0 / steps) if self.shortcut else None       # step-size conditioning
        demb = self._demb(d) if self.shortcut else None
        for k in range(steps):                        # param="v": Euler-integrate dx/dtau = v, tau=1 -> 0
            tau = cond.new_full(ts, 1.0 - k / steps)
            x = x - self.velocity(x, self._temb(tau), cond, demb) * (1.0 / steps)
            if record_path:
                path.append(x)
        return (x, path) if record_path else x


class _TokenMixBlock(nn.Module):
    """One pre-norm block: bidirectional SDPA across the N bag tokens + MLP. No positional encoding — slot
    identity already rides in `cond` (the backbone adds a learned per-slot embedding), and N is tiny."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = int(heads)
        assert dim % self.heads == 0, f"hidden {dim} must be divisible by heads {self.heads}"
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        # ZERO-INIT BOTH RESIDUAL BRANCHES so the block is an EXACT identity at init and has to earn its
        # contribution (ControlNet / DiT adaLN-zero; the same _zero_init_last trick vision.py uses on the
        # adapter). Without this the block perturbs the velocity field from step 0, and MEASURED
        # (2026-08-11) the flow's gradients then explode inside the rollout: grad/norm/flow went
        # 0.98 -> 1.48 -> 766 on tf_affine and 2.10 -> 65.6 -> 10151 on tf_bespoke, while the per-token MLP
        # denoiser stayed under 1.0 the whole time. gradient_clip_val=1.0 then LAUNDERS the blowup -- postclip
        # is exactly 1.000 every epoch, so training keeps taking confident unit-norm steps in a direction
        # dominated by the exploded component. That degrades smoothly instead of NaN-ing, which is why it
        # looked like a modelling problem rather than an optimisation one.
        for _lin in (self.proj, self.mlp[-1]):
            nn.init.zeros_(_lin.weight)
            if _lin.bias is not None:
                nn.init.zeros_(_lin.bias)

    def forward(self, y: Tensor) -> Tensor:                       # (M, N, d)
        M, N, d = y.shape
        q, k, v = self.qkv(self.n1(y)).reshape(M, N, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v)                # plain SDPA: no FlexAttention constraints
        y = y + self.proj(o.transpose(1, 2).reshape(M, N, d))
        return y + self.mlp(self.n2(y))


class FlowField(TransportHead):
    """MLP velocity `v([x || emb(tau) || cond (|| emb(d))]) -> dz`. The dynamics head over the token bag
    (cond = backbone context h, dz = d) AND the proprio decoder (cond = the proprio token, dz = 6)."""

    def __init__(self, dz: int, h_dim: int, hidden: int, *, cond: str = "concat", param: str = "v",
                 shortcut: bool = False, n_freq: int = 16, time_dim: int = 32, no_noise: bool = False,
                 arch: str = "mlp", n_tokens: int = 0, depth: int = 2, heads: int = 4, chunk: int = 0):
        """`arch` selects how the velocity net sees the OTHER tokens being denoised at the same time.

          mlp          (default, bit-identical to every run before 2026-08-10) a per-token MLP. Token k's
                       velocity depends ONLY on its own x_k and its own conditioning h_k, so the K denoising
                       steps sample p(z_1..z_N | h) = PROD_k p(z_k | h_k) -- a FACTORIZED model. Cross-token
                       correlation can only ride in h, which is frozen for the whole trajectory, and no
                       amount of information in h fixes it: a product of correct marginals is not the correct
                       joint. Harmless while eval is DETERMINISTIC (the mean of the factorized model equals
                       the true marginal means), but it makes stochastic sampling produce INCOHERENT bags.
          transformer  token-mixing: every denoising step attends across the N tokens, so the trajectory
                       samples a proper JOINT. Prerequisite for stochastic_eval to buy sharpness rather than
                       incoherence. Plain SDPA (not FlexAttention) so compile_rollout's head_dim>=16
                       power-of-2 constraint does not apply -- N is ~10, attention over it is negligible."""
        if cond != "concat":
            raise NotImplementedError(f"FlowField cond={cond!r} not implemented; use 'concat' (adaln is a future upgrade).")
        super().__init__(param=param, shortcut=shortcut, event_dims=1, n_freq=n_freq, time_dim=time_dim,
                         no_noise=no_noise, chunk=chunk)
        self.dz = dz
        self.arch = str(arch).lower()
        if self.arch not in ("mlp", "transformer"):
            raise ValueError(f"FlowField arch must be 'mlp' or 'transformer', got {arch!r}")
        in_dim = dz + time_dim + h_dim + (time_dim if self.shortcut else 0)   # self.shortcut (x0 forces it off)
        if self.arch == "mlp":
            self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                     nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, dz))
        else:
            # n_tokens is REQUIRED: velocity() must know which axis is the token axis. Guessing dim -2 would
            # silently mix over TIME when this class backs the action head (whose lead axes are (B,T)).
            if int(n_tokens) <= 0:
                raise ValueError("FlowField arch='transformer' requires n_tokens>0 (the bag's token count)")
            self.n_tokens = int(n_tokens)
            self.inp = nn.Linear(in_dim, hidden)
            self.blocks = nn.ModuleList([_TokenMixBlock(hidden, heads) for _ in range(int(depth))])
            self.out = nn.Linear(hidden, dz)

    def velocity(self, x: Tensor, temb: Tensor, cond: Tensor, demb: Tensor | None = None) -> Tensor:
        parts = [x, temb, cond] + ([demb] if self.shortcut else [])
        u = torch.cat(parts, dim=-1)
        if self.arch == "mlp":
            return self.net(u)
        n = self.n_tokens
        assert u.shape[-2] == n, f"token axis is dim -2 and must be n_tokens={n}, got {tuple(u.shape)}"
        lead = u.shape[:-2]
        y = self.inp(u).reshape(-1, n, self.inp.out_features)     # (M, N, hidden) — N is the sequence
        for b in self.blocks:
            y = b(y)
        return self.out(y).reshape(*lead, n, self.dz)

    def sample(self, h: Tensor, *, steps: int, deterministic: bool, eps: Tensor | None = None,
               record_path: bool = False):
        """Preserves the original signature (cond=h, dz-shaped output) so the dynamics call sites are unchanged."""
        return self._sample(h, event_shape=(self.dz,), lead=h.shape[:-1], steps=steps,
                            deterministic=deterministic, eps=eps, record_path=record_path)


class ImageFlowHead(TransportHead):
    """ViT velocity denoiser over an image, conditioned on the predicted latent tokens: a GENERATIVE image
    decoder. v(noised_img (M,H,W,C), tau, latent_tokens (M,num_tokens,d)[, d_step]) -> velocity (M,H,W,C).
    Reuses vision.ViTBlock/CrossAttn + linear (de)patchify (own weights, separate from the AE)."""

    def __init__(self, ae_cfg, *, depth: int = 4, param: str = "v", shortcut: bool = False, n_freq: int = 16,
                 time_dim: int = 32, no_noise: bool = False, chunk: int = 0):
        super().__init__(param=param, shortcut=shortcut, event_dims=3, n_freq=n_freq, time_dim=time_dim,
                         no_noise=no_noise, chunk=chunk)
        from .vision import CrossAttn, ViTBlock, img_hw
        c = ae_cfg
        self.cfg = c
        H, W = img_hw(c.img_size)
        self.gh, self.gw = H // c.patch, W // c.patch
        self.np = self.gh * self.gw
        pdim = c.patch * c.patch * c.channels
        d = c.d
        self.patch_embed = nn.Linear(pdim, d)
        self.pos = nn.Parameter(torch.zeros(1, self.np, d))
        self.t_proj = nn.Linear(time_dim, d)
        self.d_proj = nn.Linear(time_dim, d) if self.shortcut else None      # self.shortcut (x0 forces it off)
        self.from_latent = CrossAttn(d, c.heads)                      # patches attend to the conditioning tokens
        self.blocks = nn.ModuleList([ViTBlock(d, c.heads, c.mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        self.unpatch = nn.Linear(d, pdim)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def _patchify(self, img):                                          # (M,H,W,C) -> (M, np, patch*patch*C)
        p, gh, gw, C = self.cfg.patch, self.gh, self.gw, self.cfg.channels
        M = img.shape[0]
        return img.reshape(M, gh, p, gw, p, C).permute(0, 1, 3, 2, 4, 5).reshape(M, gh * gw, p * p * C)

    def _unpatchify(self, x):                                          # (M, np, patch*patch*C) -> (M,H,W,C)
        p, gh, gw, C = self.cfg.patch, self.gh, self.gw, self.cfg.channels
        x = x.reshape(x.shape[0], gh, gw, p, p, C).permute(0, 1, 3, 2, 4, 5)
        return x.reshape(x.shape[0], gh * p, gw * p, C)

    def velocity(self, x: Tensor, temb: Tensor, cond: Tensor, demb: Tensor | None = None) -> Tensor:
        M = x.shape[0]
        h = self.patch_embed(self._patchify(x)) + self.pos            # (M, np, d)
        h = h + self.t_proj(temb.reshape(M, -1)).unsqueeze(1)         # broadcast one tau-embed over patches
        if self.shortcut and demb is not None:
            h = h + self.d_proj(demb.reshape(M, -1)).unsqueeze(1)
        h = self.from_latent(h, cond)                                 # condition on the predicted latent tokens
        for blk in self.blocks:
            h = blk(h)
        return self._unpatchify(self.unpatch(self.norm(h)))           # (M, H, W, C)

    def sample(self, cond: Tensor, *, steps: int, deterministic: bool, eps: Tensor | None = None,
               record_path: bool = False):
        from .vision import img_hw
        c = self.cfg
        return self._sample(cond, event_shape=(*img_hw(c.img_size), c.channels), lead=cond.shape[:-2],
                            steps=steps, deterministic=deterministic, eps=eps, record_path=record_path)


class ImageUNetFlowHead(TransportHead):
    """CNN U-Net velocity denoiser over an image (the conv alternative to ImageFlowHead's ViT), conditioned on
    the predicted latent tokens. Same TransportHead contract + `sample()` signature — so it drops in wherever
    ImageFlowHead does. No patch grid -> smooth color fields don't block (see the ep24 ViT-decode blocking)."""

    def __init__(self, ae_cfg, *, base: int = 32, param: str = "v", shortcut: bool = False,
                 n_freq: int = 16, time_dim: int = 32, no_noise: bool = False, chunk: int = 0):
        super().__init__(param=param, shortcut=shortcut, event_dims=3, n_freq=n_freq, time_dim=time_dim,
                         no_noise=no_noise, chunk=chunk)
        from .vision import ConditionalUNet
        self.cfg = ae_cfg
        self.unet = ConditionalUNet(ae_cfg, base=base, time_dim=time_dim)

    def velocity(self, x: Tensor, temb: Tensor, cond: Tensor, demb: Tensor | None = None) -> Tensor:
        return self.unet.velocity(x, temb, cond, demb)

    def sample(self, cond: Tensor, *, steps: int, deterministic: bool, eps: Tensor | None = None,
               record_path: bool = False):
        from .vision import img_hw
        c = self.cfg
        return self._sample(cond, event_shape=(*img_hw(c.img_size), c.channels), lead=cond.shape[:-2],
                            steps=steps, deterministic=deterministic, eps=eps, record_path=record_path)
