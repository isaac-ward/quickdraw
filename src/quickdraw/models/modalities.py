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

from .flow import FlowField, ImageFlowHead, ImageUNetFlowHead, TransportHead
from .vision import (ConvImageEncoder, GridToTokens, ImageAutoencoder, TokensToGrid, VisionAEConfig, img_hw)


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
    ae_bottleneck: int = 8   # conv-pyramid bottleneck target (px, short side); 8 = previous behaviour
    decoder_cond: str = "seed"   # U-Net decoder conditioning (record §8.32): "seed" = the historical 2x2
    #                              cond_to_spatial seed (bit-identical default) | "xattn" = a learned query grid
    #                              at the FULL bottleneck resolution cross-attending over the latent tokens.
    #                              §8.31 measured why: the seed path hands the decoder 384 of the 4096 latent
    #                              numbers and cannot place the bunch to better than 244 um (the physical signal
    #                              is 157 um); the two-lobe structure survives only at the bottleneck grid.
    seed_upsample: str = "nearest"   # up-path F.interpolate mode: "nearest" (bit-identical) | "bilinear"
    #                                  (§8.29/§8.30: nearest is what turns coarse structure into plateaus).
    channels: int = 3
    # pretrained image AE (TAESD) — issue #12. pretrained=false -> the bespoke AE above (BIT-IDENTICAL default).
    pretrained: bool = False                     # master on/off for the pretrained-AE image trunk
    pretrained_name: str = "madebyollin/taesd"   # HF repo id, loaded via diffusers.AutoencoderTiny
    pretrained_init: bool = True                 # true=load HF weights; false=random-init SAME arch (prior-vs-arch ablation)
    freeze: bool = True                          # freeze the AE weights (Perceiver adapters still train); false=fine-tune
    adapter: dict | None = None                  # down/up-adapter config: {depth: N, dense: bool}. The BASE path is a
    #                              parameter-free index rearrangement (exactly invertible) + a zero-init residual, so
    #                              the round-trip is the IDENTITY at step 0 whenever num_tokens*d >= latent floats
    #                              (mode EXACT or PADDED). dense=true swaps the pad for a learned per->d projection
    #                              (mode PROJECTED: dense tokens, no idle decode width, but NO identity guarantee).
    ln_carrier: bool = False     # VECTOR modalities (record §8.54): make the bag LayerNorm LOSSLESS for this
    #                              token. LN destroys (mean, std) — 2 scalars/token (the documented −3.51 dB).
    #                              With this flag the encoder emits d−2 content dims, normalizes them INTERNALLY
    #                              (mean 0 / biased std 1), and writes the two destroyed scalars into the last 2
    #                              dims as direction components. The known content statistics then let the
    #                              decoder undo the outer LN in CLOSED FORM (t = y·s + m with s = 1/std(y[:d−2]),
    #                              m = −mean(y[:d−2])·s) — measured exact to 7e-7 and worth 0.032 → 0.011
    #                              heldout roundtrip nRMSE. The bag stays exactly unit-LN'd (the dynamics'
    #                              scale-free geometry is untouched). Default off = bit-identical.
    roundtrip_detach_enc: bool = False   # roundtrip anchor trains the DECODER ONLY (encoder detached inside
    #                              roundtrip_losses). Use with latent_loss_weight>0 on a warm-started run to hold
    #                              the decode head at the codec optimum while the encoder stays frozen (§8.61:
    #                              llw=0 froze the encoder but unanchored the decoder, 0.018 -> 0.29 in 3 epochs).
    linear_skip: bool = False    # VECTOR modalities: parallel LINEAR enc/dec paths beside the MLP trunk/head
    #                              (record §8.53). The obs vector is a NEAR-LINEAR signal (128 PCs = 99.999% of
    #                              variance -> a 128-wide linear code is ~lossless), and the hidden-64 GELU MLP
    #                              both pinches it (0.054 in-distribution) and extrapolates 1.6x worse onto the
    #                              run-tail heldout (0.088; the in-run 0.112). Measured fix: +linear skip ->
    #                              0.032 heldout under the bag LayerNorm (3.5x). Default off = bit-identical.
    obs_slice: tuple[int, int] | list | None = None   # VECTOR modalities: read this [start, stop) slice of the
    #                              SHARED observation_vector instead of a batch stream of its own. None = the
    #                              whole vector (bit-identical). Added 2026-08-30 for the physics modality
    #                              (record §8.43): the loader and the LeRobot schema carry exactly two streams
    #                              (observation_vector + one camera), so a third modality would otherwise mean
    #                              a new dataset column, a new loader tuple and new window stacking. Slicing the
    #                              obs vector gives the physics features their OWN BAG TOKEN -- which is the
    #                              whole point, since only a token's flow-loss weight reaches `act_enc` -- at
    #                              zero cost to the data pipeline.
    fourier_fmax: float = 100.0                  # VECTOR modalities: top band of that ladder. 100.0 = the
    #                              historical hardcoded value (bit-identical). Record §8.40 F3 measured 60.1%
    #                              of the 4416 proprio fourier features as shot-to-shot WHITE at f_max=100;
    #                              lag-1 autocorr falls 0.356 (raw) -> 0.012 (top band). ~8 suits this data.
    fourier_freqs: int = 0                       # VECTOR modalities: sin/cos feature bands prepended to the
    #                                              encoder input (0 = off, bit-identical). See models/features.py.
    latent_loss_weight: float | None = None      # weight of the adapter ROUND-TRIP loss ||up(down(g))-g||^2 (#12).
    #                        None -> per-class default: image 1.0 (unchanged), vector 0.0 (OFF — bit-identical;
    #                        set explicitly, e.g. +model.modalities.0.latent_loss_weight=1, to give a vector
    #                        encoder the encode->decode anchor. Added 2026-08-21: xtcav run-3 arm 1d/1e left the
    #                        proprio encoder gradient-free because nothing set this attr on VectorModality).
    #                              The ONLY term that supervises the adapter pair directly; decode_loss only ever
    #                              trains up() on the dynamics' predicted bag. 0 -> off.


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
        _sl = getattr(spec, "obs_slice", None)
        self.obs_slice = None if _sl is None else (int(_sl[0]), int(_sl[1]))
        self.roundtrip_detach_enc = bool(getattr(spec, "roundtrip_detach_enc", False))
        _llw = getattr(spec, "latent_loss_weight", None)     # None -> 0.0 = roundtrip OFF (bit-identical for
        self.latent_loss_weight = 0.0 if _llw is None else float(_llw)   # every pre-existing vector run)
        self.noise_std = float(spec.noise_std)
        self.decode_kind = spec.decode_kind
        self.dim = spec.dim
        # fourier_freqs>0: [raw | sin/cos] before the trunk. Same rationale as the action encoder -- proprio is
        # z-scored and unbounded, and its small step-to-step differences ARE the motion. 0 = off = bit-identical.
        from .multimodal import FourierMLP
        self.ln_carrier = bool(getattr(spec, "ln_carrier", False))
        nc = d - 2 if self.ln_carrier else d      # carrier: last 2 dims carry (mean, log std) of the content
        self.nc = nc
        self.enc = FourierMLP(spec.dim, nc, hidden, n_freq=int(getattr(spec, "fourier_freqs", 0) or 0),
                              f_max=float(getattr(spec, "fourier_fmax", 100.0) or 100.0))
        self.decode_steps = int(spec.decode_steps)
        no_noise = self.decode_kind == "mse"      # mse = the DEGENERATE no-noise FlowField (unified net; cond = the token)
        self.decode_head = FlowField(dz=spec.dim, h_dim=nc, hidden=hidden,
                                     param=("x0" if no_noise else spec.decode_param),
                                     shortcut=(spec.decode_shortcut and not no_noise), no_noise=no_noise)
        # linear_skip (record §8.53): parallel linear paths. The decode head then learns the RESIDUAL
        # target − dec_lin(token) (see decode/decode_loss), so decode() and decode_loss() stay consistent.
        if bool(getattr(spec, "linear_skip", False)):
            self.enc_lin = nn.Linear(spec.dim, nc)
            self.dec_lin = nn.Linear(nc, spec.dim)
        else:
            self.enc_lin = self.dec_lin = None

    def _encode(self, obs):                      # (M, dim) -> (M, 1, d)
        z = self.enc(obs)
        if self.enc_lin is not None:
            z = z + self.enc_lin(obs)
        if self.ln_carrier:                       # §8.54: internally normalize; re-encode (mu, log sd) as dims
            mu = z.mean(dim=-1, keepdim=True)
            sd = z.std(dim=-1, unbiased=False, keepdim=True).clamp_min(1e-6)
            z = torch.cat([(z - mu) / sd, mu, sd.log()], dim=-1)
        return z.unsqueeze(1)

    def _decode_cond(self, flat_tok):             # (M, 1, d) -> (M, nc)
        y = flat_tok[:, 0]
        if not self.ln_carrier:
            return y
        # closed-form inversion of the bag's outer LN: content dims have mean 0 / biased std 1 by
        # construction, giving the two constraints that recover the destroyed (mean, std). Identity when
        # the token was never LN'd (latent_norm=none/affine), so this is norm-agnostic.
        yc = y[:, :self.nc]
        s = 1.0 / yc.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-8)
        m = -yc.mean(dim=1, keepdim=True) * s
        t = y * s + m
        return t[:, :self.nc] * t[:, self.nc + 1:].exp() + t[:, self.nc:self.nc + 1]

    def decode(self, tok):
        out = super().decode(tok)
        if self.dec_lin is not None:
            lead = tok.shape[:-2]
            cond = self._decode_cond(tok.reshape(-1, tok.shape[-2], tok.shape[-1]))
            out = out + self.dec_lin(cond).reshape(*lead, self.dim)
        return out

    def decode_loss(self, tok, target):
        if self.dec_lin is None:
            return super().decode_loss(tok, target)
        lead = tok.shape[:-2]
        flat = tok.reshape(-1, tok.shape[-2], tok.shape[-1])
        cond = self._decode_cond(flat)
        tgt = target.reshape(-1, self.dim) - self.dec_lin(cond)   # head fits the residual; grads reach dec_lin
        return self.decode_head.loss(cond, tgt)


class ImageModality(Modality):
    """An image stream (egocentric FPV). num_tokens tokens via the 100%-ViT autoencoder (models/vision.py)."""
    _obs_ndim = 3                                 # (H, W, C)

    def __init__(self, spec: ModalitySpec, d: int):
        super().__init__()
        self.name, self.n_tokens, self.weight = spec.name, spec.num_tokens, spec.weight
        _llw = getattr(spec, "latent_loss_weight", 1.0)     # ROUND-TRIP anchor weight (design/collapse.md): a
        self.latent_loss_weight = float(1.0 if _llw is None else _llw)   # bespoke AE needs Dec(Enc(x))->x to not
        #                          collapse. Exposed here (was PretrainedImageModality-only) so roundtrip_losses
        #                          can anchor a TRAINABLE encoder too. Explicit 0 disables it (None -> default 1).
        self.noise_std = float(spec.noise_std)
        self.decode_kind = spec.decode_kind
        self.decode_arch = getattr(spec, "decode_arch", "vit")
        self.encode_arch = getattr(spec, "encode_arch", "vit")
        # ln_carrier (record §8.54, extended to image tokens §8.55): each of the num_tokens tokens gives up
        # 2 dims to carry its own (mean, log std), making the bag's per-token LN closed-form invertible.
        # The AE keeps its native width d (its attention needs d % heads == 0; 126 is not divisible), a
        # modality-level Linear(d → d−2) projects to content, and the decode head is built at d−2 via a
        # cfg copy. The rank-(d−2) projection is a STATIC learned subspace the encoder co-adapts to —
        # unlike the LN it replaces, nothing per-sample is destroyed.
        self.ln_carrier = bool(getattr(spec, "ln_carrier", False))
        self.roundtrip_detach_enc = bool(getattr(spec, "roundtrip_detach_enc", False))
        ae_cfg = VisionAEConfig(
            img_size=spec.img_size, patch=spec.patch, d=d, enc_depth=spec.ae_depth,
            dec_depth=spec.ae_depth, num_tokens=spec.num_tokens, channels=spec.channels,
            bottleneck=int(getattr(spec, "ae_bottleneck", 8)),
            decoder_cond=str(getattr(spec, "decoder_cond", "seed") or "seed"),
            seed_upsample=str(getattr(spec, "seed_upsample", "nearest") or "nearest"),
            build_decoder=False)                   # encoder-only; the unified decode_head IS the decoder
        # `self.ae` is the encoder AND the cfg-holder the decode head reads (both variants expose .cfg + .encode()).
        self.ae = (ConvImageEncoder(ae_cfg, base=int(getattr(spec, "encode_base", 32)))
                   if self.encode_arch == "conv" else ImageAutoencoder(ae_cfg))
        self.decode_steps = int(spec.decode_steps)
        if self.ln_carrier:
            import dataclasses
            self.carrier_proj = nn.Linear(d, d - 2)
            _dec_cfg = dataclasses.replace(self.ae.cfg, d=d - 2)   # the head conditions on the CONTENT width
        else:
            self.carrier_proj = None
            _dec_cfg = self.ae.cfg
        no_noise = self.decode_kind == "mse"       # mse = the DEGENERATE no-noise head (unified net; cond = latent tokens)
        param, sc = ("x0" if no_noise else spec.decode_param), (spec.decode_shortcut and not no_noise)
        if self.decode_arch == "unet":
            self.decode_head = ImageUNetFlowHead(_dec_cfg, base=int(getattr(spec, "decode_base", 32)),
                                                 param=param, shortcut=sc, no_noise=no_noise)
        else:
            self.decode_head = ImageFlowHead(_dec_cfg, depth=spec.ae_depth, param=param, shortcut=sc, no_noise=no_noise)

    def _encode(self, obs):                       # (M, H, W, C) [0,1] -> (M, num_tokens, d)
        z = self.ae.encode(obs)
        if self.ln_carrier:                       # §8.55: per-token internal normalize + 2 carrier dims
            z = self.carrier_proj(z)
            mu = z.mean(dim=-1, keepdim=True)
            sd = z.std(dim=-1, unbiased=False, keepdim=True).clamp_min(1e-6)
            z = torch.cat([(z - mu) / sd, mu, sd.log()], dim=-1)
        return z

    def _decode_cond(self, flat_tok):             # (M, num_tokens, d) -> (M, num_tokens, d[-2]) latent tokens
        if not self.ln_carrier:
            return flat_tok
        nc = flat_tok.shape[-1] - 2               # closed-form LN inversion per token (see VectorModality)
        yc = flat_tok[..., :nc]
        sc = 1.0 / yc.std(dim=-1, unbiased=False, keepdim=True).clamp_min(1e-8)
        mc = -yc.mean(dim=-1, keepdim=True) * sc
        t = flat_tok * sc + mc
        return t[..., :nc] * t[..., nc + 1:].exp() + t[..., nc:nc + 1]


class _AEHolder:
    """Plain (non-Module) holder exposing `.cfg` so eval routines that read `mod.ae.cfg.img_size` work for the
    pretrained modality too (the pretrained AE is TAESD, not an ImageAutoencoder). No params — not registered."""

    def __init__(self, cfg):
        self.cfg = cfg


class PretrainedImageHead(TransportHead):
    """decode_head for the pretrained-AE image modality (issue #12 §3): the DETERMINISTIC (no_noise=mse) net
    whose forward is `up-adapter -> frozen TAESD decoder`. Keeping it a TransportHead means decode()/decode_loss()
    /recon_frac/eval_ae_floor all compose UNCHANGED — the pretrained decoder is just another 'net' behind the
    unified head. TAESD is held by a NON-registered ref (registered once on the modality, not double-counted)."""

    def __init__(self, *, taesd, up_adapter, img_size, channels, decode_pm1: bool = True):
        super().__init__(param="x0", shortcut=False, event_dims=3, no_noise=True)
        self._taesd = (taesd,)                     # tuple -> NOT a registered submodule (no double-count)
        self.up_adapter = up_adapter
        self.img_size, self.channels, self._pm1 = img_size, channels, decode_pm1

    def velocity(self, x, temb, cond, demb=None):  # no_noise: x is zeros; cond = latent tokens (M, num_tokens, d)
        grid = self.up_adapter(cond)               # (M, C, gh, gw) — back to TAESD latent-grid space
        own = getattr(self, "_owner", None)        # latent_norm=affine: EXACT inverse of the encode-side affine
        if own is not None and own[0].latent_affine:
            grid = grid * own[0].lat_std + own[0].lat_mean
        img = self._taesd[0].decode(grid).sample   # (M, 3, H, W) in TAESD range
        if self._pm1:
            img = (img + 1) / 2                    # [-1,1] -> [0,1] (probe-confirmed TAESD convention)
        return img.permute(0, 2, 3, 1)             # (M, H, W, 3) in [0,1] — obs space

    def sample(self, cond, *, steps, deterministic, eps=None, record_path=False):
        H, W = img_hw(self.img_size)
        return self._sample(cond, event_shape=(H, W, self.channels), lead=cond.shape[:-2],
                            steps=steps, deterministic=deterministic, eps=eps, record_path=record_path)


class PretrainedImageModality(Modality):
    """Image stream backed by a PRETRAINED AE (TAESD) bridged to the token bag by learned Perceiver adapters
    (issue #12). encode: TAESD.encode -> down-adapter -> (num_tokens, d). decode (unified no_noise head):
    up-adapter -> TAESD.decode -> image. The AE may be frozen (adapters + dynamics still train)."""
    _obs_ndim = 3

    def __init__(self, spec: ModalitySpec, d: int):
        super().__init__()
        from diffusers import AutoencoderTiny
        self.name, self.n_tokens, self.weight = spec.name, spec.num_tokens, spec.weight
        self.noise_std = float(spec.noise_std)
        self.decode_kind = "mse"                   # pretrained path = the deterministic no_noise decode
        self.decode_steps = 1
        H, W = img_hw(spec.img_size)
        if spec.pretrained_init:                   # load the HF weights (the pretrained pixel prior)
            taesd = AutoencoderTiny.from_pretrained(spec.pretrained_name)
        else:                                      # random-init the SAME arch (ablation: prior vs architecture)
            taesd = AutoencoderTiny.from_config(AutoencoderTiny.load_config(spec.pretrained_name))
        self.taesd = taesd                         # registered ONCE here (the head holds a non-registered ref)
        self.taesd_frozen = bool(spec.freeze)
        if self.taesd_frozen:                      # freeze weights; keep permanently in eval (see train() below)
            for p in self.taesd.parameters():
                p.requires_grad_(False)
            self.taesd.eval()
        with torch.no_grad():                      # derive the true latent grid shape (robust to downsample/channels)
            lat = self.taesd.encode(torch.zeros(1, spec.channels, H, W)).latents
        lat_ch, gh, gw = int(lat.shape[1]), int(lat.shape[2]), int(lat.shape[3])
        heads = max(1, d // 16)
        _ad = dict(spec.adapter) if spec.adapter else {}
        depth = int(_ad.get("depth", 2))
        dense = bool(_ad.get("dense", False))      # PROJECTED (dense tokens, no identity) vs PADDED (identity)
        self.down_adapter = GridToTokens(lat_ch, (gh, gw), spec.num_tokens, d, heads, depth, dense)
        up = TokensToGrid(lat_ch, (gh, gw), spec.num_tokens, d, heads, depth, dense)
        self.adapter_info = self.down_adapter.info          # mode/L/M/per/identity_at_init — reported at build
        _llw = getattr(spec, "latent_loss_weight", 1.0)
        self.latent_loss_weight = float(1.0 if _llw is None else _llw)   # None (new spec default) -> 1.0, as before
        self.decode_head = PretrainedImageHead(taesd=self.taesd, up_adapter=up, img_size=spec.img_size,
                                               channels=spec.channels)
        self.decode_head._owner = (self,)          # tuple -> NOT a registered submodule (no cycle in state_dict)
        # latent_norm=affine: fixed per-CHANNEL scale+shift on the AE latent, calibrated once from the training
        # set (calibrate_latent_affine) and inverted exactly before decode. Unit-variance input for the dynamics
        # at ZERO reconstruction cost, unlike the per-token LN on the bag. Identity until calibrated.
        self.latent_affine = False
        self.register_buffer("lat_mean", torch.zeros(1, lat_ch, 1, 1))
        self.register_buffer("lat_std", torch.ones(1, lat_ch, 1, 1))
        self.register_buffer("lat_calibrated", torch.zeros((), dtype=torch.bool))
        self.ae = _AEHolder(VisionAEConfig(img_size=spec.img_size, patch=spec.patch, d=d,
                                           num_tokens=spec.num_tokens, channels=spec.channels, build_decoder=False))

    def _encode(self, obs):                        # (M,H,W,C)[0,1] -> (M, num_tokens, d)
        grid = self.taesd.encode(obs.permute(0, 3, 1, 2) * 2 - 1).latents   # [0,1]->[-1,1]-> latent grid
        if self.latent_affine:                     # invertible; PretrainedImageHead.velocity undoes it
            grid = (grid - self.lat_mean) / self.lat_std
        return self.down_adapter(grid)

    def enable_latent_affine(self):
        self.latent_affine = True

    def _decode_cond(self, flat_tok):              # (M, num_tokens, d) -> tokens (head decodes via up-adapter+TAESD)
        return flat_tok


    def train(self, mode: bool = True):            # keep a frozen TAESD in eval permanently (Lightning can't flip it)
        super().train(mode)
        if getattr(self, "taesd_frozen", False):
            self.taesd.eval()
        return self


@torch.no_grad()
def calibrate_latent_affine(model, frames) -> dict:
    """Fill lat_mean/lat_std for every affine-normalized pretrained trunk from real TRAINING frames.

    Per CHANNEL over (N,H,W) -- the same statistic Stable Diffusion bakes into `scaling_factor` (a single latent
    std) and that modern video VAEs store as latents_mean/latents_std. Runs ONCE at fit start, before the AE
    floor gate, so the reported floor already reflects the normalization the run will actually use."""
    stats = {}
    for name, mod in getattr(model, "modalities", {}).items():
        if not getattr(mod, "latent_affine", False) or bool(mod.lat_calibrated):
            continue
        dev = next(mod.parameters()).device
        x = frames.to(dev).permute(0, 3, 1, 2) * 2 - 1
        lat = torch.cat([mod.taesd.encode(x[i:i + 32]).latents for i in range(0, len(x), 32)])
        mu = lat.mean(dim=(0, 2, 3), keepdim=True)
        sd = lat.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
        mod.lat_mean.copy_(mu.to(mod.lat_mean.dtype))
        mod.lat_std.copy_(sd.to(mod.lat_std.dtype))
        mod.lat_calibrated.fill_(True)
        stats[name] = {"mean": mu.flatten().tolist(), "std": sd.flatten().tolist(), "n_frames": int(len(x))}
    return stats


def make_modality(spec: ModalitySpec, d: int) -> Modality:
    if spec.kind == "vector":
        return VectorModality(spec, d)
    if spec.kind == "image":
        return PretrainedImageModality(spec, d) if getattr(spec, "pretrained", False) else ImageModality(spec, d)
    raise ValueError(f"unknown modality kind: {spec.kind!r}")


def build_modalities(specs: list[ModalitySpec], d: int) -> nn.ModuleDict:
    """Ordered name -> Modality. Iteration order = registry order = token-bag layout = log-key order."""
    return nn.ModuleDict({s.name: make_modality(s, d) for s in specs})
