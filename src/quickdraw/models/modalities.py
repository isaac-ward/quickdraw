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
    decode_chunk_train: int = 0   # >0: chunk the DECODE head's velocity forward into groups of this many
    #                               frames and checkpoint each, so its intermediates are recomputed in
    #                               backward instead of retained. 0 = OFF (bit-identical). The decoder is
    #                               ~78% of per-sample training memory across TWO passes (the decode loss and
    #                               the roundtrip anchor), so this is the one lever that buys real batch size
    #                               -- everything else lives in the other 22%. ~1.33x decode compute.
    #                               See design/decode_memory.md.
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
    fourier_freqs: int = 0                       # VECTOR modalities: sin/cos feature bands prepended to the
    #                                              encoder input (0 = off, bit-identical). See models/features.py.
    latent_loss_weight: float = 1.0              # weight of the adapter ROUND-TRIP loss ||up(down(g))-g||^2 (#12).
    #                              The ONLY term that supervises the adapter pair directly; decode_loss only ever
    #                              trains up() on the dynamics' predicted bag. 0 -> off.
    prior: str = "none"                          # decode PRIOR (design: unified decode/physics). "none" -> the head
    #                              predicts the obs outright (absolute, bit-identical default). "identity" ->
    #                              obs_next = prev_obs + head(token) (learned delta). "physics" -> obs_next =
    #                              env_fn(prev_obs, action) + head(token), chained AR (the env supplies env_fn).
    #                              For prior != "none" the head is a zero-init RESIDUAL and round-trip is illegal
    #                              (needs a context-free absolute decode) -> set latent_loss_weight=0. Only the
    #                              VECTOR (proprio) modality supports a prior today; image is always "none".


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
        self.prior = str(getattr(spec, "prior", "none") or "none")   # none|identity|physics (unified decode/physics)
        # ROUND-TRIP anchor weight (design/collapse.md): the roundtrip_losses gate reads this OFF THE MODULE, so
        # a vector modality must expose it too or the config value is silently dropped. Default 0.0 = no anchor
        # (unchanged behavior); set model.modalities.<i>.latent_loss_weight>0 to anchor the proprio codec floor.
        self.latent_loss_weight = float(getattr(spec, "latent_loss_weight", 0.0) or 0.0)
        # fourier_freqs>0: [raw | sin/cos] before the trunk. Same rationale as the action encoder -- proprio is
        # z-scored and unbounded, and its small step-to-step differences ARE the motion. 0 = off = bit-identical.
        from .multimodal import FourierMLP
        self.enc = FourierMLP(spec.dim, d, hidden, n_freq=int(getattr(spec, "fourier_freqs", 0) or 0))
        self.decode_steps = int(spec.decode_steps)
        no_noise = self.decode_kind == "mse"      # mse = the DEGENERATE no-noise FlowField (unified net; cond = the token)
        self.decode_head = FlowField(dz=spec.dim, h_dim=d, hidden=hidden,
                                     chunk=int(getattr(spec, "decode_chunk_train", 0) or 0),
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
        _llw = getattr(spec, "latent_loss_weight", 1.0)     # ROUND-TRIP anchor weight (design/collapse.md): a
        self.latent_loss_weight = float(1.0 if _llw is None else _llw)   # bespoke AE needs Dec(Enc(x))->x to not
        #                          collapse. Exposed here (was PretrainedImageModality-only) so roundtrip_losses
        #                          can anchor a TRAINABLE encoder too. Explicit 0 disables it (None -> default 1).
        self.noise_std = float(spec.noise_std)
        self.decode_kind = spec.decode_kind
        self.decode_arch = getattr(spec, "decode_arch", "vit")
        self.encode_arch = getattr(spec, "encode_arch", "vit")
        ae_cfg = VisionAEConfig(
            img_size=spec.img_size, patch=spec.patch, d=d, enc_depth=spec.ae_depth,
            dec_depth=spec.ae_depth, num_tokens=spec.num_tokens, channels=spec.channels,
            bottleneck=int(getattr(spec, "ae_bottleneck", 8)),
            build_decoder=False)                   # encoder-only; the unified decode_head IS the decoder
        # `self.ae` is the encoder AND the cfg-holder the decode head reads (both variants expose .cfg + .encode()).
        self.ae = (ConvImageEncoder(ae_cfg, base=int(getattr(spec, "encode_base", 32)))
                   if self.encode_arch == "conv" else ImageAutoencoder(ae_cfg))
        self.decode_steps = int(spec.decode_steps)
        no_noise = self.decode_kind == "mse"       # mse = the DEGENERATE no-noise head (unified net; cond = latent tokens)
        param, sc = ("x0" if no_noise else spec.decode_param), (spec.decode_shortcut and not no_noise)
        if self.decode_arch == "unet":
            self.decode_head = ImageUNetFlowHead(self.ae.cfg, base=int(getattr(spec, "decode_base", 32)),
                                                 chunk=int(getattr(spec, "decode_chunk_train", 0) or 0),
                                                 param=param, shortcut=sc, no_noise=no_noise)
        else:
            self.decode_head = ImageFlowHead(self.ae.cfg, depth=spec.ae_depth, param=param, shortcut=sc,
                                             no_noise=no_noise,
                                             chunk=int(getattr(spec, "decode_chunk_train", 0) or 0))

    def _encode(self, obs):                       # (M, H, W, C) [0,1] -> (M, num_tokens, d)
        return self.ae.encode(obs)

    def _decode_cond(self, flat_tok):             # (M, num_tokens, d) -> (M, num_tokens, d) (the latent tokens)
        return flat_tok


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
        self.latent_loss_weight = float(getattr(spec, "latent_loss_weight", 1.0))
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
