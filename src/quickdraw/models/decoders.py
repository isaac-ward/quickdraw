"""UP-ONLY image decoder: token bag -> image, with a query-grid readout.

WHY THIS FILE EXISTS (2026-08-26). `vision.ConditionalUNet` served BOTH jobs through one
`velocity(x, temb, cond, demb)`: a DENOISER (`decode_kind: flow`, where `x` is a genuinely noised image and an
analysis path with skips is correct) and a DECODER (`decode_kind: mse`, where `x` is a ZERO tensor). Two
measured consequences of that sharing, on the live geometry (96px, ae_bottleneck 6, base 32, d 128, T 32):

  1. THE DOWN PATH CONVOLVES ZEROS. `flow.TransportHead.loss` and `._sample` both pass an all-zero `x` on the
     no_noise branch, so the 4-level analysis path computes, measured on CPU, skip tensors whose INTERIOR
     spatial std is EXACTLY 0.000000 at levels 0-2. The only spatial structure is a zero-padding border halo,
     which engulfs the whole map by level 3 (std 0.207) -- so the up path's only absolute-position signal was
     a padding artifact (the StyleGAN3 / Xu et al. 2021 "positional information hidden in padding" pathology).
     Cost: 693,888 params (15.9% of the 4,373,763-param decoder) and ~33% of its activation volume, at FULL
     resolution, paid TWICE per training step (the decode loss AND the roundtrip anchor).

  2. THE LATENT REACHED PIXELS THROUGH A RANK-640 CHOKE. Exactly two routes existed:
        cond_to_spatial = Linear(T*d = 4096 -> chs[-1]*2*2 = 512)   2,097,664 params = 48% of the decoder
        g = cond.mean(1)                                            rank <= 128, the token MEAN
     Total <= 640 of the latent's 4,096 floats: 15.6% visible, 84.4% in the null space, i.e. changes to the
     latent in ~3,456 dimensions produced PIXEL-IDENTICAL images. That is a 6.4x compression INSIDE the
     decoder, after the latent -- tighter than ae_bottleneck, tighter than num_tokens, tighter than anything
     swept. It quantitatively explains record section 17's nulls: at num_tokens=8 the bag is already 1,024
     floats > 640, so the entire 8->64 sweep saturated the readout and the floor was flat at 18.7-20.4 dB.
     And `g` being the token MEAN annihilated token identity on the only per-block conditioning path.

THIS CLASS FIXES BOTH: no down path (nothing to analyse), and a learned query grid cross-attending the tokens
so every one of the bott_h*bott_w cells gets its own d-dim, attention-selected view of ALL tokens. Readout
bandwidth becomes bott_h*bott_w*d (4,608 at 6x6x128) which MATCHES the latent, so nothing is structurally
discarded; `grid_q` also supplies the positional signal the padding halo was illegitimately providing; and
num_tokens becomes free to vary, since no weight shape is baked to T*d any more.

LITERATURE. Up-only is what every published tokens/latent -> image decoder does: VQ-GAN / Stable-Diffusion VAE
(ResBlocks + upsample, mid self-attention, NO analysis path, no encoder skips), StyleGAN (learned constant ->
progressive upsample with per-layer modulation), MAE, Perceiver IO. The query readout is Perceiver IO's output
queries, and it has an unimpeachable in-repo control: `vision.ConvImageEncoder` uses the identical Perceiver
readout in every successful run including the record holder. Cross-attention at EVERY resolution is a
DENOISER-conditioning pattern (SD's U-Net), not a decoder-readout pattern -- no published latent decoder
re-reads the latent per level -- so it is deliberately NOT done here (held in reserve).

PRIOR FAILURE, ADJUDICATED. `decode_arch=vit` is on record as having failed ("motion 0.141, and it collapsed",
conf/model/bsp32mse.yaml). An adversarial audit of the run logs found that indictment is not supported: the
`bsp32vit` run posted the best bespoke floor of the whole program and then died at epoch 15 from a
RECURRENT-PATH gradient explosion -- inf on flow/backbone/encoders while BOTH decoders stayed finite at ~1.9 --
the same fingerprint as the bott16 and predict=absolute collapses, neither of which involved a ViT decoder.
The 0.141 is epoch 7's single lowest value of `motion_ratio`, which the record itself calls direction-blind.
What IS on record against the ViT decoder is patch-grid SEAMS (the "ep24 blocking"), a property of linear
unpatchify SYNTHESIS -- which this class does not use, keeping the conv up path instead.

WHY IT MIGHT MOVE THE OBJECTIVE (raw OL LPIPS@+128; all-time best 0.2783). At matched settings `decode_base`
32 -> 64 improved OL LPIPS@+128 from 0.333 to 0.310, so widening the decoder has already moved the objective
once, not merely the floor. The record holder pairs the WEAK readout with recon_frac=1.0, and design/flow.md
notes the AR decode loss is the only autoregressive gradient in the model -- it reaches the dynamics ONLY
through the decoder's Jacobian, so a rank-640 Jacobian makes 84% of predicted-bag error invisible to it.

AND THE HONEST RISK. That 0.333 -> 0.310 win came from `decode_base=64` widening the ENTIRE conv trunk (2.3x
params), not the readout alone. This class widens the readout while SHRINKING the trunk ~3x. If the win was
trunk capacity rather than readout rank, this loses it. Settle it with the frozen-encoder harness in
_oneoff_decode_kind.py (same encoder, fresh heads, same seed/data/steps) BEFORE spending a GPU pair.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .flow import TransportHead
from .vision import CrossAttn, ViTBlock, _FiLMResBlock, img_hw


class TokenGridReadout(nn.Module):
    """Token SET -> spatial GRID, by cross-attention from a learned query grid.

    Replaces `ConditionalUNet.cond_to_spatial` (a Linear(T*d -> chs[-1]*4) that flattened the bag in fixed
    order into 4 spatial cells). Written as its own module so the DENOISER can adopt the identical readout
    later without re-deriving it -- leaving `ConditionalUNet`'s dense flatten in place while fixing it here
    would re-confound any future mse-vs-flow comparison, which is exactly the confound conf/model/mm_flow.yaml
    already warns about.

    A fixed-order flatten is wrong for a SET three ways: it caps the readout at its output width regardless of
    token count, it cannot scale when num_tokens changes (the weight shape is baked to T*d), and it denies the
    decoder per-token addressing. Cross-attention fixes all three -- and `grid_q` doubles as the positional
    encoding the conv path otherwise lacks entirely.

    THE MID SELF-ATTENTION IS NOT OPTIONAL (adversarial audit, 2026-08-26). `grid_q` is content-INDEPENDENT, so
    the attention logits are a fixed query against learned keys: before those keys learn to discriminate slots,
    attention is near-uniform and EVERY cell receives approximately the token mean -- i.e. at init this module
    momentarily reproduces the very mean-pooling pathology it exists to remove. One self-attention block over
    the grid cells breaks that symmetry for ~198k params at 6x6, where attention is nearly free. Every
    published grid decoder with quality claims also has attention at its lowest resolution (VQ-GAN / SD-VAE
    mid block), so this is standard rather than a workaround.
    """

    def __init__(self, d: int, heads: int, grid_hw: tuple[int, int], *, mlp_ratio: float = 4.0):
        super().__init__()
        self.grid_hw = grid_hw
        gh, gw = grid_hw
        self.grid_q = nn.Parameter(torch.zeros(1, gh * gw, d))
        nn.init.trunc_normal_(self.grid_q, std=0.02)          # house style (vision.py pos/latent_q)
        self.readout = CrossAttn(d, heads)                    # cells attend the token bag
        self.mix = ViTBlock(d, heads, mlp_ratio)              # see the init-degeneracy note above

    def forward(self, cond: Tensor) -> Tensor:                # (M,T,d) -> (M,d,gh,gw)
        M = cond.shape[0]
        h = self.readout(self.grid_q.expand(M, -1, -1), cond)  # (M, gh*gw, d)
        h = self.mix(h)
        gh, gw = self.grid_hw
        return h.transpose(1, 2).reshape(M, -1, gh, gw)        # (M,d,gh,gw)


class TokenPool(nn.Module):
    """Token SET -> one global vector, by attention pooling (a single learned query).

    Replaces `g = cond.mean(1)`. The mean's crime was not being a mean -- it was being the ONLY per-block
    conditioning, so token identity was annihilated on that path. Attention pooling is content-adaptive for
    ~66k params, shared across every block. Feeding the full bag instead would need Linear(T*d -> 2*cout) per
    block: ~1M params per block, ~8M total, and it would reintroduce the fixed-order flatten.

    FIRST ABLATION TO RUN: with the spatial path now carrying per-cell information, FiLM may be droppable
    entirely -- SD-VAE has no global conditioning at all. Kept for now because removing it is a second change.
    """

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.q = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.q, std=0.02)
        self.pool = CrossAttn(d, heads)

    def forward(self, cond: Tensor) -> Tensor:                # (M,T,d) -> (M,d)
        return self.pool(self.q.expand(cond.shape[0], -1, -1), cond)[:, 0]


class TokenGridDecoder(TransportHead):
    """`decode_arch: "up"` -- tokens -> image, UP ONLY. Deterministic decoder (no_noise), never a denoiser.

    Stays a `TransportHead` subclass and keeps the `velocity(x, temb, cond, demb)` signature even though it
    ignores x/temb/demb, because three things depend on that contract: `multimodal.py`'s frozen-decoder probe
    calls `functional_call(head, pb, (x0, temb, cond, None))`; `TransportHead._chunked_velocity` provides the
    checkpointed chunking that `decode_chunk_train` needs (still exact here -- attention runs over the token
    axis only, so velocity is per-element on dim 0 and cat-of-chunks reproduces the whole batch); and
    smoke/decode_recon.py drives heads through it.

    `temb` is deliberately unused rather than added to `g`: in no_noise mode it is `_temb(ones)`, a learned
    CONSTANT, so folding it in is a bias with extra steps. The denoiser keeps its time embedding.
    """

    def __init__(self, ae_cfg, *, base: int = 32, chunk: int = 0):
        # x0 + no_noise: this head predicts the clean image directly and never sees noise. param/shortcut are
        # NOT configurable -- a "generative up-only decoder" would be a different object (see the denoiser).
        super().__init__(param="x0", shortcut=False, event_dims=3, no_noise=True, chunk=chunk)
        self.cfg = ae_cfg
        c = ae_cfg
        H, W = img_hw(c.img_size)
        d, T = c.d, c.num_tokens
        bott = max(1, int(getattr(c, "bottleneck", 8)))
        # SAME pyramid arithmetic as ConditionalUNet/ConvImageEncoder, so `ae_bottleneck` keeps one meaning
        # across encoder and decoder. int() TRUNCATES: at 96px a target of 8 silently gives 12x12, which is why
        # 96px pairs with 6 (-> 6x6, the exact analogue of 8 at 128px: same level counts, same 256x reduction).
        n_levels = max(1, int(math.log2(max(bott, min(H, W)) // bott)))
        chs = [base * min(4, 2 ** i) for i in range(n_levels)]      # e.g. [32,64,128,128]
        self.bott_hw = (H // (2 ** len(chs)), W // (2 ** len(chs)))
        self.readout = TokenGridReadout(d, c.heads, self.bott_hw, mlp_ratio=c.mlp_ratio)
        self.to_ch = nn.Conv2d(d, chs[-1], 1)
        self.gpool = TokenPool(d, c.heads)
        self.mid = _FiLMResBlock(chs[-1], chs[-1], d)
        self.ups, prev = nn.ModuleList(), chs[-1]
        for ch in reversed(chs):
            self.ups.append(_FiLMResBlock(prev, ch, d))              # NO concat: there are no skips to concat
            prev = ch
        self.out_norm = nn.GroupNorm(min(8, chs[0]), chs[0])
        self.out_conv = nn.Conv2d(chs[0], c.channels, 3, padding=1)

    def velocity(self, x=None, temb=None, cond=None, demb=None) -> Tensor:
        """cond (M,T,d) -> (M,H,W,C). x/temb/demb ignored (see the class docstring)."""
        assert cond is not None, "TokenGridDecoder decodes from `cond`; x carries no information."
        g = self.gpool(cond)                                         # (M,d) attention-pooled, not a mean
        h = self.to_ch(self.readout(cond))                           # (M,chs[-1],bh,bw) full-bandwidth readout
        h = self.mid(h, g)
        for up in self.ups:
            h = F.interpolate(h, scale_factor=2, mode="nearest")     # resize-conv (Odena et al.); SD-VAE/TAESD
            h = up(h, g)                                             #   do the same. NOT transposed conv.
        return self.out_conv(F.silu(self.out_norm(h))).permute(0, 2, 3, 1)

    def sample(self, cond: Tensor, *, steps: int, deterministic: bool, eps: Tensor | None = None,
               record_path: bool = False):
        c = self.cfg
        return self._sample(cond, event_shape=(*img_hw(c.img_size), c.channels), lead=cond.shape[:-2],
                            steps=steps, deterministic=deterministic, eps=eps, record_path=record_path)
