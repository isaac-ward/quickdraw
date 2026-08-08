"""AE-floor Phase-1 gate helpers (issue #12): the raw-TAESD domain-shift probe + the num_tokens-floor note.

These measure the image encode->decode CEILING WITHOUT training the dynamics — the decisive go/no-go signals
before building the TAESD modality + adapters (Phase 2). `eval_ae_floor` (evaluation/routines.py) measures the
BESPOKE AE's floor on a checkpoint; the two functions here cover the questions that need no dynamics at all.
"""
from __future__ import annotations

import numpy as np
import torch


def _psnr_ssim(gt, rec):
    """gt/rec: (N,H,W,3) float [0,1] tensors -> (psnr_db, ssim, mse)."""
    from .openloop import _ssim
    gt, rec = gt.clamp(0, 1), rec.clamp(0, 1)
    mse = float(torch.mean((rec - gt) ** 2))
    psnr = -10.0 * np.log10(max(mse, 1e-12))
    ssim = float(max(0.0, min(1.0, _ssim(rec, gt))))
    return psnr, ssim, mse


def raw_taesd_floor(cfg, sizes=(128,), n_frames=32, device=None, log=print):
    """RAW pretrained-TAESD floor: load `diffusers.AutoencoderTiny(pretrained_name)` and round-trip N REAL val
    frames with NO training and NO adapter -> PSNR/SSIM. Tests DOMAIN SHIFT (TAESD trained on natural ~512^2 web
    images vs synthetic MuJoCo/ISS renders at 128) — the one Phase-1 signal that needs zero training (#12 §5).
    diffusers is imported directly (already importable in the container; a transitive dep — declare it if Phase 2
    lands, #12 §1). Runs on CPU by default (TAESD is tiny) to avoid contending with training GPUs.

    Pixel convention is auto-detected per size: AutoencoderTiny's expected input range varies by version, so we
    try both the [-1,1] (standard-VAE) and [0,1] round-trips and report the one with the higher PSNR (the correct
    scaling wins decisively) — so the reported floor is not confounded by a wrong normalization."""
    from diffusers import AutoencoderTiny

    from ..data.dataset import load_fpv_frames
    from ..training.setup import resolve_data_root
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    name = str(_af(cfg).get("pretrained_name", "madebyollin/taesd"))
    log(f"[raw_taesd] loading {name} on {dev} (no training, no adapter)...")
    ae = AutoencoderTiny.from_pretrained(name).to(dev).eval()
    root, cam = resolve_data_root(cfg), cfg.data.get("cam", "fpv")
    out = {}
    for sz in sizes:
        frames = load_fpv_frames(root, "val", size=sz, cam=cam, max_frames=n_frames, cache=False)   # (n,sz,sz,3) uint8
        gt = torch.from_numpy(np.asarray(frames[:n_frames])).float().div(255.0).to(dev)              # (n,sz,sz,3) [0,1]
        x = gt.permute(0, 3, 1, 2)                                                                    # (n,3,sz,sz)
        best = None
        for conv, inp, back in (("[-1,1]", x * 2 - 1, lambda d: (d + 1) / 2), ("[0,1]", x, lambda d: d)):
            with torch.no_grad():
                lat = ae.encode(inp).latents
                dec = ae.decode(lat).sample
            rec = back(dec).permute(0, 2, 3, 1)                                                       # (n,sz,sz,3)
            psnr, ssim, mse = _psnr_ssim(gt, rec)
            if best is None or psnr > best["psnr"]:
                best = {"psnr": psnr, "ssim": ssim, "mse": mse, "convention": conv,
                        "latent_shape": tuple(lat.shape[1:]), "latent_floats": int(np.prod(lat.shape[1:]))}
        out[sz] = best
        log(f"[raw_taesd] {sz}x{sz}: PSNR={best['psnr']:.2f}dB SSIM={best['ssim']:.3f} MSE={best['mse']:.5f} "
            f"| pixel-conv {best['convention']} | TAESD latent {best['latent_shape']} = {best['latent_floats']} floats "
            f"(vs bespoke num_tokens*d budget)")
    return out


def num_tokens_floor_note(log=print):
    """The num_tokens ∈ {8,16,32} bespoke-AE floor sweep (#12 §5) — INFRA + honest caveat, NOT a fabricated number.

    A meaningful bespoke-AE floor requires a TRAINED AE: an untrained encoder/decoder reconstructs garbage, so
    running eval_ae_floor on a fresh model would report a floor that says nothing about the tokenizer's capacity.
    So this is intentionally a runner over TRAINED checkpoints, not a one-shot on an untrained model:

        # option A — sweep eval_ae_floor over existing trained checkpoints at each width:
        for nt in 8 16 32:
          python -m quickdraw.eval_ae_floor checkpoint=<run_trained_at_nt> 'model.modalities.1.num_tokens='$nt

        # option B — short AE-only runs per width (freeze dynamics / few epochs), then eval_ae_floor on each.

    The single measurement it yields — floor(num_tokens) — decides whether the ceiling is the BOTTLENECK (floor
    rises with num_tokens) or the TOKENIZER (floor flat), i.e. whether raising num_tokens or swapping in a
    pretrained AE is the lever. It needs no dynamics, but it DOES need a trained AE, hence checkpoints."""
    log(num_tokens_floor_note.__doc__)


def _af(cfg):
    """The optional `ae_floor` config block (added via `+ae_floor.<k>=<v>` overrides); {} if absent."""
    try:
        return cfg.get("ae_floor", {}) or {}
    except Exception:
        return {}
