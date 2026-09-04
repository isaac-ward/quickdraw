"""XTCAV two-bunch physics features — the SINGLE source of truth for the deployed extractor.

Moved out of `wizard/scripts/xtcav_physics_eval.py` (2026-08-30) so that the TRAINING targets of the
physics modality (record §8.43) and the EVAL response gate are literally the same code. If these ever
diverge, a model can score well on a target that the gate does not measure.

The default path is BIT-IDENTICAL to the deployed extractor. `interpolate=True` opts into the
sub-pixel median of §8.40 F6 and is NOT the default — every number in record §7-§8.42 was produced
with the integer version, and silently changing it would make them incomparable.
"""
from __future__ import annotations

import numpy as np

# ---- geometry / gating constants (v4 centered-crop frames at the 64x192 training resolution)
PX_UM = 30.5 * 384.0 / 192.0        # streak-axis um per model px (= 61.0)
ROW_UM = 30.5 * 128.0 / 64.0        # energy-axis um per model px (= 61.0)
NOISE_U8 = 5                        # Tier-0 noise_threshold 35 raw counts / (2000/255) counts-per-uint8
QMIN_FRAC = 0.08
IMIN_SUM = 1500.0

# The physics-modality channel order. sep_um is the quantity the campaign's response gate measures.
FEATURE_NAMES = ("sep_um", "q_lo", "q_hi", "row_sep_um", "sig_lo_um", "sig_hi_um")
N_FEATURES = len(FEATURE_NAMES)


def frame_grey(fr):
    a = np.asarray(fr, dtype=np.float32)
    if a.max() <= 1.5:
        a = a * 255.0
    g = a.mean(-1) if a.ndim == 3 else a
    return np.where(g >= NOISE_U8, g, 0.0)          # Tier-0-equivalent noise threshold


def proj_median(p, interpolate: bool = False):
    """Median position of a 1-D non-negative profile (the deployed energy-gated basis uses medians).

    interpolate=False reproduces the deployed integer index exactly. interpolate=True returns the
    sub-pixel 50% crossing: §8.40 F6 measured that the integer version pins every (episode, L2, sign)
    cell median onto the 61 um grid, with |integer - interpolated| p50 15.0 um, 0.41x those cells'
    sampling SE. Note a separation is a DIFFERENCE of two medians, so the interpolation removes noise
    rather than a bias -- any constant offset between the two conventions cancels.
    """
    c = np.cumsum(p)
    if c[-1] <= 0:
        return 0.0
    t = 0.5 * c[-1]
    i = int(np.searchsorted(c, t))
    if not interpolate:
        return float(i)
    if i <= 0:
        return 0.0
    i = min(i, len(c) - 1)
    c0, c1 = float(c[i - 1]), float(c[i])
    return float(i - 1) + ((t - c0) / (c1 - c0) if c1 > c0 else 0.5)


def energy_gated_sep(fr, interpolate: bool = False):
    """Energy-gated two-bunch features from one (64,192[,3]) frame (v4 centered-crop geometry).
    Returns ok, sep_um (streak-axis band-median separation), q_lo/q_hi (charge fractions),
    row_sep_um (energy-axis band-centroid separation), sig_lo/sig_hi_um (per-band streak sigma)."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    g = frame_grey(fr)
    tot = float(g.sum())
    bad = {"ok": False, "sep_um": np.nan}
    if tot < IMIN_SUM:
        return bad
    eproj = gaussian_filter1d(g.sum(1), 1.0)
    pk, props = find_peaks(eproj, prominence=0.04 * eproj.max(), distance=3)
    if len(pk) < 2:
        return bad
    top2 = np.sort(pk[np.argsort(props["prominences"])[-2:]])
    r1, r2 = int(top2[0]), int(top2[1])
    split = r1 + int(np.argmin(eproj[r1:r2 + 1]))
    lo, hi = g[:split], g[split:]
    q_lo, q_hi = lo.sum(), hi.sum()
    if min(q_lo, q_hi) < QMIN_FRAC * tot:
        return bad
    p_lo, p_hi = lo.sum(0), hi.sum(0)
    c_lo, c_hi = proj_median(p_lo, interpolate), proj_median(p_hi, interpolate)
    rows = np.arange(g.shape[0], dtype=np.float32)
    r_lo = float((lo.sum(1) * rows[:split]).sum() / q_lo)
    r_hi = float((hi.sum(1) * rows[split:]).sum() / q_hi)
    cols = np.arange(g.shape[1], dtype=np.float32)

    def _sig(p, c):
        w = p / p.sum()
        return float(np.sqrt(((cols - c) ** 2 * w).sum()))
    return {"ok": True, "sep_um": abs(c_hi - c_lo) * PX_UM,
            "q_lo": float(q_lo / tot), "q_hi": float(q_hi / tot),
            "row_sep_um": abs(r_hi - r_lo) * ROW_UM,
            "sig_lo_um": _sig(p_lo, c_lo) * PX_UM, "sig_hi_um": _sig(p_hi, c_hi) * PX_UM}


def extract_features(frames, interpolate: bool = False):
    """Batch helper for the processor: (N,H,W[,3]) -> (values (N,6) float32, ok (N,) bool).

    Invalid rows are returned as 0.0 with ok=False. They must be MASKED OUT of the loss, not imputed
    -- record §8.40 F4/F5 documents impute-plus-mask as an active hazard in this corpus (a channel
    that is 88% imputed had its real values re-amplified 2.76x by the second z-scoring, and the
    only proprio channel with 'positive skill' won by predicting its own imputation constant).
    """
    frames = np.asarray(frames)
    n = len(frames)
    vals = np.zeros((n, N_FEATURES), np.float32)
    ok = np.zeros(n, bool)
    for i in range(n):
        e = energy_gated_sep(frames[i], interpolate=interpolate)
        if not e["ok"]:
            continue
        ok[i] = True
        vals[i] = [e[k] for k in FEATURE_NAMES]
    return vals, ok
