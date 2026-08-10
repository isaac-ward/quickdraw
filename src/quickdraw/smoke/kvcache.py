"""Guard for the temporal KV-cache rollout (models/transformer.py KVRing + spacetime.forward_cached +
multimodal._rollout_cached). The cache is an INFERENCE optimization that must not change the answer: the
cached rollout has to match the parallel (full-window recompute) rollout numerically. We assert that in
fp32 (tight — isolates the cache from bf16/decode), then print the realized GPU speedup + bf16 divergence.
Run: uv run python -m quickdraw.smoke.kvcache"""

from __future__ import annotations

import torch

from ..models.modalities import ModalitySpec
from ..models.multimodal import MultiModalFlow
from ..models.spacetime import SpaceTimeTransformer
from ..models.transformer import _block_mask

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _model(window):
    specs = [ModalitySpec(name="proprio", kind="vector", dim=6, decode_kind="flow", decode_param="x0"),
             ModalitySpec(name="image", kind="image", num_tokens=4, img_size=32, patch=8, ae_depth=2,
                          decode_kind="flow", decode_param="x0")]
    m = MultiModalFlow(specs, d=32, depth=2, heads=4, window=window, mlp_ratio=4.0, rope_theta=10000.0,
                       action_dim=2, sampling_steps=4,
                       # PIN: stochastic_eval defaults TRUE since 2026-08-10 (train and eval must roll on the
                       # same distribution). Cached-vs-uncached PARITY is only defined for a deterministic
                       # readout -- with sampling each rollout draws fresh eps and |delta| is ~4, not ~1e-6.
                       stochastic_eval=False).to(DEV).eval()
    return m


def _inputs(m, B, P, horizon):
    ctx = {"proprio": torch.randn(B, P, 6, device=DEV),
           "image": torch.rand(B, P, 32, 32, 3, device=DEV)}
    act = torch.randn(B, P - 1 + horizon, 2, device=DEV)
    return ctx, act


def main():
    torch.manual_seed(0)
    ok = []

    def check(name, cond, extra=""):
        ok.append(bool(cond)); print(f"[{'OK' if cond else 'FAIL'}] {name}{(' — ' + extra) if extra else ''}")

    # CORRECTNESS (the real invariant): the cached step-by-step backbone == a FULL-SEQUENCE forward with the
    # sliding-window causal mask (the true windowed-attention semantics). This holds for T > window, where the
    # cache preserves each position's true [j-W+1, j] window at every layer. (The legacy per-step rollout
    # does NOT match this beyond the window — it re-feeds only the last W bags, truncating deep-layer
    # receptive fields at the window's left edge; the cache is the more faithful computation.)
    for W, T in [(8, 6), (8, 30)]:
        d, depth, heads, N = 32, 3, 4, 5
        bb = SpaceTimeTransformer(d, depth, heads, W, 4.0, n_slots=N).to(DEV).eval()
        x = torch.randn(1, T, N, d, device=DEV)
        with torch.no_grad():
            full = bb(x, temporal_block_mask=_block_mask(W, T, x.device))          # (1,T,N,d) true window
            cache = bb.make_cache()
            cached = torch.stack([bb.forward_cached(x[:, t], cache, t) for t in range(T)], dim=1)
        diff = float((full - cached).abs().max())
        check(f"cache == full masked forward (W={W},T={T},depth={depth})", diff < 1e-4, f"max|Δ| {diff:.2e}")

    # ROLLOUT parity when the whole rollout fits in the window (P+horizon <= W): no position's window is
    # truncated, so the cached rollout equals the legacy per-step rollout EXACTLY (end-to-end incl. readout).
    m = _model(16)
    ctx, act = _inputs(m, B=2, P=3, horizon=12)                                    # 3+12=15 <= 16
    with torch.no_grad():
        bag_par = m._rollout(ctx, act, 12, 0.0, None, 0, use_cache=False)
        bag_kv = m._rollout(ctx, act, 12, 0.0, None, 0, use_cache=True)
    diff = float((bag_par - bag_kv).abs().max())
    check("rollout parity within window (P+H<=W)", diff < 1e-4, f"max|Δ| {diff:.2e}")

    # SHAPE: imagine_eval (bf16 + decode) still returns the right obs shapes with the cache on.
    m = _model(8)
    ctx, act = _inputs(m, B=2, P=3, horizon=6)
    out = m.imagine_eval(ctx, act, 6)
    check("imagine_eval proprio shape", out["proprio"].shape == (2, 6, 6))
    check("imagine_eval image shape", out["image"].shape == (2, 6, 32, 32, 3))

    # SPEEDUP report (GPU only, longer horizon where the O(H) vs O(H*W) gap shows).
    if DEV == "cuda":
        m = _model(32)
        ctx, act = _inputs(m, B=8, P=3, horizon=64)
        rep = m.kvcache_report(ctx, act, 64, heads=["proprio"])
        print("  kvcache_report:", {k: round(v, 4) for k, v in rep.items()})
        check("bf16 divergence bounded", rep["kvcache/latent_max_abs_diff"] < 0.2,
              f"bf16 max|Δ| {rep['kvcache/latent_max_abs_diff']:.3f}")
        check("cache is faster", rep["kvcache/speedup"] > 1.0, f"{rep['kvcache/speedup']:.2f}x")

    print(f"\n{'ALL OK' if all(ok) else 'FAILURES'} ({sum(ok)}/{len(ok)})")
    raise SystemExit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
