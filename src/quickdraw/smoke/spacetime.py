"""Greenlight smoke for the factorized space-time transformer (models/spacetime.py). Checks shape,
TEMPORAL causality (a token at step t cannot see step t+1), SPATIAL coupling (tokens within a step DO
see each other), and that the eager (double-backward) path runs. Run: uv run python -m quickdraw.smoke.spacetime
"""
import torch

from quickdraw.models.spacetime import SpaceTimeTransformer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
R = []


def check(name, cond, extra=""):
    R.append(bool(cond))
    print(f"[{'OK' if cond else 'FAIL'}] {name}{(' — ' + extra) if extra else ''}")


def main():
    torch.manual_seed(0)
    B, T, N, d = 2, 12, 5, 64
    m = SpaceTimeTransformer(dim=d, depth=2, heads=4, window=T, mlp_ratio=4.0, n_slots=N).to(DEV).eval()
    x = torch.randn(B, T, N, d, device=DEV)
    with torch.no_grad():
        y = m(x)
    check("shape (B,T,N,d) preserved", y.shape == (B, T, N, d), str(tuple(y.shape)))

    # TEMPORAL causality: perturb input at step t1 (all its tokens); outputs at steps < t1 must not change.
    # Use NON-uniform noise: a uniform shift would be removed by the input LayerNorm (mean-subtraction).
    t1 = 7
    x2 = x.clone()
    x2[:, t1] += torch.randn(B, N, d, device=DEV)
    with torch.no_grad():
        y2 = m(x2)
    pre_same = torch.allclose(y[:, :t1], y2[:, :t1], atol=1e-5)
    post_diff = not torch.allclose(y[:, t1], y2[:, t1], atol=1e-5)
    check("temporal causal: steps < t unchanged by perturbing step t", pre_same)
    check("temporal: step t itself changes", post_diff)

    # SPATIAL coupling: perturb ONE token at step t; the OTHER tokens at the SAME step must change.
    x3 = x.clone()
    x3[:, t1, 0] += torch.randn(B, d, device=DEV)
    with torch.no_grad():
        y3 = m(x3)
    other_tokens_change = not torch.allclose(y[:, t1, 1:], y3[:, t1, 1:], atol=1e-5)
    check("spatial: other tokens in the step react (within-step attention)", other_tokens_change)

    # eager (double-backward) path runs and matches shape
    xe = torch.randn(B, T, N, d, device=DEV, requires_grad=True)
    ye = m(xe, attn_eager=True)
    g = torch.autograd.grad(ye.sum(), xe, create_graph=True)[0]
    gg = torch.autograd.grad(g.sum(), xe)[0]               # second-order (contraction needs this)
    check("eager path: double-backward works", torch.isfinite(gg).all() and ye.shape == (B, T, N, d))

    print(f"\n{'ALL OK' if all(R) else 'SOME FAILED'} ({sum(R)}/{len(R)})")
    import sys
    sys.exit(0 if all(R) else 1)


if __name__ == "__main__":
    main()
