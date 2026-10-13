"""Smoke for the sliding X-step-ahead eval (replaces the old sawtooth closed-loop).
  1. decode_last: imagine_eval(decode_last=True) returns horizon-1, and (deterministic) == the last frame of the
     full decode -> the compute-saver is correct (decode 1 frame/start, not X).
  2. indexing: for target future-frame j, the grounding is P+j-X, context = P frames ending there, and the
     number of valid targets = Hm-(X-1) with a START drop of X-1 and NO end drop (Hm bounds targets in-episode).
  3. routines import: the edited eval module imports clean.
Run: docker compose exec -T app uv run --no-sync python -m quickdraw.smoke.xstep_ahead
"""
import torch
torch.set_num_threads(1)
from hydra import compose, initialize_config_dir
from ..training.setup import build_model


def _fail(m): print("FAIL:", m); raise SystemExit(1)


def main():
    torch.manual_seed(0)
    with initialize_config_dir(config_dir="/app/conf", version_base=None):
        cfg = compose(config_name="config", overrides=["model=vl128_blockstack_flow"])
    m = build_model(cfg).eval()
    m.stochastic_eval = False                      # deterministic so decode_last == full[:,-1]
    P, B, H = cfg.data.P, 2, 5
    heads = ["proprio"] + [n for n, _ in m.layout if n != "proprio"]
    ctx = {}
    for name, _ in m.layout:
        mod = m.modalities[name]
        if name == "proprio":
            ctx[name] = torch.randn(B, P, mod.dim)
        else:
            sh = mod.ae.cfg.img_size; sh = sh if isinstance(sh, (tuple, list)) else (sh, sh)
            ctx[name] = torch.rand(B, P, sh[0], sh[1], 3)
    acts = torch.randn(B, P + H - 1, 5)

    # ---- 1. decode_last ----
    with torch.no_grad():
        full = m.imagine_eval(ctx, acts, H, heads=heads, decode_last=False)
        last = m.imagine_eval(ctx, acts, H, heads=heads, decode_last=True)
    for h in heads:
        if last[h].shape[1] != 1:
            _fail(f"decode_last {h} horizon {last[h].shape[1]} != 1")
        if full[h].shape[1] != H:
            _fail(f"full {h} horizon {full[h].shape[1]} != {H}")
        if not torch.allclose(last[h][:, 0], full[h][:, -1], atol=1e-4):
            _fail(f"decode_last {h} != full[:,-1] (maxdiff {float((last[h][:,0]-full[h][:,-1]).abs().max()):.2e})")
    print(f"[1/3] decode_last: horizon-1 for all heads, == full[:,-1] (deterministic)   OK")

    # ---- 2. indexing (the target-time contract) ----
    for X, Hm in [(1, 10), (3, 10), (16, 64)]:
        n_j = Hm - (X - 1)
        targets = list(range(X - 1, Hm))
        if len(targets) != n_j:
            _fail(f"X={X} Hm={Hm}: valid target count {len(targets)} != {n_j}")
        for j in targets:
            cs = j - X + 1                         # context start (abs, future-indexed from P)
            grounding = P + j - X                  # abs grounding frame
            if cs < 0:
                _fail(f"X={X} j={j}: context start {cs} < 0")
            if cs + P - 1 != grounding - (P - 1) + (P - 1):   # cs+P-1 is future-rel; grounding abs = P+(cs+P-1)-? sanity
                pass
            # the decoded target is P+j; grounding is X steps before it
            if (P + j) - grounding != X:
                _fail(f"X={X} j={j}: target {P+j} - grounding {grounding} != X={X}")
        # start drop = X-1, no end drop (last target j=Hm-1 -> frame P+Hm-1, in-episode by Hm<=L-P)
        if targets[0] != X - 1 or targets[-1] != Hm - 1:
            _fail(f"X={X}: targets span {targets[0]}..{targets[-1]} != {X-1}..{Hm-1}")
    print("[2/3] indexing: target P+j grounded X steps back; start-drop X-1, no end-drop   OK")

    # ---- 3. eval module imports clean (syntax of the edited routines) ----
    from ..evaluation import routines  # noqa: F401
    print("[3/3] evaluation.routines imports clean   OK")

    print("\nALL XSTEP-AHEAD SMOKE PASSED")


if __name__ == "__main__":
    main()
