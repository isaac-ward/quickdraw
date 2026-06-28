"""Bit-verify the fixed-window rollout refactor (option 3) against the current variable-length rollout.
Small config (window=8, P=4, H=12) so it exercises padded steps (L<window) AND sliding steps (L>=window).
  ref   : current code -> save state_dict + inputs + imagine_eval output.
  check : refactored code -> reload, rerun, assert allclose.
Run: uv run python -m quickdraw.smoke.rollout_bitref {ref|check}
"""
import sys

import torch

from quickdraw.models.base import BaseModelConfig, BaseWorldModel

PATH = "/tmp/rollout_bitref.pt"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
B, P, H = 2, 4, 12


def make():
    torch.manual_seed(0)
    return BaseWorldModel(BaseModelConfig(d=32, depth=2, heads=2, window=8)).to(DEV).eval()


def run(m, ctx, act):
    with torch.no_grad():
        return m.imagine_eval(ctx, act, H)


mode = sys.argv[1] if len(sys.argv) > 1 else "ref"
if mode == "ref":
    m = make()
    g = torch.Generator(device=DEV).manual_seed(7)
    ctx = torch.randn(B, P, 6, generator=g, device=DEV)
    act = torch.randn(B, P + H - 1, 2, generator=g, device=DEV)
    out = run(m, ctx, act)
    torch.save({"sd": m.state_dict(), "ctx": ctx, "act": act, "out": out}, PATH)
    print(f"[ref] saved rollout out {tuple(out.shape)} sum={float(out.sum()):.6f}")
else:
    ref = torch.load(PATH, map_location=DEV)
    m = make()
    m.load_state_dict(ref["sd"])
    out = run(m, ref["ctx"], ref["act"])
    d = float((out - ref["out"]).abs().max())
    eq = torch.allclose(out, ref["out"], atol=1e-5, rtol=1e-4)
    print(f"[check] ROLLOUT match={eq}  max|d|={d:.3e}")
    print("IDENTICAL" if eq else "MISMATCH")
