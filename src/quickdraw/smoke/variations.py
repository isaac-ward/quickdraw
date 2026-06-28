"""Smoke test for the three train-time variations (design/models/variations.md).

Verifies, on a tiny DSAR + a tiny LSAR:
  - noise_injection: injects on inputs (training only), measured ~ desired, off when std=0 / eval;
  - physical_loss:   finite penalty, grad to the representation, LSAR decoder stays frozen;
  - contraction:     finite sigma_max + hinge, grad to backbone weights, via the eager sdpa(MATH) path;
  - the composability CONTRACT: every model satisfies one_step_states + physical_state;
  - integration: a LitWorldModel step with all three on -> finite objective + all {variation}/ log keys.

Run: uv run python -m quickdraw.smoke.variations
"""
import torch

from quickdraw.data.dataset import Normalizer
from quickdraw.models.dsar import BaseModelConfig, DataSpaceAR
from quickdraw.models.lsar import LSARConfig, LatentSpaceAR
from quickdraw.models.collapse import Naked
from quickdraw.training.variations import (VarContext, NoiseInjection, PhysicalLoss, Contraction,
                                           make_variation_suite)
from quickdraw.training.lit import LitWorldModel

DEV = "cuda" if torch.cuda.is_available() else "cpu"
R, r, VS, DT = 0.75, 0.25, 1.0, 1.0 / 60.0
NORM = Normalizer({"observation_vector": {"mean": [0.0] * 6, "std": [1.0] * 6},
                   "action": {"mean": [0.0, 0.0], "std": [1.0, 1.0]}})
results = []


def check(name, ok, extra=""):
    results.append(ok)
    print(f"  [{'OK' if ok else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")


def dsar():
    return DataSpaceAR(BaseModelConfig(d=32, depth=2, heads=2, window=8)).to(DEV).train()


def lsar():
    return LatentSpaceAR(LSARConfig(d=32, dz=8, depth=2, heads=2, window=8), Naked()).to(DEV).train()


# ---------------------------------------------------------------- noise injection
def test_noise():
    print("noise_injection:")
    obs = torch.randn(3, 12, 6, device=DEV)
    ni = NoiseInjection(0.1)
    out, logs = ni.transform_obs(obs, training=True)
    check("perturbs inputs in train", not torch.equal(out, obs))
    meas = float(logs["sigma_measured"])
    check("measured ~ desired", abs(meas - 0.1) < 0.03, f"desired=0.1 measured={meas:.3f}")
    out_eval, logs_eval = ni.transform_obs(obs, training=False)
    check("no-op at eval", torch.equal(out_eval, obs) and not logs_eval)
    out0, _ = NoiseInjection(0.0).transform_obs(obs, training=True)
    check("std=0 -> identity", torch.equal(out0, obs))


# ---------------------------------------------------------------- physical loss
def _ctx(model, preds, obs, act):
    return VarContext(model, preds, obs[:, 4:], obs, act, NORM, R, r, VS, DT, True)


def test_physical():
    print("physical_loss:")
    # DSAR: physical_state is identity -> grad to the prediction directly.
    m = dsar()
    preds = torch.randn(3, 8, 6, device=DEV, requires_grad=True)
    obs, act = torch.randn(3, 12, 6, device=DEV), torch.randn(3, 11, 2, device=DEV)
    L, logs = PhysicalLoss(1.0).loss(_ctx(m, preds, obs, act))
    Lf = float(L.detach())
    L.backward()
    check("DSAR finite penalty + grad to prediction",
          torch.isfinite(L) and preds.grad is not None and torch.isfinite(preds.grad).all(),
          f"L={Lf:.3f}")
    # continuity term (kinematic v = dp/dt): finite, logged, grad flows to the prediction.
    m2 = dsar()
    preds2 = torch.randn(3, 8, 6, device=DEV, requires_grad=True)
    Lc, logs_c = PhysicalLoss(0.0, continuity=1.0).loss(_ctx(m2, preds2, obs, act))
    Lc.backward()
    check("continuity term finite + logged + grad",
          "continuity" in logs_c and torch.isfinite(Lc) and preds2.grad is not None
          and torch.isfinite(preds2.grad).all(), f"continuity={float(logs_c.get('continuity', float('nan'))):.3f}")
    # LSAR: decoder must stay FROZEN (no grad), but the latent/encoder must receive grad.
    m = lsar()
    obs = torch.randn(3, 12, 6, device=DEV)
    z = m.encode_state(obs[:, 4:])                       # real latent depending on enc weights
    L, _ = PhysicalLoss(1.0).loss(_ctx(m, z, obs, torch.randn(3, 11, 2, device=DEV)))
    L.backward()
    dec_grad = any(p.grad is not None for p in m.dec.parameters())
    enc_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in m.enc.parameters())
    check("LSAR decoder frozen (no grad)", not dec_grad)
    check("LSAR encoder receives grad", enc_grad)
    # vision-style stub: physical_state None -> variation auto-skips.
    class NoPhys:
        def physical_state(self, pred): return None
    term, logs = PhysicalLoss(1.0).loss(_ctx(NoPhys(), torch.randn(3, 8, 6, device=DEV), obs, torch.randn(3, 11, 2, device=DEV)))
    check("physical_state None -> skipped", term is None and logs.get("skipped") == 1.0)


# ---------------------------------------------------------------- contraction
def test_contraction():
    print("contraction:")
    for name, m, tau in [("DSAR", dsar(), 1.02), ("LSAR", lsar(), 1.0)]:
        obs, act = torch.randn(4, 12, 6, device=DEV), torch.randn(4, 11, 2, device=DEV)
        m.zero_grad()
        L, logs = Contraction(1.0, tau, power_iters=2, n_sample_steps=2).loss(_ctx(m, None, obs, act))
        sig = float(logs["sigma_max"])
        Lf = float(L.detach())
        finite = torch.isfinite(L) and (sig == sig) and Lf >= 0
        L.backward()
        gp = [p.grad for p in m.parameters() if p.grad is not None]
        wgrad = len(gp) > 0 and all(torch.isfinite(g).all() for g in gp)
        check(f"{name} sigma_max finite + hinge + weight grads", bool(finite and wgrad),
              f"sigma_max={sig:.3f} L={Lf:.3e} grads={len(gp)}")


# ---------------------------------------------------------------- composability contract
def test_contract():
    print("contract (hook surface for any model):")
    for name, m, sdim in [("DSAR", dsar(), 6), ("LSAR", lsar(), 8)]:
        sw = torch.randn(2, 8, sdim, device=DEV); aw = torch.randn(2, 8, 2, device=DEV)
        nxt = m.one_step_states(sw, aw, attn_eager=True)
        ok_step = tuple(nxt.shape) == (2, sdim)
        phys = m.physical_state(torch.randn(2, 5, sdim, device=DEV))
        ok_phys = phys is None or phys.shape[-1] == 6
        check(f"{name} one_step_states + physical_state", ok_step and ok_phys)


# ---------------------------------------------------------------- integration via LitWorldModel
def test_integration():
    print("integration (LitWorldModel, all variations on):")
    varcfg = {"noise_injection": {"std": 0.1}, "physical_loss": {"weight": 1.0, "continuity": 1.0},
              "contraction": {"weight": 1.0, "target": 1.0, "power_iters": 2, "n_sample_steps": 2}}
    for name, m in [("DSAR", dsar()), ("LSAR", lsar())]:
        lit = LitWorldModel(m, NORM, R, r, VS, 4, 8, 0.0, 0.0, 0, 1e-3, 0.0, 0, variations=varcfg).to(DEV)
        keys = []
        lit.log = lambda k, v=None, **kw: keys.append(k)            # capture log keys (no Trainer)
        batch = {"obs_seq": torch.randn(4, 12, 6, device=DEV), "act_seq": torch.randn(4, 11, 2, device=DEV)}
        # run under bf16 autocast to mirror real training precision (bf16-mixed); contraction disables
        # autocast internally for its second-order AD. The objective + full backward must stay finite.
        with torch.autocast(device_type=DEV, dtype=torch.bfloat16, enabled=(DEV == "cuda")):
            obj = lit._step(batch, "train")
        want = {"noise_injection/sigma_desired", "noise_injection/sigma_measured",
                "train/loss/physical", "train/loss/contraction",   # loss components in the {tag}/loss/ breakdown
                "physical_loss/d_off", "physical_loss/v_off", "physical_loss/continuity",
                "contraction/sigma_max"}
        missing = want - set(keys)
        check(f"{name} finite objective (bf16 autocast)", bool(torch.isfinite(obj)), f"obj={float(obj.detach()):.3e}")
        check(f"{name} all variation log keys present", not missing, f"missing={missing or 'none'}")
        obj.backward()  # full backward through noise + physical + contraction must be finite
        gp = [p.grad for p in lit.parameters() if p.grad is not None]
        check(f"{name} end-to-end backward finite", all(torch.isfinite(g).all() for g in gp))


def main():
    torch.manual_seed(0)
    test_noise(); test_physical(); test_contraction(); test_contract(); test_integration()
    print("\nALL OK" if all(results) else "\nSOME FAILED")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
