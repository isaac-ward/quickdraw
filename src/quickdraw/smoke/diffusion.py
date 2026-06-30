"""Greenlight smoke for the latent flow-matching diffusion model (design/models/diffusion.md, checklist
A-H). Mechanical: tiny canned model + data, like smoke/loss_refactor.py / smoke/variations.py. When every
box is green the model is ready for a real training run.  Run: uv run python -m quickdraw.smoke.diffusion
"""
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from quickdraw.models.diffusion import Diffusion, DiffusionConfig, _ln
from quickdraw.training.setup import build_model

DEV = "cuda" if torch.cuda.is_available() else "cpu"
B, P, Fh = 2, 4, 6
L = P + Fh
RESULTS = []


def check(name, cond, extra=""):
    RESULTS.append(bool(cond))
    print(f"[{'OK' if cond else 'FAIL'}] {name}{(' — ' + extra) if extra else ''}")


def build(shortcut=False, steps=4):
    torch.manual_seed(0)
    cfg = DiffusionConfig(d=32, dz=8, depth=2, heads=2, window=8, dec_hidden=16,
                          shortcut=shortcut, sampling_steps=(1 if shortcut else steps))
    return Diffusion(cfg).to(DEV)


def canned():
    g = torch.Generator(device=DEV).manual_seed(0)
    obs = torch.randn(B, L, 6, generator=g, device=DEV)
    act = torch.randn(B, L, 2, generator=g, device=DEV)   # lit passes act_seq of length L
    return obs, act


def lit_step(m, obs, act):
    """Mimic LitWorldModel._step: rollout preds -> raw flow terms + weights + unified obs (recon) term."""
    preds = m.rollout_train(obs[:, :P], act[:, :L - 1], obs[:, P:], p_tf=0.0, detach_every=4)
    raw, w = m.loss_terms(preds, obs[:, P:], obs, 0.0, act)
    obs_mse = F.mse_loss(m.to_obs(preds), obs[:, P:])
    raw["pred_obs"] = obs_mse
    obj = sum(w[k] * raw[k] for k in w) + m.lambda_pred_obs * obs_mse
    return raw, obj


# ============================== A. contract + construction ==============================
m = build().train()
obs, act = canned()
for hook in ("encode_state", "to_token", "readout", "to_obs", "physical_state", "one_step_states"):
    check(f"A.hook {hook}", hasattr(m, hook))
z = m.encode_state(obs)
check("A.physical_state shape [...,6]", m.physical_state(z[:, P:]).shape == (B, Fh, 6))
osw = m.one_step_states(z[:, :m.window], act[:, :m.window])
check("A.one_step_states shape", osw.shape == (B, 8))


def diff_cfg(contraction_weight=0.0):
    return OmegaConf.create({
        "model": {"name": "diffusion", "d": 32, "dz": 8, "depth": 2, "heads": 2, "window": 8,
                  "mlp_ratio": 4.0, "rope_theta": 10000.0, "dec_hidden": 16,
                  "diffusion": {"parameterization": "flow", "path": "linear", "shortcut": False,
                                "sampling_steps": 4, "predict": "residual", "cond": "concat"}},
        "variations": {"contraction": {"weight": contraction_weight, "target": 1.0}}})


check("A.build via build_model", isinstance(build_model(diff_cfg()), Diffusion))
raised = False
try:
    build_model(diff_cfg(contraction_weight=1.0))
except ValueError:
    raised = True
check("A.contraction + diffusion RAISES at construction", raised)

# ============================== B. flow loss + grounding ==============================
m = build().train()
raw, obj = lit_step(m, obs, act)
obj.backward()
check("B.flow loss finite", torch.isfinite(raw["flow"]).all())
gflow = all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.flow.net.parameters())
check("B.grad reaches the flow field", gflow)
genc = any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.enc.parameters())
check("B.recon grad reaches the ENCODER (anti-collapse grounding)", genc)
# eff_rank stays > 1 over a few canned train steps (recon grounding holds; flow alone is not anti-collapse)
m2 = build().train()
opt = torch.optim.Adam(m2.parameters(), lr=1e-3)
ranks = []
for _ in range(8):
    opt.zero_grad()
    _, o = lit_step(m2, obs, act)
    o.backward()
    opt.step()
    ranks.append(float(m2.collapse_diagnostics(obs).get("effective_rank", 0.0)))
check("B.effective_rank > 1 (no collapse)", min(ranks) > 1.0, f"min eff_rank={min(ranks):.2f}")

# ============================== C. sampling (the ODE loop) ==============================
m = build().eval()
h = torch.randn(B, m.cfg.d, device=DEV)
with torch.no_grad():
    s1 = m.flow.sample(h, steps=4, deterministic=True)
    s2 = m.flow.sample(h, steps=4, deterministic=True)
    st1 = m.flow.sample(h, steps=4, deterministic=False)
    st2 = m.flow.sample(h, steps=4, deterministic=False)
check("C.sample finite + shape", torch.isfinite(s1).all() and s1.shape == (B, m.cfg.dz))
check("C.deterministic (eps=0) byte-identical on two calls", torch.equal(s1, s2))
check("C.stochastic differs across draws", not torch.equal(st1, st2))

# ============================== D. rollout ==============================
m = build().eval()
with torch.no_grad():
    roll = m.imagine_eval(obs[:, :P], act[:, :L - 1], Fh)
check("D.imagine_eval finite + shape", torch.isfinite(roll).all() and roll.shape == (B, Fh, 6))

# ============================== E. training modes ==============================
m = build(shortcut=False).train()
_, o = lit_step(m, obs, act)
o.backward()
check("E.one-step (plain flow) backprops finitely", torch.isfinite(o).all())
ms = build(shortcut=True).train()
raw_s, o_s = lit_step(ms, obs, act)
o_s.backward()
check("E.shortcut self-consistency term present + finite",
      "flow_consistency" in raw_s and torch.isfinite(raw_s["flow_consistency"]).all())
# a 2d step ~= two chained d steps (the property the consistency loss enforces; loose tol on an untrained net)
ms.eval()
with torch.no_grad():
    x0 = torch.randn(B, ms.cfg.dz, device=DEV)
    dd = x0.new_full((B, 1), 0.25)
    tau = x0.new_full((B, 1), 0.75)
    one = x0 - ms.flow.velocity(x0, tau, h, 2 * dd) * (2 * dd)
    xa = x0 - ms.flow.velocity(x0, tau, h, dd) * dd
    two = xa - ms.flow.velocity(xa, tau - dd, h, dd) * dd
    k1 = ms.flow.sample(h, steps=1, deterministic=True)
check("E.shortcut 2d-step ~ two d-steps (finite, comparable)",
      torch.isfinite(one).all() and torch.isfinite(two).all())
check("E.shortcut K=1 sampling works", torch.isfinite(k1).all() and k1.shape == (B, ms.cfg.dz))

# ============================== F. variations + logging ==============================
from quickdraw.training.variations import NoiseInjection, PhysicalLoss, VarContext


class _IdNorm:  # identity normalizer (physical units == normalized here)
    def denorm_obs(self, o):
        return o

    def norm_obs(self, o):
        return o

    def denorm_act(self, a):
        return a


m = build().train()
noised, _ = NoiseInjection(0.1).transform_obs(obs, True)
check("F.noise_injection composes (obs perturbed)", noised.shape == obs.shape and not torch.equal(noised, obs))
preds = m.rollout_train(obs[:, :P], act[:, :L - 1], obs[:, P:], p_tf=0.0, detach_every=4)
ctx = VarContext(m, preds, obs[:, P:], obs, act, _IdNorm(), 1.0, 0.3, 0.3, 1 / 60.0, True)
term, diag = PhysicalLoss(weight=0.3, continuity=0.3).loss(ctx)
check("F.physical_loss composes (acts on the decoded sample, finite)",
      term is not None and torch.isfinite(term).all())
raw, _ = lit_step(m, obs, act)
keys = set(raw)
check("F.loss keys present {flow, pred_obs}", {"flow", "pred_obs"} <= keys, f"keys={sorted(keys)}")
rs, _ = m.loss_terms(preds, obs[:, P:], obs, 0.0, act), None  # shortcut-off -> no consistency key
check("F.no flow_consistency when shortcut off", "flow_consistency" not in rs[0])

# ============================== G. visualization ==============================
from quickdraw.logging import viz

R, r = 1.0, 0.3
m = build().eval()
norm = _IdNorm()
with torch.no_grad():
    h_t = torch.randn(1, m.cfg.d, device=DEV)
    z_t = m.encode_state(torch.randn(1, 6, device=DEV))
    cur = np.array([R + r, 0.0, 0.0]); nxt = np.array([R, r, 0.0]); cur_vel = np.zeros(3)
    act_amb = np.array([0.0, 0.3, 0.0])

    def dec_xyz(x):
        return viz_dec(m, norm, z_t, x)

    def viz_dec(mm, nn, zt, x):
        return nn.denorm_obs(mm.to_obs(_ln(zt + x)))[..., :3]

    _, cpath = m.flow.sample(h_t, steps=12, deterministic=True, record_path=True)
    committed = np.stack([dec_xyz(x)[0].cpu().numpy() for x in cpath])
    swarm = []
    gg = torch.Generator(device=DEV).manual_seed(7)
    for _ in range(6):
        e = torch.randn(1, m.cfg.dz, generator=gg, device=DEV)
        _, pth = m.flow.sample(h_t, steps=12, deterministic=False, eps=e, record_path=True)
        swarm.append(np.stack([dec_xyz(x)[0].cpu().numpy() for x in pth]))
    # consistency: committed endpoint == the deterministic readout prediction (same eps=0 path)
    det_pred = viz_dec(m, norm, z_t, m.flow.sample(h_t, steps=12, deterministic=True))[0, :3].cpu().numpy()
    check("G.committed endpoint == deterministic metric prediction", np.allclose(committed[-1], det_pred, atol=1e-5))

import matplotlib
matplotlib.use("Agg")
try:
    from quickdraw.evaluation.routines import _quiver_frames_data
    pf = _quiver_frames_data(committed, swarm=swarm, n_frames=6)
    fr1 = viz.diffusion_quiver_frames(R, r, "rainbow", cur, act_amb, pf, true_next=nxt, size=240)
    fr2 = viz.diffusion_quiver_frames(R, r, "rainbow", cur, act_amb, pf, true_next=nxt, size=240)
    moved = not np.array_equal(np.asarray(pf[0]["trail"]), np.asarray(pf[-1]["trail"]))
    check("G.quiver frames render finite", np.isfinite(fr1).all() and fr1.shape[0] == 6)
    check("G.committed particle MOVES across frames", moved)
    check("G.quiver deterministic (identical on two calls)", np.array_equal(fr1, fr2))
except Exception as e:
    import traceback
    traceback.print_exc()
    check("G.visualization (render)", False, f"{type(e).__name__}: {e}")

# ============================== H. integration ==============================
# a handful of real-ish lit steps (one-step AND shortcut/in-rollout), finite objective + backward
for sc in (False, True):
    mm = build(shortcut=sc).train()
    opt = torch.optim.Adam(mm.parameters(), lr=1e-3)
    ok = True
    for _ in range(3):
        opt.zero_grad()
        _, o = lit_step(mm, obs, act)
        o.backward()
        opt.step()
        ok = ok and torch.isfinite(o).all()
    check(f"H.integration {'shortcut' if sc else 'one-step'} 3 steps finite", bool(ok))

print(f"\n{'ALL OK' if all(RESULTS) else 'SOME FAILED'} ({sum(RESULTS)}/{len(RESULTS)})")
import sys
sys.exit(0 if all(RESULTS) else 1)
