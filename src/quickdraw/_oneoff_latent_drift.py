"""ONE-OFF (2026-08-24): two no-training diagnostics on a trained checkpoint, run from the CLI:

    uv run python -m quickdraw._oneoff_latent_drift <run>/checkpoints/best.ckpt out.json

Committed as a one-off (same pattern as _oneoff_action_sensitivity.py) rather than promoted to an eval
routine, because both answers are architectural facts about this model family, not per-epoch telemetry.

WHY IT EXISTS. The redesigned `eval_denoising_filmstrip` showed the latent flow's refinement gaining
+1.2 to +2.0 dB at rollout depth h<=32 and EXACTLY NOTHING at h=64 (PSNR 13.07 -> 12.91 across 8 Euler
steps). Two candidate causes, both cheap to settle without retraining.

WHAT IT FOUND on bott_recon1/best.ckpt (recon_frac=1.0, the best long-horizon config measured):

  TEST 1 -- LATENT DRIFT. |pred|/|true| and var-ratio are EXACTLY 1.0000 at every horizon, because
  latent_norm=layernorm pins them (predict_next applies _ln every step) -- the latent cannot drift in
  SCALE. What it does is ROTATE AWAY:
        h:        1      8     16     32     64    128
        cos(pred,true)   0.980  0.709  0.479  0.264  0.138  0.059
        cos(pred,z_ctx)  0.989  0.806  0.618  0.392  0.175  0.053
  By h=64 the rolled-out latent is ~orthogonal to the true latent AND to the context latent it started
  from: not frozen, not exploding, DIFFUSING over the sphere. That is why refinement does nothing at
  depth -- the flow is asked for a velocity at a point it has no information about.
  CONSEQUENCE: the long-horizon blur is NOT the mse decoder hedging over an uncertain latent (the
  hypothesis this probe was written to test). The latent is not uncertain, it is WRONG, so we are
  decoding a different scene. A generative decoder would render that sharply -- a sharp WRONG image.
  Confirmed independently: decoding TRUE latents (the AE floor) gives LPIPS 0.19 against 0.31 for
  predicted latents, so the decoder renders sharply when handed a correct latent. The dynamics is the
  limiter, not the codec.

  TEST 2 -- SAMPLING-STEPS SWEEP: a clean NULL. The filmstrip's mid-path decline (h=16 peaking at k5,
  h=32 at k7) does NOT translate to the rollout. Everything K>=2 is within noise; K=16 is marginally
  best on LPIPS (0.385 vs 0.392 @h64) for 2.7x the sampling cost, and the default K=6 is fine. But
  K=1 is BADLY broken at depth (10.62 dB / 0.498 LPIPS @h64, 2.5 dB below K=2), so single-step
  shortcut sampling is not viable on this model.
"""
import os, sys, json

import numpy as np
import torch
from omegaconf import OmegaConf

CKPT, OUT = sys.argv[1], sys.argv[2]
RUN = os.path.dirname(os.path.dirname(CKPT))
cfg = OmegaConf.load(os.path.join(RUN, "checkpoints", "config.resolved.yaml"))
from quickdraw.training.setup import build_model, normalizer, resolve_data_root
from quickdraw.data.dataset import load_split_episodes_mm
from quickdraw.evaluation.openloop import _lpips_net

dev = torch.device("cuda")
model = build_model(cfg).to(dev).eval()
sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
miss, unex = model.load_state_dict(sd, strict=False)
print(f"[load] {os.path.basename(RUN)} missing={len(miss)} unexpected={len(unex)}", flush=True)
m = getattr(model, "_orig_mod", model)
norm = normalizer(cfg)
img_head = next((n for n, _ in m.layout if n != "proprio"), None)
img_size = next((mod.ae.cfg.img_size for mod in m.modalities.values() if hasattr(mod, "ae")), 128)
# frames come back as a DICT keyed by camera (data/dataset.py). Single-camera script:
# name the key once rather than indexing position 2 as if there were only ever one view.
_CAMK = str(cfg.data.get("cam", "fpv"))
eps_ds = load_split_episodes_mm(resolve_data_root(cfg), "val", img_size=img_size,
                                cam=cfg.data.get("cam", "fpv"), repo_id=cfg.data.get("repo_id", "torus"))
P = int(cfg.data.P)
HZ = [1, 8, 16, 32, 64, 128]
HMAX = max(HZ)
rng = np.random.default_rng(0)
picks = []
for _ in range(200):
    i = int(rng.integers(len(eps_ds)))
    o, a, _fr = eps_ds[i]
    im = _fr[_CAMK]
    if len(o) > P + HMAX + 2:
        t0 = int(rng.integers(P, len(o) - HMAX - 1))
        picks.append((i, t0))
    if len(picks) == 6:
        break
print(f"[data] {len(eps_ds)} val episodes, using {len(picks)} (episode, t0) samples", flush=True)
res = {}

# ---------------- TEST 1: latent drift ----------------
def _ln(x):
    return torch.nn.functional.layer_norm(x, (x.shape[-1],))

# cos_frozen_true is THE baseline this probe was missing on its first run: how well a DO-NOTHING predictor
# (emit the context latent forever) scores against the true latent. Without it, cos(pred,true)=0.138 at h=64
# cannot be read -- it is only a failure if frozen does BETTER, and only a success if frozen does worse.
drift = {h: {"norm_ratio": [], "cos_true": [], "cos_frozen": [], "cos_frozen_true": [], "var_ratio": []} for h in HZ}
with torch.no_grad():
    for (ei, t0) in picks:
        o, a, _fr = eps_ds[ei]
        im = _fr[_CAMK]
        obs = {"proprio": norm.norm_obs(torch.from_numpy(o)).float()[None].to(dev),
               img_head: torch.from_numpy(im).float().div(255.0)[None].to(dev)}
        act = norm.norm_act(torch.from_numpy(a)).float()[None].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = m.encode_state(obs).float()                      # (1,T,n_state,d) ENCODER latents = the manifold
        hist = z[:, :t0 + 1]
        z_ctx = z[0, t0]                                          # the frozen/no-op reference
        for hstep in range(1, HMAX + 1):
            t = t0 + hstep - 1
            w = min(m.window, hist.shape[1])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hb = m.backbone(m._to_input(hist[:, -w:], act[:, t - w + 1:t + 1]))[:, -1]
                hs = m._cond(hb, act[:, t])
                nb = m.predict_next(hs, hist[:, -1]).float()      # (1,n_state,d) committed rollout step
            hist = torch.cat([hist, nb[:, None]], dim=1)
            if hstep in HZ and t0 + hstep < z.shape[1]:
                p, tr = nb[0], z[0, t0 + hstep]
                fl = lambda x: x.reshape(-1)
                drift[hstep]["norm_ratio"].append(float(p.norm() / tr.norm()))
                drift[hstep]["cos_true"].append(float(torch.nn.functional.cosine_similarity(fl(p), fl(tr), 0)))
                drift[hstep]["cos_frozen"].append(float(torch.nn.functional.cosine_similarity(fl(p), fl(z_ctx), 0)))
                drift[hstep]["cos_frozen_true"].append(                       # DO-NOTHING baseline
                    float(torch.nn.functional.cosine_similarity(fl(z_ctx), fl(tr), 0)))
                drift[hstep]["var_ratio"].append(float(p.var() / tr.var()))
print("\n=== TEST 1: LATENT DRIFT vs the encoder's own latents ===")
print(f"{'h':>5}{'cos(pred,true)':>16}{'cos(FROZEN,true)':>18}{'gain':>8}"
      f"{'cos(pred,z_ctx)':>17}{'|p|/|t|':>9}{'var':>7}")
for h in HZ:
    d = drift[h]
    if not d["cos_true"]:
        continue
    res[f"drift_h{h}"] = {k: float(np.mean(v)) for k, v in d.items()}
    ct, cf = np.mean(d["cos_true"]), np.mean(d["cos_frozen_true"])
    print(f"{h:>5}{ct:>16.4f}{cf:>18.4f}{ct - cf:>+8.4f}"
          f"{np.mean(d['cos_frozen']):>17.4f}{np.mean(d['norm_ratio']):>9.4f}{np.mean(d['var_ratio']):>7.4f}")

# ---------------- TEST 2: sampling_steps sweep ----------------
lp = _lpips_net(dev)
K_ORIG = int(m.sampling_steps)
print(f"\n=== TEST 2: sampling_steps sweep (training/inference default = {K_ORIG}) ===")
def ol_metrics(K):
    m.sampling_steps = K
    ps, lps = {h: [] for h in HZ}, {h: [] for h in HZ}
    with torch.no_grad():
        for (ei, t0) in picks:
            o, a, _fr = eps_ds[ei]
            im = _fr[_CAMK]
            ctx = {"proprio": norm.norm_obs(torch.from_numpy(o[t0 - P + 1:t0 + 1])).float()[None].to(dev),
                   img_head: torch.from_numpy(im[t0 - P + 1:t0 + 1]).float().div(255.0)[None].to(dev)}
            acts = norm.norm_act(torch.from_numpy(a[t0 - P + 1:t0 + HMAX])).float()[None].to(dev)
            out = m.imagine_eval(ctx, acts, HMAX, heads=[img_head], decode_chunk=16)[img_head][0].clamp(0, 1)
            for h in HZ:
                if t0 + h >= len(im):
                    continue
                pr = out[h - 1]
                gt = torch.from_numpy(im[t0 + h].astype(np.float32) / 255.0).to(dev)
                ps[h].append(float(10 * torch.log10(1.0 / torch.clamp(((pr - gt) ** 2).mean(), min=1e-10))))
                if lp is not None:
                    a4 = pr.permute(2, 0, 1)[None].float(); b4 = gt.permute(2, 0, 1)[None].float()
                    lps[h].append(float(lp(a4, b4)))
    return ({h: float(np.mean(v)) for h, v in ps.items() if v},
            {h: float(np.mean(v)) for h, v in lps.items() if v})
hdr = "".join(f"{'h'+str(h):>9}" for h in HZ)
print(f"{'K':>4}  PSNR{hdr}")
sweep = {}
for K in [1, 2, 4, 8, 16]:
    p, l = ol_metrics(K)
    sweep[K] = {"psnr": p, "lpips": l}
    print(f"{K:>4}       " + "".join(f"{p.get(h, float('nan')):>9.2f}" for h in HZ), flush=True)
print(f"{'K':>4}  LPIPS{hdr}")
for K in sweep:
    l = sweep[K]["lpips"]
    print(f"{K:>4}       " + "".join(f"{l.get(h, float('nan')):>9.4f}" for h in HZ))
m.sampling_steps = K_ORIG

# ---------------- TEST 3: does the rollout beat DO-NOTHING in PIXEL space? ----------------
# TEST 1 says the learned dynamics is WORSE than the identity map in latent cosine at every horizon. The
# pixel metrics have always been read as if the rollout adds value. Only one of those can be right, so
# measure the do-nothing baseline through the SAME decode path: freeze the context latent and decode it at
# every horizon (through the codec, so this is the model's own ceiling on "predict no change"), plus the
# raw copy-the-last-frame baseline which needs no model at all.
print("\n=== TEST 3: rollout vs DO-NOTHING, in pixel space ===")
frz_p, frz_l, cpy_p, cpy_l, mdl_p, mdl_l = ({h: [] for h in HZ} for _ in range(6))
m.sampling_steps = K_ORIG
with torch.no_grad():
    for (ei, t0) in picks:
        o, a, _fr = eps_ds[ei]
        im = _fr[_CAMK]
        ctx = {"proprio": norm.norm_obs(torch.from_numpy(o[t0 - P + 1:t0 + 1])).float()[None].to(dev),
               img_head: torch.from_numpy(im[t0 - P + 1:t0 + 1]).float().div(255.0)[None].to(dev)}
        acts = norm.norm_act(torch.from_numpy(a[t0 - P + 1:t0 + HMAX])).float()[None].to(dev)
        out = m.imagine_eval(ctx, acts, HMAX, heads=[img_head], decode_chunk=16)[img_head][0].clamp(0, 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            zc = m.encode_state(ctx)[:, -1:]                       # the frozen context bag
            frz = m.to_obs(zc[:, :, None][:, 0], heads=[img_head])[img_head][0, 0].clamp(0, 1).float()
        last = torch.from_numpy(im[t0].astype(np.float32) / 255.0).to(dev)
        for h in HZ:
            if t0 + h >= len(im):
                continue
            gt = torch.from_numpy(im[t0 + h].astype(np.float32) / 255.0).to(dev)
            for tag, pr, dp, dl in (("m", out[h - 1], mdl_p, mdl_l), ("f", frz, frz_p, frz_l), ("c", last, cpy_p, cpy_l)):
                dp[h].append(float(10 * torch.log10(1.0 / torch.clamp(((pr - gt) ** 2).mean(), min=1e-10))))
                if lp is not None:
                    dl[h].append(float(lp(pr.permute(2, 0, 1)[None].float(), gt.permute(2, 0, 1)[None].float())))
hdr2 = "".join(f"{'h'+str(h):>9}" for h in HZ)
print(f"{'':>22}{hdr2}")
for nm, d in (("rollout   PSNR", mdl_p), ("frozen-latent PSNR", frz_p), ("copy-frame  PSNR", cpy_p)):
    print(f"{nm:>22}" + "".join(f"{np.mean(d[h]):>9.2f}" if d[h] else f"{'-':>9}" for h in HZ))
for nm, d in (("rollout  LPIPS", mdl_l), ("frozen-latent LPIPS", frz_l), ("copy-frame  LPIPS", cpy_l)):
    print(f"{nm:>22}" + "".join(f"{np.mean(d[h]):>9.4f}" if d[h] else f"{'-':>9}" for h in HZ))
res["pixel_baselines"] = {"rollout_psnr": {str(h): float(np.mean(v)) for h, v in mdl_p.items() if v},
                          "frozen_psnr": {str(h): float(np.mean(v)) for h, v in frz_p.items() if v},
                          "copy_psnr": {str(h): float(np.mean(v)) for h, v in cpy_p.items() if v},
                          "rollout_lpips": {str(h): float(np.mean(v)) for h, v in mdl_l.items() if v},
                          "frozen_lpips": {str(h): float(np.mean(v)) for h, v in frz_l.items() if v},
                          "copy_lpips": {str(h): float(np.mean(v)) for h, v in cpy_l.items() if v}}
res["sampling_sweep"] = {str(k): v for k, v in sweep.items()}
res["default_K"] = K_ORIG
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=2)
print(f"\n[done] {OUT}")
