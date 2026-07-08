"""Language-steered MPPI: plan actions that maximize a learned language reward R(latent, text) in latent
space (decode-free), so the agent steers toward a natural-language request (e.g. "red"). Single controller
(no oracle race), no spatial goals — the objective is the reward. Executes on the true env; renders ep0 FPV
so the steering is watchable. See design/language_steering.md."""

from __future__ import annotations

import time

import numpy as np
import torch

from ..environments.torus import TorusEnv


@torch.no_grad()
def run_language_control(model, normalizer, env_cfg, reward, request, mppi, device="cpu", fpv=None, log=None):
    """MPPI maximizing R(latent, request). Returns ep0 path, the realized reward curve, and (if an image head
    + fpv) the ep0 predicted-vs-actual FPV video. `reward` is a language.reward.LanguageReward."""
    core = getattr(model, "_orig_mod", model)
    img_head = next((n for n, _ in core.layout if n != "proprio"), None)
    use_fpv = img_head is not None and fpv is not None
    from ..logging import viz
    fpv_rend = viz.FPVRenderer(env_cfg.R, env_cfg.r, fpv["coloring"], fpv["fov"], fpv["size"]) if use_fpv else None

    def _fpv(states):
        return torch.from_numpy(fpv_rend.render(states.detach().cpu().numpy())).float().div_(255.0).to(device)

    t_e = reward.text_embedding(request).to(device)
    B, P, H, a_max, K = mppi.n_episodes, model.window, mppi.horizon, env_cfg.a_max, mppi.num_samples
    env = TorusEnv(env_cfg, batch=B, device=device)
    env.reset(torch.Generator(device=device).manual_seed(2))
    obs, act = [env.observe()], []
    fpv_buf = [_fpv(obs[-1])] if use_fpv else None
    mean = torch.zeros(B, H, 2, device=device)
    g = torch.Generator(device=device).manual_seed(0)
    chunk = max(1, min(mppi.chunk, H))
    reward_curve, actual_ep0, pred_ep0 = [], [], []
    t0, step, n_chunks = time.perf_counter(), 0, 0

    while step < mppi.max_steps:
        n_chunks += 1
        ctx = torch.stack(obs[-P:], dim=1)                                       # (B,P,6)
        pa = torch.stack(act[-(P - 1):], dim=1) if act else torch.zeros(B, 0, 2, device=device)
        ctxd = {"proprio": normalizer.norm_obs(ctx)}
        if use_fpv:
            ctxd[img_head] = torch.stack(fpv_buf[-P:], dim=1)                    # (B,P,s,s,3)
        cand = (mean[:, None] + torch.randn(B, K, H, 2, device=device, generator=g) * mppi.noise_sigma).clamp(-a_max, a_max)
        paK = pa[:, None].expand(B, K, pa.shape[1], 2).reshape(B * K, pa.shape[1], 2)
        actK = normalizer.norm_act(torch.cat([paK, cand.reshape(B * K, H, 2)], dim=1))
        bag = model.imagine_shared(ctxd, actK, H, K, heads=["proprio"], return_bag=True)["_bag"]   # (B*K,H,n_state,d)
        r = reward.score(bag.reshape(B, K, H, -1), t_e)                          # (B,K,H) per-step reward
        w = torch.softmax(r.sum(-1) / max(mppi.lambda_, 1e-6), dim=1)            # (B,K)
        plan = (w[..., None, None] * cand).sum(1)                               # (B,H,2)
        if use_fpv:                                                             # ep0 selected-plan imagined FPV
            ctx0 = {"proprio": normalizer.norm_obs(ctx[0:1]), img_head: ctxd[img_head][0:1]}
            act0 = normalizer.norm_act(torch.cat([pa[0:1], plan[0:1]], dim=1))
            plan_fpv = model.imagine_eval(ctx0, act0, H, heads=[img_head])[img_head][0].clamp(0, 1)  # (H,s,s,3)
        for j in range(chunk):
            if step >= mppi.max_steps:
                break
            new_obs = env.step(plan[:, j])
            obs.append(new_obs); act.append(plan[:, j])
            # realized reward of ep0's actual new state (encode it, score against the request)
            ro = {"proprio": normalizer.norm_obs(new_obs[0:1, None])}
            if use_fpv:
                nf = _fpv(new_obs)
                ro[img_head] = nf[0:1, None]
                fpv_buf.append(nf); fpv_buf = fpv_buf[-P:]
                actual_ep0.append(nf[0].detach().cpu().numpy()); pred_ep0.append(plan_fpv[j].detach().cpu().numpy())
            zt = core.encode_state(ro).reshape(1, -1)                            # (1, n_state*d)
            reward_curve.append(float(reward.score(zt, t_e)[0]))
            step += 1
        mean = torch.cat([plan[:, chunk:], torch.zeros(B, chunk, 2, device=device)], dim=1) * mppi.mean_decay
        if log is not None and n_chunks % 5 == 0:
            log(f"step {step}/{mppi.max_steps} | reward {reward_curve[-1]:+.3f} | {n_chunks} replans "
                f"({1000 * (time.perf_counter() - t0) / n_chunks:.0f} ms/replan)")

    if fpv_rend is not None:
        fpv_rend.close()
    out = {"request": request, "path": np.stack([o[0, :3].cpu().numpy() for o in obs]),
           "reward_curve": np.asarray(reward_curve, dtype=np.float32), "n_steps": step}
    if use_fpv and actual_ep0:
        n = min(len(pred_ep0), len(actual_ep0))
        out["fpv_video"] = {"pred": np.stack(pred_ep0)[:n], "actual": np.stack(actual_ep0)[:n]}
    return out
