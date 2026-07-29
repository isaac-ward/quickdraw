"""MPPI control through a random sequence of 8 torus goals (design/training.md eval/control/).

Two controllers race the SAME task (same random init + same random goal permutation), both
executing on the TRUE env and differing only in the dynamics used to score MPPI candidates:
  - true: the true TorusEnv dynamics (oracle baseline -- best achievable control)
  - pred: the learned world model (the controller under test)
Each advances its own goal pointer when IT settles within `tol` of its current goal for
`settle_steps` steps. All episodes x candidates are batched into one rollout per control step.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from ..environments.torus import TorusConfig, TorusEnv, control_goals


@dataclass
class MPPIConfig:
    horizon: int = 24
    chunk: int = 4             # execute this many steps of each plan before replanning (action chunking)
    num_samples: int = 512
    noise_sigma: float = 0.5
    lambda_: float = 1.0
    mean_decay: float = 1.0
    tol: float = 0.15          # within this ambient distance of the goal counts as "at" it
    settle_steps: int = 4      # consecutive in-tol steps before advancing to the next goal
    max_steps: int = 800       # per-episode step budget for the whole goal sequence
    beta_vel: float = 0.3      # velocity penalty weight, gated to near-goal (encourages settling)
    r_settle: float = 0.5      # distance under which the velocity penalty turns on
    beta_ctrl: float = 0.0     # control (action-magnitude) cost weight: penalizes sum_h ||a_h||^2 over the
    #                            horizon, so the planner prefers cheaper thrust (and settles with less jitter).
    #                            Applies to BOTH controllers (shared _score). 0 = off (no control cost).
    n_episodes: int = 16       # parallel control episodes (random inits/orders); video is episode 0
    n_goals: int = 5           # goals visited per episode (random subset of the 8 NESW in/out goals)


def _score(p_xyz, v_xyz, cand, goal, mppi, dist=None, reward_fn=None):
    """MPPI return for each candidate: the env's per-step reward summed over the horizon, plus an optional
    control (action-magnitude) cost. p_xyz/v_xyz: (G,K,H,3); cand: (G,K,H,2); goal: (G,3) -> (G,K).
    reward_fn(obs, goal) is the env-agnostic scorer (WorldEnv.reward; torus == the old inline
    -distance - gated velocity penalty, so torus numbers are unchanged — gym_refactor.md Phase 4).
    dist (G,K,H) optional: use this precomputed per-step distance INSTEAD of the env reward (language
    steering passes 1 - reward, so 'closer' == 'redder') — the velocity-settling + control shaping is
    then applied identically, which is what makes language control structurally the same as goal control."""
    if dist is not None:                                       # language fast path: distance precomputed on the bag
        gate = (dist < mppi.r_settle).float()
        ret = (-dist - mppi.beta_vel * gate * v_xyz.norm(dim=-1)).sum(dim=-1)
    else:                                                      # env reward per step; goal broadcast over (G,K,H)
        ret = reward_fn(torch.cat([p_xyz, v_xyz], dim=-1), goal[:, None, None]).sum(dim=-1)
    if mppi.beta_ctrl > 0.0:                                   # cheaper thrust preferred (energy/jitter)
        ret = ret - mppi.beta_ctrl * cand.pow(2).sum(dim=-1).sum(dim=-1)   # sum_h ||a_h||^2  (G,K)
    return ret


def _mppi_step(rollout_fn, mean, goal, mppi, a_max, g, reward_fn=None):
    """One MPPI update: sample candidates, score via rollout_fn, return the new weighted mean (G,H,2)
    and the first action (G,2). rollout_fn(cand) -> (p_xyz, v_xyz, dist): dist is a per-step distance
    override (G,K,H) for the language reward, or None -> score with reward_fn (the env's reward). _score
    applies the same velocity/control shaping either way."""
    G, H = mean.shape[0], mppi.horizon
    K = mppi.num_samples
    noise = torch.randn(G, K, H, 2, device=mean.device, generator=g) * mppi.noise_sigma
    cand = (mean[:, None] + noise).clamp(-a_max, a_max)        # (G,K,H,2)
    p_xyz, v_xyz, dist = rollout_fn(cand)                     # dist: per-step (G,K,H) override, or None
    ret = _score(p_xyz, v_xyz, cand, goal, mppi, dist=dist, reward_fn=reward_fn)   # (G,K) higher = better
    w = torch.softmax(ret / max(mppi.lambda_, 1e-6), dim=1)   # (G,K)
    new_mean = (w[..., None, None] * cand).sum(dim=1)         # (G,H,2)
    return new_mean, new_mean[:, 0], p_xyz, ret               # p_xyz/ret expose the candidate fan


def _true_rollout_fn(env: TorusEnv, cfg: TorusConfig, device):
    """Roll candidate action sequences through the TRUE dynamics from env's current state."""
    def fn(cand):                                             # cand: (G,K,H,2)
        G, K, H = cand.shape[:3]
        sim = TorusEnv(cfg, batch=G * K, device=device)
        sim.theta = env.theta.repeat_interleave(K)
        sim.phi = env.phi.repeat_interleave(K)
        sim.theta_dot = env.theta_dot.repeat_interleave(K)
        sim.phi_dot = env.phi_dot.repeat_interleave(K)
        a = cand.reshape(G * K, H, 2)
        obs = torch.stack([sim.step(a[:, h]) for h in range(H)], dim=1)   # (G*K,H,6)
        obs = obs.view(G, K, H, 6)
        return obs[..., :3], obs[..., 3:], None               # dist=None -> goal distance (true dynamics = oracle only)
    return fn


def _mm_model_rollout_fn(model, normalizer, ctx_pro, ctx_fpv, pa, img_head, dist_bag=None):
    """Learned rollout for the spine: proprio (+ rendered FPV context when img_head is set). The image
    context is encoded ONCE and shared across the K candidates (imagine_shared). Returns (p_xyz, v_xyz, dist):
    dist = dist_bag(rolled latent bag) is a per-step (G,K,H) distance for the learned-reward objective
    (1 - reward), else None (goal-distance). Either way _score applies its velocity/control shaping."""
    def fn(cand):                                           # cand: (G,K,H,2)
        G, K, H = cand.shape[:3]
        ctx = {"proprio": normalizer.norm_obs(ctx_pro)}                      # (G,p,6)
        if img_head is not None:
            ctx[img_head] = ctx_fpv                                          # (G,p,s,s,3) rendered FPV context
        paK = pa[:, None].expand(G, K, pa.shape[1], 2).reshape(G * K, pa.shape[1], 2)
        actK = normalizer.norm_act(torch.cat([paK, cand.reshape(G * K, H, 2)], dim=1))  # (G*K, p-1+H, 2)
        out = model.imagine_shared(ctx, actK, H, K, heads=["proprio"], return_bag=(dist_bag is not None))
        pr = normalizer.denorm_obs(out["proprio"]).view(G, K, H, 6)
        dist = dist_bag(out["_bag"].view(G, K, H, -1)) if dist_bag is not None else None   # (G,K,H) reward distance
        return pr[..., :3], pr[..., 3:], dist
    return fn


def _init_controller(cfg, B, device, seed):
    env = TorusEnv(cfg, batch=B, device=device)
    env.reset(torch.Generator(device=device).manual_seed(seed))
    return {"env": env, "obs": [env.observe()], "act": [],
            "gidx": torch.zeros(B, dtype=torch.long, device=device),
            "settle": torch.zeros(B, dtype=torch.long, device=device), "goal_log": []}


@torch.no_grad()
def run_control(model, normalizer, env_cfg: TorusConfig, mppi: MPPIConfig, device="cpu", log=None, fpv=None,
                reward=None, request=None, oracle=True, n_plot=1, requests=None, reward_fn=None):
    """MPPI control on the torus. Default: race the oracle (true dynamics) vs the learned model through
    spatial goals. `oracle=False` -> learned controller only (same code spine). `reward` (a
    language.reward.LanguageReward) + `request` -> the learned controller maximizes R(latent, request)
    instead of reaching goals: goal advancement is OFF and each controller's `dist_curve` holds the realized
    reward per step (this is the language-steered eval_control). `fpv` renders FPV context in the loop.
    n_plot: render per-episode products (paths/fan/FPV) for the first n_plot of the n_episodes parallel
    episodes (all run in ONE batched rollout; n_plot only controls how many we keep for visuals).
    reward_fn(obs, goal) -> per-step (…,) reward: the env-agnostic MPPI scorer (WorldEnv.reward), used by
    BOTH controllers; None -> the true env's reward with the config's beta_vel/r_settle (torus default)."""
    core = getattr(model, "_orig_mod", model)
    img_head = next((n for n, _ in core.layout if n != "proprio"), None)   # image head name, or None (proprio-only)
    use_fpv = img_head is not None and fpv is not None                     # render FPV in the loop ONLY with a real image head
    from ..logging import viz
    fpv_rend = viz.FPVRenderer(env_cfg.R, env_cfg.r, fpv["coloring"], fpv["fov"], fpv["size"]) if use_fpv else None

    def _fpv(states):                                   # (B,6) -> (B,s,s,3) [0,1] on device (persistent plotter)
        return torch.from_numpy(fpv_rend.render(states.detach().cpu().numpy())).float().div_(255.0).to(device)
    goals = control_goals(env_cfg.R, env_cfg.r, device=device)
    names = [n for n, _ in goals]
    n_goals = min(mppi.n_goals, len(goals))                   # visit this many per episode (subset of the 8)
    tgt = torch.stack([p for _, p in goals]).to(device)       # (8,3) all goal points
    B, P, H, a_max = mppi.n_episodes, model.window, mppi.horizon, env_cfg.a_max
    NP = max(1, min(n_plot, B))                               # episodes to keep per-episode visuals for
    g = torch.Generator(device=device).manual_seed(0)         # candidate-noise stream
    # per episode: a random n_goals-subset of the 8 goals, in random order (variety across episodes)
    order = torch.rand(B, len(goals), generator=torch.Generator(device=device).manual_seed(1),
                       device=device).argsort(dim=1)[:, :n_goals]    # (B, n_goals)
    chunk = max(1, min(mppi.chunk, H))
    # multi-query: `requests` (len == n_episodes) -> a PER-EPISODE target t_e (B, embed), so each episode steers to
    # its OWN request in ONE batched rollout (they share the torus). Single `request` -> one (embed,) target for all.
    ep_requests = list(requests) if requests else ([request] * B if request is not None else None)
    if reward is None:
        t_e = None
    elif requests:
        assert len(requests) == B, f"requests ({len(requests)}) must match n_episodes ({B})"
        t_e = torch.stack([reward.text_embedding(r) for r in requests]).to(device)   # (B, embed)
    else:
        t_e = reward.text_embedding(request).to(device)                              # (embed,)
    # language reward as a DISTANCE: d = 1 - cos(f_z(z), t_e) per step (G,K,H). Fed to _score exactly like
    # the goal distance, so beta_vel/r_settle (near-target braking) + beta_ctrl apply identically -> the agent
    # settles ON red instead of orbiting it, and the "score" reads as a distance (0 = perfectly red).
    dist_bag = (lambda bag: 1.0 - reward.score(bag, t_e)) if reward is not None else None   # (G,K,H,D)->(G,K,H)
    kinds = ["true", "pred"] if oracle else ["pred"]          # oracle=False -> learned controller only (same spine)
    ctrls = {k: _init_controller(env_cfg, B, device, 2) for k in kinds}
    if reward_fn is None:   # default scorer: the TRUE env's reward with the config's shaping knobs
        reward_fn = lambda o, gl: ctrls[kinds[0]]["env"].reward(o, gl, beta_vel=mppi.beta_vel,
                                                                r_settle=mppi.r_settle)
    arange = torch.arange(B, device=device)
    for c in ctrls.values():
        c["mean"] = torch.zeros(B, H, 2, device=device)
        c["done_step"] = torch.full((B,), -1, dtype=torch.long, device=device)
        c["dist_log"] = []  # per executed step: distance to current goal, OR (reward mode) the realized reward
    if use_fpv and "pred" in ctrls:   # learned controller needs the FPV context (proprio comes from the env)
        ctrls["pred"]["fpv"] = [_fpv(ctrls["pred"]["obs"][-1])]   # GPU: last P frames only (context)

    t0 = time.perf_counter()
    step = 0
    n_chunks = 0        # number of MPPI replans (one per action chunk) — for per-step timing
    next_log = 100
    fan_logs = [[] for _ in range(NP)]        # per episode: per-step pred candidate fan {pts (K,H+1,3), ret (K,)}
    cur_fans = [None] * NP
    pred_fpv_logs = [[] for _ in range(NP)]   # (MM) per episode: per-step model-imagined FPV for the SELECTED plan
    fpv_actual_logs = [[] for _ in range(NP)] # (MM) per episode: per-step actual rendered FPV
    latent_logs = [[] for _ in range(NP)]     # (reward mode) per episode: per-step agent latent (the rolled bag, the
    #                                           same space eval_interpret fit its reducers on) -> latent-space animation
    imag_head_logs = [[] for _ in range(NP)]  # (reward mode) imagined reward-head reward of the CHOSEN plan, per executed step
    imag_xyz_logs = [[] for _ in range(NP)]   # (reward mode) imagined xyz of the chosen plan -> imagined GROUND-TRUTH reward
    cur_plan_fpv = cur_imag = None
    replan_steps = []          # executed-step index at each MPPI replan (where the imagined belief is refreshed)
    while step < mppi.max_steps:
        n_chunks += 1
        replan_steps.append(step)          # this executed step begins a fresh plan (imagined belief refreshed)
        for kind, c in ctrls.items():  # plan once per chunk (re-grounded on the latest true state)
            cur = tgt[order[arange, c["gidx"].clamp(max=n_goals - 1)]]
            if kind == "pred":
                ctx = torch.stack(c["obs"][-P:], dim=1)
                pa = torch.stack(c["act"][-(P - 1):], dim=1) if c["act"] else torch.zeros(B, 0, 2, device=device)
                ctx_fpv = torch.stack(c["fpv"][-P:], dim=1) if use_fpv else None   # (B,p,s,s,3) FPV context, or None (proprio-only)
                rollout = _mm_model_rollout_fn(model, normalizer, ctx, ctx_fpv, pa, img_head if use_fpv else None,
                                               dist_bag=dist_bag)   # reward mode -> per-step distance 1-R on the rolled bag
            else:
                rollout = _true_rollout_fn(c["env"], env_cfg, device)
            c["plan"], _, p_xyz, ret = _mppi_step(rollout, c["mean"], cur, mppi, a_max, g, reward_fn)
            if kind == "pred":  # per-episode candidate fan, ANCHORED at the current known position: prepend
                # the dot (last true obs) so the first segment joins where-we-are -> first prediction.
                anchor = c["obs"][-1][:NP, :3].cpu().numpy()                    # (NP,3) current positions
                pts = p_xyz[:NP].cpu().numpy()                                  # (NP,K,H,3)
                retn = ret[:NP].cpu().numpy()                                   # (NP,K)
                cur_fans = [{"pts": np.concatenate([np.broadcast_to(anchor[e], (pts.shape[1], 1, 3)), pts[e]], axis=1),
                             "ret": retn[e]} for e in range(NP)]                # each (K, H+1, 3)
                if use_fpv:  # per-episode model-imagined FPV for the SELECTED plan -> pred-vs-actual video
                    ctxN = {"proprio": normalizer.norm_obs(ctx[:NP]), img_head: ctx_fpv[:NP]}
                    actN = normalizer.norm_act(torch.cat([pa[:NP], c["plan"][:NP]], dim=1))  # (NP, p-1+H, 2)
                    cur_plan_fpv = model.imagine_eval(ctxN, actN, H, heads=[img_head])[img_head].clamp(0, 1)  # (NP,H,s,s,3)
                if reward is not None:  # imagined rollout of the CHOSEN plan -> per-step imagined reward-head + xyz (trace)
                    a_im = normalizer.norm_act(torch.cat([pa[:NP], c["plan"][:NP]], dim=1))
                    ric = {"proprio": normalizer.norm_obs(ctx[:NP])}
                    if use_fpv:
                        ric[img_head] = ctx_fpv[:NP]
                    # KV-cache: inference/no-grad rollout -> cache applies (faster) AND it's the faithful sliding-
                    # window computation MPPI's planner uses (imagine_shared), so the imagined line matches the belief.
                    with torch.no_grad(), torch.autocast(device_type=("cuda" if "cuda" in str(device) else "cpu"),
                                                         dtype=torch.bfloat16, enabled=("cuda" in str(device))):
                        ibag = core._rollout(ric, a_im, H, 0.0, None, 0, use_cache=True)   # (NP,H,n_state,d)
                        iprop = normalizer.denorm_obs(core.to_obs(ibag, heads=["proprio"])["proprio"])  # (NP,H,6)
                    te = t_e[:NP] if t_e.dim() > 1 else t_e                            # per-episode target (multi-query)
                    cur_imag = {"head": reward.score(ibag.reshape(NP, H, -1).float(), te).cpu().numpy(),   # (NP,H) imagined R
                                "xyz": iprop[..., :3].float().cpu().numpy()}           # (NP,H,3) imagined path
        for j in range(chunk):  # execute `chunk` actions of each plan open-loop, then replan
            if step >= mppi.max_steps:
                break
            for e in range(NP):
                fan_logs[e].append(cur_fans[e])  # same plan's fan governs each of the chunk's executed steps
            for kind, c in ctrls.items():
                cur = tgt[order[arange, c["gidx"].clamp(max=n_goals - 1)]]   # (B,3)
                new_obs = c["env"].step(c["plan"][:, j])
                c["obs"].append(new_obs)
                if use_fpv and kind == "pred":                 # render the new FPV for the model's context
                    c["fpv"].append(_fpv(new_obs))
                    frame = c["fpv"][-1][:NP].detach().cpu().numpy()             # (NP,s,s,3) actual frames -> CPU
                    c["fpv"] = c["fpv"][-P:]                     # keep ONLY the last P frames on GPU (context) — bounds memory
                    for e in range(NP):
                        fpv_actual_logs[e].append(frame[e])
                        pred_fpv_logs[e].append(cur_plan_fpv[e, j].detach().cpu().numpy())  # predicted FPV for this obs
                c["act"].append(c["plan"][:, j])
                if reward is not None:   # LANGUAGE steering: no goals; log the realized reward of the ACTUAL state.
                    # FAITHFUL latent: re-ground on the real last-P states and roll ONE dynamics step via the SAME
                    # open-loop rollout path eval_interpret trained the reward on (core._rollout) — NOT encode_state,
                    # whose encoder latent lives in a different subspace than the rolled bags the reward/planner use.
                    o_ctx = torch.stack(c["obs"][-P:], dim=1)                   # (B,nc,6) real states ending at new_obs
                    nc = o_ctx.shape[1]
                    rc_obs = {"proprio": normalizer.norm_obs(o_ctx)}
                    if use_fpv and kind == "pred":
                        rc_obs[img_head] = torch.stack(c["fpv"][-P:], dim=1)    # (B,nc,s,s,3) real rendered FPV context
                    a_ctx = c["act"][-(nc - 1):] if nc > 1 else []              # nc-1 real inter-state actions
                    a_roll = normalizer.norm_act(torch.stack(a_ctx + [c["plan"][:, min(j + 1, H - 1)]], dim=1))  # (B,nc,2): +1 to roll
                    with torch.autocast(device_type=("cuda" if "cuda" in str(device) else "cpu"),
                                        dtype=torch.bfloat16, enabled=("cuda" in str(device))):
                        bag = core._rollout(rc_obs, a_roll, 1, 0.0, None, 0)    # (B,1,n_state,d) rolled bag (step 0)
                    bf = bag.reshape(B, -1).float()                             # (B, n_state*d) the reward-space input latent
                    c["dist_log"].append((1.0 - reward.score(bf, t_e)).cpu().numpy())  # (B,) realized dist 1-R
                    for e in range(NP):
                        latent_logs[e].append(bf[e].cpu().numpy())              # per-episode agent latent -> LDA animation
                        if cur_imag is not None:                                # the chosen plan's imagined belief for this step
                            imag_head_logs[e].append(float(cur_imag["head"][e, min(j, H - 1)]))
                            imag_xyz_logs[e].append(cur_imag["xyz"][e, min(j, H - 1)])
                    c["goal_log"].append(new_obs[:, :3].cpu().numpy())          # no target -> mark the agent itself
                else:                    # goal-reaching control: distance to current goal + advance on settle
                    c["goal_log"].append(cur.cpu().numpy())
                    d = (new_obs[:, :3] - cur).norm(dim=-1)                      # (B,)
                    c["dist_log"].append(d.cpu().numpy())
                    c["settle"] = torch.where(d < mppi.tol, c["settle"] + 1, torch.zeros_like(c["settle"]))
                    advance = (c["settle"] >= mppi.settle_steps) & (c["gidx"] < n_goals)
                    c["gidx"] = c["gidx"] + advance.long()
                    c["settle"] = torch.where(advance, torch.zeros_like(c["settle"]), c["settle"])
                    just_done = (c["gidx"] >= n_goals) & (c["done_step"] < 0)
                    c["done_step"] = torch.where(just_done, torch.full_like(c["done_step"], step + 1), c["done_step"])
            step += 1
        for c in ctrls.values():  # warm-start: shift the executed chunk off the plan
            c["mean"] = torch.cat([c["plan"][:, chunk:], torch.zeros(B, chunk, 2, device=device)], dim=1) * mppi.mean_decay
        if log is not None and step >= next_log:  # periodic progress (so eval_control time is visible live)
            el = time.perf_counter() - t0
            eta = el / max(1, step) * max(0, mppi.max_steps - step)   # upper bound (may end early once all goals hit)
            finish = time.strftime("%H:%M:%S", time.localtime(time.time() + eta))
            if reward is not None:
                msg = f"distance to '{request}' pred={float(np.mean(ctrls['pred']['dist_log'][-1])):.3f}"
            else:
                msg = "mean goals " + " ".join(f"{k}={float(c['gidx'].clamp(max=n_goals).float().mean()):.1f}"
                                                for k, c in ctrls.items()) + f"/{n_goals}"
            log(f"step {step}/{mppi.max_steps} ({int(100 * step / mppi.max_steps)}%) | elapsed {el:.0f}s "
                f"ETA {eta:.0f}s (~{finish}) | {n_chunks} replans ({1000 * el / max(1, n_chunks):.0f} ms/replan) | {msg}")
            next_log += 50                                            # every 50 steps (was 100) — denser live progress
        if reward is None and all((c["gidx"] >= n_goals).all() for c in ctrls.values()):
            break
    dt = env_cfg.dt

    out = {"goals": [(n, p.cpu().numpy()) for n, p in goals], "n_goals": n_goals, "n_chunks": n_chunks,
           "n_steps": step, "dt": dt, "n_plot": NP,
           "fan_seqs": fan_logs}       # per episode: pred candidate fan per executed step
    if use_fpv and pred_fpv_logs[0]:   # per episode: pred-vs-actual FPV over the whole run (from the selected plans)
        out["pred_fpv_videos"] = []
        for e in range(NP):
            pred, actual = np.stack(pred_fpv_logs[e]), np.stack(fpv_actual_logs[e])
            n = min(len(pred), len(actual))
            out["pred_fpv_videos"].append({"pred": pred[:n], "actual": actual[:n]})
    if reward is not None and latent_logs[0]:   # per episode: (T, n_state*d) agent latent trajectory (reward space)
        out["agent_latents"] = [np.stack(latent_logs[e]) for e in range(NP)]
    if reward is not None and imag_head_logs[0]:   # per episode: the chosen plan's IMAGINED belief per executed step
        out["imag_head_curves"] = [np.array(imag_head_logs[e]) for e in range(NP)]   # (T,) imagined reward-head reward
        out["imag_paths"] = [np.stack(imag_xyz_logs[e]) for e in range(NP)]          # (T,3) imagined xyz -> imagined GT
    if ep_requests is not None:
        out["requests"] = ep_requests[:NP]         # per-episode request (multi-query: each episode's own text)
    out["replan_steps"] = [s for s in replan_steps if s < step]   # replan boundaries (for the reward-trace markers)
    for kind, c in ctrls.items():
        done = c["done_step"]
        completed = done >= 0
        steps_tc = float(done[completed].float().mean()) if completed.any() else float("nan")
        gseq = np.stack(c["goal_log"])                                          # (T,B,3)
        dcur = np.stack(c["dist_log"])                                          # (T,B)
        out[kind] = {
            # per episode e<NP: (NP,T,3)/(NP,T-1,2)/(NP,T,3)/(NP,T). Per-episode (not mean) is smooth and
            # matches each episode's own video + goal-change markers.
            "paths": np.stack([np.stack([o[e, :3].cpu().numpy() for o in c["obs"]]) for e in range(NP)]),
            "obs_seqs": np.stack([np.stack([o[e].cpu().numpy() for o in c["obs"]]) for e in range(NP)]),  # full obs (render_obs fallback)
            "actions": np.stack([np.stack([a[e].cpu().numpy() for a in c["act"]]) for e in range(NP)]),
            "goal_seqs": gseq[:, :NP].transpose(1, 0, 2),
            "dist_curves": dcur[:, :NP].T,
            "success_rate": float(completed.float().mean()),                   # aggregate over ALL B episodes
            "mean_goals_reached": float(c["gidx"].clamp(max=n_goals).float().mean()),
            "mean_steps_to_complete": steps_tc,                                # over completed episodes
            "mean_seconds_to_complete": steps_tc * dt,
        }
    if fpv_rend is not None:
        fpv_rend.close()
    return out, names
