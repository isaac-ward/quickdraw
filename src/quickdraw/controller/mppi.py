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

from ..environments.examples.torus import TorusEnv
from ..environments.torus_utils import control_goals


@dataclass
class MPPIConfig:
    # NOTE: defaults MIRROR conf/control/mppi.yaml (the authoritative values used at runtime) — keep them in
    # sync. The yaml overrides these anyway; matching just stops the dataclass from misleading readers.
    horizon: int = 64
    chunk: int = 8             # execute this many steps of each plan before replanning (action chunking)
    num_samples: int = 128
    noise_sigma: float = 2.0
    lambda_: float = 0.3
    mean_decay: float = 1.0
    tol: float = 0.30          # within this ambient distance of the goal counts as "at" it (2x'd 2026-08-05)
    settle_steps: int = 45     # consecutive in-tol steps before advancing (~0.75s @60Hz dwell to confirm arrival)
    max_steps: int = 1000      # per-episode step budget for the whole goal sequence
    beta_vel: float = 0.0      # near-goal velocity penalty OFF: it made the controller stall at the gate boundary
    r_settle: float = 0.5      # distance under which the velocity penalty would turn on (unused while beta_vel=0)
    beta_ctrl: float = 0.02    # control (action-magnitude) cost weight: penalizes sum_h ||a_h||^2 over the
    #                            horizon, so the planner prefers cheaper thrust (and settles with less jitter).
    #                            Applies to BOTH controllers (shared _score). 0 = off (no control cost).
    n_episodes: int = 8        # parallel control episodes (random inits/orders); video is episode 0
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
    G, H, A = mean.shape[0], mppi.horizon, mean.shape[-1]      # A: env action_dim (torus: 2, unchanged)
    K = mppi.num_samples
    noise = torch.randn(G, K, H, A, device=mean.device, generator=g) * mppi.noise_sigma
    cand = (mean[:, None] + noise).clamp(-a_max, a_max)        # (G,K,H,A)
    p_xyz, v_xyz, dist = rollout_fn(cand)                     # dist: per-step (G,K,H) override, or None
    ret = _score(p_xyz, v_xyz, cand, goal, mppi, dist=dist, reward_fn=reward_fn)   # (G,K) higher = better
    w = torch.softmax(ret / max(mppi.lambda_, 1e-6), dim=1)   # (G,K)
    new_mean = (w[..., None, None] * cand).sum(dim=1)         # (G,H,2)
    return new_mean, new_mean[:, 0], p_xyz, ret               # p_xyz/ret expose the candidate fan


def _true_rollout_fn(env):
    """Roll candidate action sequences through the TRUE dynamics from env's current state. Env-agnostic:
    the state fork goes through the env's OPTIONAL `fork(k)` hook (base.py) — a batch-(G*K) copy with each
    state repeat_interleaved K times. TorusEnv.fork reproduces the exact fork this function always inlined,
    so the torus oracle rollout is byte-identical. Envs without `fork` fall back to per-candidate deepcopy
    forks (the reward-only oracle's mechanism, _true_rollout_obs_fn)."""
    def fn(cand):                                             # cand: (G,K,H,A)
        G, K, H = cand.shape[:3]
        if hasattr(env, "fork"):
            sim = env.fork(K)
            a = cand.reshape(G * K, H, cand.shape[-1])
            obs = torch.stack([sim.step(a[:, h]) for h in range(H)], dim=1)   # (G*K,H,obs_dim)
            obs = obs.view(G, K, H, obs.shape[-1])
        else:
            obs = _true_rollout_obs_fn(env)(cand)             # (G,K,H,obs_dim) via deepcopy forks
        return obs[..., :3], obs[..., 3:], None               # dist=None -> goal distance (true dynamics = oracle only)
    return fn


def _mm_model_rollout_fn(model, normalizer, ctx_pro, ctx_fpv, pa, img_head, dist_bag=None):
    """Learned rollout for the spine: proprio (+ rendered FPV context when img_head is set). The image
    context is encoded ONCE and shared across the K candidates (imagine_shared). Returns (p_xyz, v_xyz, dist):
    dist = dist_bag(rolled latent bag) is a per-step (G,K,H) distance for the learned-reward objective
    (1 - reward), else None (goal-distance). Either way _score applies its velocity/control shaping."""
    def fn(cand):                                           # cand: (G,K,H,A)
        G, K, H, A = cand.shape
        ctx = {"proprio": normalizer.norm_obs(ctx_pro)}                      # (G,p,obs_dim)
        if img_head is not None:
            ctx[img_head] = ctx_fpv                                          # (G,p,s,s,3) rendered FPV context
        paK = pa[:, None].expand(G, K, pa.shape[1], A).reshape(G * K, pa.shape[1], A)
        actK = normalizer.norm_act(torch.cat([paK, cand.reshape(G * K, H, A)], dim=1))  # (G*K, p-1+H, A)
        out = model.imagine_shared(ctx, actK, H, K, heads=["proprio"], return_bag=(dist_bag is not None))
        pr = normalizer.denorm_obs(out["proprio"]).view(G, K, H, -1)
        dist = dist_bag(out["_bag"].view(G, K, H, -1)) if dist_bag is not None else None   # (G,K,H) reward distance
        return pr[..., :3], pr[..., 3:], dist
    return fn


def _init_controller(cfg, B, device, seed, env_factory=None):
    """One controller's live env + logs. env_factory (the real env's factory, e.g. registry.make_env bound
    to the run's config) makes this env-agnostic; None keeps the legacy TorusEnv(cfg) construction. Either
    way the reset seed is the same, so the torus race inits are byte-identical."""
    env = env_factory() if env_factory is not None else TorusEnv(cfg, batch=B, device=device)
    env.reset(torch.Generator(device=device).manual_seed(seed))
    return {"env": env, "obs": [env.observe()], "act": [],
            "gidx": torch.zeros(B, dtype=torch.long, device=device),
            "settle": torch.zeros(B, dtype=torch.long, device=device), "goal_log": []}


@torch.no_grad()
def run_control(model, normalizer, env_cfg, mppi: MPPIConfig, device="cpu", log=None, fpv=None,
                reward=None, request=None, oracle=True, n_plot=1, requests=None, reward_fn=None,
                env=None, env_factory=None):
    """MPPI GOAL-RACE control (torus: byte-identical to the legacy torus-only version). Default: race the
    oracle (true dynamics) vs the learned model through
    spatial goals. `oracle=False` -> learned controller only (same code spine). `reward` (a
    language.reward.LanguageReward) + `request` -> the learned controller maximizes R(latent, request)
    instead of reaching goals: goal advancement is OFF and each controller's `dist_curve` holds the realized
    reward per step (this is the language-steered eval_control). `fpv` renders FPV context in the loop.
    n_plot: render per-episode products (paths/fan/FPV) for the first n_plot of the n_episodes parallel
    episodes (all run in ONE batched rollout; n_plot only controls how many we keep for visuals).
    reward_fn(obs, goal) -> per-step (…,) reward: the env-agnostic MPPI scorer (WorldEnv.reward), used by
    BOTH controllers; None -> the true env's reward with the config's beta_vel/r_settle (torus default).
    env (the run's live WorldEnv) + env_factory (fresh batch-B instances for the controllers) make this
    env-agnostic: goals come from env.control_goals, the oracle forks the real env (env.fork), and a
    generic image-head env renders its in-loop image context via env.render_obs. Both None (legacy torus
    callers) -> the exact torus-only construction (TorusEnv + module-level control_goals), unchanged."""
    core = getattr(model, "_orig_mod", model)
    img_head = next((n for n, _ in core.layout if n != "proprio"), None)   # image head name, or None (proprio-only)
    use_fpv = img_head is not None and fpv is not None                     # render FPV in the loop ONLY with a real image head
    from ..logging import viz
    # The torus in-loop FPV FAST PATH (parity-critical — render_obs would change torus control scalars):
    # kept whenever the env cfg carries the torus geometry. A generic env falls back to env.render_obs.
    fpv_rend = viz.FPVRenderer(env_cfg.R, env_cfg.r, fpv["coloring"], fpv["fov"], fpv["size"]) \
        if (use_fpv and hasattr(env_cfg, "R")) else None

    def _fpv(states):                                   # (B,obs_dim) -> (B,s,s,3) [0,1] on device
        if fpv_rend is not None:                        # torus: persistent FPV plotter (unchanged)
            return torch.from_numpy(fpv_rend.render(states.detach().cpu().numpy())).float().div_(255.0).to(device)
        return env.render_obs(states).to(device).float().div_(255.0)   # generic: the env's image modality
    goals = (env.control_goals(mppi.n_episodes, mppi.n_goals, None, device) if env is not None
             else control_goals(env_cfg.R, env_cfg.r, device=device))   # torus env returns the SAME 8 goals
    names = [n for n, _ in goals]
    n_goals = min(mppi.n_goals, len(goals))                   # visit this many per episode (subset of the 8)
    tgt = torch.stack([p for _, p in goals]).to(device)       # (n,3) all goal points
    B, P, H = mppi.n_episodes, model.window, mppi.horizon
    a_max = getattr(env, "a_max", None) if env is not None else None    # TorusEnv carries it on cfg only
    a_max = env_cfg.a_max if a_max is None else float(a_max)
    A = int(env.action_dim) if env is not None else 2         # env action_dim (torus: 2, unchanged)
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
    ctrls = {k: _init_controller(env_cfg, B, device, 2, env_factory) for k in kinds}
    if reward_fn is None:   # default scorer: the TRUE env's reward with the config's shaping knobs
        reward_fn = lambda o, gl: ctrls[kinds[0]]["env"].reward(o, gl, beta_vel=mppi.beta_vel,
                                                                r_settle=mppi.r_settle)
    arange = torch.arange(B, device=device)
    for c in ctrls.values():
        c["mean"] = torch.zeros(B, H, A, device=device)
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
                pa = torch.stack(c["act"][-(P - 1):], dim=1) if c["act"] else torch.zeros(B, 0, A, device=device)
                ctx_fpv = torch.stack(c["fpv"][-P:], dim=1) if use_fpv else None   # (B,p,s,s,3) FPV context, or None (proprio-only)
                rollout = _mm_model_rollout_fn(model, normalizer, ctx, ctx_fpv, pa, img_head if use_fpv else None,
                                               dist_bag=dist_bag)   # reward mode -> per-step distance 1-R on the rolled bag
            else:
                rollout = _true_rollout_fn(c["env"])
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
                    gp = getattr(c["env"], "goal_point", None)                   # optional obs -> goal-space map
                    d = ((gp(new_obs) if gp is not None else new_obs[:, :3]) - cur).norm(dim=-1)   # (B,)
                    c["dist_log"].append(d.cpu().numpy())
                    c["settle"] = torch.where(d < mppi.tol, c["settle"] + 1, torch.zeros_like(c["settle"]))
                    advance = (c["settle"] >= mppi.settle_steps) & (c["gidx"] < n_goals)
                    c["gidx"] = c["gidx"] + advance.long()
                    c["settle"] = torch.where(advance, torch.zeros_like(c["settle"]), c["settle"])
                    just_done = (c["gidx"] >= n_goals) & (c["done_step"] < 0)
                    c["done_step"] = torch.where(just_done, torch.full_like(c["done_step"], step + 1), c["done_step"])
            step += 1
        for c in ctrls.values():  # warm-start: shift the executed chunk off the plan
            c["mean"] = torch.cat([c["plan"][:, chunk:], torch.zeros(B, chunk, A, device=device)], dim=1) * mppi.mean_decay
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


# --------------------------------------------------------------------------------------
# REWARD-ONLY control: MPPI for a WorldEnv with NO goal source (env.control_goals -> None, e.g. a gym
# Pendulum) — the objective is the env's OWN reward, not visiting goal points. A SEPARATE branch from
# `run_control` above (which stays byte-identical for the torus): structurally the language mode (no goal
# sequence, no gidx advancement, runs to max_steps) with env.reward(obs, None) as the per-step objective.
# Generic action_dim/obs_dim (the goal-based helpers above are torus-shaped: 2D actions, obs split p/v).
# --------------------------------------------------------------------------------------
def _mppi_step_reward(rollout_fn, mean, mppi, a_max, g, reward_fn):
    """One MPPI update for reward-only control: sample candidates around `mean` (G,H,A), roll them out to
    obs (G,K,H,obs_dim), score sum_h env.reward(obs_h, None) (+ the optional control cost), softmax-weight.
    `reward_fn(obs, None)` must broadcast over leading dims ((...,obs_dim) -> (...))."""
    G, H, A = mean.shape
    K = mppi.num_samples
    noise = torch.randn(G, K, H, A, device=mean.device, generator=g) * mppi.noise_sigma
    cand = (mean[:, None] + noise).clamp(-a_max, a_max)        # (G,K,H,A)
    obs = rollout_fn(cand)                                     # (G,K,H,obs_dim)
    ret = reward_fn(obs, None).sum(dim=-1)                     # (G,K) env reward summed over the horizon
    if mppi.beta_ctrl > 0.0:                                   # cheaper thrust preferred (energy/jitter)
        ret = ret - mppi.beta_ctrl * cand.pow(2).sum(dim=-1).sum(dim=-1)
    w = torch.softmax(ret / max(mppi.lambda_, 1e-6), dim=1)    # (G,K)
    return (w[..., None, None] * cand).sum(dim=1)              # (G,H,A) new mean == the plan


def _model_rollout_obs_fn(model, normalizer, ctx_pro, pa, obs_dim):
    """Learned proprio rollout for reward-only control: context encoded ONCE, K action variants per episode
    (imagine_shared), decode proprio, denorm -> (G,K,H,obs_dim). Proprio-only: a generic env has no in-loop
    FPV renderer (the torus FPV path lives in _mm_model_rollout_fn, unchanged)."""
    def fn(cand):                                              # cand: (G,K,H,A)
        G, K, H, A = cand.shape
        ctx = {"proprio": normalizer.norm_obs(ctx_pro)}                      # (G,p,obs_dim)
        paK = pa[:, None].expand(G, K, pa.shape[1], A).reshape(G * K, pa.shape[1], A)
        actK = normalizer.norm_act(torch.cat([paK, cand.reshape(G * K, H, A)], dim=1))  # (G*K, p-1+H, A)
        out = model.imagine_shared(ctx, actK, H, K, heads=["proprio"])
        return normalizer.denorm_obs(out["proprio"]).view(G, K, H, obs_dim)
    return fn


def _true_rollout_obs_fn(env):
    """TRUE-dynamics rollout for reward-only control (the oracle's planner): the WorldEnv protocol has no
    state get/set, so fork the live env by deepcopy — one fork per candidate, stepped batched over episodes.
    Cheap for batched-tensor envs; envs that can't deepcopy+step should run with oracle=False."""
    import copy
    def fn(cand):                                              # cand: (G,K,H,A)
        G, K, H = cand.shape[:3]
        cols = []
        for k in range(K):
            sim = copy.deepcopy(env)
            cols.append(torch.stack([sim.step(cand[:, k, h]) for h in range(H)], dim=1))  # (G,H,obs_dim)
        return torch.stack(cols, dim=1)                        # (G,K,H,obs_dim)
    return fn


@torch.no_grad()
def run_control_reward_only(model, normalizer, env_factory, mppi: MPPIConfig, device="cpu", log=None,
                            oracle=True, n_plot=1, reward_fn=None):
    """REWARD-ONLY MPPI control for an env with NO goal source: maximize the env's own reward. `env_factory`
    builds a fresh batched WorldEnv (batch == mppi.n_episodes, exposing action_dim/obs_dim/a_max); each
    controller gets its OWN instance, reset with the same seed (2, matching _init_controller) so oracle and
    learned race the same inits. oracle=True -> dual true-dynamics vs learned controllers (same spine as the
    goal race); the oracle plans via deepcopy env forks (_true_rollout_obs_fn). No goal sequence / advancement
    / markers: every episode runs the full max_steps and logs its realized per-step env reward.
    reward_fn(obs, goal) -> per-step reward, broadcasting over leading dims; None -> env.reward."""
    B, P, H = mppi.n_episodes, model.window, mppi.horizon
    NP = max(1, min(n_plot, B))
    chunk = max(1, min(mppi.chunk, H))
    g = torch.Generator(device=device).manual_seed(0)          # candidate-noise stream (as in run_control)
    kinds = ["true", "pred"] if oracle else ["pred"]
    ctrls = {}
    for k in kinds:
        e = env_factory()
        obs0 = e.reset(torch.Generator(device=device).manual_seed(2))
        ctrls[k] = {"env": e, "obs": [obs0], "act": [], "reward_log": []}
    env0 = ctrls[kinds[0]]["env"]
    A, obs_dim, a_max = int(env0.action_dim), int(env0.obs_dim), float(env0.a_max)
    core = getattr(model, "_orig_mod", model)
    assert all(n == "proprio" for n, _ in core.layout), \
        "reward-only control is proprio-only (a generic env has no in-loop image renderer)"
    if reward_fn is None:
        reward_fn = env0.reward
    for c in ctrls.values():
        c["mean"] = torch.zeros(B, H, A, device=device)

    t0 = time.perf_counter()
    step = 0
    n_chunks = 0
    next_log = 100
    while step < mppi.max_steps:
        n_chunks += 1
        for kind, c in ctrls.items():  # plan once per chunk (re-grounded on the latest true state)
            if kind == "pred":
                ctx = torch.stack(c["obs"][-P:], dim=1)
                pa = torch.stack(c["act"][-(P - 1):], dim=1) if c["act"] else torch.zeros(B, 0, A, device=device)
                rollout = _model_rollout_obs_fn(model, normalizer, ctx, pa, obs_dim)
            else:
                rollout = _true_rollout_obs_fn(c["env"])
            c["plan"] = _mppi_step_reward(rollout, c["mean"], mppi, a_max, g, reward_fn)
        for j in range(chunk):         # execute `chunk` actions of each plan open-loop, then replan
            if step >= mppi.max_steps:
                break
            for kind, c in ctrls.items():
                new_obs = c["env"].step(c["plan"][:, j])
                c["obs"].append(new_obs)
                c["act"].append(c["plan"][:, j])
                # realized per-step reward of the ACTUAL state, from the controller's OWN env (so envs whose
                # reward is step-native, e.g. the gym adapter, report the right controller's reward).
                c["reward_log"].append(c["env"].reward(new_obs, None).cpu().numpy())   # (B,)
            step += 1
        for c in ctrls.values():       # warm-start: shift the executed chunk off the plan
            c["mean"] = torch.cat([c["plan"][:, chunk:], torch.zeros(B, chunk, A, device=device)], dim=1) * mppi.mean_decay
        if log is not None and step >= next_log:
            el = time.perf_counter() - t0
            eta = el / max(1, step) * max(0, mppi.max_steps - step)
            finish = time.strftime("%H:%M:%S", time.localtime(time.time() + eta))
            msg = "reward " + " ".join(f"{k}={float(np.mean(c['reward_log'][-1])):.3f}" for k, c in ctrls.items())
            log(f"step {step}/{mppi.max_steps} ({int(100 * step / mppi.max_steps)}%) | elapsed {el:.0f}s "
                f"ETA {eta:.0f}s (~{finish}) | {n_chunks} replans ({1000 * el / max(1, n_chunks):.0f} ms/replan) | {msg}")
            next_log += 50

    out = {"n_chunks": n_chunks, "n_steps": step, "n_plot": NP}
    for kind, c in ctrls.items():
        rew = np.stack(c["reward_log"])                                     # (T,B)
        out[kind] = {
            "obs_seqs": np.stack([np.stack([o[e].cpu().numpy() for o in c["obs"]]) for e in range(NP)]),  # (NP,T+1,obs_dim)
            "actions": np.stack([np.stack([a[e].cpu().numpy() for a in c["act"]]) for e in range(NP)]),   # (NP,T,A)
            "reward_curves": rew[:, :NP].T,                                 # (NP,T) realized reward per step
            "mean_reward": float(rew.mean()),                               # over ALL steps x episodes
            "final_reward": float(rew[-1].mean()),                          # last executed step, over episodes
        }
        if obs_dim >= 3:               # world-space paths for envs with a diagnostic scene renderer
            out[kind]["paths"] = out[kind]["obs_seqs"][..., :3]
    return out
