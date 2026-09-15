"""Language-steered planning INSIDE the world model's imagination: sample action chunks from the learned
prior, roll them through the world model, score the imagined latents with the language reward head, keep the
best, repeat. No environment, no true dynamics -- the question is where the model's own imagination goes when
asked for something in words.

WHY THIS IS NOT eval_control. `run_control` is a closed-loop goal race that compares learned dynamics
against an env's true dynamics, and starling has no simulator to compare against. This is one long open-loop
plan: the only ground truth available is what the CHOSEN ACTIONS say the drone was commanded to do, which is
exact and independent of the reward head (see `motion_readout`).

THE PROPOSAL IS AN INTERFACE, and that is the point. MPPI's candidates were hardcoded as gaussian noise
around a running mean (controller/mppi.py), so the action prior -- the whole reason it was trained -- was
never in the planner. A planner should not care where its candidates come from:

    GaussianProposal   mean + sigma * noise. What MPPI always did; kept so the old behaviour is available
                       and so a run can be compared against "structured noise".
    PriorProposal      draws from the trained action prior, conditioned on the imagined context. Proposals
                       then look like real pilot behaviour rather than white noise, which matters most
                       exactly where white noise is worst: it never holds a stick still, and the recorded
                       commands are at rest 23-58% of the time.

THE HONEST LIMITATION, stated because it bounds every number this produces: lookahead is ONE CHUNK. The
prior emits `action_head_chunk` actions conditioned on one context, so scoring a 128-step sequence up front
would mean sampling 16 chunks from a context that only describes the first. Instead this plans one chunk
ahead, commits it, and re-plans -- a receding-horizon planner with a 2.1 s lookahead executed out to the
full horizon. It cannot trade a bad chunk now for a good one later.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class GaussianProposal:
    """mean + sigma * noise, clamped. The candidate distribution MPPI has always used."""

    def __init__(self, sigma: float, a_max: float = 1.0):
        self.sigma, self.a_max = float(sigma), float(a_max)
        self.name = f"gaussian(sigma={sigma})"
        self.commit = "mean"        # the softmax-weighted mean of white noise is still white noise
        self.needs_ctx = False      # white noise is state-blind, so a planner need not build a context

    def sample(self, mean, k: int, ctx=None, g=None, prefix=None) -> torch.Tensor:
        """(G,H,A) -> (G,k,H,A). `ctx` and `prefix` are ignored: white noise knows nothing about the state,
        and has no structure to be continuous WITH -- recentring it on the last action would still jump by
        sigma from wherever it was centred, so there is no gaussian analogue of prefix guidance.

        The randn call is kept EXACTLY as controller/mppi.py always wrote it -- same shape, same order, same
        generator -- so routing MPPI through this class leaves the torus candidate stream bit-identical."""
        G, H, A = mean.shape
        noise = torch.randn(G, k, H, A, device=mean.device, dtype=mean.dtype, generator=g) * self.sigma
        return (mean[:, None] + noise).clamp(-self.a_max, self.a_max)


class PriorProposal:
    """Draws candidate action chunks from the model's trained action prior, conditioned on `ctx` (h_ctx).

    The prior returns NORMALIZED actions of width action_dim * chunk, time-major, so a draw is reshaped to
    (chunk, action_dim) -- `sample_action` already inverts the percentile transform when the head was trained
    with one, so callers see ordinary normalized actions either way.

    `norm` (a data.dataset.Normalizer) makes this usable by an ENV-SPACE planner: eval_steer plans entirely
    inside the imagination and works in normalized units, while eval_control's candidates are raw actions
    clamped to the env's a_max (controller/mppi.py normalizes them again on its way into the model). Pass
    `norm` there and draws come back denormalized; leave it None and they stay normalized.

    PREFIX GUIDANCE (`prefix_guidance=True`, and a `prefix` at sample time) makes the draw CONTINUE the
    previous chunk instead of starting wherever it likes -- the one harness that is genuinely prior-only,
    because it steers the flow while it integrates rather than editing the finished sample. See
    `_prefix_hook`. There is no equivalent for the data or gaussian proposals: a bank cannot be steered,
    only searched (measured, its 64th-nearest opening sits 6.3x the recorded step change away), and white
    noise has no structure to be continuous with.

    NOTE on temperature, removed 2026-09-14: scaling the flow's initial noise did widen the choice set and
    did buy obedience, but measured it pushed EVERY axis past the time-shuffled null (fore/aft step-to-step
    change 0.067 -> 0.173 at T=1.8), i.e. it bought steering by making the commands unflyable. Do not
    reintroduce it without a smoothness readout beside the steering one."""

    def __init__(self, model, a_max: float = 1e9, norm=None, prefix_guidance: bool = True,
                 prefix_freeze: int = 1, prefix_decay: float = 0.5):
        self.m = model
        self.a_max = float(a_max)
        self.norm = norm
        self.prefix_guidance = bool(prefix_guidance)
        self.prefix_freeze, self.prefix_decay = int(prefix_freeze), float(prefix_decay)
        self.K = int(getattr(model, "action_head_chunk", 1))
        self.name = (f"prior(chunk={self.K}, steps={getattr(model, 'action_head_sampling_steps', 0)}"
                     f"{', guided' if prefix_guidance else ''}{', raw' if norm is not None else ''})")
        self.commit = "best"        # averaging two plausible manoeuvres gives an implausible one
        self.needs_ctx = True       # the planner MUST hand over h_ctx (the pooled backbone context)
        assert getattr(model, "action_head_enabled", False), (
            "PriorProposal needs a model with a trained action head -- load a checkpoint from "
            "train_action_model, not a bare world model.")

    def _prefix_hook(self, prefix, k, A, eps):
        """RTC prefix guidance: re-impose the prefix on the flow's canvas after every Euler step.

        At time tau the state is a known blend of noise and the eventual sample, so the CORRECTLY NOISED
        prefix is (1-tau)*target + tau*eps -- writing that into the first positions keeps the canvas
        self-consistent and lets the velocity field carry the free positions toward values coherent with
        the fixed ones. Weights are 1 on the first `prefix_freeze` positions and decay exponentially after,
        so the handoff is gradual rather than a step discontinuity one position later; the rest of the
        chunk is untouched (weight 0) and generates freely.

        The target must live in the space the flow was TRAINED in: under target_transform=pit that is
        z-space, so the prefix goes through the same percentile map the training targets did."""
        G, p, _ = prefix.shape
        tgt = prefix if self.norm is None else self.norm.norm_act(prefix)     # the flow speaks normalized
        mode = getattr(self.m, "action_head_target_transform", "none")
        if mode == "pit":
            from ..data.transforms import PIT
            tgt = PIT(self.m.action_pit_knots).apply(tgt)
        elif mode == "pit_delta":
            # THE TARGET SPACE IS MIXED under pit_delta: slot 0 is a VALUE and slots 1.. are INCREMENTS.
            # Mapping the whole prefix through the value transform would write increments into value slots
            # -- a silently wrong constraint that still produces a plausible-looking plan, which is the
            # worst kind. Convert the prefix exactly as action_pairs builds its target.
            from ..data.transforms import PIT
            assert p >= 1
            head = PIT(self.m.action_pit_knots).apply(tgt[:, :1])
            tgt = (head if p == 1 else torch.cat(
                [head, PIT(self.m.action_delta_pit_knots).apply(tgt[:, 1:] - tgt[:, :-1])], dim=1))
        tgt = tgt.reshape(G, 1, p * A).expand(G, k, p * A).reshape(G * k, p * A)
        i = torch.arange(p, device=prefix.device, dtype=tgt.dtype)
        w = torch.exp(-self.prefix_decay * (i - self.prefix_freeze + 1).clamp(min=0.0))
        w = w.repeat_interleave(A)[None]                                      # (1, p*A), time-major
        e = eps[:, :p * A]

        def hook(x, tau, _eps):
            x = x.clone()
            x[:, :p * A] = (1.0 - w) * x[:, :p * A] + w * ((1.0 - tau) * tgt + tau * e)
            return x
        return hook

    def sample(self, mean, k: int, ctx=None, g=None, prefix=None) -> torch.Tensor:
        """(G,H,A) -> (G,k,H,A), conditioned on `ctx` (G,d). `mean` supplies only the shape/device: the prior
        is conditioned on the STATE, which is strictly more information than a running average of past plans.

        H MUST NOT EXCEED THE CHUNK. One context describes `self.K` steps; a longer horizon would be scoring
        actions drawn for a state the plan has already left. eval_control enforces horizon == chunk for this
        reason, which turns MPPI into the chunk-wise receding-horizon planner eval_steer already is.

        `prefix` (G,p,A) are the actions the new chunk must open on -- the unexecuted tail of the previous
        chunk, which exists only when the planner commits fewer steps than it scores."""
        G, H, A = mean.shape
        assert ctx is not None, "PriorProposal needs the pooled context h_ctx (G,d)"
        assert H <= self.K, (f"horizon {H} exceeds the prior's chunk {self.K}: the prior only describes "
                             f"{self.K} steps from one context")
        assert ctx.shape[0] == G, f"ctx has {ctx.shape[0]} rows but mean has {G} episodes"
        # (G,d) -> (G*k,d): every episode draws its own k candidates from its own context, in one call.
        h = ctx[:, None].expand(G, k, ctx.shape[-1]).reshape(G * k, -1)
        # SEEDED DRAWS. `sample_action` integrates the flow from x=eps and, left to itself, takes eps from
        # the GLOBAL rng -- so the planner's seed did not reach its candidates and two runs of the same
        # config gave different plans. Drawing eps here puts the whole plan back under `generator`.
        dz = int(self.m.action_flow.dz)
        eps = torch.randn((G * k, dz), device=h.device, dtype=h.dtype, generator=g)
        hook = None
        if self.prefix_guidance and prefix is not None and prefix.shape[1] > 0:
            hook = self._prefix_hook(prefix[:, :self.K], k, A, eps)
        a = self.m.sample_action(h, eps=eps, hook=hook).reshape(G, k, self.K, A)[:, :, :H]
        if self.norm is not None:
            a = self.norm.denorm_act(a)
        return a.clamp(-self.a_max, self.a_max)


class Crossfade:
    """Blend the drawn chunk's opening onto the prefix with a decaying weight. ANY proposal.

    The generic, post-hoc way to hide a seam, and the one the literature says not to trust: this is ACT's
    temporal ensembling, and RTC reports it "fails catastrophically on multi-modal distributions, producing
    invalid averaged actions". The average of `bank left` and `bank right` is `fly straight`, which neither
    the prior nor the data ever assigned probability to -- the same reason this planner commits the best
    candidate instead of MPPI's softmax-weighted mean. It is here as the BASELINE that prefix conditioning
    has to beat, because measuring that failure ourselves is worth more than citing it."""

    def __init__(self, inner, decay: float = 0.5):
        self.inner, self.decay = inner, float(decay)
        self.name = f"crossfade(decay={decay:g}) o {inner.name}"
        for a in ("commit", "needs_ctx", "K", "a_max"):
            setattr(self, a, getattr(inner, a, None))

    def sample(self, mean, k: int, ctx=None, g=None, prefix=None) -> torch.Tensor:
        a = self.inner.sample(mean, k, ctx=ctx, g=g, prefix=prefix)
        if prefix is None or prefix.shape[1] == 0:
            return a
        p = min(prefix.shape[1], a.shape[2])
        w = torch.exp(-self.decay * torch.arange(p, device=a.device, dtype=a.dtype))[None, None, :, None]
        a = a.clone()
        a[:, :, :p] = (1.0 - w) * a[:, :, :p] + w * prefix[:, None, :p]
        return a


class Held:
    """Snap a step to the previous one when the change is below `tol`, turning near-holds into exact holds.
    ANY proposal.

    The cheap stand-in for the head's missing hold behaviour: measured, the recorded fore/aft stick is held
    13.4 consecutive steps and the prior holds it 2.0, which is worse than a time-shuffled null. Unlike a
    crossfade this cannot invent an action between two modes -- it moves a value ONTO the data's rest-atom,
    a point with real probability mass -- so it is safe in a way averaging is not. It is approximate in two
    ways worth stating: it also flattens genuine small movements, and it cannot manufacture a hold out of a
    draw that drifts steadily, only remove jitter around an already-flat stretch.

    THIS IS THE PROBE for increment-PIT. If Held recovers a useful share of DataProposal's advantage, the
    hold-atom really is the defect and refitting the percentile transform to increments is worth a retrain;
    if it does not, the head's problem is bigger than the atom and that retrain would be wasted."""

    def __init__(self, inner, tol: float = 0.02):
        self.inner, self.tol = inner, float(tol)
        self.name = f"held(tol={tol:g}) o {inner.name}"
        for a in ("commit", "needs_ctx", "K", "a_max"):
            setattr(self, a, getattr(inner, a, None))

    def sample(self, mean, k: int, ctx=None, g=None, prefix=None) -> torch.Tensor:
        a = self.inner.sample(mean, k, ctx=ctx, g=g, prefix=prefix)
        # SEQUENTIAL on purpose: a hold has to propagate, so step t is compared with the already-snapped
        # t-1, not with the raw draw. Vectorising over (G,k,axes) leaves only the H-loop, H <= 32.
        out = [a[:, :, 0]] if prefix is None or prefix.shape[1] == 0 else \
              [torch.where((a[:, :, 0] - prefix[:, None, 0]).abs() < self.tol, prefix[:, None, 0], a[:, :, 0])]
        for t in range(1, a.shape[2]):
            out.append(torch.where((a[:, :, t] - out[-1]).abs() < self.tol, out[-1], a[:, :, t]))
        return torch.stack(out, dim=2)


def wrap(proposal, names, *, held_tol: float = 0.02, crossfade_decay: float = 0.5):
    """Apply harness wrappers by name, outermost last. `names` is a list like ["held"] or ["held",
    "crossfade"]. Unknown names raise rather than being ignored -- a silently dropped harness would look
    exactly like a harness that did not help."""
    for n in names or []:
        if n == "crossfade":
            proposal = Crossfade(proposal, crossfade_decay)
        elif n == "held":
            proposal = Held(proposal, held_tol)
        else:
            raise ValueError(f"unknown steer.harness {n!r} (crossfade | held)")
    return proposal


def chunk_return(s: torch.Tensor, objective: str) -> torch.Tensor:
    '''WHAT A CANDIDATE CHUNK IS JUDGED BY. s (k, steps) is the reward at every imagined step. -> (k,)

    level     mean R over the chunk. R is a STATE similarity ("does this latent look like climb") and it
              SATURATES -- measured, a plan reaches its reachable ceiling by step ~12 of 128 -- so the
              gradient thins out. Despite that it is the BETTER objective, measured over 15 contexts:
              opposing-pair separation on yaw +0.494 (15/15 contexts) against progress's +0.194, and
              +0.101 on vertical where progress has none.
    progress  R(end of chunk) - R(start of chunk). Keeps a gradient after the level saturates, which was
              the reason to try it, but it is a difference of two noisy scores and it discards the level
              information that turned out to be doing the work. An earlier 6/8-vs-4/8 result favouring it
              was a 4-context artifact; at 15 contexts the ordering reverses.
    terminal  R at the last step of the chunk. Between the two: not diluted by steps already at the
              ceiling, but not a difference either.

    NOTE why an advantage/baseline is NOT offered: subtracting any per-block constant (a no-op rollout, the
    candidate mean) leaves the argmax within that block untouched, so it would change the number reported
    and nothing about the plan. The chunk's SHAPE is the only part of the objective that selection sees.'''
    if objective == "level":
        return s.mean(dim=1)
    if objective == "progress":
        return s[:, -1] - s[:, 0]
    if objective == "terminal":
        return s[:, -1]
    raise ValueError(f"unknown steer.objective {objective!r} (level | progress | terminal)")


class DataProposal:
    """Candidates are REAL action chunks, drawn uniformly from the recorded flights.

    The null that isolates SMOOTHNESS from CONDITIONING, which no other proposal does. Gaussian noise is
    state-blind and unflyable; the prior is state-aware but its draws hold the stick for 2 steps where a
    pilot holds it for 13. A bank of recorded chunks is state-BLIND but perfectly flyable by construction --
    every candidate is something a human actually flew, with the data's own hold lengths and jerk. So if
    steering improves with this proposal, the action head's realism is the bottleneck; if it does not, the
    bottleneck is elsewhere and no amount of fixing the head will help.

    Smoothness within a chunk is free here; continuity BETWEEN chunks is a separate problem, and
    `prefix_retrieval` is this proposal's answer to it -- draw from the chunks that already open where the
    plan needs to continue rather than editing any chunk. See `sample`.

    `bank` is (n, K, action_dim) NORMALIZED chunks; `norm` denormalizes on the way out for an env-space
    planner, exactly as PriorProposal does."""

    def __init__(self, bank: torch.Tensor, a_max: float = 1e9, norm=None,
                 prefix_retrieval: bool = True, retrieval_tau: float = 0.05):
        self.bank = bank
        self.a_max = float(a_max)
        self.norm = norm
        self.prefix_retrieval = bool(prefix_retrieval)
        self.retrieval_tau = float(retrieval_tau)
        self.K = int(bank.shape[1])
        self.name = (f"data(bank={len(bank)} chunks, K={self.K}"
                     f"{', retrieved' if prefix_retrieval else ''})")
        self.commit = "best"        # real chunks are structured; averaging them is as wrong as averaging draws
        self.needs_ctx = False      # state-BLIND on purpose: that is what makes it the smoothness null

    def sample(self, mean, k: int, ctx=None, g=None, prefix=None) -> torch.Tensor:
        """`prefix` (G,p,A): PREFIX-CONDITIONED RETRIEVAL. A bank cannot be steered the way a flow can, but
        it can be SEARCHED -- so instead of editing a chunk (which would destroy the one property this
        proposal has), draw from the chunks that already open where the plan needs to continue, sampling
        from a softmax over -distance so the choice stays stochastic.

        This was rejected once on a measurement and the measurement was wrong: on a 365-chunk bank (val at
        stride 8) the 64th-nearest opening sat 6.3x the recorded step change away and NO seam had a full
        candidate pool within one step, so conditioning would have bought continuity by shrinking the pool
        to nothing. On the real bank (train at stride 1, 24,737 chunks) the 64th-nearest is 1.4x and 46% of
        seams have a full pool. The verdict was about the bank, not about the idea.

        Only the OPENING is matched, not the whole overlap: nothing here is committed to executing the
        previous plan's tail, so the constraint that matters is joining smoothly onto it, and matching all
        p positions would collapse the pool for no gain."""
        G, H, A = mean.shape
        assert H <= self.K, f"horizon {H} exceeds the bank's chunk {self.K}"
        if not (self.prefix_retrieval and prefix is not None and prefix.shape[1] > 0):
            i = torch.randint(len(self.bank), (G * k,), device=self.bank.device, generator=g)
        else:
            d = torch.cdist(prefix[:, 0].to(self.bank.dtype), self.bank[:, 0])      # (G, n)
            i = torch.multinomial(torch.softmax(-d / max(self.retrieval_tau, 1e-6), dim=-1),
                                  k, replacement=True, generator=g).reshape(-1)
        a = self.bank[i].reshape(G, k, self.K, A)[:, :, :H].to(mean.device, mean.dtype)
        if self.norm is not None:
            a = self.norm.denorm_act(a)
        return a.clamp(-self.a_max, self.a_max)


@torch.no_grad()
def plan(model, lang, t_e, ctx_obs, ctx_act, proposal, *, horizon: int, lookahead: int,
         n_samples: int, lam: float, objective: str = "level", commit: int = 0,
         beta_jerk: float = 0.0, generator=None, log=None):
    """Receding-horizon plan inside the imagination. Returns (bags, actions, scores).

    bags    (horizon, n_state, d) the imagined internal state at every planned step
    actions (horizon, action_dim) the NORMALIZED actions the planner committed to
    scores  (horizon,)            the reward of each committed step, for a progress curve

    At each block: draw `n_samples` candidate chunks from `proposal`, roll each through the world model from
    the CURRENT imagined state, score every imagined latent against the request, and commit the chunk of the
    MPPI-weighted best. Committing the softmax-weighted MEAN of candidate actions (as MPPI does) is wrong for
    a proposal with structure -- averaging two plausible manoeuvres gives an implausible one -- so the best
    single candidate is committed instead."""
    m = model
    bag_buf = list(m.encode_state(ctx_obs).unbind(dim=1))              # P bags of (1, n_state, d)
    acts = list(ctx_act.unbind(dim=1))                                 # P-1.. actions leading into them
    out_bags, out_acts, out_scores = [], [], []
    # THE UNEXECUTED TAIL of the chunk just committed, which is what any continuity harness conditions on.
    # It exists only when `commit` < the steps scored: the leftover actions were planned for exactly the
    # timesteps the NEXT draw will occupy, so they align with its first positions. At commit == lookahead
    # (the default) there is no overlap, nothing to be continuous with, and every harness is a no-op --
    # which is the regime all the jerk measurements were taken in.
    prev_tail = None
    W = int(getattr(m, "window", 32) or 32)

    while len(out_bags) < horizon:
        steps = min(lookahead, horizon - len(out_bags))                # steps SCORED
        # STEPS EXECUTED before re-planning, which need not equal the steps scored and is the difference
        # between "propose a coherent manoeuvre" and "commit to one". The proposal's value is that its
        # chunk hangs together; committing the WHOLE chunk also commits its wandering, and with chunk=32
        # that is 8.5 s of open loop from a single decision -- measured, the steps a chunk scored highest on
        # carried the requested stick (+0.33) while the chunk's average went the other way (-0.15). Commit
        # 1-2 steps of a 32-step proposal and the executed trajectory becomes the per-step argmax of a
        # coherent proposal instead. 0 -> commit everything scored (what every run before 2026-09-14 did).
        ncom = max(1, min(int(commit) or steps, steps))
        # context vector for the prior: the backbone over the recent window, pooled, at the LAST step
        zs = torch.stack(bag_buf[-W:], dim=1)                          # (1, w, n_state, d)
        aa = torch.stack(acts[-zs.shape[1]:], dim=1)                   # (1, w, a)
        h = m.pool_context(m.backbone(m._to_input(zs, aa)))[:, -1]     # (1, d_ctx)
        # proposals are MPPI-shaped -- (G,H,A) mean in, (G,k,H,A) out -- so the same objects serve
        # eval_control's batched episodes. One context here, hence G == 1 and the squeeze.
        # zeros, not a running average of past plans: this planner commits the best candidate rather than a
        # weighted mean, so there is no plan to warm-start from. For the gaussian arm that makes candidates
        # ordinary white noise about rest -- which is the control we want. (It also fixes the arm outright:
        # the old signature took `mean` and asserted it was not None, so proposal=gaussian could not run.)
        mean = torch.zeros(1, steps, acts[-1].shape[-1], device=zs.device, dtype=zs.dtype)
        cand = proposal.sample(mean, n_samples, ctx=h, g=generator, prefix=prev_tail)[0]  # (k, steps, a)

        # roll every candidate from the current imagined state. `_rollout_from` indexes `actions` by the
        # ABSOLUTE step position (a_win = actions[:, Lh-real:Lh] with Lh = len(bag_buf)), so it needs the
        # whole history aligned with the bag buffer, not just the new chunk -- passing only the chunk makes
        # the action window one short of the state window and the concat inside _to_input fails.
        buf = [b.expand(n_samples, -1, -1) for b in bag_buf]
        hist = torch.cat([a.expand(n_samples, -1).unsqueeze(1) for a in acts], dim=1)    # (k, L, a)
        rolled = m._rollout_from(buf, torch.cat([hist, cand], dim=1), steps, 0.0, None, 0)
        # `lang` is language/reward.LanguageReward -- the same scorer MPPI language steering uses, so a plan
        # is judged by exactly the reward a controller would see, and open-vocabulary phrasings work.
        s = lang.score(rolled.reshape(n_samples * steps, -1), t_e).reshape(n_samples, steps)
        ret = chunk_return(s, objective)                               # see chunk_return: `level` saturates
        if beta_jerk > 0.0:
            # CONTINUITY, which nothing else in this planner asks for. Measured on real flights, the sticks
            # change by 0.055 (raw) per step; a plan's commands change 2.5-4x that inside a chunk and
            # 8-9x at the SEAM, where one committed chunk ends and a freshly drawn one begins with nothing
            # connecting them. Penalising mean |da| -- with the previous committed action prepended, so the
            # seam is inside the penalty rather than outside it -- is what makes "smoothly" part of the
            # objective instead of an accident of the prior. Normalized action units.
            seq = torch.cat([acts[-1].expand(n_samples, -1).unsqueeze(1), cand], dim=1)   # (k,steps+1,a)
            ret = ret - beta_jerk * (seq[:, 1:] - seq[:, :-1]).abs().mean(dim=(1, 2))
        best = int(torch.argmax(ret))
        w = torch.softmax(ret / max(lam, 1e-6), dim=0)                 # logged for diagnostics only
        # NOTE: best is committed for EITHER proposal, ignoring proposal.commit -- the gaussian arm is a
        # control, and it is only a control if the selection rule is held fixed and the candidate source is
        # the only thing that changes. eval_control honours proposal.commit instead, because there the
        # weighted mean IS the algorithm being evaluated.
        prev_tail = cand[best, ncom:].unsqueeze(0) if ncom < steps else None   # (1, steps-ncom, a)
        for t in range(ncom):
            out_bags.append(rolled[best, t:t + 1])
            out_acts.append(cand[best, t:t + 1])
            out_scores.append(float(s[best, t]))
            bag_buf.append(rolled[best, t:t + 1])
            acts.append(cand[best, t:t + 1])
        if log is not None:
            log(f"planned {len(out_bags)}/{horizon} | scored {steps} committed {ncom} | block return "
                f"{float(ret[best]):+.4f} (over candidates: mean {float(ret.mean()):+.4f} "
                f"sd {float(ret.std()):.4f}, top weight {float(w.max()):.3f})")
    return (torch.cat(out_bags, 0), torch.cat(out_acts, 0), np.asarray(out_scores, dtype=np.float32))


def motion_readout(actions: np.ndarray, axes: list, n_axes: int) -> dict:
    """What the COMMITTED actions say the drone was told to do -- exact, and computed without the reward head.

    This is the independent check. Scoring imagined latents with the reward head and then reporting that the
    reward went up is circular: the planner maximises exactly that number. The commanded motion is derived
    from the chosen actions alone, so "the plan scores well" and "the plan actually climbs" stay separate."""
    a = np.asarray(actions, dtype=np.float32)
    a = a.reshape(len(a), a.shape[-1] // n_axes, n_axes).mean(axis=1)   # fold concat sub-steps, keep sign
    out = {}
    for j, ax in enumerate(axes):
        mu = float(a[:, j].mean())
        out[ax["name"]] = {"mean": mu, "frac_positive": float((a[:, j] > 0.05).mean()),
                           "frac_negative": float((a[:, j] < -0.05).mean()),
                           "direction": ax["positive"] if mu > 0 else ax["negative"]}
    return out
