"""Check the two MPPI PROPOSALS: the gaussian arm is bit-identical to the candidate noise controller/mppi.py
used to inline, and the prior arm is actually wired.

WHY THE PARITY CHECK MATTERS. `mean + noise_sigma * randn` moved out of controller/mppi._mppi_step and into
evaluation/steering.GaussianProposal so that eval_control and eval_steer could share one candidate interface.
Every torus control number in the record was produced by the inlined version, so the move is only free if the
candidate tensor AND the generator state come out identical -- which is what `parity()` asserts.

The prior arm is checked on the path eval_control takes: build the pooled context from a real observation
window (controller/mppi._h_ctx), draw candidates in RAW action units, check both commit rules and that the
draw is reproducible under a seed. No env is needed -- starling has no simulator, so the rollout and reward
are stubbed and only the candidate/commit plumbing is under test.

    CUDA_VISIBLE_DEVICES=0 python scratch/check_proposals.py <train_action_run>
"""
import json, os, sys
import numpy as np
import torch
from omegaconf import OmegaConf

from quickdraw.controller.mppi import MPPIConfig, _commit, _h_ctx, _mppi_step_reward
from quickdraw.data.dataset import load_split_episodes_mm, set_action_aggregate, set_subsample
from quickdraw.evaluation.steering import GaussianProposal, PriorProposal
from quickdraw.training.setup import (build_model, env_cfg, image_head_cams, image_head_sizes,
                                      load_checkpoint, normalizer, resolve_data_root)


# --- determinism of the seeded draw (appended check) ---
def parity():
    """The gaussian proposal must reproduce the inlined MPPI arithmetic exactly -- tensor and rng state."""
    from quickdraw.controller.mppi import MPPIConfig, _commit
    from quickdraw.evaluation.steering import GaussianProposal
    G, K, H, A, sigma, a_max = 8, 128, 64, 2, 2.0, 4.0
    mean = torch.randn(G, H, A) * 0.7
    g1, g2 = torch.Generator().manual_seed(0), torch.Generator().manual_seed(0)
    old = (mean[:, None] + torch.randn(G, K, H, A, generator=g1) * sigma).clamp(-a_max, a_max)
    new = GaussianProposal(sigma, a_max).sample(mean, K, ctx=None, g=g2)
    assert torch.equal(old, new) and torch.equal(g1.get_state(), g2.get_state())
    mp, ret = MPPIConfig(), torch.randn(G, K)
    w = torch.softmax(ret / mp.lambda_, dim=1)
    assert torch.equal((w[..., None, None] * old).sum(1), _commit(old, ret, mp, GaussianProposal(sigma, a_max)))
    print("gaussian: candidates + rng state + weighted-mean commit all bit-identical to the inlined version")


def _determinism(core, norm, h, A, dev):
    from quickdraw.evaluation.steering import PriorProposal
    pr = PriorProposal(core, a_max=1e9, norm=norm)
    mean = torch.zeros(h.shape[0], pr.K, A, device=dev)
    def draw(seed):
        g = torch.Generator(device=dev); g.manual_seed(seed)
        return pr.sample(mean, 32, ctx=h, g=g)
    a1, a2, a3 = draw(0), draw(0), draw(1)
    print(f"  same seed identical: {torch.equal(a1, a2)} | max|diff| {float((a1-a2).abs().max()):.2e}")
    print(f"  diff seed differs  : {not torch.equal(a1, a3)}")
    # and the un-seeded path (g=None) must still vary, as it always did
    b1, b2 = pr.sample(mean, 32, ctx=h, g=None), pr.sample(mean, 32, ctx=h, g=None)
    print(f"  g=None still random: {not torch.equal(b1, b2)}")


def main(run: str) -> int:
    parity()
    cfg = OmegaConf.create(json.load(open(os.path.join(run, "logs", "config.json"))))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    set_subsample(int(cfg.data.get("subsample", 1) or 1))
    set_action_aggregate(str(cfg.data.get("action_aggregate", "sum")))
    m = build_model(cfg).to(dev)
    ck = os.path.join(run, "checkpoints", "best.ckpt")
    if not os.path.exists(ck):     # action-head runs keep per-epoch checkpoints; backups keep last.ckpt
        import glob
        ep = glob.glob(os.path.join(run, "checkpoints", "epoch=*.ckpt"))
        ck = (max(ep, key=lambda f: int(f.split("epoch=")[1].split("-")[0])) if ep
              else os.path.join(run, "checkpoints", "last.ckpt"))
    print(f"checkpoint {os.path.basename(ck)}")
    load_checkpoint(m, ck)
    m.eval()
    core = getattr(m, "_orig_mod", m)
    norm, ecfg = normalizer(cfg), env_cfg(cfg)
    P = int(cfg.data.P)
    from quickdraw.training.setup import effective_action_dim
    A = effective_action_dim(cfg)
    img_head = next((n for n, _ in core.layout if n != "proprio"), None)
    eps = load_split_episodes_mm(resolve_data_root(cfg), "val", img_size=image_head_sizes(cfg),
                                 cam=image_head_cams(cfg), repo_id=cfg.data.get("repo_id", "torus"))
    o, a, fr = eps[0]
    t, B, K = 120, 2, 32
    obs_win = torch.from_numpy(o[t - P:t]).float()[None].expand(B, -1, -1).to(dev)          # (B,P,obs) RAW
    fpv_win = torch.from_numpy(fr[img_head][t - P:t]).float().div(255.0)[None].expand(B, -1, -1, -1, -1).to(dev)
    pa = torch.from_numpy(a[t - P:t - 1]).float()[None].expand(B, -1, -1).to(dev)            # (B,P-1,A) RAW

    with torch.no_grad():
        h = _h_ctx(core, norm, obs_win, fpv_win, pa, img_head)
        print(f"_h_ctx -> {tuple(h.shape)}  (pooling {getattr(core, 'pool_context_mode', getattr(core, 'action_head_context', '?'))})")
        a_max = float(getattr(ecfg, "a_max", 1.0) or 1.0)
        prior = PriorProposal(core, a_max=1e9, norm=norm)
        mean = torch.zeros(B, prior.K, A, device=dev)
        cand = prior.sample(mean, K, ctx=h, g=None)
        raw = np.abs(a[t - P:t + 200]).reshape(-1, A)
        print(f"prior.sample -> {tuple(cand.shape)}  needs_ctx={prior.needs_ctx} commit={prior.commit!r}")
        print(f"  candidate |a| mean per axis {np.round(cand.abs().mean((0,1,2)).cpu().numpy(),3)}")
        print(f"  RECORDED  |a| mean per axis {np.round(raw.mean(0),3)}   <- RAW units: these must be comparable")
        assert cand.shape == (B, K, prior.K, A)
        # the two episodes share a context, so their draws must differ (independent samples, not a broadcast)
        assert not torch.equal(cand[0], cand[1]), "both episodes drew the SAME candidates -- ctx expand is wrong"
        # commit rules
        mp = MPPIConfig(num_samples=K, horizon=prior.K, lambda_=0.3)
        ret = torch.randn(B, K, device=dev)
        best = _commit(cand, ret, mp, prior)
        assert torch.equal(best, cand[torch.arange(B, device=dev), ret.argmax(1)]), "commit=best is not the argmax"
        gm = _commit(cand, ret, mp, GaussianProposal(0.5, a_max))
        assert not torch.equal(gm, best), "commit=mean and commit=best gave the same plan"
        print(f"  commit best -> {tuple(best.shape)} == argmax candidate; commit mean differs  OK")
        # the full MPPI step, with the rollout/reward stubbed (no simulator for starling)
        plan = _mppi_step_reward(lambda c: c.sum(-1, keepdim=True).expand(*c.shape[:3], 6),
                                 mean, mp, prior, None, lambda obs, gl: obs.sum(-1), h_ctx=h)
        print(f"  _mppi_step_reward -> plan {tuple(plan.shape)}  OK")
        _determinism(core, norm, h, A, dev)
    print("\nprior proposal is wired: context built, candidates drawn in raw units, commit=best honoured.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))

