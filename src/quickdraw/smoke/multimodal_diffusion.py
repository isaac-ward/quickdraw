"""Greenlight smoke for MultiModalFlow (models/multimodal.py) — latent flow-matching over the token
bag on the shared spine. Checks: forward/rollout/imagine_eval shapes, flow loss present+finite and drops,
DETERMINISTIC (ε=0) committed prediction byte-stable, per-head recon drops, and proprio-only generality.
Run: uv run python -m quickdraw.smoke.multimodal_diffusion
"""
import torch
import torch.nn.functional as F

from quickdraw.models.modalities import ModalitySpec
from quickdraw.models.multimodal import MultiModalFlow

DEV = "cuda" if torch.cuda.is_available() else "cpu"
R = []


def check(name, cond, extra=""):
    R.append(bool(cond))
    print(f"[{'OK' if cond else 'FAIL'}] {name}{(' — ' + extra) if extra else ''}")


def build(specs, d=256, depth=2, shortcut=False):
    torch.manual_seed(0)
    return MultiModalFlow(specs, d=d, depth=depth, heads=4, window=16, mlp_ratio=4.0,
                               rope_theta=10000.0, action_dim=2, sampling_steps=4, shortcut=shortcut).to(DEV)


def main():
    B, P, Fh, d = 2, 8, 6, 256
    L = P + Fh
    specs = [ModalitySpec("proprio", "vector", dim=6), ModalitySpec("image", "image", num_tokens=8)]
    m = build(specs, d=d)
    g = torch.Generator(device=DEV).manual_seed(1)
    obs = {"proprio": torch.randn(B, L, 6, generator=g, device=DEV),
           "image": torch.rand(B, L, 128, 128, 3, generator=g, device=DEV)}
    act = torch.randn(B, L, 2, generator=g, device=DEV)

    pf = m(obs, act)
    check("forward bag (B,L,n_state,d)", pf.shape == (B, L, 9, d), str(tuple(pf.shape)))
    raw, w = m.loss_terms(pf.detach(), {k: v[:, P:] for k, v in obs.items()}, obs, 1.0, act)
    check("flow loss present + finite", "flow/latent" in raw and torch.isfinite(raw["flow/latent"]).all())

    # DETERMINISTIC committed prediction: eval + eps=0 -> byte-identical across two calls
    m.eval()
    s = m.encode_state(obs)
    h = m.backbone(m._to_input(s, act))
    with torch.no_grad():
        r1 = m.readout(h, s)
        r2 = m.readout(h, s)
    check("deterministic (eps=0) readout byte-identical", torch.equal(r1, r2))
    m.train()

    # rollout + imagine_eval
    ctx = {k: v[:, :P] for k, v in obs.items()}
    roll = m.rollout_train(ctx, act[:, : L - 1], {k: v[:, P:] for k, v in obs.items()}, p_tf=0.5, detach_every=4)
    check("rollout_train (B,F,n_state,d)", roll.shape == (B, Fh, 9, d), str(tuple(roll.shape)))
    with torch.no_grad():
        im = m.imagine_eval(ctx, act[:, : L - 1], Fh)
    check("imagine_eval decodes both heads", im["proprio"].shape == (B, Fh, 6) and im["image"].shape == (B, Fh, 128, 128, 3))

    # train: flow loss + per-head recon drop. Flow-matching loss is high-variance (random tau/eps each
    # step), so compare WINDOWED MEANS over 150 steps rather than single noisy endpoints.
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    future = {k: obs[k][:, P:] for k, _ in m.layout}
    flow_hist, pro_hist, img_hist = [], [], []
    for it in range(150):
        opt.zero_grad()
        preds = m({k: v[:, :-1] for k, v in obs.items()}, act[:, :-1])[:, P - 1:]
        dec = m.to_obs(preds)
        rl = {k: F.mse_loss(dec[k], future[k]) for k in future}
        raw, w = m.loss_terms(preds, future, obs, 1.0, act)
        (sum(rl.values()) + sum(w[k] * raw[k] for k in raw)).backward(); opt.step()
        flow_hist.append(float(raw["flow/latent"])); pro_hist.append(float(rl["proprio"])); img_hist.append(float(rl["image"]))
    mean = lambda xs: sum(xs) / len(xs)
    f0, f1 = mean(flow_hist[:15]), mean(flow_hist[-15:])
    check("flow loss drops (windowed mean)", f1 < f0, f"{f0:.4f}->{f1:.4f}")
    check("recon (proprio+image) drops",
          mean(pro_hist[-15:]) < mean(pro_hist[:15]) and mean(img_hist[-15:]) < mean(img_hist[:15]),
          f"pro {mean(pro_hist[:15]):.3f}->{mean(pro_hist[-15:]):.3f}, img {mean(img_hist[:15]):.3f}->{mean(img_hist[-15:]):.3f}")

    # proprio-only generality
    mp = build([ModalitySpec("proprio", "vector", dim=6)], d=d)
    op = {"proprio": obs["proprio"]}
    check("proprio-only diffusion forward+rollout",
          mp(op, act).shape == (B, L, 1, d) and
          mp.rollout_train({"proprio": op["proprio"][:, :P]}, act[:, : L - 1], {"proprio": op["proprio"][:, P:]}, 0.5).shape == (B, Fh, 1, d))

    print(f"\n{'ALL OK' if all(R) else 'SOME FAILED'} ({sum(R)}/{len(R)})")
    import sys
    sys.exit(0 if all(R) else 1)


if __name__ == "__main__":
    main()
