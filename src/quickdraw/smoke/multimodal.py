"""Greenlight smoke for the multimodal world-model spine (models/multimodal.py), on MultiModalLSAR.
Synthetic obs (proprio + image). Checks: forward/to_obs/rollout shapes, the token-bag layout, that a few
train steps drive BOTH per-head recon losses + pred_latent down, AND that proprio-only (image disabled in
config) runs through the same spine. Run: uv run python -m quickdraw.smoke.multimodal
"""
import torch
import torch.nn.functional as F

from quickdraw.models.modalities import ModalitySpec
from quickdraw.models.multimodal import MultiModalLSAR

DEV = "cuda" if torch.cuda.is_available() else "cpu"
R = []


def check(name, cond, extra=""):
    R.append(bool(cond))
    print(f"[{'OK' if cond else 'FAIL'}] {name}{(' — ' + extra) if extra else ''}")


def build(specs, d=256, depth=2):
    torch.manual_seed(0)
    return MultiModalLSAR(specs, d=d, depth=depth, heads=4, window=16, mlp_ratio=4.0,
                          rope_theta=10000.0, action_dim=2).to(DEV)


def synth(B, L):
    g = torch.Generator(device=DEV).manual_seed(1)
    return ({"proprio": torch.randn(B, L, 6, generator=g, device=DEV),
             "image": torch.rand(B, L, 128, 128, 3, generator=g, device=DEV)},
            torch.randn(B, L, 2, generator=g, device=DEV))


def recon_losses(m, preds, future):
    dec = m.to_obs(preds)
    return {k: F.mse_loss(dec[k], future[k]) for k in future}


def main():
    B, P, Fh, d = 2, 8, 6, 256
    L = P + Fh
    specs = [ModalitySpec("proprio", "vector", dim=6), ModalitySpec("image", "image", num_tokens=8)]
    m = build(specs, d=d)
    obs, act = synth(B, L)

    # token-bag layout
    check("layout (proprio:1, image:8) -> n_state=9, n_input=10",
          m.n_state == 9 and m.n_input == 10, f"n_state={m.n_state} n_input={m.n_input}")

    # forward + to_obs shapes
    preds_full = m(obs, act)                                  # (B,L,n_state,d)
    check("forward bag (B,L,n_state,d)", preds_full.shape == (B, L, 9, d), str(tuple(preds_full.shape)))
    dec = m.to_obs(preds_full)
    check("to_obs proprio (B,L,6)", dec["proprio"].shape == (B, L, 6))
    check("to_obs image (B,L,128,128,3)", dec["image"].shape == (B, L, 128, 128, 3))

    # rollout (training) + imagine_eval shapes
    ctx = {k: v[:, :P] for k, v in obs.items()}
    fut = {k: v[:, P:] for k, v in obs.items()}
    roll = m.rollout_train(ctx, act[:, : L - 1], fut, p_tf=0.5, detach_every=4)
    check("rollout_train (B,F,n_state,d)", roll.shape == (B, Fh, 9, d), str(tuple(roll.shape)))
    img = m.imagine_eval(ctx, act[:, : L - 1], Fh)
    check("imagine_eval decodes both heads", img["proprio"].shape == (B, Fh, 6) and img["image"].shape == (B, Fh, 128, 128, 3))

    # a few train steps: per-head recon + pred_latent should drop
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    future = {k: obs[k][:, P:] for k, _ in m.layout}
    first, last = None, None
    for it in range(40):
        opt.zero_grad()
        pf = m({k: v[:, :-1] for k, v in obs.items()}, act[:, :-1])
        preds = pf[:, P - 1:]
        rl = recon_losses(m, preds, future)
        raw, w = m.loss_terms(preds, future, obs, 1.0, act)
        loss = sum(rl.values()) + sum(w[k] * raw[k] for k in raw)
        loss.backward(); opt.step()
        vals = {**{f"recon_{k}": float(v) for k, v in rl.items()}, "pred_latent": float(raw["pred_latent"])}
        if it == 0:
            first = vals
        last = vals
    check("recon proprio drops", last["recon_proprio"] < first["recon_proprio"] * 0.9,
          f"{first['recon_proprio']:.3f}->{last['recon_proprio']:.3f}")
    check("recon image drops", last["recon_image"] < first["recon_image"] * 0.9,
          f"{first['recon_image']:.4f}->{last['recon_image']:.4f}")
    check("pred_latent finite + drops", last["pred_latent"] < first["pred_latent"],
          f"{first['pred_latent']:.3f}->{last['pred_latent']:.3f}")

    # GENERALITY: proprio-only through the SAME spine (image disabled in config)
    mp = build([ModalitySpec("proprio", "vector", dim=6)], d=d)
    op = {"proprio": obs["proprio"]}
    pfp = mp(op, act)
    rp = mp.rollout_train({"proprio": op["proprio"][:, :P]}, act[:, : L - 1],
                          {"proprio": op["proprio"][:, P:]}, p_tf=0.5)
    check("proprio-only: n_state=1, forward+rollout run",
          mp.n_state == 1 and pfp.shape == (B, L, 1, d) and rp.shape == (B, Fh, 1, d))

    print(f"\n{'ALL OK' if all(R) else 'SOME FAILED'} ({sum(R)}/{len(R)})")
    import sys
    sys.exit(0 if all(R) else 1)


if __name__ == "__main__":
    main()
