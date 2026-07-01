"""Greenlight smoke for MultiModalDSAR (data-space AR over the token bag). Checks forward/rollout shapes,
that carry_transform is the data-space re-encode (== encode(decode(bag)) and != raw bag), per-head recon
drops, and proprio-only generality. Run: uv run python -m quickdraw.smoke.multimodal_dsar
"""
import torch
import torch.nn.functional as F

from quickdraw.models.modalities import ModalitySpec
from quickdraw.models.multimodal import MultiModalDSAR

DEV = "cuda" if torch.cuda.is_available() else "cpu"
R = []


def check(name, cond, extra=""):
    R.append(bool(cond))
    print(f"[{'OK' if cond else 'FAIL'}] {name}{(' — ' + extra) if extra else ''}")


def build(specs, d=256, depth=2):
    torch.manual_seed(0)
    return MultiModalDSAR(specs, d=d, depth=depth, heads=4, window=16, mlp_ratio=4.0,
                          rope_theta=10000.0, action_dim=2).to(DEV)


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
    check("no model-specific loss term (obs recon grounds DSAR)", m.loss_terms(pf, obs, obs, 1.0, act)[0] == {})

    # carry_transform is the data-space re-encode
    with torch.no_grad():
        bag = m.encode_state({k: v[:, :P] for k, v in obs.items()})[:, -1]   # (B,n_state,d)
        carried = m.carry_transform(bag)
        reencoded = m.encode_state(m.to_obs(bag))
    check("carry_transform == encode(decode(bag)) (data-space)", torch.allclose(carried, reencoded, atol=1e-5))
    check("carry_transform != identity (re-encoded, not raw)", not torch.allclose(carried, bag, atol=1e-3))

    # rollout + recon drop
    ctx = {k: v[:, :P] for k, v in obs.items()}
    roll = m.rollout_train(ctx, act[:, : L - 1], {k: v[:, P:] for k, v in obs.items()}, p_tf=0.5, detach_every=4)
    check("rollout_train (B,F,n_state,d)", roll.shape == (B, Fh, 9, d), str(tuple(roll.shape)))
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    future = {k: obs[k][:, P:] for k, _ in m.layout}
    first = last = None
    for _ in range(30):
        opt.zero_grad()
        preds = m({k: v[:, :-1] for k, v in obs.items()}, act[:, :-1])[:, P - 1:]
        dec = m.to_obs(preds)
        rl = {k: F.mse_loss(dec[k], future[k]) for k in future}
        sum(rl.values()).backward(); opt.step()
        v = {k: float(x) for k, x in rl.items()}
        first = first or v
        last = v
    check("recon (proprio+image) drops",
          last["proprio"] < first["proprio"] and last["image"] < first["image"],
          f"pro {first['proprio']:.3f}->{last['proprio']:.3f}, img {first['image']:.3f}->{last['image']:.3f}")

    mp = build([ModalitySpec("proprio", "vector", dim=6)], d=d)
    check("proprio-only DSAR forward+rollout",
          mp({"proprio": obs["proprio"]}, act).shape == (B, L, 1, d) and
          mp.rollout_train({"proprio": obs["proprio"][:, :P]}, act[:, : L - 1], {"proprio": obs["proprio"][:, P:]}, 0.5).shape == (B, Fh, 1, d))

    print(f"\n{'ALL OK' if all(R) else 'SOME FAILED'} ({sum(R)}/{len(R)})")
    import sys
    sys.exit(0 if all(R) else 1)


if __name__ == "__main__":
    main()
