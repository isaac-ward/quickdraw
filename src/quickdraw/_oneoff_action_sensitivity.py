"""ONE-OFF (2026-08-12): is the frozen long-horizon prediction ACTION-DEPENDENT, and if so, does the
action signal survive the DECODER?

`motion_ratio` says the rollout stops moving by ~+16 steps, but it is a magnitude ratio (direction-blind)
and cannot say WHY. This asks the sharper question directly: hold the context fixed, change only the FUTURE
actions, and see whether the imagined future changes at all -- in the latent bag and in pixels separately.

WHY BOTH SPACES: if shuffling actions moves the LATENT but not the PIXELS, the dynamics is action-sensitive
and the decoder is washing the difference out -- a recon-capacity problem. If it moves neither, the dynamics
itself ignores actions -- an objective problem. Those need opposite fixes, and one rollout pair answers it
because to_obs decodes the very bags _rollout returns.

THE CONTROL THAT MAKES THIS MEAN ANYTHING: model.diffusion.stochastic_eval is true, so two rollouts with
IDENTICAL actions already differ by sampler noise. Every variant is therefore rolled under the SAME reseeded
RNG (identical noise draws), and `true_reseed` -- same actions, different seed -- measures the noise floor.
A variant only demonstrates action-dependence if it diverges MORE than true_reseed does.

Runs in fp32 (no autocast) on purpose: bf16 nondeterminism is another noise source, and this is a
measurement of small differences.

Perturbs only actions from index P-1 onward -- the ones that drive predicted steps -- so the context bags
and their aligned actions stay intact and the only variable is the COMMANDED FUTURE.

    python -m quickdraw._oneoff_action_sensitivity checkpoint=<ckpt> ...
"""

from __future__ import annotations

import json
import os

import hydra
import numpy as np
import torch

from .data.dataset import load_split_episodes_mm
from .training.setup import build_model, env_cfg, load_checkpoint, normalizer, resolve_data_root

H_REPORT = [1, 8, 16, 32, 64]


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """a,b in [0,1], any shape -> scalar dB."""
    mse = torch.mean((a.float() - b.float()) ** 2).item()
    return float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)


@torch.no_grad()
@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg):
    from omegaconf import OmegaConf
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = cfg.checkpoint
    run = os.path.dirname(os.path.dirname(ck))
    saved = OmegaConf.create(json.load(open(os.path.join(run, "logs", "config.json"))))
    OmegaConf.set_struct(cfg, False)
    cfg.model = saved.model
    m = build_model(cfg).to(device)
    load_checkpoint(m, ck)
    m.eval()
    norm = normalizer(cfg)
    env_cfg(cfg)

    P = int(cfg.data.P)
    H = 64
    n_clips = int(cfg.get("n_clips", 8))
    img_head = next(n for n, _ in m.layout if hasattr(m.modalities[n], "ae"))
    img_size = m.modalities[img_head].ae.cfg.img_size

    eps = load_split_episodes_mm(resolve_data_root(cfg), "val", img_size=img_size,
                                 cam=cfg.data.get("cam", "fpv"), repo_id=cfg.data.get("repo_id", "torus"))
    rng = np.random.RandomState(0)
    slices = [(ei, t) for ei in range(len(eps)) for t in range(P, len(eps[ei][0]) - H - 1)]
    rng.shuffle(slices)
    slices = slices[:n_clips]
    print(f"[setup] {len(eps)} val eps | {len(slices)} clips | P={P} H={H} img={img_head}@{img_size} "
          f"| stochastic_eval={cfg.model.diffusion.get('stochastic_eval')} steps={cfg.model.diffusion.get('sampling_steps')}")

    ctx = {"proprio": torch.stack([norm.norm_obs(torch.from_numpy(eps[ei][0][t - P:t])) for ei, t in slices]).float().to(device),
           img_head: torch.stack([torch.from_numpy(eps[ei][2][t - P:t]) for ei, t in slices]).float().div(255.0).to(device)}
    a_true = torch.stack([norm.norm_act(torch.from_numpy(eps[ei][1][t - P:t + H])) for ei, t in slices]).float().to(device)
    gt = torch.stack([torch.from_numpy(eps[ei][2][t:t + H]) for ei, t in slices]).float().div(255.0).to(device)

    # --- action variants. Only indices >= P-1 are touched: those drive the PREDICTED steps.
    g = np.random.RandomState(1)
    variants = {}
    variants["true"] = a_true
    a = a_true.clone()                                        # time-permute the commanded future
    perm = torch.from_numpy(g.permutation(a.shape[1] - (P - 1))).to(device)
    a[:, P - 1:] = a[:, P - 1:][:, perm]
    variants["shuffled_time"] = a
    a = a_true.clone()                                        # actions from a DIFFERENT clip (decorrelated, in-dist)
    roll = torch.from_numpy(g.permutation(len(slices))).to(device)
    a[:, P - 1:] = a_true[roll][:, P - 1:]
    variants["other_clip"] = a
    a = a_true.clone()                                        # zero in NORMALISED space == the mean action
    a[:, P - 1:] = 0.0
    variants["zero"] = a
    # ORDER-ONLY perturbations that are actually BIG. A random permutation is a weak perturbation on this
    # dataset (measured 0.771 relative L2 vs 1.936 for a clip swap, and EXACTLY 0 in >10% of windows) because
    # robocasa actions are near-smooth: lag-1 r = +0.988, consecutive steps differ by 13%. Reversal and a
    # half-window shift preserve the multiset but misalign a SMOOTH trajectory, so they perturb much harder --
    # which is what makes "does ORDER matter" separable from "did you change the action distribution".
    a = a_true.clone()
    a[:, P - 1:] = torch.flip(a[:, P - 1:], dims=[1])
    variants["reversed_time"] = a
    a = a_true.clone()
    a[:, P - 1:] = torch.roll(a[:, P - 1:], shifts=H // 2, dims=1)
    variants["shift_half"] = a

    # how big is each perturbation, in the action space the model consumes? (relative L2 on the driving slice)
    ref_a = a_true[:, P - 1:]
    pert = {n: (torch.norm(v[:, P - 1:] - ref_a) / torch.norm(ref_a)).item() for n, v in variants.items()}
    print("[perturbation size] " + "  ".join(f"{n}={p:.3f}" for n, p in pert.items() if n != "true"))

    # --- roll every variant under the SAME noise, plus N same-actions/different-noise floor draws.
    # N>1 because the whole interpretation hinges on the floor: with a single draw, any ratio near 1.0x is
    # indistinguishable from "the floor happened to land there". Three draws give a range to judge against.
    SEED = 1234
    N_FLOOR = 3
    bags, imgs = {}, {}
    floors = [f"true_reseed{i}" for i in range(N_FLOOR)]
    order = list(variants) + floors
    for name in order:
        act = variants["true"] if name.startswith("true_reseed") else variants[name]
        torch.manual_seed(SEED + 1 + int(name[-1]) if name.startswith("true_reseed") else SEED)
        bag = m._rollout(ctx, act, H, 0.0, None, 0)                     # (B,H,n_state,d)
        bags[name] = bag.float()
        imgs[name] = m.to_obs(bag, heads=[img_head])[img_head].float().clamp(0, 1)
        print(f"[roll] {name:<14} bag={tuple(bags[name].shape)} img={tuple(imgs[name].shape)}")

    ref_b, ref_i = bags["true"], imgs["true"]
    print("\n" + "=" * 100)
    print("DIVERGENCE FROM THE TRUE-ACTION ROLLOUT  (identical sampler noise except true_reseed)")
    print("  lat_rel = ||z_v - z_true||_F / ||z_true||_F      pix_psnr = PSNR(variant, true) dB, HIGH = identical")
    print("=" * 100)
    hdr = f"{'variant':<14}{'metric':<10}" + "".join(f"{'+'+str(h):>9}" for h in H_REPORT)
    print(hdr + "\n" + "-" * len(hdr))
    out = {}
    for name in order:
        if name == "true":
            continue
        lat = [ (torch.norm(bags[name][:, h - 1] - ref_b[:, h - 1]) / torch.norm(ref_b[:, h - 1])).item() for h in H_REPORT ]
        pix = [ psnr(imgs[name][:, h - 1], ref_i[:, h - 1]) for h in H_REPORT ]
        out[name] = {"lat_rel": lat, "pix_psnr": pix}
        print(f"{name:<14}{'lat_rel':<10}" + "".join(f"{v:9.4f}" for v in lat))
        print(f"{'':<14}{'pix_psnr':<10}" + "".join(f"{v:9.2f}" for v in pix))

    print("\n" + "=" * 100)
    print("ACCURACY vs GROUND TRUTH — do the TRUE actions actually help?")
    print("=" * 100)
    hdr2 = f"{'variant':<14}" + "".join(f"{'+'+str(h):>9}" for h in H_REPORT)
    print(hdr2 + "\n" + "-" * len(hdr2))
    for name in order:
        gp = [psnr(imgs[name][:, h - 1], gt[:, h - 1]) for h in H_REPORT]
        out.setdefault(name, {})["gt_psnr"] = gp
        print(f"{name:<14}" + "".join(f"{v:9.2f}" for v in gp))

    print("\n" + "=" * 100)
    print(f"VERDICT — perturbations scored against the noise floor ({N_FLOOR} reseed draws; x = ratio to the")
    print("          floor MEAN, and (lo-hi) is what the floor's own spread would score as)")
    print("=" * 100)
    mse = lambda db: 10 ** (-db / 10)                       # PSNR is logarithmic -- ratio MSE, never dB
    for i, h in enumerate(H_REPORT):
        fl = [out[f]["lat_rel"][i] for f in floors]
        fp = [mse(out[f]["pix_psnr"][i]) for f in floors]
        fl_m, fp_m = sum(fl) / len(fl), sum(fp) / len(fp)
        print(f"  +{h:<4} floor: latent {fl_m:.4f} ({min(fl):.4f}-{max(fl):.4f})  "
              f"pixel-mse {fp_m:.2e} (spread {max(fp)/min(fp):.2f}x)")
        for name in ("shuffled_time", "reversed_time", "shift_half", "other_clip", "zero"):
            lr = out[name]["lat_rel"][i] / max(fl_m, 1e-12)
            pr = mse(out[name]["pix_psnr"][i]) / max(fp_m, 1e-30)
            # /pert = response PER UNIT of action change, which is what makes an order-only perturbation
            # comparable to one that also moves the action distribution
            print(f"        {name:<14} latent {lr:6.2f}x | pixel {pr:6.2f}x | pert {pert[name]:.3f} "
                  f"-> pixel/pert {pr / max(pert[name], 1e-9):6.2f}")
    json.dump(out, open("/tmp/action_sensitivity.json", "w"), indent=2)
    print("\n[done] raw numbers -> /tmp/action_sensitivity.json")


if __name__ == "__main__":
    main()
