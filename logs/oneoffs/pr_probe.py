"""ONE-OFF: is the latent's low participation ratio REAL (scene is low-DOF / data limit) or a 2-episode
artifact / an objective collapse? Recompute PR over MANY train episodes, and compare the LATENT's effective
dim to the RAW DATA's (proprio + a pixel-PCA proxy). Read-only on a checkpoint.
  python logs/oneoffs/pr_probe.py '+eval_ckpt="<...>.ckpt"' +n_ep=43 +stride=4
"""
from __future__ import annotations
import os, hydra, numpy as np, torch
from omegaconf import OmegaConf

def pr(x):  # x: (N, D) -> participation ratio (effective #dims)
    x = x.float(); x = x - x.mean(0, keepdim=True)
    lam = torch.linalg.svdvals(x / (x.shape[0] ** 0.5)) ** 2
    return float((lam.sum() ** 2) / (lam ** 2).sum())

@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg):
    from quickdraw.data.dataset import DataConfig, load_split_episodes_mm
    from quickdraw.training.setup import (build_model, image_head_cams, image_head_sizes, load_checkpoint,
                                          normalizer, resolve_data_root)
    ckpt = os.path.expanduser(str(cfg.get("eval_ckpt", "") or "")); assert os.path.exists(ckpt)
    run_dir = os.path.dirname(os.path.dirname(ckpt))
    rcfg = OmegaConf.load(os.path.join(run_dir, "checkpoints", "config.resolved.yaml")); OmegaConf.set_struct(rcfg, False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = build_model(rcfg).to(dev).eval(); load_checkpoint(m, ckpt); m = getattr(m, "_orig_mod", m)
    norm = normalizer(rcfg); cams, sizes = image_head_cams(rcfg), image_head_sizes(rcfg)
    stride = int(cfg.get("stride", 4))   # temporal subsample to decorrelate consecutive frames
    eps = load_split_episodes_mm(resolve_data_root(rcfg), "train", dcfg=DataConfig.from_cfg(rcfg),
                                 img_size=sizes, cam=cams, repo_id=str(rcfg.data.get("repo_id", "torus")))
    n_ep = min(int(cfg.get("n_ep", 43)), len(eps)); eps = eps[:n_ep]
    print(f"[pr] {os.path.basename(run_dir)}: {n_ep} TRAIN episodes, temporal stride {stride}")
    bags, prop = [], []
    with torch.no_grad():
        for o, a, fr in eps:
            T = len(o); idx = np.arange(0, T, stride)
            obs = {"proprio": norm.norm_obs(torch.from_numpy(o[idx])).float()[None].to(dev)}
            for h in cams: obs[h] = torch.from_numpy(fr[h][idx]).float().div(255.0)[None].to(dev)
            z = m.encode_state(obs)[0]                      # (t, n_state, d)
            bags.append(z.reshape(z.shape[0], -1).cpu()); prop.append(torch.from_numpy(o[idx]).float())
    Z = torch.cat(bags); Ztok = Z.reshape(-1, m.d)          # full-bag (N, n_state*d) and per-token (N*n_state, d)
    Pr = torch.cat(prop)
    print(f"[pr] samples: full-bag N={Z.shape[0]}  per-token N={Ztok.shape[0]}  bag_dim={Z.shape[1]}  d={m.d}")
    print(f"[pr] LATENT per-token PR = {pr(Ztok):.1f} / {m.d}")
    print(f"[pr] LATENT full-bag  PR = {pr(Z):.1f} / {Z.shape[1]}")
    print(f"[pr] DATA   proprio   PR = {pr(Pr):.1f} / {Pr.shape[1]}  (raw obs intrinsic dim)")
    # pixel-PCA proxy for the scene's visual intrinsic dim (downsample frames hard, one cam)
    h0 = list(cams)[0]; px = []
    for _, _, fr in eps:
        f = torch.from_numpy(fr[h0][::stride]).float().div(255.0)
        f = torch.nn.functional.interpolate(f.permute(0,3,1,2), size=(12,16), mode="area").reshape(f.shape[0],-1)
        px.append(f)
    PX = torch.cat(px)
    print(f"[pr] DATA   pixels(12x16) PR = {pr(PX):.1f} / {PX.shape[1]}  (visual intrinsic dim proxy)")
    print("[pr] DONE")

if __name__ == "__main__":
    main()
