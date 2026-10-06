"""Long-horizon OPEN-LOOP GT | Prediction videos + gifs (IND val). From the first P GT frames the model free-runs
for H model-steps (driven only by true actions); we render Ground truth (left) | Prediction (right) over the whole
horizon -- the showcase that the prediction tracks the truth for minutes of open-loop rollout. Same 2-panel setup /
font as the gallery. mp4 plays at 30 fps (= ~10x robot time); gif downscaled to fit <= 40 MB.
Out: logs/ood/rollout_error/longhorizon_*.{mp4,gif}
Run: docker compose exec -T -e CUDA_VISIBLE_DEVICES=1 app uv run --no-sync python logs/oneoffs/long_horizon_video.py <ckpt>
"""
import os, sys, dataclasses
import numpy as np, torch, cv2
from omegaconf import OmegaConf
from quickdraw.data.dataset import DataConfig, load_split_episodes_mm
from quickdraw.logging.viz import save_mp4, save_gif, label_panels
from quickdraw.training.setup import (build_model, env_cfg, image_head_cams, image_head_sizes,
                                      load_checkpoint, normalizer, resolve_data_root)

HEAD, H, SCALE, OUT = "cam_scene", 900, 4, "logs/ood/rollout_error"
TITLES = ["Ground truth", "Prediction"]
GIF_LADDER = [(0.5, 20), (0.45, 20), (0.4, 15), (0.35, 15), (0.3, 12)]


def main():
    ck = sys.argv[1]; dev = "cuda"
    os.makedirs(OUT, exist_ok=True)
    rc = OmegaConf.load(os.path.join(os.path.dirname(os.path.dirname(ck)), "checkpoints", "config.resolved.yaml"))
    OmegaConf.set_struct(rc, False)
    m = build_model(rc).to(dev).eval(); m = getattr(m, "_orig_mod", m); load_checkpoint(m, ck); m.stochastic_eval = False
    norm = normalizer(rc); P = rc.data.P; s = int(rc.data.subsample)
    native = 1.0 / float(env_cfg(rc).dt); model_hz = native / s
    eps = load_split_episodes_mm(resolve_data_root(rc), "val", dcfg=DataConfig.from_cfg(rc),
                                 img_size=image_head_sizes(rc) or 128,
                                 cam=image_head_cams(rc) or rc.data.get("cam", "fpv"),
                                 repo_id=rc.data.get("repo_id", "torus"))
    fr_keys = list(eps[0][2])

    for idx, (o, a, fr) in enumerate(eps):
        start = P; Hc = min(H, len(o) - start - P - 1)
        ctx = {"proprio": norm.norm_obs(torch.from_numpy(o[start:start + P])).float()[None].to(dev)}
        for h in fr_keys:
            ctx[h] = torch.from_numpy(fr[h][start:start + P]).float().div(255.0)[None].to(dev)
        acts = norm.norm_act(torch.from_numpy(a[start:start + P + Hc - 1])).float()[None].to(dev)
        with torch.no_grad():
            pred = m.imagine_eval(ctx, acts, Hc, heads=[HEAD], decode_chunk=64, norm=norm)[HEAD][0]  # (Hc,h,w,3)
        gt = fr[HEAD][start + P:start + P + Hc].astype(np.uint8)
        pu = (pred.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        frames = [label_panels([gt[t], pu[t]], TITLES, scale=SCALE) for t in range(Hc)]
        name = f"longhorizon_{idx:02d}"
        save_mp4(os.path.join(OUT, f"{name}.mp4"), np.stack(frames), fps=30)        # 30 fps ~ 10x robot time
        for ds, fps in GIF_LADDER:
            step = max(1, round(30 / fps)); eff = 30 / step
            gf = [cv2.resize(f, (int(f.shape[1] * ds), int(f.shape[0] * ds)), interpolation=cv2.INTER_AREA) for f in frames[::step]]
            gpath = os.path.join(OUT, f"{name}.gif"); save_gif(gpath, gf, eff)
            sz = os.path.getsize(gpath)
            if sz <= 40e6:
                break
        print(f"[{name}] {Hc} steps = {Hc/model_hz:.0f}s robot time | mp4 30fps | gif {ds}x {eff:.0f}fps {sz/1e6:.1f}MB", flush=True)


if __name__ == "__main__":
    main()
