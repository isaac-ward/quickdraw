"""Long-horizon IND rollout videos + gifs. TWO separate files per episode, each 2-panel [Ground truth | Prediction]:
  *_openloop.{mp4,gif} : from the first P GT frames the model free-runs H steps (no re-grounding), true actions only.
  *_1step.{mp4,gif}    : at every step the model predicts ONE step ahead from the TRUE context (re-grounded each step).
Same setup / font as the gallery. mp4 30 fps (~10x robot time); gif resolution-first, no dither, <=40 MB.
Out: logs/ood/rollout_error/longhorizon_*_{openloop,1step}.{mp4,gif}
Run: docker compose exec -T -e CUDA_VISIBLE_DEVICES=1 app uv run --no-sync python logs/oneoffs/long_horizon_video.py <ckpt>
"""
import os, sys
import numpy as np, torch, cv2
from omegaconf import OmegaConf
from quickdraw.data.dataset import DataConfig, load_split_episodes_mm
from quickdraw.logging.viz import save_mp4, save_gif, label_panels
from quickdraw.training.setup import (build_model, env_cfg, image_head_cams, image_head_sizes,
                                      load_checkpoint, normalizer, resolve_data_root)

HEAD, H, SCALE, OUT = "cam_scene", 900, 4, "logs/ood/rollout_error"
TITLES = ["Ground truth", "Prediction"]
GIF_LADDER = [(0.75, 10), (0.7, 10), (0.6, 10), (0.55, 8), (0.5, 8)]


@torch.no_grad()
def onestep_batch(m, norm, o, a, fr, fr_keys, P, ts, dev, chunk=64):
    """Batched 1-step (re-grounded) predictions: for each target frame t, predict it from the TRUE context [t-P:t]."""
    out = []
    for i in range(0, len(ts), chunk):
        ch = ts[i:i + chunk]
        ctx = {"proprio": torch.stack([norm.norm_obs(torch.from_numpy(o[t - P:t])) for t in ch]).float().to(dev)}
        for h in fr_keys:
            ctx[h] = torch.stack([torch.from_numpy(fr[h][t - P:t]) for t in ch]).float().div(255.0).to(dev)
        acts = torch.stack([norm.norm_act(torch.from_numpy(a[t - P:t])) for t in ch]).float().to(dev)
        p = m.imagine_eval(ctx, acts, 1, heads=[HEAD], norm=norm)[HEAD][:, 0]       # (B,h,w,3)
        out.append((p.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
    return np.concatenate(out, 0)


def main():
    ck = sys.argv[1]; dev = "cuda"
    os.makedirs(OUT, exist_ok=True)
    rc = OmegaConf.load(os.path.join(os.path.dirname(os.path.dirname(ck)), "checkpoints", "config.resolved.yaml"))
    OmegaConf.set_struct(rc, False)
    m = build_model(rc).to(dev).eval(); m = getattr(m, "_orig_mod", m); load_checkpoint(m, ck); m.stochastic_eval = False
    norm = normalizer(rc); P = rc.data.P; s = int(rc.data.subsample)
    model_hz = (1.0 / float(env_cfg(rc).dt)) / s
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
            ol = m.imagine_eval(ctx, acts, Hc, heads=[HEAD], decode_chunk=64, norm=norm)[HEAD][0]   # open-loop (Hc,h,w,3)
        ol = (ol.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        os1 = onestep_batch(m, norm, o, a, fr, fr_keys, P, list(range(start + P, start + P + Hc)), dev)  # 1-step
        gt = fr[HEAD][start + P:start + P + Hc].astype(np.uint8)
        for mode, pred in [("openloop", ol), ("1step", os1)]:
            frames = [label_panels([gt[t], pred[t]], TITLES, scale=SCALE) for t in range(Hc)]
            name = f"longhorizon_{idx:02d}_{mode}"
            save_mp4(os.path.join(OUT, f"{name}.mp4"), np.stack(frames), fps=30)
            for ds, fps in GIF_LADDER:
                step = max(1, round(30 / fps)); eff = 30 / step
                gf = [cv2.resize(f, (int(f.shape[1] * ds), int(f.shape[0] * ds)), interpolation=cv2.INTER_AREA) for f in frames[::step]]
                gpath = os.path.join(OUT, f"{name}.gif"); save_gif(gpath, gf, eff, dither=False)
                sz = os.path.getsize(gpath)
                if sz <= 40e6:
                    break
            print(f"[{name}] {Hc} steps = {Hc/model_hz:.0f}s robot | mp4 30fps | gif {ds}x {eff:.0f}fps {sz/1e6:.1f}MB", flush=True)


if __name__ == "__main__":
    main()
