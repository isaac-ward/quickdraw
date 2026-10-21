"""IN-DISTRIBUTION open-loop error-over-time: how bounded is the error after EXTREMELY long OPEN-LOOP rollouts?
From a single GT context (first P frames) the world model free-runs (no re-grounding) for up to MAXH model steps,
driven only by the true actions. Per-horizon L1 / L2(RMSE) / LPIPS vs the true frames (reuses openloop.image_curves).
Several start points across the val episodes -> mean +/- std band. Also reports prediction speed (Hz) and whether
the model runs faster than real time. Separate from the gallery (this is open-loop; the gallery is 1-step regrounded).
Out: logs/ood/rollout_error/error_over_time.png (+ timing printed, saved to timing.json)
Run: docker compose exec -T -e CUDA_VISIBLE_DEVICES=1 app uv run --no-sync python logs/oneoffs/rollout_error_over_time.py <ckpt>
"""
import os, sys, json, time, dataclasses
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from quickdraw.data.dataset import DataConfig, load_split_episodes_mm
from quickdraw.evaluation.openloop import image_curves
from quickdraw.training.setup import (build_model, env_cfg, image_head_cams, image_head_sizes,
                                      load_checkpoint, normalizer, resolve_data_root)

HEAD, MAXH, STARTS_PER_EP, OUT = "cam_scene", 1000, 8, "logs/ood/rollout_error"   # 8 starts x 2 val eps = 16 rollouts


@torch.no_grad()
def rollout(m, norm, o, a, fr, P, start, H, dev, heads):
    ctx = {"proprio": norm.norm_obs(torch.from_numpy(o[start:start + P])).float()[None].to(dev)}
    for h in fr:
        ctx[h] = torch.from_numpy(fr[h][start:start + P]).float().div(255.0)[None].to(dev)
    acts = norm.norm_act(torch.from_numpy(a[start:start + P + H - 1])).float()[None].to(dev)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = m.imagine_eval(ctx, acts, H, heads=heads, decode_chunk=64, norm=norm)
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    return out, dt


def main():
    ck = sys.argv[1]; dev = "cuda"
    os.makedirs(OUT, exist_ok=True)
    rc = OmegaConf.load(os.path.join(os.path.dirname(os.path.dirname(ck)), "checkpoints", "config.resolved.yaml"))
    OmegaConf.set_struct(rc, False)
    m = build_model(rc).to(dev).eval(); m = getattr(m, "_orig_mod", m); load_checkpoint(m, ck); m.stochastic_eval = False
    norm = normalizer(rc); P = rc.data.P; s = int(rc.data.subsample)
    native = 1.0 / float(env_cfg(rc).dt); model_hz = native / s           # model step rate (block-stack: 30/10 = 3 Hz)
    eps = load_split_episodes_mm(resolve_data_root(rc), "val", dcfg=DataConfig.from_cfg(rc),
                                 img_size=image_head_sizes(rc) or 128,
                                 cam=image_head_cams(rc) or rc.data.get("cam", "fpv"),
                                 repo_id=rc.data.get("repo_id", "torus"))
    fr_keys = [k for k in eps[0][2]]

    L1, L2, LP, PS, times, lat_times = [], [], [], [], [], []
    for o, a, fr in eps:
        Lep = len(o)
        for st in np.linspace(P, max(P, Lep - MAXH - P - 1), STARTS_PER_EP).astype(int):
            H = min(MAXH, Lep - st - P - 1)
            if H < 50:
                continue
            out, dt = rollout(m, norm, o, a, {k: fr[k] for k in fr_keys}, P, int(st), H, dev, [HEAD])
            pred = out[HEAD]
            true = torch.from_numpy(fr[HEAD][st + P:st + P + H]).float().div(255.0)[None].to(dev)
            c = image_curves(pred, true)
            L1.append(c["l1"]); L2.append(np.sqrt(c["mse"])); LP.append(c.get("lpips", np.full(H, np.nan))); PS.append(c["psnr"])
            times.append(dt / H)
            _, dl = rollout(m, norm, o, a, {k: fr[k] for k in fr_keys}, P, int(st), H, dev, ["proprio"])  # latent/dynamics-only
            lat_times.append(dl / H)
            print(f"  start {int(st):4d} H {H} | {dt/H*1e3:.1f} ms/step (img) {dl/H*1e3:.1f} ms/step (dynamics) | end L1 {c['l1'][-1]:.3f}", flush=True)

    hm = min(len(x) for x in L1)
    L1, L2, LP, PS = (np.stack([x[:hm] for x in a]) for a in (L1, L2, LP, PS))
    secs = np.arange(hm) / model_hz
    img_ms, dyn_ms = np.mean(times) * 1e3, np.mean(lat_times) * 1e3
    img_hz, dyn_hz = 1e3 / img_ms, 1e3 / dyn_ms

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axs = plt.subplots(4, 1, figsize=(7.5, 11), sharex=True)
    for ax, data, name in [(axs[0], L1, "L1"), (axs[1], L2, "L2 (RMSE)"), (axs[2], LP, "LPIPS"), (axs[3], PS, "PSNR")]:
        mu, sd = np.nanmean(data, 0), np.nanstd(data, 0)
        ax.plot(secs, mu, color="#1f77b4", lw=2); ax.fill_between(secs, mu - sd, mu + sd, color="#1f77b4", alpha=0.22)
        ax.set_title(name, fontsize=13); ax.grid(alpha=0.3)
        ax.set_ylabel("dB" if name == "PSNR" else "")
        if name != "PSNR":
            ax.set_ylim(bottom=0)
    axs[-1].set_xlabel("open-loop rollout time (seconds of robot time)")
    fig.suptitle("Trajectory averaged open-loop rollout error", fontsize=15, y=0.997)
    fig.text(0.5, 0.965, f"{L1.shape[0]} rollouts  ·  up to {hm/model_hz:.0f}s ({hm} open-loop steps)", ha="center", fontsize=10, color="#555")
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(os.path.join(OUT, "error_over_time.png"), dpi=130); plt.close(fig)

    rt = {"model_rate_hz": model_hz, "img_rollout_ms_per_step": img_ms, "img_rollout_hz": img_hz,
          "dynamics_only_ms_per_step": dyn_ms, "dynamics_only_hz": dyn_hz,
          "realtime_factor_img": img_hz / model_hz, "realtime_factor_dynamics": dyn_hz / model_hz,
          "horizon_steps": int(hm), "horizon_seconds": hm / model_hz, "n_starts": int(L1.shape[0])}
    json.dump(rt, open(os.path.join(OUT, "timing.json"), "w"), indent=2)
    print("\n=== SPEED / REAL-TIME ===", flush=True)
    print(f"model operates at {model_hz:.1f} Hz (1 step = {s} native frames = {1/model_hz:.3f}s robot time)", flush=True)
    print(f"full image rollout : {img_ms:.1f} ms/step -> {img_hz:.0f} Hz  = {img_hz/model_hz:.0f}x real time", flush=True)
    print(f"dynamics only      : {dyn_ms:.1f} ms/step -> {dyn_hz:.0f} Hz  = {dyn_hz/model_hz:.0f}x real time", flush=True)
    print(f"-> {'REAL-TIME CAPABLE' if img_hz > model_hz else 'NOT real-time'} (image decode incl.); dynamics-only has far more headroom", flush=True)


if __name__ == "__main__":
    main()
