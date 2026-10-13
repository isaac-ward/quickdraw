"""Oracle cube-state ablation (2026-09-25).

Question: if the model KNEW the true cube position/size every step, would the cubes stop
drifting? We feed GROUND-TRUTH cube-state back into the rollout each step (teacher-force the
cube channel only) and compare decoded video against the normal (free-running) rollout.

  normal  : pure open-loop -- the model predicts EVERYTHING including cube-state.
  oracle  : same rollout, but each step the proprio token's CUBE dims are overwritten with GT
            (decode proprio -> splice GT cube dims -> re-encode -> feed back AND decode from it).

If oracle cubes stay put  -> the problem is PREDICTING cube-state (decode can use it).
If oracle cubes still drift-> the problem is the DECODE BINDING (position never reaches pixels).

Out: logs/oracle_cube/<arm>/ep<i>_<cam>.mp4  (GT | normal | oracle) + ep<i>_<cam>.png filmstrip.
Run: docker compose exec -T app uv run --no-sync python -m logs.oneoffs.oracle_cube <ckpt> <arm>
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from quickdraw.data.dataset import DataConfig, load_split_episodes_mm
from quickdraw.models.multimodal import _ln
from quickdraw.models.transformer import pad_block_mask
from quickdraw.training.setup import (build_model, image_head_cams, image_head_sizes,
                                      load_checkpoint, normalizer, resolve_data_root)

CUBE_N = 32          # observation.objects.cube dims (appended last in obs_fields)
N_EP = 4
HORIZON = 64


def rollout(m, ctx, acts, H, gt_future_norm=None, mode="normal"):
    """Eager open-loop rollout (mirrors _rollout_from, p_tf=0, uncached).
    mode: 'normal' = pure open-loop (no splice).
          'rt'     = decode proprio -> re-encode PREDICTED proprio (roundtrip CONTROL: same mechanism as
                     the oracle but injects nothing -- isolates the decode->encode codec cost).
          'cube'   = splice GT CUBE dims into the re-encoded proprio (the oracle under test).
          'full'   = re-encode the FULL GT proprio (strongest oracle: true arm pose AND cubes)."""
    W = m.window
    po, pn = m._proprio_off, m._proprio_n
    obs_dim = m.modalities["proprio"].dim
    cube0 = obs_dim - CUBE_N
    pmod = m.modalities["proprio"]
    bag_buf = list(m.encode_state(ctx).unbind(dim=1))     # P bags of (B,n_state,d)
    preds = []
    for h in range(H):
        Lh = len(bag_buf)
        real = min(Lh, W)
        pad = W - real
        s_win = torch.stack(bag_buf[-real:], dim=1)
        a_win = acts[:, Lh - real:Lh]
        if pad:
            s_win = F.pad(s_win, (0, 0, 0, 0, pad, 0))
            a_win = F.pad(a_win, (0, 0, pad, 0))
        bm = pad_block_mask(W, pad, s_win.device)
        s_pred = m._rollout_step(s_win, a_win, bm, bag_buf[-1])       # (B,n_state,d)
        if mode != "normal":
            ptok = s_pred[..., po:po + pn, :]                        # (B,1,d)
            p_pred = pmod.decode(ptok, commit=True)                  # (B,obs_dim) normalized
            p_orc = p_pred.clone()
            if mode == "cube":
                p_orc[..., cube0:] = gt_future_norm[:, h, cube0:]    # force GT cube dims only
            elif mode == "full":
                p_orc = gt_future_norm[:, h].clone()                 # force ALL GT proprio dims
            # mode == "rt": p_orc stays = predicted (roundtrip control, no injection)
            newtok = pmod.encode(p_orc)                              # (B,1,d)
            if m.latent_norm:
                newtok = _ln(newtok)
            s_pred = s_pred.clone()
            s_pred[..., po:po + pn, :] = newtok
        preds.append(s_pred)
        bag_buf.append(s_pred)                                       # carry_transform = identity (latent)
    return torch.stack(preds, dim=1)                                 # (B,H,n_state,d)


def to01(x):
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def label(img, text):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, len(text) * 6 + 4, 12], fill=(0, 0, 0))
    d.text((2, 1), text, fill=(255, 255, 255))
    return np.asarray(im)


def main():
    ckpt, arm = sys.argv[1], sys.argv[2]
    run_dir = os.path.dirname(os.path.dirname(ckpt))
    rcfg = OmegaConf.load(os.path.join(run_dir, "checkpoints", "config.resolved.yaml"))
    OmegaConf.set_struct(rcfg, False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(rcfg).to(dev).eval()
    load_checkpoint(model, ckpt)
    m = getattr(model, "_orig_mod", model)
    norm = normalizer(rcfg)
    P = rcfg.data.P
    img_heads = [n for n, _ in m.layout if n != "proprio"]
    img_size = next((mod.ae.cfg.img_size for mod in m.modalities.values() if hasattr(mod, "ae")), 128)
    eps = load_split_episodes_mm(resolve_data_root(rcfg), "val",
                                 dcfg=DataConfig.from_cfg(rcfg),
                                 img_size=image_head_sizes(rcfg) or img_size,
                                 cam=image_head_cams(rcfg) or rcfg.data.get("cam", "fpv"),
                                 repo_id=rcfg.data.get("repo_id", "torus"))[:N_EP]
    H = min(HORIZON, min(len(o) for o, _, _ in eps) - P - 1)
    print(f"[oracle] arm={arm} eps={len(eps)} P={P} H={H} heads={img_heads} obs_dim={m.modalities['proprio'].dim}", flush=True)

    ctx = {"proprio": torch.stack([norm.norm_obs(torch.from_numpy(o[:P])) for o, _, _ in eps]).float().to(dev)}
    for hd in img_heads:
        ctx[hd] = torch.stack([torch.from_numpy(fr[hd][:P]) for _, _, fr in eps]).float().div(255.0).to(dev)
    acts = torch.stack([norm.norm_act(torch.from_numpy(a[:P + H - 1])) for _, a, _ in eps]).float().to(dev)
    gt_future = torch.stack([norm.norm_obs(torch.from_numpy(o[P:P + H])) for o, _, _ in eps]).float().to(dev)
    gt_img = {hd: torch.stack([torch.from_numpy(fr[hd][P:P + H]) for _, _, fr in eps]).float().div(255.0)
              for hd in img_heads}

    ac = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available())
    MODES = ["floor", "normal", "rt", "cube", "full"]
    imgs = {}
    with torch.no_grad(), ac:
        # AE FLOOR: encode the GT frames+proprio and decode straight back -- NO dynamics, NO rollout.
        # This is the codec ceiling: the sharpest any rollout could ever render (Ceiling A).
        gt_obs = {"proprio": gt_future}
        for hd in img_heads:
            gt_obs[hd] = gt_img[hd].to(dev)
        imgs["floor"] = m.to_obs(m.encode_state(gt_obs), heads=img_heads, commit=True)
        for md in ["normal", "rt", "cube", "full"]:
            bag = rollout(m, ctx, acts, H, gt_future_norm=gt_future, mode=md)
            imgs[md] = m.to_obs(bag, heads=img_heads, commit=True)
    img_n, img_o = imgs["normal"], imgs["cube"]

    out = f"logs/oracle_cube/{arm}"
    os.makedirs(out, exist_ok=True)
    try:
        import imageio.v2 as imageio
        have_mp4 = True
    except Exception:
        have_mp4 = False

    panels = [("GT", None), ("AE-floor", "floor"), ("normal", "normal"),
              ("cube-oracle", "cube"), ("full-oracle", "full")]
    for i in range(len(eps)):
        for hd in img_heads:
            g = gt_img[hd][i].cpu().numpy()                         # (H,ih,iw,3)
            frames = []
            for t in range(H):
                cells = []
                for name, md in panels:
                    arr = g[t] if md is None else imgs[md][hd][i].float().cpu().numpy()[t]
                    cells.append(label(to01(arr), name))
                frames.append(np.concatenate(cells, axis=1))
            frames = np.stack(frames)                               # (H, ih, 4*iw, 3)
            base = f"{out}/ep{i}_{hd}"
            if have_mp4:
                imageio.mimwrite(f"{base}.mp4", list(frames), fps=8, quality=8)
            # filmstrip PNG: every ~H/8 frame down the columns
            cols = list(range(0, H, max(1, H // 8)))[:8]
            strip = np.concatenate([frames[c] for c in cols], axis=0)
            Image.fromarray(strip).save(f"{base}.png")
            print(f"[oracle] wrote {base}.png{' + .mp4' if have_mp4 else ''}", flush=True)

    # numeric: mean pixel L1 to GT per head, per quarter-horizon, for every mode.
    # rt is the CONTROL: cube/full only mean something RELATIVE to rt (same splice mechanism).
    print("\n[oracle] pixel L1 to GT (lower=closer). rt=roundtrip control; compare cube/full vs rt.", flush=True)
    print(f"   {'head':22s} {'h':>3s} | " + " ".join(f"{md:>7s}" for md in MODES), flush=True)
    for hd in img_heads:
        g = gt_img[hd].cpu().numpy()
        arr = {md: imgs[md][hd].float().cpu().numpy() for md in MODES}
        for q in [H // 4, H // 2, 3 * H // 4, H - 1]:
            vals = {md: np.abs(arr[md][:, q] - g[:, q]).mean() for md in MODES}
            print(f"   {hd:22s} {q:3d} | " + " ".join(f"{vals[md]:7.4f}" for md in MODES), flush=True)


if __name__ == "__main__":
    main()
