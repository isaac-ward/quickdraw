"""Card products per clip: (1) 1-step-ahead image pred video GT|pred; (2) ae_floor encode->decode recon
filmstrip (GT top / recon bottom); (3) open-loop proprio 3D plot + per-axis plot, cut at STEPS. Runs in-container."""
import os, json, numpy as np, torch, imageio.v2 as imageio
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (enables projection='3d')
from omegaconf import OmegaConf
from quickdraw.training.setup import build_model, load_checkpoint, normalizer, image_head_cams, image_head_sizes
from quickdraw.data.dataset import load_split_episodes_mm, set_subsample, set_obs_keep
from quickdraw.evaluation.routines import _pos_idx
from quickdraw.environments.registry import make_env

RUN, CKPT, OUT = os.environ["RUN"], os.environ["CKPT"], os.environ["OUT"]
EPIDX = [int(x) for x in os.environ["EPIDX"].split(",")]
EPNAMES = os.environ["EPNAMES"].split(",")
STEPS = int(os.environ.get("STEPS", "200"))
cfg = OmegaConf.create(json.load(open(f"{RUN}/logs/config.json")))
set_subsample(int(cfg.data.subsample)); set_obs_keep(cfg.data.get("obs_keep", None))
dev = "cuda"
model = build_model(cfg).to(dev); load_checkpoint(model, CKPT); model.eval()
norm = normalizer(cfg); P = int(cfg.data.P)
img_heads = [n for n, _ in model.layout if n != "proprio"]; H = img_heads[0]
cams = image_head_cams(cfg) or cfg.data.get("cam", "fpv"); sizes = image_head_sizes(cfg) or 128
env = make_env(cfg.environments.get("name", "recorded"), cfg.environments, 1, "cpu")
pos, _ = _pos_idx(cfg, env=env)
eps = load_split_episodes_mm(cfg.data.root, "val", img_size=sizes, cam=cams, repo_id=cfg.data.repo_id)
eps = sorted(eps, key=lambda e: -len(e[0]))
os.makedirs(OUT, exist_ok=True); CH = 8
for idx, name in zip(EPIDX, EPNAMES):
    o, a, fr = eps[idx]; T = len(o)
    ot = norm.norm_obs(torch.from_numpy(o).float()).to(dev)
    at = norm.norm_act(torch.from_numpy(a).float()).to(dev)
    it = torch.from_numpy(fr[H]).float().div(255.0).to(dev)
    # (1) 1-step image pred video
    gs = list(range(P - 1, T - 1)); preds = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for s in range(0, len(gs), CH):
            gb = gs[s:s + CH]
            ctx = {"proprio": torch.stack([ot[g - P + 1:g + 1] for g in gb]), H: torch.stack([it[g - P + 1:g + 1] for g in gb])}
            acts = torch.stack([at[g - P + 1:g - P + 1 + P] for g in gb])
            preds.append(model.imagine_eval(ctx, acts, 1, heads=[H], norm=norm)[H][:, 0].float())
    pr = torch.cat(preds).clamp(0, 1).cpu().numpy(); gt1 = it[P:T].cpu().numpy()
    imageio.mimwrite(f"{OUT}/{name}_1step.mp4", (np.concatenate([gt1, pr], axis=2) * 255).clip(0, 255).astype(np.uint8),
                     fps=15, codec="libx264", quality=8)
    # (2) ae_floor: encode->decode the whole clip
    obs_full = {"proprio": ot[None, :T], H: it[None, :T]}
    anchor = model.rel_anchor(obs_full) if model._rel_on() else None
    rec_acc = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for s in range(0, T, 64):
            sub = {k: v[:, s:s + 64] for k, v in obs_full.items()}
            rec_acc.append(model.to_obs(model.encode_state(sub, anchor), heads=[H], anchor=anchor)[H].float())
    recon = torch.cat(rec_acc, dim=1)[0].clamp(0, 1).cpu().numpy()
    jd = np.linspace(0, T - 1, 8).astype(int)
    strip = np.concatenate([np.concatenate([it[j].cpu().numpy(), recon[j]], axis=0) for j in jd], axis=1)  # GT top / recon bottom
    imageio.imwrite(f"{OUT}/{name}_aefloor.png", (strip * 255).clip(0, 255).astype(np.uint8))
    # (3) open-loop proprio rollout (proprio head), cut at STEPS -> 3D + axes
    Hp = min(STEPS, T - P)
    ctx0 = {"proprio": ot[None, :P], H: it[None, :P]}
    acts0 = at[None, :P - 1 + Hp]
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outp = model.imagine_eval(ctx0, acts0, Hp, heads=["proprio"], norm=norm)
    ph = norm.denorm_obs(outp["proprio"][0].float()).cpu().numpy()[:, pos]
    pt = o[P:P + Hp][:, pos]
    fig = plt.figure(figsize=(6, 6)); ax = fig.add_subplot(111, projection="3d")
    ax.plot(pt[:, 0], pt[:, 1], pt[:, 2], color="k", lw=2, label="GT")
    ax.plot(ph[:, 0], ph[:, 1], ph[:, 2], color="tab:red", lw=2, label="pred")
    ax.legend(); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    fig.savefig(f"{OUT}/{name}_proprio3d.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    for i, lab in enumerate("xyz"):
        axs[i].plot(pt[:, i], color="k", lw=2, label="GT"); axs[i].plot(ph[:, i], color="tab:red", lw=2, label="pred")
        axs[i].set_ylabel(lab); axs[i].grid(alpha=0.3)
    axs[0].legend(loc="best"); axs[-1].set_xlabel("step")
    fig.savefig(f"{OUT}/{name}_proprio_axes.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  {name}: 1step {len(pr)}f | aefloor | proprio {Hp} steps", flush=True)
print("DONE", flush=True)
