"""1-step-ahead pred|GT videos for the model card. For each chosen val episode (sorted longest, to match
eval.longest), every frame t is predicted from the TRUE context ending at t-1 (ground on [t-P..t-1], roll 1
step, take the prediction). Output GT|pred side-by-side mp4 per clip. Runs in-container (torch 2.6)."""
import os, json, numpy as np, torch, imageio.v2 as imageio
from omegaconf import OmegaConf
from quickdraw.training.setup import build_model, load_checkpoint, normalizer, image_head_cams, image_head_sizes
from quickdraw.data.dataset import load_split_episodes_mm, set_subsample, set_obs_keep

RUN = os.environ["RUN"]; CKPT = os.environ["CKPT"]; OUT = os.environ["OUT"]
EPIDX = [int(x) for x in os.environ["EPIDX"].split(",")]
EPNAMES = os.environ["EPNAMES"].split(",")
CH, FPS = 8, 15
cfg = OmegaConf.create(json.load(open(f"{RUN}/logs/config.json")))
set_subsample(int(cfg.data.subsample))
set_obs_keep(cfg.data.get("obs_keep", None))   # subset proprio to ego-13, as training/eval do
dev = "cuda"
model = build_model(cfg).to(dev); load_checkpoint(model, CKPT); model.eval()
norm = normalizer(cfg); P = int(cfg.data.P)
img_heads = [n for n, _ in model.layout if n != "proprio"]
H = img_heads[0]
cams = image_head_cams(cfg) or cfg.data.get("cam", "fpv")
sizes = image_head_sizes(cfg) or 128
eps = load_split_episodes_mm(cfg.data.root, "val", img_size=sizes, cam=cams, repo_id=cfg.data.repo_id)
eps = sorted(eps, key=lambda e: -len(e[0]))
print(f"loaded {len(eps)} val eps; head={H}; picking idx {EPIDX} -> {EPNAMES}", flush=True)
os.makedirs(OUT, exist_ok=True)
for idx, name in zip(EPIDX, EPNAMES):
    o, a, fr = eps[idx]; T = len(o)
    ot = norm.norm_obs(torch.from_numpy(o).float()).to(dev)
    at = norm.norm_act(torch.from_numpy(a).float()).to(dev)
    it = torch.from_numpy(fr[H]).float().div(255.0).to(dev)     # (T,S,S,3)
    gs = list(range(P - 1, T - 1)); preds = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for s in range(0, len(gs), CH):
            gb = gs[s:s + CH]
            ctx = {"proprio": torch.stack([ot[g - P + 1:g + 1] for g in gb]),
                   H:         torch.stack([it[g - P + 1:g + 1] for g in gb])}
            acts = torch.stack([at[g - P + 1:g - P + 1 + P] for g in gb])   # P actions -> 1-step
            out = model.imagine_eval(ctx, acts, 1, heads=[H], norm=norm)
            preds.append(out[H][:, 0].float())
    pr = torch.cat(preds).clamp(0, 1).cpu().numpy()             # frames [P .. T-1]
    gt = it[P:T].cpu().numpy()
    vid = (np.concatenate([gt, pr], axis=2).clip(0, 1) * 255).astype(np.uint8)   # GT | pred
    imageio.mimwrite(os.path.join(OUT, f"{name}.mp4"), vid, fps=FPS, codec="libx264", quality=8)
    print(f"  {name}: idx{idx} {len(pr)} frames (GT|pred) -> {OUT}/{name}.mp4", flush=True)
print("DONE", flush=True)
