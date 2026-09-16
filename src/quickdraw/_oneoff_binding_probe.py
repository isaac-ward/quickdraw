"""Do the image tokens BIND to objects, or is identity smeared across the whole bag?

    python -m quickdraw._oneoff_binding_probe <run_dir> [--ckpt best.ckpt] [--cam cam_scene]

WHY THIS EXISTS. Record 8.25: once the codec floor is subtracted, `OL - floor` is pinned at ~0.063
LPIPS across THIRTEEN arms -- weight decay, capacity in both directions, window, loss reweighting,
p_tf_dynamics, df_scale, detach_every, action_aggregate, subsampling. Every apparent open-loop win
was the CODEC improving. The dynamics contribution has never moved. That is the shape of a
structural limit, and the standing suspicion is that nothing in this architecture binds one token
to one object: the bag is undifferentiated, so "which block is which" must be re-derived diffusely
at every rollout step.

Before building an unsupervised binding loss, MEASURE whether there is any binding to sharpen.
This is checkpoint-only -- no training, runs alongside live arms.

HOW. Ablate one image token at a time (replace it with its own temporal mean, which is in
distribution, unlike zeroing), decode, and look at where the image changed. That gives each token a
spatial FOOTPRINT. Three numbers decide it:

  concentration  fraction of the footprint's energy in its top 5% of pixels. A token that owns a
                 region concentrates; one that contributes diffusely does not. Uniform = 0.05.
  overlap        mean pairwise IoU of the tokens' top-10% masks. High overlap = every token touches
                 the same pixels = no spatial division of labour.
  centroid drift std over time of the footprint's centroid, in pixels. THIS IS THE ONE THAT
                 MATTERS, and it distinguishes the two ways a token can be localised:
                     drift ~ 0   -> POSITION-bound. Each token owns a fixed tile of the image.
                                   Localised but useless for identity: when a block moves between
                                   tiles, its identity has to be handed from one token to another,
                                   and that hand-off is exactly what could compound into a colour
                                   swap over 30 steps.
                     drift >> 0  -> CONTENT-bound. Footprints follow moving stuff, i.e. there IS
                                   emergent object binding to sharpen with a loss.

So "localised" alone is NOT evidence of binding, and reading concentration without drift would give
the comforting answer. Compare drift against how far the scene's own moving content travels over
the same frames (`content motion` below).

AND THAT IS STILL NOT ENOUGH -- this was got wrong once. If the DECODER is simply content dependent,
ablating ANY token produces a footprint near wherever the action currently is, so every token drifts
with content while owning nothing. Two further checks are required, and they are what actually
decide it:

  inter-token spread  std ACROSS tokens of footprint centroid at a FIXED instant. If small, every
                      token points at the same place and "content-bound" is an artifact of the
                      decoder, not binding. Measured against the image diagonal.
  drift decomposition split the motion into COMMON-MODE (all tokens moving together) and INDIVIDUAL
                      (each token relative to that common mode). Only individual motion is
                      per-object tracking; common-mode is the whole bag sliding toward the action.

On bs_stride10 ep9 the first read said CONTENT-BOUND on concentration 0.498 and drift 0.77x content.
The checks above then gave inter-token spread 8.33 px on a 160 px diagonal (0.05) and common-mode
6.20 px against individual 4.89 px -- i.e. all 32 footprints clustered in one region, drifting
together. NOT bound. Record 8.26.
"""
from __future__ import annotations
import glob, json, sys
import numpy as np, pandas as pd, torch
from omegaconf import OmegaConf

DEV = "cuda:0"


def _load(run_dir: str, ckpt: str | None):
    from .training.setup import build_model, load_checkpoint
    cfg = OmegaConf.load(f"{run_dir}/checkpoints/config.resolved.yaml")
    torch.manual_seed(0)
    m = build_model(cfg)
    load_checkpoint(m, f"{run_dir}/checkpoints/{ckpt or 'best.ckpt'}")
    return m.to(DEV).eval(), cfg


def main() -> int:
    run = sys.argv[1] if len(sys.argv) > 1 else "logs/train_world_model_2026_09_11_03_19_45_bs_stride10"
    ckpt = sys.argv[sys.argv.index("--ckpt") + 1] if "--ckpt" in sys.argv else None
    cam  = sys.argv[sys.argv.index("--cam") + 1] if "--cam" in sys.argv else "cam_scene"
    model, cfg = _load(run, ckpt)
    ROOT, STRIDE, T = cfg.data.root, int(cfg.data.subsample), 24

    df = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{ROOT}/val/data/*/*.parquet"))])
    e0 = df[df.episode_index == 0].sort_values("frame_index")
    st = json.load(open(f"{ROOT}/normalization_stats.json"))
    O = (np.stack(e0.observation_vector.values).astype(np.float32)
         - np.array(st["observation_vector"]["mean"], np.float32)) / np.array(st["observation_vector"]["std"], np.float32)
    npy = {"cam_scene": "scene_right", "cam_wrist": "gripper_right_top"}
    cams = {k: np.load(f"{ROOT}/val/{v}_96x128.npy", mmap_mode="r") for k, v in npy.items()}
    S0 = 4000

    def fr(c, n):
        return torch.from_numpy(np.stack([cams[c][S0 + i * STRIDE] for i in range(n)])).float().div(255.)[None].to(DEV)

    obs = {"proprio": torch.from_numpy(O[S0:S0 + T * STRIDE:STRIDE])[None].to(DEV),
           "cam_scene": fr("cam_scene", T), "cam_wrist": fr("cam_wrist", T)}

    off, n_tok = 0, None
    for name, n in model.layout:
        if name == cam: n_tok = n; break
        off += n
    assert n_tok, f"{cam} not in layout {model.layout}"
    mod = model.modalities[cam]
    print(f"\nrun   {run.split('/')[-1]}\nckpt  {ckpt or 'best.ckpt'}\ncam   {cam}: {n_tok} tokens at bag offset {off}\n")

    with torch.no_grad():
        bag = model.encode_state(obs)[0]                       # (T, n_state, d)
        tok = bag[:, off:off + n_tok].contiguous()             # (T, n_tok, d)
        base = mod.decode(tok, commit=True).float()            # (T,H,W,C)
        mean_tok = tok.mean(0, keepdim=True)                   # temporal mean per slot
        foot = torch.empty(n_tok, T, *base.shape[1:3])
        for i in range(n_tok):
            t2 = tok.clone(); t2[:, i] = mean_tok[:, i]
            foot[i] = (mod.decode(t2, commit=True).float() - base).abs().mean(-1).cpu()

    HW = foot.shape[-2] * foot.shape[-1]
    f = foot.reshape(n_tok, T, HW)
    k5 = max(1, int(0.05 * HW)); k10 = max(1, int(0.10 * HW))
    tot = f.sum(-1).clamp_min(1e-12)
    conc = (f.topk(k5, -1).values.sum(-1) / tot).mean(1)                       # (n_tok,)

    ys, xs = torch.meshgrid(torch.arange(foot.shape[-2]).float(),
                            torch.arange(foot.shape[-1]).float(), indexing="ij")
    w = (f / tot[..., None])
    cy = (w * ys.reshape(-1)).sum(-1); cx = (w * xs.reshape(-1)).sum(-1)       # (n_tok,T)
    drift = torch.sqrt(cy.var(1) + cx.var(1))

    masks = torch.zeros(n_tok, HW, dtype=torch.bool)
    for i in range(n_tok):
        masks[i, f[i].mean(0).topk(k10, -1).indices] = True
    iou = []
    for i in range(n_tok):
        for j in range(i + 1, n_tok):
            inter = (masks[i] & masks[j]).sum().item(); union = (masks[i] | masks[j]).sum().item()
            iou.append(inter / max(union, 1))
    iou = float(np.mean(iou))

    # how far does the scene's own moving content travel over these frames? the yardstick for drift
    img = base.cpu()
    dif = (img[1:] - img[:-1]).abs().mean(-1).reshape(T - 1, HW)
    dw = dif / dif.sum(-1, keepdim=True).clamp_min(1e-12)
    my = (dw * ys.reshape(-1)).sum(-1); mx = (dw * xs.reshape(-1)).sum(-1)
    content = float(torch.sqrt(my.var() + mx.var()))

    print(f"=== CONCENTRATION: energy in top 5% of pixels (uniform = 0.050) ===")
    print(f"    mean {conc.mean():.3f}   min {conc.min():.3f}   max {conc.max():.3f}")
    print(f"    tokens above 0.25: {(conc > 0.25).sum().item()}/{n_tok}")
    print(f"\n=== OVERLAP: mean pairwise IoU of top-10% masks (0 = disjoint, 1 = identical) ===")
    print(f"    {iou:.3f}")
    H, W = foot.shape[-2], foot.shape[-1]
    diag = float(np.hypot(H, W))
    spread = torch.sqrt(cy.var(0) + cx.var(0))                      # ACROSS tokens, fixed instant
    gy, gx = cy.mean(0), cx.mean(0)
    common = float(torch.sqrt(gy.var() + gx.var()))
    indiv = float(torch.sqrt((cy - gy).var(1) + (cx - gx).var(1)).mean())

    print(f"=== CENTROID DRIFT over {T} frames ({T*STRIDE/30:.1f}s), in pixels ===")
    print(f"    tokens: mean {drift.mean():.2f}   min {drift.min():.2f}   max {drift.max():.2f}")
    print(f"    the scene's own moving content: {content:.2f}")
    print(f"    ratio token/content = {drift.mean()/max(content,1e-9):.2f}x")
    print(f"\n=== INTER-TOKEN SPREAD at a fixed instant -- THE CONFOUND CHECK ===")
    print(f"    mean {spread.mean():.2f} px on a {diag:.0f} px diagonal = {spread.mean()/diag:.3f}")
    print(f"    small => every token points at the same place => drift is the decoder, not binding")
    print(f"\n=== DRIFT DECOMPOSITION (px) ===")
    print(f"    COMMON-MODE (bag sliding together) {common:.2f}   INDIVIDUAL (per-token) {indiv:.2f}")

    print(f"\n--- verdict")
    if conc.mean() < 0.15:
        print("    DIFFUSE: tokens do not own regions at all. Identity is distributed across the bag;")
        print("    binding would be an architecture rewrite, not a loss.")
    elif spread.mean() / diag < 0.12 or common > indiv:
        print("    NOT BOUND. Footprints are localised but they cluster in ONE region and drift")
        print("    together, so the apparent content-following is the decoder being content")
        print("    dependent, not tokens owning objects. There is no division of labour for a")
        print("    binding loss to sharpen -- which matches the 8.25 invariant: with nothing tying a")
        print("    token to an object, identity must be re-derived diffusely at every rollout step.")
    elif drift.mean() < 0.35 * content:
        print("    POSITION-BOUND: localised, spatially distinct, but STATIC -- tokens own fixed")
        print("    tiles. A moving block is handed tile to tile, a plausible mechanism for identity")
        print("    compounding away over a rollout.")
    else:
        print("    CONTENT-BOUND: localised, spatially DISTINCT, and individually following content.")
        print("    Emergent object binding is present and a binding loss has something to sharpen.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
