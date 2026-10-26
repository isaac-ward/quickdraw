"""The ENCODE -> DECODE round trip the intermediates figure shows, run on the real checkpoint.

The intermediates figure makes one claim -- that each stream can be pushed through a learned encoder into
a code and pulled back out again -- so the items it draws on the right have to be actual reconstructions,
not the input items repeated. Vision and proprioception go through the model's OWN codec path with the
dynamics taken out: encode_state() straight into to_obs(), exactly as roundtrip_losses does it (LayerNorm
on the bag included, relative-position anchor included), at the SAME checkpoint make_prediction.py uses.

NO DYNAMICS, NO CONTEXT. Each drawn item is encoded on its own as a one-step window, because that is what
the figure says is happening -- nothing here reads a neighbouring frame.

THE ACTION DECODER IS FITTED HERE, AND IS NOT PART OF THE MODEL. The trained model has no action decoder:
`act_enc` lifts the 16-D command (4 sticks x subsample 4) into the same d=128 token the transformer reads,
and nothing ever maps it back -- actions are conditioning, never a prediction target. To show the action
stream making the same round trip, this fits a least-squares readout of the FROZEN encoder on half the
episodes and scores it on the other half. A real decode of a real code, trained after the fact, and said
out loud rather than drawn as the input twice.

    python src/paper_specific/figures/make_intermediates.py     # CPU; writes intermediates/recon.npz
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np
import torch
from omegaconf import OmegaConf

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = sorted(glob.glob("/app/logs/train_world_model_*_s2_sub4_concat"))[-1]
CKPT = f"{RUN}/checkpoints/ah_base_ep38.ckpt"         # the same weights the predicted frame comes from
OUT = "/app/logs/paper_icra_2027/intermediates"
ROOT = glob.glob("/app/scratch/recording_*_starling-2")[0]

# the figure's own constants, read from the figure script so the two cannot drift apart
_src = open(f"{HERE}/make_sequence_figs.py").read()
N = int(re.search(r"^N\s+= (\d+)", _src, re.M).group(1))
N_STATE = int(re.search(r"^N_STATE\s+= (\d+)", _src, re.M).group(1))
PROPRIO_CELLS = int(re.search(r"^PROPRIO_CELLS\s+= (\d+)", _src, re.M).group(1))
END = int(re.search(r"^END\s+= (\d+)", _src, re.M).group(1))
PANEL_EVERY = int(re.search(r"^PANEL_EVERY\s+= (\d+)", _src, re.M).group(1))

from quickdraw.data.dataset import set_action_aggregate, set_obs_keep, set_subsample   # noqa: E402
from quickdraw.training.setup import build_model, load_checkpoint, normalizer          # noqa: E402


def fit_action_decoder(m, norm, act_all, eps, S):
    """Least-squares readout act_enc(a) -> a, fitted on the EVEN episodes and scored on the odd ones.

    The split is what makes the number mean anything: 128 features and a bias against a 16-D target will
    fit any 300 samples, so a training residual would say nothing about whether the code carries the
    command. Episodes are the unit because consecutive strided steps are anything but independent."""
    def windows(rows):                                   # (T,4) raw -> (T//S, S*4), the loader's concat
        n = len(rows) // S
        return rows[: n * S].reshape(n, S * rows.shape[1])
    tr = np.concatenate([windows(act_all[eps == e]) for e in np.unique(eps) if e % 2 == 0])
    te = np.concatenate([windows(act_all[eps == e]) for e in np.unique(eps) if e % 2 == 1])
    with torch.no_grad():
        a_tr, a_te = norm.norm_act(torch.from_numpy(tr).float()), norm.norm_act(torch.from_numpy(te).float())
        z_tr, z_te = m.act_enc(a_tr).float(), m.act_enc(a_te).float()
    X = np.concatenate([z_tr.numpy(), np.ones((len(z_tr), 1), np.float32)], 1)
    W, *_ = np.linalg.lstsq(X, a_tr.numpy(), rcond=None)
    Xe = np.concatenate([z_te.numpy(), np.ones((len(z_te), 1), np.float32)], 1)
    err = np.abs(norm.denorm_act(torch.from_numpy(Xe @ W)).numpy() - te)
    print(f"  action decoder: fitted on {len(tr)} windows from {len(np.unique(eps)) // 2} episodes, "
          f"scored on {len(te)} held-out -> mean |error| {err.mean():.4f} stick units "
          f"(max {err.max():.3f}; commands span [-1, 1])")
    return W


def write_shapes(path, cfg, m, bag, act_n, win_img, win_pro, win_act):
    """What every arrow in the two figures is actually carrying, MEASURED on a real 8-step window.

    Every number here comes from a tensor that was built, not from the config and not from a comment: the
    bag widths, the conditioning width and the predicted slice are read off the forward path, so the file
    cannot drift from the model the figures are drawn against."""
    P, S = int(cfg.data.P), int(cfg.data.get("subsample", 1) or 1)
    step_ms = 1000.0 * S / 15.0                          # starling-2 captures at 15 fps
    # THE RAW ACTION, not the token: _to_input and _cond both encode it themselves, and handing them an
    # already-encoded one is a 128-wide tensor walking into a Linear(16, 128).
    with torch.no_grad():
        a_tok = m.act_enc(act_n)
        inp = m._to_input(bag, act_n)
        h = m.backbone(inp)
        cond = m._cond(h[:, -1], act_n[:, -1], m._proprio_slot(bag[:, -1]))
        nxt = m.predict_next(cond, bag[:, -1])
    n_img, n_pro = dict(m.layout)["image"], dict(m.layout)["proprio"]
    d = bag.shape[-1]
    px = int(np.prod(win_img.shape[1:]))

    def line(name, per_step, window, note=""):
        return f"  {name:<12} {per_step:<34} {window:<22} {note}\n"

    t = []
    t.append(f"Shapes behind the intermediates figures -- measured, not quoted\n")
    t.append(f"  dataset    starling-2, captured at 15 fps, subsampled x{S} -> one model step = {step_ms:.0f} ms\n")
    t.append(f"  window     {P} model steps = {P * step_ms / 1000:.2f} s of flight (the braced span in the figures)\n")
    t.append(f"  checkpoint {CKPT}\n\n")
    t.append("INPUTS                per model step                     per window\n")
    t.append(line("vision", f"{' x '.join(str(v) for v in win_img.shape[1:])} = {px:,} values",
                  str(tuple(win_img.shape)), "RGB, 0..1"))
    t.append(line("proprio", f"{win_pro.shape[-1]} values", str(tuple(win_pro.shape)),
                  f"the figures draw {PROPRIO_CELLS} of the {win_pro.shape[-1]} cells"))
    t.append(line("action", f"4 sticks x {S} raw commands = {win_act.shape[-1]}", str(tuple(win_act.shape)),
                  "concatenated, not summed"))
    t.append("\nCODES                 per model step                     per window\n")
    t.append(line("vision", f"{n_img} tokens x {d} = {n_img * d:,} floats", f"({P}, {n_img}, {d})",
                  f"{px / (n_img * d):.1f}x SMALLER than the pixels"))
    t.append(line("proprio", f"{n_pro} token x {d} = {n_pro * d:,} floats", f"({P}, {n_pro}, {d})",
                  f"{n_pro * d / win_pro.shape[-1]:.0f}x LARGER than the {win_pro.shape[-1]} numbers"))
    t.append(line("action", f"1 token x {a_tok.shape[-1]} = {a_tok.shape[-1]:,} floats",
                  f"({P}, {a_tok.shape[-1]})",
                  f"{a_tok.shape[-1] / win_act.shape[-1]:.0f}x larger; conditioning, never a target"))
    t.append(line("state bag", f"{bag.shape[-2]} tokens x {d}", str(tuple(bag.shape[1:])),
                  "vision + proprio: what the decoders read"))
    t.append(line("input bag", f"{inp.shape[-2]} tokens x {d}", str(tuple(inp.shape[1:])),
                  "+ the action token: what the summarizer reads"))
    t.append("\nDYNAMICS\n")
    t.append(f"  summarizer   {tuple(inp.shape[1:])} -> {tuple(h.shape[1:])}   (space-time transformer, "
             f"depth {cfg.model.get('depth', '?')})\n")
    t.append(f"  conditioning {tuple(cond.shape[1:])} per step: {cond.shape[-1]} = "
             f"{d} summarized state + {d} action slot + {d} raw action embedding\n")
    t.append(f"  observation head denoises the next state bag: {tuple(nxt.shape[1:])} "
             f"({n_img} image + {n_pro} proprio tokens)\n")
    t.append(f"  the action token is carried through, never predicted -- which is why the predicted slice "
             f"is {nxt.shape[-2]} tokens and not {inp.shape[-2]}\n")
    open(path, "w").writelines(t)
    print(f"  wrote {path}")


def main() -> int:
    cfg = OmegaConf.create(json.load(open(f"{RUN}/logs/config.json")))
    OmegaConf.set_struct(cfg, False)
    S = int(cfg.data.get("subsample", 1) or 1)
    set_subsample(S)
    set_action_aggregate(str(cfg.data.get("action_aggregate", "sum")))
    set_obs_keep(cfg.data.get("obs_keep", None))

    idx = [END - (N - 1 - i) * S * PANEL_EVERY for i in range(N)]
    print(f"  round-tripping the figure's {N} items: frames {idx[0]}..{idx[-1]}")

    import imageio.v3 as iio3
    import pyarrow.parquet as pq
    tab = pq.read_table(sorted(glob.glob(f"{ROOT}/train/data/**/*.parquet", recursive=True))[0]).to_pydict()
    eps = np.asarray(tab["episode_index"])
    act_all = np.stack([np.asarray(x, np.float32) for x in tab["action"]])
    keep = np.flatnonzero(eps == eps[0])                 # the figure's items all come from episode 0
    obs_ep = np.stack([np.asarray(x, np.float32) for x in tab["observation_vector"]])[keep]
    act_ep = act_all[keep]
    vid = sorted(glob.glob(f"{ROOT}/train/videos/**/*.mp4", recursive=True))[0]
    frames = np.stack([f for k, f in enumerate(iio3.imiter(vid, plugin="pyav")) if k in set(idx)])

    norm = normalizer(cfg)
    m = build_model(cfg)
    load_checkpoint(m, CKPT)
    m = m.eval()

    # ONE WINDOW PER ITEM (T=1): the codec is per-frame, so the batch axis carries the items and the time
    # axis stays at 1. With T=1 the relative-position anchor IS that item's own position, which is exactly
    # what a stateless encode of a single frame has available.
    img = torch.from_numpy(frames).float().div(255.0).unsqueeze(1)                 # (N,1,H,W,3)
    pro = norm.norm_obs(torch.from_numpy(obs_ep[idx]).float()).unsqueeze(1)        # (N,1,obs)
    actual = {"image": img, "proprio": pro}
    with torch.no_grad():
        anchor = m.rel_anchor(actual) if m._rel_on() else None
        out = m.to_obs(m.encode_state(actual, anchor), heads=["image", "proprio"], anchor=anchor,
                       commit=True)
    rec_img = out["image"][:, 0].clamp(0, 1).numpy()
    rec_pro = norm.denorm_obs(out["proprio"][:, 0].cpu()).numpy()

    # THE ACTION STREAM. The drawn tile is the raw 4-D command at the item's frame; the model's code is
    # over the 16-D window that STARTS there, whose first four numbers ARE that command -- so the round
    # trip runs on the window and the tile takes the first four back out of it.
    W = fit_action_decoder(m, norm, act_all, eps, S)
    win = np.stack([act_ep[t: t + S].reshape(-1) for t in idx])                    # (N, S*4)
    with torch.no_grad():
        z = m.act_enc(norm.norm_act(torch.from_numpy(win).float())).float().numpy()
    rec_act = norm.denorm_act(torch.from_numpy(np.concatenate(
        [z, np.ones((len(z), 1), np.float32)], 1) @ W)).numpy()[:, : act_ep.shape[1]]

    os.makedirs(OUT, exist_ok=True)
    np.savez(f"{OUT}/recon.npz", image=(rec_img * 255).astype(np.uint8),
             proprio=rec_pro.astype(np.float32), action=rec_act.astype(np.float32),
             frames=np.array(idx))
    print(f"  wrote {OUT}/recon.npz  image {rec_img.shape}  proprio {rec_pro.shape}  action {rec_act.shape}")

    # THE SHAPE SHEET. The same window the figures brace: P consecutive strided steps ending on the last
    # braced item, which is the window the model is actually given.
    P = int(cfg.data.P)
    ctx_raw = [idx[N_STATE - 1] - (P - 1 - j) * S for j in range(P)]
    w_img = torch.from_numpy(np.stack([f for k, f in enumerate(iio3.imiter(vid, plugin="pyav"))
                                       if k in set(ctx_raw)])).float().div(255.0)
    w_pro = torch.from_numpy(obs_ep[ctx_raw]).float()
    w_act = torch.from_numpy(np.stack([act_ep[t: t + S].reshape(-1) for t in ctx_raw])).float()
    with torch.no_grad():
        ctx = {"image": w_img.unsqueeze(0), "proprio": norm.norm_obs(w_pro).unsqueeze(0)}
        anc = m.rel_anchor(ctx) if m._rel_on() else None
        bag = m.encode_state(ctx, anc)
        act_n = norm.norm_act(w_act).unsqueeze(0)
    write_shapes(f"{OUT}/shapes.txt", cfg, m, bag, act_n, w_img, w_pro, w_act)
    true_img = frames.astype(np.float32) / 255.0
    for i, t in enumerate(idx):
        mse = float(((rec_img[i] - true_img[i]) ** 2).mean())
        pos = float(np.linalg.norm(rec_pro[i][:3] - obs_ep[t][:3]))
        act = float(np.abs(rec_act[i] - act_ep[t]).mean())
        print(f"    item {i:2d} (frame {t:3d}): image {10 * np.log10(1.0 / max(mse, 1e-12)):5.2f} dB "
              f"| proprio position error {pos:.3f} m | action {act:.4f} stick units")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
