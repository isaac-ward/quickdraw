"""Architecture figure for the ICRA 2027 paper — vector SVG + matching PNG from one source.

Every layer, shape and parameter count is read off the module tree of a model built with the config the
runs actually use (scratch/dump_arch.py), so the figure states what IS training, not what is available.
Anything the code supports but this configuration does not use is listed as OFF rather than drawn.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, MUT, DIM = "#141414", "#5f5f5f", "#8c8c8c"
C = {"in": "#eef2f7", "enc": "#dbe7f3", "bag": "#f7edda", "bb": "#e7def1",
     "dyn": "#dcecdc", "dec": "#f7dfdc", "act": "#fdf4d2"}
E = {"in": "#7d93ab", "enc": "#5b82ab", "bag": "#b08d3c", "bb": "#7a63a8",
     "dyn": "#5f9463", "dec": "#b8695f", "act": "#c09a2e"}

fig, ax = plt.subplots(figsize=(19.5, 10.4))
ax.set_xlim(0, 195); ax.set_ylim(0, 106); ax.axis("off")


def box(x, y, w, h, title, layers="", params="", fc="#fff", ec="#777", ls="solid",
        tfs=9.2, lfs=7.2, mono=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.4",
                                fc=fc, ec=ec, lw=1.6, linestyle=ls, zorder=2))
    ax.text(x + w / 2, y + h - 3.0, title, ha="center", va="center", fontsize=tfs,
            fontweight="bold", color=INK, zorder=3)
    bot = y + 2.2
    if params:
        ax.text(x + w / 2, bot, params, ha="center", va="center", fontsize=lfs - 0.4,
                color=DIM, style="italic", zorder=3)
        bot += 3.4
    if layers:
        ax.text(x + 2.4, (y + h - 6.4 + bot) / 2, layers, ha="left", va="center", fontsize=lfs,
                color=INK, zorder=3, linespacing=1.62,
                family="monospace" if mono else None)
    return (x, y, w, h)


def arrow(a, b, label="", ls="solid", col=MUT, rad=0.0, dy_a=0, dy_b=0, ly=2.0, lfs=7.0):
    x0, y0 = a[0] + a[2], a[1] + a[3] / 2 + dy_a
    x1, y1 = b[0], b[1] + b[3] / 2 + dy_b
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13,
                                 lw=1.4, color=col, linestyle=ls, zorder=1,
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + ly, label, ha="center", va="bottom",
                fontsize=lfs, color=col, zorder=3, family="monospace")


# ---- inputs ------------------------------------------------------------------------------------
i1 = box(1.5, 79, 21, 12, "proprio  $o_t$", "(B,T,16)\nB=20, T=72", fc=C["in"], ec=E["in"], lfs=7.4)
i2 = box(1.5, 47, 21, 14, "image  $I_t$", "(B,T,112,192,3)\nego camera, RGB", fc=C["in"], ec=E["in"], lfs=7.4)
i3 = box(1.5, 13, 21, 15, "action  $a_t$", "(B,T,16)\n4 joy axes x\n4 concat slots", fc=C["in"], ec=E["in"], lfs=7.4)

# ---- encoders ----------------------------------------------------------------------------------
e1 = box(29, 76, 31, 16, "proprio encoder",
         "Linear(16->64)\nGELU\nLinear(64->128)\n-> (B,T,1,128)", "9,408", fc=C["enc"], ec=E["enc"])
e2 = box(29, 36, 31, 38, "image encoder (conv)",
         "Conv2d(3->32,k3,s2)      56x96\n"
         "ResBlk(32->32)  +pool    28x48\n"
         "ResBlk(32->64)  +pool    14x24\n"
         "ResBlk(64->128) +pool     7x12\n"
         "Conv2d(128->128,k1)\n"
         "+ learned pos (84,128)\n"
         "CrossAttn: 32 learned Q\n"
         "  attend the 84 cells\n"
         "LayerNorm\n"
         "-> (B,T,32,128)", "405,600", fc=C["enc"], ec=E["enc"])
e3 = box(29, 12, 31, 17, "act_enc",
         "Linear(16->128)\nGELU\nLinear(128->128)\nFourier OFF (n_freq=0)\n-> (B,T,1,128)",
         "18,688", fc=C["enc"], ec=E["enc"])
for a, b in ((i1, e1), (i2, e2), (i3, e3)):
    arrow(a, b)

# ---- token bag ---------------------------------------------------------------------------------
bag = box(66, 34, 24, 50, "token bag",
          "per step:\n\n 1 proprio token\n+32 image tokens\n+ 1 action token\n = 34 x d=128\n\n"
          "per-token\nLayerNorm\n(non-affine)\n\n(B,T,34,128)", fc=C["bag"], ec=E["bag"], lfs=7.5)
for a, dy in ((e1, 17), (e2, 0), (e3, -17)):
    arrow(a, bag, dy_b=dy)

# ---- backbone ----------------------------------------------------------------------------------
bb = box(96, 34, 29, 50, "space-time backbone",
         "4 x SpaceTimeBlock (pre-norm)\nd=128, heads=4, window 32\n\n"
         "  LN -> SPATIAL attn\n     across 34 tokens\n     within a step\n\n"
         "  LN -> TEMPORAL attn\n     causal, across steps\n\n"
         "  LN -> MLP 128->512->128\n\nLayerNorm out\n-> $h_t$ (B,T,34,128)",
         "1,062,912", fc=C["bb"], ec=E["bb"], lfs=7.2)
arrow(bag, bb)

# ---- dynamics ----------------------------------------------------------------------------------
dyn = box(131, 60, 30, 26, "predict_next   (flow)",
          "cond = 3d = 384\n [state | slot(a x s) | act]\n"
          "inp Linear(544->128)\n2 x ViTBlock over the\n  33 state tokens (JOINT)\nout Linear(128->128)\n"
          "residual: next = prev + d", "484,864", fc=C["dyn"], ec=E["dyn"], lfs=7.0)
arrow(bb, dyn, "$h_t$", dy_a=13, rad=-0.10)

# ---- action head -------------------------------------------------------------------------------
act = box(131, 16, 30, 24, "action head   (OFF)",
          "pooled $h_{t-1}$ (mean over\n  the 34 tokens) -> FlowField\n"
          "p(a_t .. a_t+K-1 | h_t-1)\nchunk K, joint, leak-free\n\n"
          "post-hoc on a FROZEN WM\nnot enabled in these runs",
          fc=C["act"], ec=E["act"], ls="dashed", lfs=7.0)
arrow(bb, act, "", ls="dashed", dy_a=-15, rad=0.12)

# ---- decoders ----------------------------------------------------------------------------------
d1 = box(167, 74, 26, 16, "proprio decode",
         "[0(16) | tau(32) | cond(128)]\nLinear(176->64) GELU\nLinear(64->64)  GELU\n"
         "Linear(64->16)", "18,640", fc=C["dec"], ec=E["dec"], lfs=6.8)
d2 = box(167, 20, 26, 48, "TokenGridDecoder",
         "NO U-Net skips: the only\npath is the 32 tokens.\n\n"
         "readout: learned 7x12 Q grid\n  CROSS-ATTENDS the bag,\n  then 1 ViTBlock over cells\n"
         "gpool:   1 learned Q -> g(128)\n         (FiLM, re-read below)\n\n"
         "to_ch Conv(128->256)   7x12\n"
         "mid   FiLMRes(256)     7x12\n"
         "up x2 FiLMRes(256)    14x24\n"
         "up x2 FiLMRes(256)    28x48\n"
         "up x2 FiLMRes(->128)  56x96\n"
         "up x2 FiLMRes(-> 64) 112x192\n"
         "GN+SiLU, Conv(64->3,k3)\n"
         "SIGMOID", "4,766,083", fc=C["dec"], ec=E["dec"], lfs=6.8)
arrow(dyn, d1, "", dy_a=6, rad=-0.06)
arrow(dyn, d2, "", dy_a=-6, rad=0.06)

# ---- the autoregressive feedback: latent -> latent, never through the decoders -----------------
# _rollout_from: s_pred = readout(backbone(window)) is a BAG OF LATENT TOKENS, and `bag_buf.append(s_feed)`
# puts it straight back in as the next step's state tokens. carry_transform is the identity here (it is a
# data-space re-encode only for DSAR), and lit.py explicitly does NOT decode inside the loop -- to_obs
# would sample the decoder every step for nothing. Decoding happens ONCE, off the loop, to score the loss.
FB_Y = 92.0
ax.plot([dyn[0] + 4, dyn[0] + 4], [dyn[1] + dyn[3], FB_Y], color="#5f9463", lw=1.6, zorder=4)
ax.plot([dyn[0] + 4, bag[0] + bag[2] / 2], [FB_Y, FB_Y], color="#5f9463", lw=1.6, zorder=4)
ax.add_patch(FancyArrowPatch((bag[0] + bag[2] / 2, FB_Y), (bag[0] + bag[2] / 2, bag[1] + bag[3]),
                             arrowstyle="-|>", mutation_scale=14, lw=1.6, color="#5f9463", zorder=4))
ax.text((dyn[0] + 4 + bag[0] + bag[2] / 2) / 2, FB_Y + 1.2,
        "AUTOREGRESSION — the predicted bag re-enters as the next step's state tokens.\n"
        "Latent space only: it is never decoded to pixels inside the loop. Teacher forcing mixes in the\n"
        "TRUE encoded latent with probability p_tf; the graph is detached every 16 steps.",
        ha="center", va="bottom", fontsize=7.0, color="#3f7048", linespacing=1.5)

# ---- titles / notes ----------------------------------------------------------------------------
ax.text(97, 101.8, "Quickdraw world model as trained — 6.77 M parameters, $d$ = 128", ha="center",
        fontsize=15, fontweight="bold", color=INK)
ax.text(97, 98.6, "starling-2 ego flight  |  15 Hz capture, frame stride 4 = 267 ms/step  |  "
                  "P = 8 context + F = 64 rollout  |  runs s2_sub4_concat, s2_sub4_concat_deriv",
        ha="center", fontsize=9.4, color=MUT)

ax.text(1.5, 101.8, "OFF in this configuration:", ha="left", fontsize=8.2, fontweight="bold", color=INK)
ax.text(1.5, 99.2, "action-head  |  act_enc Fourier  |  action squash  |  decoder inject  |\n"
                   "decoder per-level x-attn  |  diffusion forcing  |  decoder shortcut",
        ha="left", va="top", fontsize=7.2, color=DIM, linespacing=1.5)

ax.text(97, 9.0, "LOSSES    decode/<head>  AR decode of rolled latents, every frame (recon_frac=1) — the only AR gradient    |    "
                 "codec/roundtrip  real frames, w=10", ha="center", fontsize=7.4, color=MUT, family="monospace")
ax.text(97, 6.0, "          dynamics/latent  flow matching, w=10    |    derivative/<head>  temporal difference, stride 1, w=1 (deriv run only)    |    "
                 "VisualLoss = 3.0 L1 + 1.0 LPIPS", ha="center", fontsize=7.4, color=MUT, family="monospace")
ax.text(97, 2.6, "objective:  eval_ood_horizon/open_loop/image/lpips/@+128   —   128 steps = 34.1 s at this stride",
        ha="center", fontsize=8.6, color=INK, style="italic")

fig.tight_layout()
for ext in ("svg", "png"):
    p = f"/app/logs/paper_icra_2027/architecture.{ext}"
    fig.savefig(p, format=ext, dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", p)
