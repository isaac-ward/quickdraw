"""Sequence figures for the ICRA 2027 paper.

Emits every component on ONE SHARED CANVAS so the PNGs overlay exactly in a figure editor:

    architecture.png                everything, in the folder root -- this is paper Fig. 4
    elements/items_vision.png       the image cascade alone, no braces, no tokenizer
    elements/items_proprio.png      the observation-vector cascade alone
    elements/items_action.png       the action-vector cascade alone
    elements/braces.png             every brace + stem, nothing else
    elements/latent_dynamics.png    the dashed group around the two transformers, nothing else
    elements/tokenizers.png         the three tokenizer blocks, nothing else

Because the layout is solved once and every render draws into it, dropping all six on top of each other
reproduces architecture.png pixel for pixel.

Real data: frames from starling-2's train split and the matching rows of its parquet. The action is
rebuilt the way data/dataset.py::_subsample_episodes does under action_aggregate=concat.
"""
from __future__ import annotations

import glob
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrow, PathPatch, Polygon, Rectangle
from matplotlib.path import Path

# ----------------------------------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------------------------------
ROOT = glob.glob("/app/scratch/recording_*_starling-2")[0]
OUT = "/app/logs/paper_icra_2027"
RUN = sorted(glob.glob("/app/logs/train_world_model_*_s2_sub4_concat"))[-1]
FPS = 15.0                                   # starling-2 capture rate

N            = 12        # items per modality: 8 braced + 4 after (no unused lead-in)
END          = 304       # anchor on the LAST item (the ladder view); earlier items count backwards
PANEL_EVERY  = 4         # MODEL STEPS between items. 4 => ~1.07 s apart, so the camera visibly moves
BRACES      = [(0, 7)]             # inclusive, 0-based -- the 8 in-window steps, from the very first item
BRACES_ACT  = [(0, 7), (8, 11)]    # ...and the action stream braces the 4 steps AFTER it: the nub's input
N_STATE     = 8        # vision/proprio stop at the last braced step; only the action stream runs to N
# *1 = act[t], the last action INSIDE the window: act[t] drives t -> t+1, so this is the one that
# produces the predicted state.  *2 = act[t+1], the single-braced action that makes the nub.
ASTERISKS    = ((7, "*\u2081"), (8, "*\u2082"))

OFFSET_X     = 48.0      # per-item step in x
OFFSET_Y     = OFFSET_X / 2          # ...and half that in y: the cascade angle
IMG_W        = 300.0     # image item width
PRED_BAND_PX = 16.0      # predicted IMAGES are hatched only in a band this many IMAGE PIXELS in from
#                          the edge: hatching the whole frame greyed out the thing the figure is for
CELL         = 32.0      # square vector cell
PROPRIO_CELLS = 7        # observation dims DRAWN (the vector is 16-D; this is a schematic)
# ONE COLOURMAP PER MODALITY, and everything that modality touches is drawn from it: the cascade's cells,
# its tokenizer, its token block, its decoder. Before this every stream was the same rainbow inside the
# same blue outline, so which stream a block belonged to had to be traced along an arrow; now it is the
# colour. `gist_rainbow` is gone from the cells for the same reason it made the distribution glyph fail --
# a hue cycle is not ordered, so a cell's colour said "which value" only by lookup, never by rank.
MOD_CMAP     = {"vision": "Greens", "proprio": "Oranges", "action": "Blues"}
ACT_OUT_CMAP = "Purples"   # the SAMPLED chunk at the far right. The whole action stream -- both brace
#                            groups, both blocks -- is one blue; the only thing that changes colour is
#                            what the model MADE UP rather than read, which is the point worth marking.
VIS_EC       = plt.get_cmap(MOD_CMAP["vision"])(0.78)   # image tiles are OUTLINED in the vision hue
CMAP_LO      = 0.20      # cells use this much of the map upward: a sequential map's bottom end is white,
CMAP_HI      = 0.95      # and a white cell reads as empty rather than as low
MUTE         = 0.0       # the rainbow needed desaturating; these do not
ALPHA_LO     = 0.32      # oldest item's opacity; newest is 1.0

LW           = 1.2       # one line width: item outlines and cell separators
EDGE         = "#000000"
CORNER_R     = 8.0       # ONE corner radius: braces and tokenizers alike

BRACE_M      = 30.0      # clearance above the items
BRACE_T      = 8.0       # gap between a brace nib and the corner it marks
STREAM_FRAC  = 1.0      # vertical spacing between the three cascades, as a fraction of the riser gap.
#                         MEASURED 2026-09-15 and left at 1.0: the arrow of time sits 601 units (33% of
#                         the height) below the bottom of the blue action block, but compressing this
#                         moves the canvas height NOT AT ALL -- 1814.9 units at 1.0, 0.5 and 0.25 alike,
#                         because the height is pinned by the RIGHT-HAND column (summariser, the two
#                         heads, the token blocks, the decoders), not by the streams. Lifting the streams
#                         only moves whitespace around inside the same box.
TIME_M       = 40.0      # clearance between the last cascade and the arrow of time
TIME_LAB     = "$t$"    # at the ARROWHEAD, not the middle of the axis
BRACE_HALO   = 6.0       # white halo width, as a multiple of LW

ENC_GAP      = 170.0     # gap between the cascade and its tokenizer (holds the stem risers)
ENC_W        = 225.0     # tokenizer width
ENC_H        = 240.0     # tokenizer input-edge height = 0.8 x the original 300 (all three match)
ENC_LW       = 3.2
BOX_EC       = "#4d4d4d" # the TRANSFORMER boxes only. Neutral on purpose: every hue in the figure
#                          now names a modality, and the summariser and the two heads are not one -- give
#                          them a colour and a reader ties them to whichever stream shares it.
ENC_FC       = "white"
ENC_FS       = 26.0      # label size
# ---- token blocks: an isometric cuboid per tokenizer, cells = (depth x height x width) tokens --------
BLK_RUN      = 150.0     # horizontal run of the 45-degree arrow from a tokenizer to its block
#                          (45 deg means the vertical drop equals it)
BLK_ROW_SEP  = 150.0     # vertical separation between the two feed points on the action tokenizer
BLK_DIAG     = 210.0     # length of the 45-degree feed arrow, from the tokenizer edge to the block
BLK_LABELS   = False      # draw A/B/C on the three faces so they can be referred to unambiguously

# ---- THE REWARD MODEL, bottom right ----------------------------------------------------------------
# It is not part of the world model, so it is drawn as its own block rather than solved into the stack:
# a brace along the BOTTOM face of the token column (mirroring the summariser's, and including the
# predicted slice, because the reward is read on IMAGINED latents), a sentence entering from below, and
# one scalar out.
RM_ON        = True
RM_TITLE     = "Reward head\n(contrastive)"
RM_LAT       = "Project to $f_z$"     # the latent branch, fed by the brace, drawn ON TOP
RM_TXT       = "Project to $f_t$"     # the text branch, drawn BELOW the latent one
RM_ENC       = "MiniLM"               # ...and its sentence encoder is its OWN block: it is frozen
RM_COS       = "$\\cos(f_z, f_t)$"
RM_REQ       = "\u201cGo forward to the ladder\nin the middle of the room\u201d"   # no box, two lines
RM_REQ_PAD   = 90.0      # clear space between the request and the box it feeds
RM_OUT       = "Similarity score"
RM_ROW_H     = 58.0      # internal row height
RM_ROW_SEP   = 16.0      # between the two stacked branches
RM_COS_GAP   = 110.0     # branch boxes -> the cosine: the two feed arrows have to be visible in it
#                          tall and the figure has no room under the token column for that
RM_PAD       = 26.0
RM_GAP_BR    = BRACE_M   # column bottom -> brace line: THE SAME LIFT the summariser's input brace uses
RM_GAP_BOX   = 78.0      # brace line -> box top (16 units lower than it was, at the author's ask)
RM_GAP_TXT   = 70.0      # box -> the sentence, which sits to its LEFT
# ---- Z ORDER (one place, because the stacking rules are not obvious) --------------------------------
# RULE: A FEED ARROW MUST NEVER CROSS A TOKENIZER OUTLINE. The tokenizers therefore sit ABOVE every
# arrow, and the arrows sit above the blocks (the extension feed has to thread past the stacked column
# to reach its block). Blocks are ordered explicitly so the seams of the stacked column read correctly:
# vision on top, then proprioception, then action.
Z_ARROW      = 9000
Z_DECODE     = 7000      # the decoder feeds run UNDER every block, so their tails (and halos) are hidden
#                          INSIDE the red column exactly as the tokenizer feeds' tails hide in the blue
Z_TOKENIZER  = 20000
Z_BLOCK      = {"vision": 8000, "proprio": 7900, "action": 7700, "action_extra": 7800}
BLK_S        = 31.5      # edge length of one token cell
BLK_DEPTH    = 8         # depth = the 8 braced time steps
NUB_DEPTH    = 4         # ...and the nub is as long as ITS brace: the 4 action steps after the window
BLK_DROP     = 0.0       # extra nudge on top of the rule below (the column's BOTTOM is flush with the
#                          transformer boxes hang off the slice brace, so they follow it down
BLK_FC       = ("#f2f5f9", "#d9e3ee", "#b9cbdf")   # top, depth face, width face
BLK_EC       = "#5b82ab"
# THE PREDICTED STEP: one time step deep, flush against the near end of the column, in red. 33 tall and
# not 34 -- predict_next emits the STATE tokens only (32 image + 1 proprio); the action token is carried
# through, never predicted -- so it is flush with the TOP of the column and stops one row short at the
# bottom, exactly where the action row is.
# THE PREDICTED SLICE IS NOT A COLOUR OF ITS OWN. It holds image tokens and a proprio token, so each of
# its two blocks takes the colour of the stream it predicts -- red for vision, green for proprioception --
# and what marks it as PREDICTED is diagonal hatching, not hue. That separates the two questions a reader
# asks of a block ("which modality" and "observed or predicted") onto two channels instead of making one
# hue answer both, which is what forced the slice to orange and made it look like a third stream.
PRED_HATCH   = "//"     # THICKER AND SPARSER than the default: "///" at the block line weight read as a
#                         grey wash at print size rather than as a texture, so the stripe count is halved
#                         and hatch.linewidth is tripled below. Halved a second time to a single
#                         stripe. Every predicted thing carries it: the
#                         slice in the token column, both
#                         decoded outputs, the sampled chunk. The hatch always takes the object's OWN
#                         outline colour, so it never introduces a hue -- it only adds a texture.
LEG_S        = 58.0     # legend swatch
LEG_SEP      = 22.0     # between the two rows
LEG_GAP      = 26.0     # between a swatch and its label
LEG_TRUE     = "Truth"
LEG_PRED     = "Predicted"
# THE HAT, over each of the three things the model emits. A caret and nothing else: the item IS the
# variable, so the mark sitting on it reads as x-hat without needing a letter to sit on. Drawn rather than
# set as text so its weight matches the figure's lines instead of the body font.
IMG_TOKENS   = 16      # image-token rows DRAWN. The model carries 32; halved here purely so the column
#                        is not a sliver -- the count is carried by the proportions, not by counting rows.
PRED_BLKS    = ((IMG_TOKENS, 1), (1, 1))   # (height, depth) per red block, stacked like the blue column
PRED_STEPS   = 1       # ONE predicted slice in this variant. Everything downstream follows from it and
#                        needed no separate switch: x_0's fan collapses to a single 45 (one target), the
#                        decoder feeds leave that one slice's face, and each decoder emits one item.
BLK_LW       = 1.1
plt.rcParams["hatch.linewidth"] = BLK_LW * 2.7   # 3x the old 0.9x: a hatch that reads AS a hatch
# HATCH COLOUR IS PER MODALITY, so it cannot be an rcParam: hatch.color is global, and the author wants
# the predicted proprioception striped in orange and the sampled action plan in purple. Hatches are
# therefore drawn as an OVERLAY patch whose EDGE colour carries the hue and the alpha (matplotlib draws
# hatch strokes in the patch edge colour), at HATCH_A.
HATCH_A = 0.25
# ---- decoders: mirrored trapezia off the RED blocks, producing the predicted frame and vector --------
DEC_GAP      = 170.0     # red column's right face -> decoder input edge
OUT_GAP      = 150.0     # decoder output edge -> the predicted item
DEC_LABELS   = ("Vision\ndecoder", "Proprio\ndecoder")
PRED_AT      = 8         # the predicted step IS item 8 of the cascades: its frame and vector are real
NUB_DROP     = 66.0      # how far BELOW the nub the feed runs before turning up into it
# ---- space-time backbone: a flat 2D box DIRECTLY BELOW the token column and EXACTLY its width ---------
# Every block does BOTH passes, in this order -- they do not alternate BETWEEN blocks.
# The two transformer boxes are drawn by ONE pair of functions (box_metrics / draw_box). They share a
# width and a height so they read as a matched pair; `flow` is the only thing that differs: the backbone
# runs top-to-bottom and leaves to the right, the flow head runs bottom-to-top and leaves out the top.
BB_TITLE     = "Summarizer\n(space-time transformer)"   # HORIZONTAL, in a band along the bottom of the box --
#                          the only band with no trunk running through it
BB_SUBS      = ("spatial", "temporal")
BB_DEPTH     = 4         # drawn ONCE, braced, and labelled "x 4" -- the stack is 4 identical blocks
DT_TITLE     = "Observation head\n(rectified flow)"   # NOT "diffusion": FlowField is rectified flow
#                          matching with concat conditioning (flow.py refuses cond="adaln")
# The MetaFormer pair: attention moves information ACROSS the 33 tokens of one bag, the position-wise
# MLP moves it ACROSS the 128 channels WITHIN each token. (_TokenMixBlock is the class.)
DT_SUBS      = ("token mixing", "channel mixing")
DT_DEPTH     = 2         # 2 ViTBlocks over the 33 state tokens
DT_GAP       = 120.0     # backbone right edge -> flow-head left edge
# THE SECOND HEAD. Same class as the observation head (models/flow.py::FlowField, concat conditioning,
# the same denoising loop) hanging off the same summary h -- which is exactly why it is drawn with the
# same grammar and placed in the same horizontal band. It differs only in what it emits: the observation
# head denoises the next STATE tokens, the action head denoises the next ACTION CHUNK. It is OUTSIDE the
# dashed "Latent dynamics" group on purpose: predicting what the world does next is dynamics, predicting
# what the pilot does next is a policy prior, and the group means the former.
AH_TITLE     = "Action head\n(rectified flow)"
# THE BOX TITLES ARE SET SMALLER THAN THE TOKENIZER LABELS, and the number is derived, not chosen: a box
# is already as wide as its widest ROW (residual channel + sublayer box + repeat brace + loop channel =
# 390 u), so a title narrower than that costs nothing, while one wider than it drives the whole pair --
# and the pair's x is pinned by x_0's 45, so widening it drags the summariser left until the slice stem
# can no longer make its own 45. "(rectified-flow transformer)" is 490 u at ENC_FS and 377 u here, so at
# this size the layout is EXACTLY what it was before these names were long.
BOX_FS       = ENC_FS    # the box titles are set at the TOKENIZER LABEL SIZE, and `sample` with them.
# THE "THIS BLOCK EMITS A DISTRIBUTION" MARK, in the top-right corner inside each rectified-flow box. The
# two heads carry DIFFERENT glyphs on purpose: what they put a distribution over is different, and a
# reader who sees the same mark twice reads it as the same object. Shapes and palette are
# make_dist_glyphs.py's -- that is where the candidates were drawn and compared -- but they are redrawn
# here in figure coordinates rather than pasted in as bitmaps, so they scale with the box.
#   contours  nested level sets: the observation head samples a whole 33-token BAG, a vector, not a scalar
#   bimodal   a 1-D density with two modes: the action head samples a stick, and the prior over one really
#             is multimodal (a bell would quietly deny that)
GLYPH_W      = 52.0
GLYPH_H      = 36.0
# (centre, width, weight) per mode. Two peaks each, but not the SAME two: a reader who sees one mark
# twice reads it as one distribution, and these are distributions over different things.
GLYPH_OBS    = ((-1.35, 0.55, 0.62), (1.30, 0.78, 0.40))    # observation head: a dominant mode and a tail
GLYPH_ACT    = ((-1.55, 0.70, 0.46), (0.75, 0.52, 0.62))    # action head: the smaller mode is the wider one
# THE HEADS ARE SET TIGHTER THAN THE SUMMARISER, because they are STACKED and the summariser is not: two
# boxes at the summariser's spacing would be 712 u of head above a 356 u box. These are the same three
# numbers (BB_VPAD / BB_SUBH / BB_SUBSEP) with the air taken out, and they are now per-box rather than
# global so one pair can be tight while the other stays as it was. HEAD_VPAD has a FLOOR: the "denoising"
# label hangs below the last sublayer, and y1 - tap works out to exactly vpad, so anything under
# MIN_SEG + the label's height pushes that text through the box's own outline.
PAD_TOP      = 16.0      # INSIDE margin ABOVE the title, all three boxes. Split out from vpad because
#                          only the bottom margin has work to do: for a head it is what the "denoising"
#                          label hangs in, and for the summariser it is the entry region under the stack.
#                          The top just held air.
HEAD_VPAD    = 32.0
HEAD_SUBH    = 34.0
HEAD_SUBSEP  = 14.0
HEAD_GAP     = 56.0      # between the two stacked heads
# The dashed group is OFF for now. Everything that draws or reserves space for it is gated on this one
# flag rather than deleted, so turning it back on is a one-word change.
LD_ON        = False
AH_BAR       = r"concat($h$, $\tau$, $a_\tau$)"   # ...the action chunk, not the state tokens
AH_OUT       = r"$\hat{a}_{t:t+K}$"                # the chunk it emits, labelled on its exit
# THE ACTION HEAD IS AN MLP, and is drawn as one. models/flow.py::FlowField builds it with arch="mlp" --
# multimodal.py never passes `arch` at all, so no config makes it anything else -- and that branch is
# literally Linear -> GELU -> Linear -> GELU -> Linear. So: one hidden-layer type repeated twice, a final
# projection to the velocity AFTER the stack, and NO residual bypass (nn.Sequential has no skips). Its
# target is one flat action_dim x chunk vector, which is why "token mixing" cannot apply here -- there is
# no token axis to mix over.
# THE OUTPUT IS A DISTRIBUTION, so it is drawn as one: N real draws from the trained prior, overplotted.
# A bell curve beside the arrow would say "distribution" without saying anything about THIS one -- where
# the draws agree the prior is confident, where they split it is not, and multimodality shows as the fan
# separating rather than merely widening. make_action_samples.py caches them (same checkpoint lineage as
# the predicted frames); the recorded commands over the same span go on top in black, so the figure shows
# the prior's spread AND whether it covers what the pilot did.
AS_W         = 340.0     # fan panel width
AS_ROW       = 80.0      # one stick axis
AS_SEP       = 20.0
AS_ALPHA     = 0.42
AS_LW        = 0.95
AS_N         = 16        # draws DRAWN. The cache holds more; 48 overplotted at raw command rate is a solid
#                          band with no structure left in it, and the spread is the thing being shown.
AS_FOLD      = True      # plot at the CHUNK's own granularity -- one point per model step, the 4 concat
#                          sub-commands averaged, which is the resolution K=32 actually refers to. The raw
#                          60 Hz chatter is real but at this size it reads as ink, not as a distribution.
AS_AXES      = ("yaw", "vertical", "lateral", "fore/aft")
SAMPLE_LAB   = "sample"  # over BOTH heads' output arrows. THE WORD DOES THE WORK: a head emits a
#                          distribution, and rather than trying to draw one in a notation built for single
#                          values (tried: per-cell colour spread -- the colormap is a hue cycle, so a
#                          sorted stack reads as stripes, not as a ramp), the output is ONE DRAW in the
#                          ordinary action notation and the arrow says what it is.
AS_STYLE     = "tiles"   # fan | tiles. `tiles` draws each sampled chunk in the SAME notation the input
#                          action stream uses -- the action tokenizer's own cell tile, normalised against
#                          the same lo/hi, so a sampled command is coloured exactly as a recorded one --
#                          which makes the output read as "more of the same object" rather than as a plot.
#                          It shows FEWER draws and no spread; `fan` is the quantitative one.
AS_TILES     = 8         # tiles style: chunk steps shown. The chunk is 32 long, so this is the first
#                          8 of it -- the head emits a whole chunk where the decoders emit one step.
AS_TPITCH    = 1.30      # tiles style: horizontal pitch, in cell widths
AS_SCALE     = 2.0       # tiles style: cell size relative to the INPUT stream's. Bigger on purpose -- the
#                          cell carries a whole marginal now, not one value, and 16 slivers inside a
#                          32 u cell are 6 px at print size.
AH_STEPS     = 64        # the ACTION head's own sampling_steps. NOT DT_STEPS: the dynamics flow runs 6
#                          (diffusion.sampling_steps) and the action prior runs 64
#                          (action_head.sampling_steps) -- two different heads, two different budgets.
AH_SUBS      = ("linear + GELU",)
AH_DEPTH     = 2                              # ...the two hidden layers
AH_TAIL      = (r"linear $\to$ velocity",)    # the third Linear, which is why box() grew `tail` bars
# The two transformers are ONE thing -- the latent dynamics model -- so they are wrapped in a dashed
# group. Nothing else in the figure is dashed, which is what makes the grouping read as a grouping.
LD_TITLE     = "Latent dynamics"
LD_FS        = ENC_FS * 1.5   # the GROUP title outranks the box titles, so it is set half again as big
LD_PAD       = 44.0      # clearance from the boxes to the dashed outline (and to the group label)
LD_DASH      = (0.0, (11.0, 9.0))
X0_CLEAR     = BLK_S     # how far short of the face x_0's head stops, along its own 45
X0_DIAG      = 150.0     # length of x_0's 45 arrival -- the same run the tokenizer feeds use
BB_GAP       = 200.0     # brace below the column -> box top (raised: the boxes no longer line up
#                          with the tokenizers/decoders, so they can float clear of the token column)
BB_PAD       = 20.0      # inside margin (horizontal)
BB_VPAD      = 48.0      # inside margin TOP AND BOTTOM -- the boxes are taller than their contents need
BB_SUBH      = 40.0      # height of one sublayer box
BB_SUBSEP    = 22.0      # gap between sublayer boxes -- where the residual arcs leave and rejoin
BB_ARC       = 26.0      # width of the channel the residual bypass arcs run down, left of the boxes
BB_BRACE     = 16.0      # how far right of the sublayer boxes the "x 4" brace line sits
BB_FS        = 14.0
BADGE_R      = 26.0      # the circled number in each head's top-left corner: 1 observation, 2 action,
BADGE_FS     = 20.0      #   3 reward -- the order the paper introduces them in
# The flow head's ENTRY BAR. flow.py refuses cond="adaln": every input is CONCATENATED once, here, and
# after that it is a plain transformer. Linear(544 -> 128) over [x_tau 128 | tau_emb 32 | cond 384].
# TWO bars, because they are two different operations: the concatenation (whose arguments are the whole
# point) and the projection that turns 544 back into d. in_dim = dz 128 + time_dim 32 + h_dim 384.
DT_BARS      = (r"concat($h$, $\tau$, $x_\tau$)",)
DT_BAR_SEP   = 26.0      # must clear the SHORT arrowhead between the stacked boxes
DT_LOOP      = 28.0      # width of the feedback channel, INSIDE the box, right of the "x N" brace.
#                          NARROWED 2026-09-14: the channel's own width was budgeted into the box, but the
#                          "x <steps>" label sitting to its RIGHT never was -- so "x 6" fit by luck and
#                          "x 64" (the action head's real step count) ran out through the outline. The
#                          label is now asserted to fit, and the loop is pulled in to make the room.
DT_STEPS     = 6         # sampling steps at eval
X0_HIT       = 1.0       # where on the predicted column's brace x_0 lands: 1 = its bottom CORNER      # sized so the widest label ("temporal") clears its box -- asserted, not eyeballed
ENC_SEP      = 32.0      # vertical gap BETWEEN tokenizer blocks (vision stays put; the others move to it)
MIN_SEG      = 8.0       # no line segment may be shorter than this
RISER_PAD    = 26.0      # first riser sits this far right of ALL content
RISER_STEP   = 20.0      # risers are staggered so two stems never share a vertical
ARM_SLOPE    = float(np.tan(np.deg2rad(67.5)))   # |dy/dx| of a tokenizer-output / decoder-input arm: a
#                          67.5-degree run buys the same drop for 0.41x the horizontal, which is what keeps
#                          the figure from widening every time the token column gets taller
STEM_SLOPE   = 0.5       # |dy/dx| of the SLICE STEM's diagonal -- the climb from the token column up
#                          into the summariser. The mirror of ARM_SLOPE's reasoning: those arms are
#                          horizontally constrained, so they buy drop cheaply with a steep run; this one
#                          is VERTICALLY constrained, because every unit of horizontal offset at 45 costs
#                          a unit of height above the column and the pair sits 307 u to the left. At 2:1
#                          the same offset costs half the climb, which is ~150 u of empty page.
ARM_DIAG     = 1e9       # tokenizer/decoder arms spend EVERYTHING on the diagonal: horizontal -> 45 -> horizontal,
#                          never a vertical riser. route() falls back to a right angle only if the
#                          horizontal run is genuinely too short to fit the drop at 45 degrees.
ARROW_L      = 26.0      # arrowhead length
ARROW_DEG    = 22.5      # half-angle at the tip: each barb meets the shaft at this angle

# STROKE GEOMETRY. Every outline is stroked CENTRED on its path, so the ink reaches half a line width
# OUTSIDE the shape. An arrow that stops at the geometric edge therefore lands in the middle of the
# outline. figsize is CANVAS/100 inches with xlim = CANVAS, so one data unit is DPI/100 px and one point
# is DPI/72 px -- the ratio is DPI-free.
U_PER_PT     = 100.0 / 72.0


def half_stroke(lw):
    """How far outside its path a `lw`-point stroke reaches, in data units. An arrow tip must stop
    exactly this far short to sit FLUSH AGAINST the outline instead of on top of it."""
    return lw * U_PER_PT / 2.0

# Vertical spacing between modalities is NOT a free constant. Each stream's HIGHEST BRACE POINT sits a
# fixed gap below the BOTTOM OF THE LOWEST ITEM of the stream above, and that gap is the same number as the
# horizontal distance between the proprioception riser and the right edge of the last image item -- so the
# figure has one spacing rhythm in both axes instead of two unrelated ones. Derived in solve_layout().
MARGIN       = 16.0      # slack so halos and outlines are never clipped
DPI          = 300

STRIDE = int(re.search(r"\n  subsample: (\d+)", open(glob.glob(f"{RUN}/checkpoints/config.resolved.yaml")[0]).read()).group(1))
ALPHAS = np.linspace(ALPHA_LO, 1.0, N)
SLOPE = OFFSET_Y / OFFSET_X                              # the cascade angle, as a slope
ENC_SLOPE = np.tan(np.arctan(SLOPE) / 2.0)               # tokenizer taper: HALF the cascade ANGLE

# ----------------------------------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------------------------------
import imageio.v3 as iio3                                                          # noqa: E402
import pyarrow.parquet as pq                                                       # noqa: E402

_tab = pq.read_table(sorted(glob.glob(f"{ROOT}/train/data/**/*.parquet", recursive=True))[0]).to_pydict()
_ep = np.asarray(_tab["episode_index"]); _keep = np.flatnonzero(_ep == _ep[0])
_obs = np.stack([np.asarray(x, np.float32) for x in _tab["observation_vector"]])[_keep]
_act = np.stack([np.asarray(x, np.float32) for x in _tab["action"]])[_keep]
IDX = [END - (N - 1 - i) * STRIDE * PANEL_EVERY for i in range(N)]
assert IDX[0] >= 0, f"sequence starts before the episode: {IDX[0]}"
_vid = sorted(glob.glob(f"{ROOT}/train/videos/**/*.mp4", recursive=True))[0]
_all = [f for k, f in enumerate(iio3.imiter(_vid, plugin="pyav")) if k <= IDX[-1]]
FRAMES = [_all[i] for i in IDX]
OBS = _obs[IDX][:, :PROPRIO_CELLS]
ACT = _act[IDX]                                          # raw 4-D joystick command
print(f"{N} items | model step {1000 * STRIDE / FPS:.0f} ms | items {1000 * STRIDE * PANEL_EVERY / FPS:.0f} ms apart "
      f"| span {(N - 1) * STRIDE * PANEL_EVERY / FPS:.2f} s | frames {IDX[0]}..{IDX[-1]} "
      f"| obs {OBS.shape} act {ACT.shape}")

# ----------------------------------------------------------------------------------------------------
# DRAWING PRIMITIVES
# ----------------------------------------------------------------------------------------------------
def rounded_path(pts, r):
    """Open polyline with interior corners rounded to radius ~r. A stroked polyline's `round` joinstyle
    only rounds by half the LINE WIDTH, so it cannot honour a radius; this trims each corner back by r
    along both edges and joins them with a quadratic through the vertex.

    `r` may be a SEQUENCE, one radius per interior corner, so a single corner can stay sharp -- which is
    what the predicted column's brace needs: x_0 meets it exactly at its bottom corner, and a rounded
    corner would curve away from the very point the join is made at."""
    pts = [np.asarray(q, float) for q in pts]
    rs = list(r) if np.iterable(r) else [float(r)] * max(0, len(pts) - 2)
    verts, codes = [pts[0]], [Path.MOVETO]
    for i in range(1, len(pts) - 1):
        prev, cur, nxt = pts[i - 1], pts[i], pts[i + 1]
        u_in, u_out = cur - prev, nxt - cur
        l_in, l_out = np.linalg.norm(u_in) or 1.0, np.linalg.norm(u_out) or 1.0
        d = min(rs[i - 1], 0.5 * l_in, 0.5 * l_out)
        verts += [cur - d * u_in / l_in, cur, cur + d * u_out / l_out]
        codes += [Path.LINETO, Path.CURVE3, Path.CURVE3]
    verts.append(pts[-1]); codes.append(Path.LINETO)
    return Path(verts, codes)


def rounded_polygon(pts, r):
    """CLOSED polygon with every corner rounded to radius ~r, as a Path.

    Used for BOTH the fill and the outline, which is the point: stroking a sharp polygon with a
    round-joined pen rounds the FILL by half the pen width but leaves the outline's corners sharp, so the
    two disagree. One rounded path means the blue edge follows exactly the shape it encloses."""
    P = [np.asarray(q, float) for q in pts]
    n = len(P)
    verts, codes = [], []
    for i in range(n):
        prev, cur, nxt = P[i - 1], P[i], P[(i + 1) % n]
        u_in, u_out = cur - prev, nxt - cur
        l_in, l_out = np.linalg.norm(u_in) or 1.0, np.linalg.norm(u_out) or 1.0
        d = min(r, 0.5 * l_in, 0.5 * l_out)
        verts.append(cur - d * u_in / l_in)
        codes.append(Path.MOVETO if i == 0 else Path.LINETO)
        verts += [cur, cur + d * u_out / l_out]
        codes += [Path.CURVE3, Path.CURVE3]
    verts.append(verts[0]); codes.append(Path.CLOSEPOLY)
    return Path(verts, codes)


_MEAS = plt.figure(figsize=(1, 1), dpi=DPI)


def text_extent(txt, fs, **kw):
    """Rendered size of a string in DATA UNITS (1 unit = DPI/100 px). Measured, not estimated -- the
    backbone box is exactly the token column's width, so every label in it is sized against a real
    extent instead of a guess."""
    t = _MEAS.text(0, 0, txt, fontsize=fs, **kw)
    bb = t.get_window_extent(_MEAS.canvas.get_renderer())
    t.remove()
    return bb.width * 100.0 / DPI, bb.height * 100.0 / DPI


def muted(rgba):
    return tuple(c * (1 - MUTE) + MUTE for c in rgba[:3])


def image_item(ax, x, y, w, h, i, a, z):
    ax.imshow(FRAMES[i], extent=(x, x + w, y + h, y), alpha=a, zorder=z, interpolation="bilinear")
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=VIS_EC, lw=LW * 2, alpha=a, zorder=z + 0.5))


def palette(name):
    """Every colour one modality needs, all pulled off its own map so they cannot drift apart: `ec` for
    outlines (tokenizer, decoder, block), `fc` the tint those outlines are filled with, and `blk` for the
    block's three visible faces -- the SAME hue at three lightnesses, which is what makes an isometric
    block read as one solid rather than three shapes."""
    cm = plt.get_cmap(name)
    return dict(ec=cm(0.78), fc=cm(0.04), blk=tuple(cm(v) for v in (0.06, 0.18, 0.34)))


def vector_item_factory(mat, arrows=None, cmap=None, cmap_vec=None):
    """PER-ELEMENT normalisation: each dim is scaled across its own N items, so colour shows how that dim
    CHANGES rather than which dims are largest. `arrows` (N, D) -> a compass arrow inside each cell."""
    lo, hi = mat.min(0, keepdims=True), mat.max(0, keepdims=True)
    flat = (hi - lo) < 1e-9
    norm = np.where(flat, 0.5, (mat - lo) / np.where(flat, 1.0, hi - lo))
    cm = plt.get_cmap(cmap)
    cmv = plt.get_cmap(cmap_vec) if cmap_vec else cm     # draw_vec -- the model's OWN output -- may differ
    D = mat.shape[1]

    def _c(u, m=None):                                   # normalised value -> cell colour, off the low end
        return muted((m or cm)(CMAP_LO + (CMAP_HI - CMAP_LO) * float(np.clip(u, 0.0, 1.0))))

    def draw(ax, x, y, w, h, i, a, z):
        for k in range(D):
            cy0 = y + k * CELL
            ax.add_patch(Rectangle((x, cy0), CELL, CELL, fc=_c(norm[i, k]),
                                   ec=EDGE, lw=LW, alpha=a, zorder=z))
            if arrows is not None:
                ang, L = arrows[i, k], CELL * 0.54
                cx, cy = x + CELL / 2, cy0 + CELL / 2
                dx, dy = L * np.sin(ang), -L * np.cos(ang)       # y is inverted: -cos points N
                ax.add_patch(FancyArrow(cx - dx / 2, cy - dy / 2, dx, dy, width=LW * 0.8,
                                        head_width=CELL * 0.30, head_length=CELL * 0.26,
                                        length_includes_head=True, fc=EDGE, ec=EDGE, alpha=a, zorder=z + 0.4))
        ax.add_patch(Rectangle((x, y), CELL, D * CELL, fill=False, ec=EDGE, lw=LW, alpha=a, zorder=z + 0.5))

    def draw_vec(ax, x, y, w, h, vec, a, z, ang=None, hatch=None, horizontal=False,
                 hatch_col=None):
        """Same tile for a vector that is NOT one of the N items -- the model's prediction. Normalised with
        the STREAM's lo/hi, so its colours mean the same thing as the inputs' do.

        `horizontal` lays the cells along x instead of y. The predicted observation leaves its decoder on
        a horizontal arrow, and a column standing on the end of that arrow reads as a wall rather than as
        the thing the arrow delivers -- flat, it sits ON the arrow, and it matches how the dataset figure
        draws the same vector."""
        v = np.asarray(vec, float)[:D][None, :]
        u = np.where(flat, 0.5, (v - lo) / np.where(flat, 1.0, hi - lo))[0]
        for k in range(D):
            cx0, cy0 = (x + k * CELL, y) if horizontal else (x, y + k * CELL)
            ax.add_patch(Rectangle((cx0, cy0), CELL, CELL, fc=_c(u[k], cmv),
                                   ec=EDGE, lw=LW, alpha=a, zorder=z))
            if hatch:                                    # the stripes, in their own hue and alpha
                ax.add_patch(Rectangle((cx0, cy0), CELL, CELL, fc="none", lw=0.0,
                                       ec=hatch_rgba(hatch_col or EDGE), hatch=hatch, zorder=z + 0.2))
            if ang is not None:                          # same glyph the stream's own tiles carry
                a_, L = ang[k], CELL * 0.54
                cx, cy = cx0 + CELL / 2, cy0 + CELL / 2
                dx, dy = L * np.sin(a_), -L * np.cos(a_)
                ax.add_patch(FancyArrow(cx - dx / 2, cy - dy / 2, dx, dy, width=LW * 0.8,
                                        head_width=CELL * 0.30, head_length=CELL * 0.26,
                                        length_includes_head=True, fc=EDGE, ec=EDGE, alpha=a,
                                        zorder=z + 0.4))
        ww, hh = (D * CELL, CELL) if horizontal else (CELL, D * CELL)
        ax.add_patch(Rectangle((x, y), ww, hh, fill=False, ec=EDGE, lw=LW, alpha=a, zorder=z + 0.5))

    return draw, CELL, D * CELL, draw_vec


def brace_between(p_a, p_b, lift, u=None, off=(0.0, 1.0)):
    """THE brace primitive: a brace running from corner p_a to corner p_b, at the CONSTANT cascade angle
    -- never the angle of the line joining the two corners, so every brace in the figure is drawn at the
    same tilt and only the length differs. `lift` is a SIGNED VERTICAL offset of the brace line from p_a:
    negative puts the brace above the thing it marks, positive below it. Vertical, not along the normal:
    a normal offset has an x-component and would slide the brace sideways by a content-dependent amount.

    `u` is that constant direction (default: the cascade's) and `off` the unit axis `lift` runs along
    (default: straight down the page). The "x 4" brace beside the backbone's sublayers is the one case
    that is not at the cascade angle -- it groups two stacked boxes, so it is vertical with a sideways
    offset, which is exactly u=(0,1), off=(1,0).

    Returns (polyline nib->line->nib, midpoint of the line). Used by the data cascades from above, the
    token column from below, and the sublayer pair from the right -- one function."""
    if u is None:
        dl = float(np.hypot(OFFSET_X, OFFSET_Y))
        u = (OFFSET_X / dl, OFFSET_Y / dl)
    dx, dy = u
    A0 = (p_a[0] + lift * off[0], p_a[1] + lift * off[1])

    def proj(q):
        t = (q[0] - A0[0]) * dx + (q[1] - A0[1]) * dy
        return (A0[0] + t * dx, A0[1] + t * dy)

    def nib(end, corner):                                # runs to the corner, stopping BRACE_T short
        vx, vy = corner[0] - end[0], corner[1] - end[1]
        L = float(np.hypot(vx, vy)) or 1.0
        k = max(0.0, L - BRACE_T) / L
        return (end[0] + k * vx, end[1] + k * vy)

    a, b = proj(p_a), proj(p_b)
    return [nib(a, p_a), a, b, nib(b, p_b)], ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def brace_segs(pw, i0, i1):
    """The cascade's brace over items i0..i1 inclusive, for any modality, whatever its item size: pick the
    two corners, then solve the lift that clears every item under the span."""
    p_a = (i0 * OFFSET_X, i0 * OFFSET_Y)                 # top-left of item i0
    p_b = (i1 * OFFSET_X + pw, i1 * OFFSET_Y)            # top-right of item i1

    def clearance(xa, xb):
        out = 0.0
        for t in np.linspace(0.0, 1.0, 256):
            x = xa + t * (xb - xa)
            j = max(0, int(np.ceil((x - pw) / OFFSET_X)))   # earliest item whose body reaches this x
            out = max(out, (p_a[1] + (x - p_a[0]) * SLOPE) - j * OFFSET_Y)
        return out + BRACE_M

    lift = clearance(p_a[0], p_b[0])
    for _ in range(2):                                   # lift and projection are solved together
        poly, mid = brace_between(p_a, p_b, -lift)
        lift = clearance(poly[1][0], poly[2][0])
    return poly, mid


def lower_profile(L, x):
    """Bottom of a stream at horizontal position x: the LOWEST item covering x, or None if none does."""
    if x < 0:
        return None
    j = min(L["n"] - 1, int(np.floor(x / OFFSET_X)))
    if x > j * OFFSET_X + L["pw"]:
        return None
    return j * OFFSET_Y + L["ph"]


def upper_profile(L, x):
    """Top of a stream at x: the highest item top there, or the brace line if a brace spans x."""
    j = max(0, int(np.ceil((x - L["pw"]) / OFFSET_X)))
    if j > L["n"] - 1:
        return None
    top = j * OFFSET_Y
    for i0, i1 in L["braces"]:
        poly, _ = brace_segs(L["pw"], i0, i1)
        a, b = poly[1], poly[2]                          # the brace diagonal
        if min(a[0], b[0]) <= x <= max(a[0], b[0]):
            t = (x - a[0]) / (b[0] - a[0])
            top = min(top, a[1] + t * (b[1] - a[1]))
    return top


def stream_clearance(above, below):
    """Smallest vertical distance between two streams measured AT THE SAME x, which is what the eye reads
    as the gap. Measuring bottom-of-one against top-of-the-other ignores x entirely: the vision stream is
    lowest at its LAST item while the next stream's brace is highest at its FIRST, ~800 units to the left,
    so a nominally small gap there leaves a huge empty wedge between them."""
    best = None
    for x in np.linspace(0.0, (N - 1) * OFFSET_X + max(above["pw"], below["pw"]), 600):
        lo, up = lower_profile(above, x), upper_profile(below, x)
        if lo is None or up is None:
            continue
        d = up - lo
        best = d if best is None else min(best, d)
    return best if best is not None else 0.0


def route(f, t, xv, diag_room=0.0, slope=1.0):
    """THE router for every arrow in the figure -- brace stems into tokenizers and tokenizer feeds out to
    blocks alike. One function, three shapes, chosen by how much horizontal room the caller says a
    diagonal may use:

      diag_room = 0   horizontal, a RIGHT-ANGLE riser at x = xv, horizontal. The brace stems get this:
                      their riser has to sit right of every cascade, so there is nowhere to put a
                      diagonal without cutting back across the items. The rounded corners carry it.
      diag_room > 0   the diagonal is made as LONG as that allows. If it can absorb the whole drop it is
                      one clean run, centred; otherwise it runs INTO the vertical at xv, so the descent
                      starts as early (and as far from whatever is below the source) as it can.

    `slope` is |dy/dx| of that diagonal: 1.0 is 45 degrees, tan(67.5) is the steep variant. A steeper
    diagonal buys the same drop for LESS horizontal run, which is the whole reason it exists -- the gaps
    either side of the token column are sized by the drop their arms have to absorb, so at 45 degrees a
    tall column makes the figure wide.
    """
    f, t = np.asarray(f, float), np.asarray(t, float)
    dy = t[1] - f[1]
    run = abs(dy)
    if run < MIN_SEG:                                    # no height to lose: straight in
        return [tuple(f), tuple(t)], "straight"
    diag = min(run, diag_room, max(0.0, xv - f[0] - MIN_SEG) * slope)
    if diag < MIN_SEG:                                   # no room for a diagonal: a plain rounded corner
        return [tuple(f), (xv, f[1]), (xv, t[1]), tuple(t)], "right-angle"
    nm = "45" if abs(slope - 1.0) < 1e-9 else f"{np.degrees(np.arctan(slope)):.4g}deg"
    if diag >= run - 1e-9:                               # the diagonal absorbs the whole drop
        x0 = f[0] + (t[0] - f[0] - run / slope) / 2.0    # centred in the horizontal run
        return [tuple(f), (x0, f[1]), (x0 + run / slope, t[1]), tuple(t)], nm
    s = float(np.sign(dy))
    return ([tuple(f), (xv - diag / slope, f[1]), (xv, f[1] + s * diag), (xv, t[1]), tuple(t)],
            nm + " + vertical")


def arrow_head(tip, direction=(1.0, 0.0), L=None):
    """Triangle at `tip` pointing along `direction`, barbs meeting the shaft at ARROW_DEG each (so the
    half-width is L*tan(ARROW_DEG), not L as a 45-degree head would give). ONE head for every arrow in
    the figure: the horizontal stem feeds and the 45-degree tokenizer->block feeds alike."""
    L = ARROW_L if L is None else L
    d = np.asarray(direction, float)
    d = d / (np.linalg.norm(d) or 1.0)
    n = np.array([-d[1], d[0]])                          # unit normal
    h = L * float(np.tan(np.deg2rad(ARROW_DEG)))
    t = np.asarray(tip, float)
    return [tuple(t), tuple(t - d * L - n * h), tuple(t - d * L + n * h)]


def draw_arrow(ax, ax_pts, z, *, tip=None, head_dir=(1.0, 0.0), head_len=None, lw=None, halo=True,
               r=CORNER_R):
    """EVERY line in the figure goes through here -- brace outlines, stems, tokenizer feeds, the backbone
    trunk and its residual arcs. White halo, then the black line, then a fill-only head. One place, so a
    line cannot end up without the halo, or with a different corner radius than the one beside it.

    halo=False only where the lines belong to ONE connected path (the backbone's trunk and the residual
    arcs that leave and rejoin it): there a halo would break the very line it is meant to protect, and
    inside the box there is nothing to protect it from."""
    lw = LW * 2.0 if lw is None else lw
    pth = rounded_path(ax_pts, r)
    if halo:
        ax.add_patch(PathPatch(pth, fill=False, ec="white", lw=lw + LW * (BRACE_HALO - 2.0),
                               capstyle="round", joinstyle="round", zorder=z - 0.1))
    ax.add_patch(PathPatch(pth, fill=False, ec=EDGE, lw=lw, capstyle="round", joinstyle="round", zorder=z))
    if tip is not None:
        ax.add_patch(head_patch(tip, head_dir, z + 0.1, L=head_len))


def draw_arrows(ax, items, z, *, lw=None, halo=True):
    """Draw a GROUP of polylines so their halos cannot erase one another: ALL halos first, then all lines,
    then all heads. draw_arrow's per-line halo sits at z-0.1, which is correct for an isolated arrow but
    wrong for a set of connected ones -- the later halo punches a hole in the earlier line. Every item is
    {pts, tip?, head_dir?, head_len?, lw?}."""
    lw0 = LW * 2.0 if lw is None else lw
    paths = [(rounded_path(it["pts"], CORNER_R), it.get("lw", lw0)) for it in items]
    if halo:
        for pth, w in paths:
            ax.add_patch(PathPatch(pth, fill=False, ec="white", lw=w + LW * (BRACE_HALO - 2.0),
                                   capstyle="round", joinstyle="round", zorder=z - 0.2))
    for pth, w in paths:
        ax.add_patch(PathPatch(pth, fill=False, ec=EDGE, lw=w, capstyle="round", joinstyle="round",
                               zorder=z))
    for it in items:
        if it.get("tip") is not None:
            ax.add_patch(head_patch(it["tip"], it.get("head_dir", (1.0, 0.0)), z + 0.1,
                                    L=it.get("head_len")))


def head_patch(tip, direction=(1.0, 0.0), z=0.0, L=None):
    """The head as a FILL WITH NO EDGE. A stroked triangle mitres at its tip and the mitre runs on for
    half_width / sin(ARROW_DEG) -- at 22.5 degrees that is 2.6x the half width, which is how the tips
    ended up several pixels inside the block outline even after being told to stop short of it."""
    return Polygon(arrow_head(tip, direction, L), closed=True, fc=EDGE, ec="none", zorder=z)


# The block's receding axes use the SAME angle as the data streams, not a textbook 30 degrees, so every
# diagonal in the figure -- cascade, brace, tokenizer taper, block -- is one of two parallel families.
_ISO_L = float(np.hypot(OFFSET_X, OFFSET_Y))
ISO_C, ISO_S = OFFSET_X / _ISO_L, OFFSET_Y / _ISO_L


def route_down(f, t, yv, diag_room=0.0):
    """route() transposed: for an arrow that leaves DOWNWARD rather than rightward. Same code, x and y
    swapped on the way in and back on the way out -- there is no second router."""
    pts, kind = route((f[1], f[0]), (t[1], t[0]), yv, diag_room)
    return [(q[1], q[0]) for q in pts], kind


def ray_exit(poly, p, d):
    """Where a ray from an interior point p along d leaves a convex polygon. Used so a feed arrow starts
    FLUSH with the tokenizer's outline instead of at its bounding box, which the taper makes different."""
    p, d = np.asarray(p, float), np.asarray(d, float)
    best = None
    for i in range(len(poly)):
        a, b = np.asarray(poly[i], float), np.asarray(poly[(i + 1) % len(poly)], float)
        e = b - a
        den = d[0] * e[1] - d[1] * e[0]
        if abs(den) < 1e-9:
            continue
        t = ((a[0] - p[0]) * e[1] - (a[1] - p[1]) * e[0]) / den   # along the ray
        u = ((a[0] - p[0]) * d[1] - (a[1] - p[1]) * d[0]) / den   # along the edge
        if t > 1e-9 and -1e-9 <= u <= 1 + 1e-9 and (best is None or t < best):
            best = t
    return p + d * (best if best is not None else 0.0)


def hatch_rgba(col, a=None):
    """The hatch stroke colour for a patch: the modality's own hue, at HATCH_A."""
    r, g, b = matplotlib.colors.to_rgb(col)
    return (r, g, b, HATCH_A if a is None else a)


def iso_block(ax, tip, depth, height, width, z, fc=BLK_FC, ec=BLK_EC, hatch=None):
    """A SOLID isometric cuboid of (depth x height x width) tokens, positioned by its front face.

    Depth recedes UP-LEFT, which is the SAME LINE DIRECTION as the down-right data streams -- the long
    edges of the block are parallel to the cascade. Width mirrors it up-right; height is straight up (the
    y axis is inverted, so 'up' is -y). Three flat-shaded faces meet at the near vertical edge: the long
    depth face on the LEFT, the short width face on the RIGHT, the top across both. No internal rules --
    the counts are carried by the proportions."""
    vd = np.array([-BLK_S * ISO_C, -BLK_S * ISO_S])  # depth: up-LEFT, i.e. PARALLEL to the cascade
    vw = np.array([BLK_S * ISO_C, -BLK_S * ISO_S])   # width: up-right, the mirror of it
    vh = np.array([0.0, -BLK_S])                     # height (up)
    # ANCHOR: the middle of the FRONT face's LEFTMOST EDGE sits on the arrow tip. The front face is the
    # long depth x height one (B below); depth runs up-left, so its leftmost edge is the far vertical edge
    # at a = depth, whose midpoint is O + vd*depth + vh*height/2.
    O = np.asarray(tip, float) - vd * depth - vh * (height / 2.0)

    def face(p0, u, nu, v, nv, face_fc):
        ax.add_patch(Polygon([p0, p0 + u * nu, p0 + u * nu + v * nv, p0 + v * nv],
                             closed=True, fc=face_fc, ec=ec, lw=BLK_LW, joinstyle="round",
                             hatch=hatch, zorder=z))

    # All three faces meet at the near vertical edge O..O+vh, which is what makes it read as a solid:
    # the LONG depth face up-left, the SHORT width face down-right, the top across the two.
    face(O + vh * height, vd, depth, vw, width, fc[0])              # A: top
    face(O, vd, depth, vh, height, fc[1])                           # B: FRONT, long, on the left
    face(O, vw, width, vh, height, fc[2])                           # C: short, on the right
    if BLK_LABELS:
        for tag, c in (("A", O + vh * height + vd * (depth / 2) + vw * (width / 2)),
                       ("B", O + vd * (depth / 2) + vh * (height / 2)),
                       ("C", O + vw * (width / 2) + vh * (height / 2))):
            ax.text(c[0], c[1], tag, ha="center", va="center", fontsize=11, color="#1a3a5c",
                    fontweight="bold", zorder=z + 0.5)


def tokenizer_pts(cy, x0, flip=False):
    """Trapezium on its side, tapering at HALF the cascade angle. A TOKENIZER narrows left-to-right (tall
    input edge, short output edge); a DECODER is the same shape mirrored -- short edge in, tall edge out --
    so the pair reads as compress / expand. One function, one taper."""
    hi = ENC_H / 2.0
    ho = max(CORNER_R * 2.0, hi - ENC_W * ENC_SLOPE)
    a, b = (ho, hi) if flip else (hi, ho)
    return [(x0, cy - a), (x0 + ENC_W, cy - b), (x0 + ENC_W, cy + b), (x0, cy + a)]


# ----------------------------------------------------------------------------------------------------
# LAYOUT — solved ONCE, shared by every component render
# ----------------------------------------------------------------------------------------------------
# THE ARROWS ARE THE DATA. These were random angles (rng(7).choice of the eight compass directions),
# which is indefensible in a figure whose point is what the data looks like -- a reader takes a glyph to
# mean something. Up is a positive stick, down negative, which is the same convention the dataset figure
# uses, so the two figures agree.
_arrows = np.where(ACT >= 0.0, 0.0, np.pi)
_p_prop, _vw, _vh_p, _p_prop_vec = vector_item_factory(OBS, cmap=MOD_CMAP["proprio"])
_p_act, _, _vh_a, _p_act_vec = vector_item_factory(ACT, arrows=_arrows, cmap=MOD_CMAP["action"],
                                                  cmap_vec=ACT_OUT_CMAP)

# THE PREDICTION the figure shows as its output: made by make_prediction.py, which rolls the real world
# model from the 8 consecutive strided frames ending at the last braced tile. Falls back to the RECORDED
# frame if that has not been run -- and says so, because a figure captioned "predicted" that is quietly
# showing ground truth is the worst of both.
try:
    _P = np.load(f"{OUT}/prediction.npz")
    PRED_IMG, PRED_OBS = _P["image"], _P["proprio"]      # (steps,H,W,3) and (steps,obs) -- ONE PER RED SLICE
    print(f"  prediction: rolled {int(_P['horizon'])} steps from frames {int(_P['ctx_first'])}.."
          f"{int(_P['ctx_last'])} -> frames {list(_P['target_frames'])}")
except OSError:
    PRED_IMG, PRED_OBS = None, None
    print("  WARNING: no prediction.npz -- the outputs show the RECORDED item, not a prediction. "
          "Run make_prediction.py.")
try:
    _A = np.load(f"{OUT}/action_samples.npz")
    ACT_SMP, ACT_TRUE = _A["samples"], _A["truth"]      # (N,T,4) draws and (T,4) recorded, RAW stick units
    if AS_FOLD:                                         # (N, K*S, 4) -> (N, K, 4), one point per model step
        _k, _s = int(_A["chunk"]), int(_A["subsample"])
        ACT_SMP = ACT_SMP.reshape(len(ACT_SMP), _k, _s, ACT_SMP.shape[-1]).mean(axis=2)
        ACT_TRUE = ACT_TRUE[: (len(ACT_TRUE) // _s) * _s].reshape(-1, _s, ACT_TRUE.shape[-1]).mean(axis=1)
    ACT_SMP = ACT_SMP[:AS_N]
    print(f"  action prior: {ACT_SMP.shape[0]} draws of {ACT_SMP.shape[1]} commands "
          f"(chunk {int(_A['chunk'])} x subsample {int(_A['subsample'])}) from frame {int(_A['ctx_last'])}")
except OSError:
    ACT_SMP, ACT_TRUE = None, None
    print("  WARNING: no action_samples.npz -- the action head's output shows no distribution. "
          "Run make_action_samples.py.")

LAYERS = [
    dict(draw=image_item, pw=IMG_W, ph=IMG_W * FRAMES[0].shape[0] / FRAMES[0].shape[1], n=N_STATE,
         braces=BRACES, label="Vision\ntokenizer", blks=[(BLK_DEPTH, IMG_TOKENS, 1)],
         blkz=[Z_BLOCK["vision"]], **palette(MOD_CMAP["vision"])),
    dict(draw=_p_prop, pw=_vw, ph=_vh_p, n=N_STATE, braces=BRACES, label="Proprio\ntokenizer",
         blks=[(BLK_DEPTH, 1, 1)], blkz=[Z_BLOCK["proprio"]], **palette(MOD_CMAP["proprio"])),
    # the action tokenizer emits TWO: the 8 in-window action tokens, and a 1-step nub that is the raw
    # act_enc channel feeding the flow head directly (it bypasses the backbone -- see _cond)
    dict(draw=_p_act, pw=_vw, ph=_vh_a, n=N, braces=BRACES_ACT, label="Action\ntokenizer",
         blks=[(BLK_DEPTH, 1, 1), (NUB_DEPTH, 1, 1)], blkz=[Z_BLOCK["action"], Z_BLOCK["action_extra"]],
         **palette(MOD_CMAP["action"])),
]


def solve_layout():
    """Per-layer offsets and the canvas, solved once so every component lands in the same place.

    The tokenizers do NOT sit at their own brace's height: VISION keeps its natural position and the
    other two are packed ENC_SEP below it, so the three blocks form a tight column. Their stems reach
    them via a 45-degree section (see route_45)."""
    spans = []
    for L in LAYERS:                                     # items + braces; stems and tokenizers come after
        xs, ys = [0.0, L["pw"] + OFFSET_X * (L["n"] - 1)], [0.0, L["ph"] + OFFSET_Y * (L["n"] - 1)]
        for i0, i1 in L["braces"]:
            for q in brace_segs(L["pw"], i0, i1)[0]:
                xs.append(q[0]); ys.append(q[1])
        # THE ARROW OF TIME runs PARALLEL to the cascades -- which is parallel to every brace, since they
        # all sit at the one angle -- and BELOW the bottom stream. Above the top one was the obvious
        # place and it does not work: a brace's stem leaves its midpoint HORIZONTALLY for the tokenizer,
        # so a line that is parallel to the brace but descending necessarily crosses that stem further
        # right. Below the last cascade nothing runs at all. Vertical offset, not a normal one, for the
        # reason brace_between gives: a normal offset slides sideways and the two stop looking parallel.
        if L is LAYERS[-1]:
            _bot = lambda x: x * SLOPE + L["ph"]         # the line through the items' bottom-left corners
            _drop = max([0.0] + [i * OFFSET_Y + L["ph"] + BRACE_T
                                 + text_extent(m, ENC_FS * 1.3)[1] - _bot(i * OFFSET_X + L["pw"] / 2)
                                 for i, m in ASTERISKS]) + TIME_M
            _x1 = (L["n"] - 1) * OFFSET_X + L["pw"]
            L["time_seg"] = ((0.0, _bot(0.0) + _drop), (_x1, _bot(_x1) + _drop))
            ys += [L["time_seg"][0][1], L["time_seg"][1][1]]
        # THE ASTERISK HANGS BELOW ITS ITEM and has to be counted. It is slack while the action stream
        # runs past the marked item (something else is then always lower), but the moment a stream ends at
        # a marked step the mark becomes the lowest ink in the figure -- and leaving it out of the span
        # clips it off the bottom of the canvas, which is exactly what happened.
        if L is LAYERS[-1] and ASTERISKS:
            ys.append(max(i for i, _ in ASTERISKS) * OFFSET_Y + L["ph"] + BRACE_T
                      + max(text_extent(m, ENC_FS * 1.3)[1] for _, m in ASTERISKS))
        spans.append([min(xs), min(ys), max(xs), max(ys)])
    x_min, x_max = min(s[0] for s in spans), max(s[2] for s in spans)
    enc_x = x_max + ENC_GAP
    j = 0                                                # one riser x per stem, left to right
    for L in LAYERS:
        L["riser_x"] = []
        for _ in L["braces"]:
            L["riser_x"].append(x_max + RISER_PAD + j * RISER_STEP); j += 1
    assert enc_x - ARROW_L - (x_max + RISER_PAD + (j - 1) * RISER_STEP) >= MIN_SEG, (
        f"ENC_GAP={ENC_GAP} too small: the last riser leaves < {MIN_SEG} before the arrowhead")
    # ONE SPACING, MEASURED not chosen: the horizontal distance between the PROPRIOCEPTION riser and the
    # right edge of the last image item. That same number is then the vertical gap between the lowest
    # point of each stream and the horizontal stem line of the stream below it, so the figure breathes at
    # one rate in both axes and cannot drift if the riser stagger changes.
    GAP = LAYERS[1]["riser_x"][0] - ((LAYERS[0]["n"] - 1) * OFFSET_X + LAYERS[0]["pw"])
    # STREAM_GAP was tied to the horizontal riser gap, on the principle that one gap should appear
    # everywhere. That principle costs a third of the figure's height: the three cascades descend
    # diagonally and stack, so the action stream ends up 600 units below the bottom of the action token
    # block, and the arrow of time below that again. STREAM_FRAC compresses the VERTICAL spacing only.
    STREAM_GAP = GAP * STREAM_FRAC
    item_bottom = [L["ph"] + OFFSET_Y * (L["n"] - 1) for L in LAYERS]
    brace_top = [min(q[1] for i0, i1 in L["braces"] for q in brace_segs(L["pw"], i0, i1)[0]) for L in LAYERS]
    # STREAM SPACING IS THE ORIGINAL RULE -- one measured gap from the lowest point of each stream to the
    # stem line of the next. Aligning every stream to its own tokenizer instead (tried 2026-09-13) makes
    # every stem horizontal but DESTROYS that spacing: the tokenizer pitch is ENC_H + ENC_SEP, which is
    # unrelated to how tall a cascade is, and proprioception and action ended up overlapping.
    LAYERS[0]["sy0"] = -spans[0][1]
    for k, L in enumerate(LAYERS):
        loc = [brace_segs(L["pw"], i0, i1)[1][1] for i0, i1 in L["braces"]]
        L["stem_mid_local"] = 0.5 * (min(loc) + max(loc))
        if k:
            L["sy0"] = (item_bottom[k - 1] + LAYERS[k - 1]["sy0"]) + STREAM_GAP - min(loc)
    _ACT_STEM = LAYERS[-1]["stem_mid_local"] + LAYERS[-1]["sy0"]   # what the tokenizer column hangs off
    stack_top = min(s[1] + L["sy0"] for L, s in zip(LAYERS, spans))
    stack_bot = max(s[3] + L["sy0"] for L, s in zip(LAYERS, spans))
    globals()["_GAP"] = GAP
    # TOKENIZER COLUMN AT THE BOTTOM (2026-09-13). It used to sit at whatever height each modality's own
    # brace stem implied, which put it level with the cascades; the figure now reads bottom-to-top --
    # cascades fold DOWN into the tokenizers, the tokenizers feed UP into the token column, the column
    # feeds UP into the transformers -- so the whole column is packed below every cascade instead.
    cy = None
    for k, L in enumerate(LAYERS):
        ys = [brace_segs(L["pw"], i0, i1)[1][1] + L["sy0"] for i0, i1 in L["braces"]]
        L["stem_ys"] = ys
        # THE COLUMN IS ANCHORED TO THE ACTION STREAM, whose stem is then dead horizontal, and stacks
        # UPWARD from there. The action stream is the one that has to line up: it carries two braces and
        # the longest run. Vision and proprioception keep a vertical component in their stems, which is
        # fine -- what is not fine is moving the CASCADES to suit the tokenizers, which is what aligning
        # all three did, and it collapsed the spacing between them.
        cy = (_ACT_STEM - (len(LAYERS) - 1) * (ENC_H + ENC_SEP)) if k == 0 else cy + ENC_H + ENC_SEP
        L["enc_cy_abs"] = cy
        # PARALLEL STEMS: the group is TRANSLATED to the tokenizer, not converged onto one point, so two
        # stems from the same modality keep exactly the spacing they had at their braces, all the way in.
        span = max(ys) - min(ys)
        room = ENC_H - 2 * MIN_SEG                       # usable height of the input edge
        k_sp = 1.0 if span <= room or span < 1e-9 else room / span
        if k_sp < 1.0:
            L["span_note"] = (span, span * k_sp)         # reported, never silent
        mid_y = 0.5 * (min(ys) + max(ys))
        L["entry_abs"] = [cy + (y - mid_y) * k_sp for y in ys]
    # Token blocks sit DOWN-RIGHT of their tokenizer, reached by a 45-degree arrow that STARTS FLUSH with
    # the tokenizer's outline (found by ray-casting to the trapezium, since its taper means the boundary
    # is not the bounding box) and ENDS at the midpoint of the block's front face.
    dfeed = np.array([1.0, 1.0]) / np.sqrt(2.0)
    hi = ENC_H / 2.0
    ho = max(CORNER_R * 2.0, hi - ENC_W * ENC_SLOPE)     # half-height of the tokenizer's RIGHT edge
    blk_w = max(BLK_S * ISO_C * (d + w) for L in LAYERS for d, _h, w in L["blks"])
    for L in LAYERS:
        n = len(L["blks"])
        cy = L["enc_cy_abs"]
        L["feed_abs"] = []
        # ONE feed per tokenizer, leaving from the MIDDLE of its right edge. A tokenizer that emits more
        # than one block (the action nub) still emits one ARROW: the extra blocks are continuations of the
        # same stream, and the column is aligned so that arrow is dead straight.
        L["feed_abs"] = [(enc_x + ENC_W, cy)] * n
    # ONE COLUMN: the three blocks are stacked flush in height (32 + 1 + 1 = the 34-token bag), sharing x
    # and depth, so they read as a single solid. Only the vision block is placed by its feed arrow; the
    # others are derived from it, and the action's extra step continues along the depth axis.
    vdu = np.array([-BLK_S * ISO_C, -BLK_S * ISO_S])
    vhu = np.array([0.0, -BLK_S])
    O = np.asarray(LAYERS[0]["feed_abs"][0]) + dfeed * BLK_DIAG - vdu * LAYERS[0]["blks"][0][0] \
        - vhu * (LAYERS[0]["blks"][0][1] / 2.0)          # origin of the vision block
    for k, L in enumerate(LAYERS):
        d, h, w = L["blks"][0]
        if k > 0:
            # Shift by THIS block's height: O is a block's BOTTOM, so putting its top on the previous
            # block's bottom means moving down by its own height, not by the previous one's.
            O = O - vhu * h
        L["tip_abs"] = [tuple(O + vdu * d + vhu * (h / 2.0))]
        L["blk_O"] = [O.copy()]
        for j in range(1, len(L["blks"])):                # the nub continues along the DEPTH axis
            dj, hj, wj = L["blks"][j]
            tip = np.asarray(L["tip_abs"][0]) - vdu * d
            L["tip_abs"].append(tuple(tip))
            L["blk_O"].append(tip - vdu * dj - vhu * (hj / 2.0))
    # The column must not make the figure taller: drop it until its top is level with whatever the rest
    # of the figure already reaches. The arrow SOURCES do not move -- only the blocks -- and the vertical
    # risers absorb the difference.
    base_top = min([stack_top] + [L["enc_cy_abs"] - ENC_H / 2 for L in LAYERS])
    col_top = min(Oj[1] - BLK_S * (h + ISO_S * (d + w)) + BLK_S * ISO_S * (d + w)
                  for L in LAYERS for (d, h, w), Oj in zip(L["blks"], L["blk_O"]))
    col_top = min(Oj[1] - BLK_S * h for L in LAYERS for (d, h, w), Oj in zip(L["blks"], L["blk_O"]))
    col_top = min(col_top, min(Oj[1] - BLK_S * h - BLK_S * ISO_S * d
                               for L in LAYERS for (d, h, w), Oj in zip(L["blks"], L["blk_O"])))
    # The column's top aligns with the top of everything else. It no longer chases the action feed: with
    # the tokenizers moved to the bottom their feeds run UP into it, so no feed can be straight anyway.
    # NO max(0,...) any more: with the tokenizers at the bottom the feed-derived position lands far too
    # low, so the column has to be able to move UP as well as down to sit level with the cascades.
    # THE COLUMN'S BOTTOM IS FLUSH WITH THE BOTTOM OF THE FIGURE. Its top used to be pinned instead,
    # which left it floating with the whole lower band empty. The red slices hang BELOW the blue blocks --
    # each step is one depth cell further down-right, and the second block of the last slice is the lowest
    # point of the whole assembly -- so their extent is folded in here analytically, because PRED itself is
    # not built until after this shift is applied.
    # THE COLUMN IS PLACED BY THE ACTION FEED: shifted so the action block's arrival point is at exactly
    # the action tokenizer's centre height, which makes that one feed dead horizontal. The bottom-flush
    # rule it replaces is still computed, and reported, so the cost in empty floor is visible.
    red_drop = PRED_STEPS * BLK_S * ISO_S + PRED_BLKS[-1][0] * BLK_S
    col_bot = max([Oj[1] for L in LAYERS for Oj in L["blk_O"]] + [LAYERS[0]["blk_O"][0][1] + red_drop])
    floor_y = max(stack_bot, max(L["enc_cy_abs"] for L in LAYERS) + ENC_H / 2.0)
    shift = LAYERS[-1]["feed_abs"][0][1] - LAYERS[-1]["tip_abs"][0][1] + BLK_DROP
    globals()["COL_LIFT"] = (floor_y - col_bot + BLK_DROP) - shift
    for L in LAYERS:
        L["tip_abs"] = [(x, y + shift) for x, y in L["tip_abs"]]
        L["blk_O"] = [Oj + np.array([0.0, shift]) for Oj in L["blk_O"]]
    # ...and now push the column far enough RIGHT that every feed can spend its whole drop on a 45 with
    # no vertical riser left over. The needed run is the largest drop plus the arrowhead and two minimum
    # segments; the column's x was previously fixed by BLK_DIAG alone, which is unrelated to the drops.
    need_dx = max(abs(L["feed_abs"][j][1] - L["tip_abs"][j][1]) / ARM_SLOPE + ARROW_L + 3 * MIN_SEG
                  for L in LAYERS for j in range(len(L["blks"])))
    have_dx = min(L["tip_abs"][0][0] - L["feed_abs"][0][0] for L in LAYERS)
    if need_dx > have_dx:
        dx_ = need_dx - have_dx
        for L in LAYERS:
            L["tip_abs"] = [(x + dx_, y) for x, y in L["tip_abs"]]
            L["blk_O"] = [Oj + np.array([dx_, 0.0]) for Oj in L["blk_O"]]
    for L in LAYERS:
        L["blk_box"] = []
        for (d, h, w), Oj in zip(L["blks"], L["blk_O"]):
            pts = [Oj + vdu * a + np.array([BLK_S * ISO_C, -BLK_S * ISO_S]) * b + vhu * c
                   for a in (0, d) for b in (0, w) for c in (0, h)]
            L["blk_box"].append((min(q[0] for q in pts), min(q[1] for q in pts),
                                 max(q[0] for q in pts), max(q[1] for q in pts)))
    # THE PREDICTED STEP, in red: depth 1, flush against the column's near (down-right) face, its top
    # level with the column's. PRED_H = 33 (not 34) because predict_next emits state tokens only.
    PRED, boxes = [], []
    for step in range(PRED_STEPS):                       # one slice per predicted step, down-right of the last
        Op = LAYERS[0]["blk_O"][0] - vdu * (step + 1)
        for j, (hj, dj) in enumerate(PRED_BLKS):
            if j:                                        # O is a block's BOTTOM: drop by ITS OWN height
                Op = Op - vhu * hj
            PRED.append(dict(O=Op.copy(), d=dj, h=hj, w=1,
                             tip=tuple(Op + vdu * dj + vhu * (hj / 2.0))))
            pts = [Op + vdu * a + np.array([BLK_S * ISO_C, -BLK_S * ISO_S]) * b + vhu * c
                   for a in (0, dj) for b in (0, 1) for c in (0, hj)]
            boxes.append((min(q[0] for q in pts), min(q[1] for q in pts),
                          max(q[0] for q in pts), max(q[1] for q in pts)))
    globals()["PRED"] = PRED
    globals()["PRED_BOX"] = (min(b[0] for b in boxes), min(b[1] for b in boxes),
                             max(b[2] for b in boxes), max(b[3] for b in boxes))
    # DECODERS: one per red block, on the far side of it. Each is centred on its block, so the arrow in
    # is horizontal, and each emits the REAL item 10 -- the step the red block predicts.
    face_red = PRED_BOX[2]                               # the red assembly's FAR right edge
    # Both decoder arms are horizontal -> 45 -> horizontal, so each gap has to be at least as long as the
    # drop it must absorb -- the same rule the tokenizer feeds get. Sizing these by a fixed constant is
    # what made the vision decode arm fall back to a riser the moment the token cell grew.
    _n_b = len(PRED_BLKS)
    _src_y = [PRED[-_n_b + i]["tip"][1] for i in range(2)]
    _dec_cy = [LAYERS[i]["enc_cy_abs"] for i in range(2)]
    _out_cy = [PRED_AT * OFFSET_Y + LAYERS[i]["sy0"] + LAYERS[i]["ph"] / 2.0 for i in range(2)]
    _pad = ARROW_L + 3 * MIN_SEG + RISER_STEP
    dec_x = face_red + max(DEC_GAP, max(abs(a - b) for a, b in zip(_src_y, _dec_cy)) / ARM_SLOPE + _pad)
    out_x = dec_x + ENC_W + max(OUT_GAP, max(abs(a - b) for a, b in zip(_dec_cy, _out_cy)) + _pad)
    DEC = []
    n_b = len(PRED_BLKS)
    for k, (P, lab, draw, ow, oh) in enumerate(((PRED[-n_b], DEC_LABELS[0], image_item, IMG_W,
                                                 IMG_W * FRAMES[0].shape[0] / FRAMES[0].shape[1]),
                                                (PRED[-n_b + 1], DEC_LABELS[1], _p_prop, _vw, _vh_p))):
        # SAME HEIGHT AS ITS TOKENIZER, so encode and decode face each other across the figure
        cy = LAYERS[k]["enc_cy_abs"]
        # ...and the OUTPUT leaves the page at the height its INPUT entered it, on the far left
        out_cy = PRED_AT * OFFSET_Y + LAYERS[k]["sy0"] + LAYERS[k]["ph"] / 2.0
        # ON the face, both of them, so the two tails line up exactly as the tokenizer feeds' do. Nudging
        # them clear of the outline instead (to stop the halo cutting a notch in it) broke that alignment;
        # the halo is dealt with by Z ORDER -- these arrows are drawn UNDER the blocks, see Z_DECODE.
        DEC.append(dict(cy=cy, out_cy=out_cy, label=lab, draw=draw, ow=ow, oh=oh,
                        src=(face_red, P["tip"][1])))   # leaves the red block at the BLOCK's height
    globals()["DEC"], globals()["DEC_X"], globals()["OUT_X"] = DEC, dec_x, out_x
    globals()["DEC_BOX"] = (dec_x, min([d["cy"] - ENC_H / 2 for d in DEC]
                                       + [d["out_cy"] - d["oh"] / 2 for d in DEC]),
                            out_x + (PRED_STEPS - 1) * OFFSET_X + max(d["ow"] for d in DEC),
                            max([d["cy"] + ENC_H / 2 for d in DEC]
                                + [d["out_cy"] + d["oh"] / 2 + (PRED_STEPS - 1) * OFFSET_Y for d in DEC]))
    blk_x = max([b[2] for L in LAYERS for b in L["blk_box"]] + [PRED_BOX[2]])
    # THE BACKBONE BOX. Exactly the column's width, directly below it. The trunk runs down the middle of
    # the sublayer boxes; the residual arcs use the channel on the left.
    col_x0 = min(b[0] for L in LAYERS for b in L["blk_box"])
    # A BRACE UNDER THE COLUMN, marking the 8 steps being sliced out. Same primitive as the cascades'
    # braces, just with the lift the other way: its two corners are the ends of the action block's bottom
    # DEPTH edge, which runs at the cascade angle like everything else.
    O = LAYERS[-1]["blk_O"][0]
    vdu = np.array([-BLK_S * ISO_C, -BLK_S * ISO_S])
    # ABOVE the column now: the transformers sit at the top, so the slice is marked on the column's TOP
    # edge and its stem runs UP into them. Same primitive, lift the other way (see brace_between).
    # THE FAR edge of the top face, not the near one: the top face rises up-RIGHT from the near edge by
    # one width cell, so the far edge is the outline the brace should sit on -- and measuring the standard
    # BRACE_M lift from it is what puts the brace clear of the face instead of across it.
    vwu = np.array([BLK_S * ISO_C, -BLK_S * ISO_S])                    # width: up-RIGHT
    O_top = (LAYERS[0]["blk_O"][0] + vhu * LAYERS[0]["blks"][0][1]
             + vwu * LAYERS[0]["blks"][0][2])                          # VISION block's FAR top corner
    bpoly, bmid = brace_between(tuple(O_top + vdu * LAYERS[0]["blks"][0][0]), tuple(O_top), -BRACE_M)
    _brace_top = min(q[1] for q in bpoly)               # the boxes sit ABOVE the slice brace
    # THE TWO TRANSFORMER BOXES. Same width, same height, same internal grammar -- only `flow` differs.
    # Width and height are SOLVED from measured text, not chosen: the sublayer labels must fit their
    # boxes and the trunk must clear the vertical title it runs past. The pair grows LEFT from the token
    # column, whose right edge the backbone stays flush with.
    def metrics(title, subs, repeat, bars=DT_BARS, tail=(), vpad=BB_VPAD, subh=BB_SUBH,
                subsep=BB_SUBSEP, barsep=DT_BAR_SEP, vtop=PAD_TOP, steps=DT_STEPS):
        ttl_w, ttl_h = text_extent(title, BOX_FS, linespacing=1.3)
        x4 = f"\u00d7 {repeat}"
        x4_w, _ = text_extent(x4, ENC_FS * 0.62)
        w_lab = max(text_extent(nm, BB_FS)[0] for nm in subs)
        w_lab = max([w_lab] + [text_extent(t, BB_FS)[0] for t in tuple(bars) + tuple(tail)])
        sub_w = w_lab + 4 * MIN_SEG                        # box width, sized to the WIDEST label it holds
        # The row is an ASSEMBLY -- residual channel, boxes, brace, "x N" -- and the flow head adds a loop
        # channel. Sizing the box off the assembly (and then CENTRING it) is what stops the row from being
        # pinned left with the leftover width dumped on the right.
        # THE LOOP'S OWN STEP COUNT IS PART OF THE ROW. It sits to the right of the feedback channel, and
        # budgeting the channel but not the label is what let "x 64" run out through the outline while
        # "x 6" fit by luck -- the two differ by 14 u and the margin was 7.5.
        x4s_w = text_extent(f"\u00d7 {steps}", ENC_FS * 0.62)[0]
        asm = {f: BB_ARC + sub_w + BB_BRACE + BRACE_T + x4_w
                  + ((DT_LOOP + MIN_SEG + x4s_w) if f < 0 else 0) for f in (+1, -1)}
        return dict(ttl_w=ttl_w, ttl_h=ttl_h, x4=x4, x4_w=x4_w, sub_w=sub_w, asm=asm,
                    vpad=vpad, vtop=vtop, subh=subh, subsep=subsep, barsep=barsep,
                    w=max(2 * BB_PAD + max(asm.values()), 2 * BB_PAD + ttl_w),
                    h=(vtop + vpad + ttl_h + 3 * subsep
                       + len(bars) * subh + (len(bars) - 1) * barsep
                       + len(subs) * subh + (len(subs) - 1) * subsep
                       + len(tail) * (subh + subsep)))

    _tight = dict(vpad=HEAD_VPAD, subh=HEAD_SUBH, subsep=HEAD_SUBSEP)
    m_bb = metrics(BB_TITLE, BB_SUBS, BB_DEPTH)
    m_dt = metrics(DT_TITLE, DT_SUBS, DT_DEPTH, steps=DT_STEPS, **_tight)
    m_ah = metrics(AH_TITLE, AH_SUBS, AH_DEPTH, bars=(AH_BAR,), tail=AH_TAIL, steps=AH_STEPS,
                   **_tight)
    w = max(blk_x - col_x0, m_bb["w"], m_dt["w"], m_ah["w"])   # at least the token column's own width
    # the two heads no longer have the same contents, so they no longer have the same height --
    # the action head carries an extra bar. They are stacked, not paired, so that is fine.

    def box(m, x1, y0, subs, repeat, title, flow, bars=DT_BARS, exit_kind="x0", tail=(),
            residual=True, steps=DT_STEPS, glyph=None):
        """Both boxes read TOP-DOWN internally, with the title band at the top, because the pair now sits
        ABOVE the token column and the data arrives from below.
          flow = +1  (space-time)     enters at the BOTTOM, stacks UP,   leaves RIGHT
          flow = -1  (rectified flow) enters at the LEFT,   stacks DOWN, leaves out the BOTTOM"""
        d = dict(x0=x1 - w, x1=x1, y0=y0, y1=y0 + m["h"], subs=subs, title=title, flow=flow,
                 x4=m["x4"], x4_fs=ENC_FS * 0.62, x4_steps=f"\u00d7 {steps}",
                 bars_txt=bars, exit=exit_kind,
                 vpad=m["vpad"], subh=m["subh"], subsep=m["subsep"], barsep=m["barsep"],
                 tail_txt=tail, residual=residual, glyph=glyph)
        if flow > 0:                                     # the BOXES themselves are centred in the box
            d["sub_x0"] = d["x0"] + 0.5 * (w - m["sub_w"])
            d["arc_x"] = d["sub_x0"] - BB_ARC
        else:                                            # the whole assembly is centred (it has a loop)
            left = d["x0"] + 0.5 * (w - m["asm"][flow])
            d["arc_x"] = left
            d["sub_x0"] = left + BB_ARC
        d["sub_x1"] = d["sub_x0"] + m["sub_w"]
        d["cx"] = 0.5 * (d["sub_x0"] + d["sub_x1"])
        d["loop_x"] = d["sub_x1"] + BB_BRACE + BRACE_T + m["x4_w"] + DT_LOOP / 2.0
        d["title_cy"] = d["y0"] + m["vtop"] + m["ttl_h"] / 2.0   # BOTH boxes: the band along the TOP
        # The glyph shares the title's band, tucked into the right margin the centred title leaves. Only
        # its X is fixed here: the Y is read off title_cy at DRAW time, because the boxes get shifted twice
        # after this (the x_0 lift, then the canvas offset) and a y cached now would be left behind -- which
        # is exactly what happened: the marks drew above the boxes and were clipped off the canvas.
        if glyph:
            assert d["x1"] - BB_PAD - GLYPH_W >= 0.5 * (d["x0"] + d["x1"]) + m["ttl_w"] / 2.0 + MIN_SEG, (
                f"{title.splitlines()[0]}: the distribution glyph collides with the title")
        span = len(subs) * m["subh"] + (len(subs) - 1) * m["subsep"]
        y = d["title_cy"] + m["ttl_h"] / 2.0 + m["subsep"]
        if flow > 0:                                       # enters BOTTOM, stacks UP, leaves RIGHT
            d["out_y"] = y                                 # the exit line, ABOVE the stack it drains
            # the stack sits LOW in the box, just above the entry, not tucked under the exit line
            d["sub_y"] = d["y1"] - m["vpad"] - m["subh"]
            d["exit_tip"] = (d["x1"] + ARROW_L * 1.6, d["out_y"])
        else:                                              # enters LEFT at the bar, stacks DOWN, leaves BOTTOM
            d["bars"] = []
            for _ in bars:
                d["bars"].append((d["sub_x0"], y, d["sub_x1"], y + m["subh"]))
                y += m["subh"] + m["barsep"]
            d["out_y"] = 0.5 * (d["bars"][0][1] + d["bars"][0][3])   # h enters at the CONCAT bar
            d["sub_y"] = y - m["barsep"] + m["subsep"]        # top of sublayer 0, the HIGHEST one
            # TRAILING BARS sit BELOW the sublayer stack -- an MLP's final projection comes after its
            # hidden layers, not before them -- and the sampling loop then closes below THOSE.
            _t0 = d["sub_y"] + len(subs) * (m["subh"] + m["subsep"])
            d["tail"] = [(d["sub_x0"], _t0 + i * (m["subh"] + m["subsep"]),
                          d["sub_x1"], _t0 + i * (m["subh"] + m["subsep"]) + m["subh"])
                         for i in range(len(tail))]
            d["loop_y"] = _t0 + len(tail) * (m["subh"] + m["subsep"])
            d["exit_tip"] = (d["cx"], d["y1"] + ARROW_L * 1.6)
        return d

    # The gap between the two boxes is centred on the MIDPOINT OF THE COLUMN'S SHORT RIGHT FACE: the
    # pair slides left or right until (bb.x1 + dt.x0) / 2 lands on it. That face is the (width x height)
    # one, so its midpoint sits half a cell out along the up-right axis from the near vertical edge O.
    ocol = LAYERS[0]["blk_O"][0]
    w_cells = LAYERS[0]["blks"][0][2]
    face_x = ocol[0] + BLK_S * ISO_C * w_cells / 2.0
    f_top = min(Oj[1] - BLK_S * h for L in LAYERS for (d, h, _w), Oj in zip(L["blks"], L["blk_O"]))
    f_bot = max(Oj[1] for L in LAYERS for Oj in L["blk_O"])
    face_mid = (face_x, 0.5 * (f_top + f_bot) - BLK_S * ISO_S * w_cells / 2.0)
    globals()["FACE_MID"] = face_mid
    # ONE TARGET PER PREDICTED SLICE: the MIDDLE OF THE TOP FACE of that slice's IMAGE block (index
    # step * n_b -- the top of the stack, the face that is actually exposed). These are the face centres
    # THEMSELVES; the X0_CLEAR standoff is applied where the arrow is drawn, because it has to run along
    # whatever direction that arrow arrives on and the arrow is no longer always a 45.
    def _face_tgt(P):
        return (P["O"][0] + BLK_S * ISO_C * (P["w"] - P["d"]) / 2.0,
                P["O"][1] - BLK_S * P["h"] - BLK_S * ISO_S * (P["w"] + P["d"]) / 2.0)

    globals()["X0_TGTS"] = [_face_tgt(PRED[st * len(PRED_BLKS)]) for st in range(PRED_STEPS)]
    globals()["X0_TGT"] = X0_TGTS[0]                     # the first slice still sets the box placement
    # THE FLOW HEAD'S TRUNK SITS X0_DIAG TO THE RIGHT OF THE FACE IT FEEDS, so x_0's last leg is a 45 of
    # exactly that length and nothing else. The trunk is the sublayer centre, not the box centre, so the
    # box's x is solved from it: box() is linear in x, so one probe at 0 gives the offset from the box's
    # right edge to its trunk.
    _dt_x1 = X0_TGT[0] + X0_DIAG - box(m_dt, 0.0, 0.0, DT_SUBS, DT_DEPTH, DT_TITLE, -1)["cx"]
    # HOW FAR ABOVE THE BRACE THE PAIR SITS IS DERIVED, NOT CHOSEN. The slice stem climbs from the brace
    # to the summariser's bottom edge as vertical -> 45 -> vertical, so it needs at least as much VERTICAL
    # room as the horizontal offset it has to cover. That offset is now set by where the flow head's trunk
    # has to be (over the predicted face), which moved the pair 150 u left -- and BB_GAP, a fixed number
    # chosen when the pair sat further right, stopped being enough. Measure the offset and take BB_GAP as
    # a FLOOR: box() is linear in x, so a probe at y=0 gives the summariser's trunk x.
    _bb_cx = box(m_bb, _dt_x1 - w - DT_GAP, 0.0, BB_SUBS, BB_DEPTH, BB_TITLE, +1)["cx"]
    _need = STEM_SLOPE * abs(_bb_cx - bmid[0]) + ARROW_L + 2 * MIN_SEG
    globals()["_STEM_DX"] = abs(_bb_cx - bmid[0])
    bb_y1 = _brace_top - max(BB_GAP, _need)
    # THE TWO HEADS ARE STACKED, action head ON TOP. The observation head keeps the bottom slot because
    # its x_0 is the one that has to reach the token column: put it above and that spine would have to
    # thread past the action head to get there. They share an x, so their left edges line up and the h
    # bus can serve both from one vertical.
    dt = box(m_dt, _dt_x1, bb_y1 - m_dt["h"], DT_SUBS, DT_DEPTH, DT_TITLE, -1, glyph=GLYPH_OBS)
    dt["badge"] = 1
    ah = box(m_ah, _dt_x1, dt["y0"] - HEAD_GAP - m_ah["h"], AH_SUBS, AH_DEPTH, AH_TITLE, -1,
             bars=(AH_BAR,), tail=AH_TAIL, exit_kind="stub", residual=False, steps=AH_STEPS,
             glyph=GLYPH_ACT)
    ah["badge"] = 2
    # The arrow carries the notation ONLY when the draws are missing: with the panel there, the caption
    # under it says the same thing properly, and a HAT on the arrow would claim a point estimate.
    ah["out_lab"] = "" if ACT_SMP is not None else AH_OUT
    # ...and the summariser is centred ON THE PAIR, by its own exit line rather than by its box, so h
    # leaves exactly halfway between the two entries it feeds and the bus splits symmetrically.
    _mid_out = 0.5 * (ah["out_y"] + dt["out_y"])
    bb = box(m_bb, dt["x0"] - DT_GAP, _mid_out - (m_bb["vtop"] + m_bb["ttl_h"] + m_bb["subsep"]),
             BB_SUBS, BB_DEPTH, BB_TITLE, +1)
    bb["brace"], bb["mid"] = bpoly, bmid
    assert abs(bb["out_y"] - _mid_out) < 1e-6, "the summariser's h is not centred on the two heads"
    globals()["STEM_DY"] = [abs(L["stem_mid_local"] + L["sy0"] - L["enc_cy_abs"]) for L in LAYERS]
    assert dt["x0"] - bb["x1"] >= DT_GAP - 1e-6, (
        f"the transformer boxes overlap: gap {dt['x0'] - bb['x1']:.0f} u < DT_GAP {DT_GAP:.0f}")
    _g0 = np.array([-1.0, 1.0]) / np.sqrt(2.0)           # x_0 arrives down-LEFT onto the face
    dt["exit_tip"] = tuple(np.asarray(X0_TGT) - _g0 * X0_CLEAR)
    # x_0 must ARRIVE on the diagonal, not flatten back to a vertical, so the whole horizontal gap has to
    # be spent on the 45 -- which needs at least that much vertical room between the box and the block.
    # Rather than hope the gap is big enough, RAISE the stack until it is. Measured from the box's BOTTOM
    # EDGE, not from the loop tap inside it: the spine is only free to branch once it is out of the box.
    _dd = dt["cx"] - dt["exit_tip"][0]                   # the 45 needs this much vertical room too
    _BOX_LIFT = max(0.0, dt["y1"] + MIN_SEG - (dt["exit_tip"][1] - _dd - MIN_SEG))
    if _BOX_LIFT:
        for D in (bb, dt, ah):
            for kk in ("y0", "y1", "title_cy", "sub_y", "out_y"):
                D[kk] -= _BOX_LIFT
            for kk in ("bars", "tail"):
                if kk in D:
                    D[kk] = [(b[0], b[1] - _BOX_LIFT, b[2], b[3] - _BOX_LIFT) for b in D[kk]]
            if "loop_y" in D:
                D["loop_y"] -= _BOX_LIFT
        for D in (bb, ah):                               # dt's exit_tip is the BLOCK: it does not move
            D["exit_tip"] = (D["exit_tip"][0], D["exit_tip"][1] - _BOX_LIFT)
    globals()["_BOX_LIFT"] = _BOX_LIFT
    # ONE h, TWO HEADS: the summariser's exit carries a tip per head and draw_box fans them off a shared
    # bus. Both tips are flush on a left edge, never on top of one.
    # THE ACTION CHUNK LEAVES SIDEWAYS. Straight down out of the box is where the grammar would put it,
    # but the observation head is now directly below, and an arrow pointing into the top of that box says
    # the action head feeds it -- which is exactly backwards. So the stub drops out of the box and turns
    # RIGHT into the gap between the two, where there is nothing to be confused with.
    ah["exit_tip"] = (ah["x1"] + ARROW_L * 1.6, ah["loop_y"])
    bb["fan"] = [(dt["x0"] - half_stroke(ENC_LW), dt["out_y"]),
                 (ah["x0"] - half_stroke(ENC_LW), ah["out_y"])]
    bb["exit_tip"] = bb["fan"][0]
    # THE FAN PANEL. Its LEFT EDGE IS OUT_X -- the same x the predicted frames and the predicted proprio
    # vector start at -- so the three things the model emits line up down the right-hand side instead of
    # each floating at whatever x its own arrow happened to end at. The action head's exit arrow is then
    # stretched to reach it, rather than the panel being hung off the arrow.
    _as_h = (len(AS_AXES) * AS_ROW + (len(AS_AXES) - 1) * AS_SEP if AS_STYLE == "fan"
             else _vh_a + (AS_TILES - 1) * OFFSET_Y)
    _as_w = AS_W if AS_STYLE == "fan" else CELL + (AS_TILES - 1) * OFFSET_X
    if ACT_SMP is not None:
        # FLUSH, AND ON THE FIRST TILE'S CENTRE -- the same contract the decoders' output arrows have
        # ("the item starts exactly at the tip", pointing at the middle of the first one). So the tip is
        # out_x itself, not short of it, and the cascade is hung so that its FIRST tile straddles the
        # arrow; the box's own centre is lower than that, because the cascade falls away down-right.
        ah["exit_tip"] = (out_x, ah["exit_tip"][1])
        assert ah["exit_tip"][0] > ah["x1"] + ARROW_L, (
            "the action head's output arrow has no room to reach the predicted column's left edge")
        _as_y0 = ah["exit_tip"][1] - (_vh_a if AS_STYLE == "tiles" else _as_h) / 2.0
        globals()["AS_BOX"] = (out_x, _as_y0, out_x + _as_w, _as_y0 + _as_h)
    else:
        globals()["AS_BOX"] = None
    globals()["BB"], globals()["DT"], globals()["AH"] = bb, dt, ah
    _ld_h = text_extent(LD_TITLE, LD_FS)[1]
    globals()["LD"] = dict(x0=bb["x0"] - LD_PAD, x1=dt["x1"] + LD_PAD,
                           y0=bb["y0"] - 2 * LD_PAD - _ld_h,      # LD_PAD above the text AND below it
                           y1=max(bb["y1"], dt["y1"]) + LD_PAD,
                           label_cy=bb["y0"] - LD_PAD - _ld_h / 2.0)
    # A feed needs no riser x of its own: route() puts its vertical (if it needs one) right next to the
    # block, which is what makes the 45 as long and as early as the gap allows.
    _ld_box = [LD["y0"], LD["y1"]] if LD_ON else []   # off -> it reserves no space either
    _as_y = ([AS_BOX[1], AS_BOX[3] + 2 * MIN_SEG + text_extent("$a$", BB_FS)[1]] if AS_BOX else [])
    top = min([stack_top] + [L["enc_cy_abs"] - ENC_H / 2 for L in LAYERS]
              + [b[1] for L in LAYERS for b in L["blk_box"]] + [PRED_BOX[1], DEC_BOX[1],
                                                                 BB["y0"], DT["y0"], AH["y0"]]
              + _ld_box[:1] + _as_y[:1])
    # LIFT THE THREE STREAMS AS A GROUP. They descend diagonally and the arrow of time hangs below the
    # last one, so the cascades -- not the right-hand column -- set the bottom of the canvas, 248 units
    # below the lowest ink anything else contributes. Lift them until the arrow is level with that, and
    # the figure loses that height outright. The tokenizers do NOT move, so every stem re-routes: they
    # are drawn from the brace midpoint (which moves with sy) to the tokenizer entry (which does not),
    # and route() re-solves each one. The lift is capped by the headroom above the topmost cascade.
    bot_ns = max([L["enc_cy_abs"] + ENC_H / 2 for L in LAYERS]
                 + [b[3] for L in LAYERS for b in L["blk_box"]]
                 + [BB["y1"], DT["y1"], AH["y1"], PRED_BOX[3], DEC_BOX[3]] + _ld_box[1:] + _as_y[1:])
    # 24 units of that lift is given back, at the author's ask: flush with the lowest other ink
    # read as slightly too high.
    lift = float(np.clip(stack_bot - bot_ns - 24.0, 0.0, max(0.0, stack_top - top)))
    globals()["STREAM_LIFT"] = lift
    bot = max(stack_bot - lift, bot_ns)
    if RM_ON:
        # THE REWARD MODEL HANGS BELOW THE TOKEN COLUMN, and after the stream lift the column is flush
        # with the bottom of the canvas -- measured, zero free height. So reserve its band here, which
        # is what makes the figure ~250 units taller than it would otherwise be, and the only place the
        # block can go without crossing something.
        # THE ANGLED BRACE DROPS AS IT RUNS. It is drawn at the cascade angle, so its midpoint sits
        # half a span times the slope BELOW where a flat brace would -- leaving that out clipped the
        # bottom of the box off the canvas.
        _rm_span = max(b[2] for b in LAYERS[1]["blk_box"]) - min(b[0] for b in LAYERS[-1]["blk_box"])
        _rm_h = RM_GAP_BR + 0.5 * _rm_span * SLOPE + RM_GAP_BOX + 2 * RM_PAD \
                + text_extent(RM_TITLE, BOX_FS)[1] + 2 * text_extent(RM_LAT, BB_FS)[1] * 2.5 \
                + RM_ROW_SEP + 4 * MIN_SEG
        bot = max(bot, max([b[3] for L in LAYERS for b in L["blk_box"]] + [PRED_BOX[3]]) + _rm_h)
    for L in LAYERS:
        L["sx"], L["sy"] = MARGIN - x_min, MARGIN - top + L["sy0"] - lift
        L["enc_cy"] = L["enc_cy_abs"] + MARGIN - top
        L["entry"] = [y + MARGIN - top for y in L["entry_abs"]]
        L["feed"] = [(x, y + MARGIN - top) for x, y in L["feed_abs"]]
        L["tip"] = [(x, y + MARGIN - top) for x, y in L["tip_abs"]]
        L["z"] = LAYERS.index(L) * 100
    dy = MARGIN - top
    for D in (BB, DT, AH):
        for k in ("y0", "y1", "title_cy", "sub_y", "out_y"):
            D[k] += dy
        if "loop_y" in D:
            D["loop_y"] += dy
        for kk in ("bars", "tail"):
            if kk in D:
                D[kk] = [(b[0], b[1] + dy, b[2], b[3] + dy) for b in D[kk]]
        D["exit_tip"] = (D["exit_tip"][0], D["exit_tip"][1] + dy)
        if "fan" in D:                                   # absolute points too -- they are box EDGES
            D["fan"] = [(t[0], t[1] + dy) for t in D["fan"]]
    for k in ("y0", "y1", "label_cy"):
        LD[k] += dy
    BB["brace"] = [(q[0], q[1] + dy) for q in BB["brace"]]
    BB["mid"] = (BB["mid"][0], BB["mid"][1] + dy)
    if AS_BOX:
        globals()["AS_BOX"] = (AS_BOX[0], AS_BOX[1] + dy, AS_BOX[2], AS_BOX[3] + dy)
    globals()["X0_TGTS"] = [(t[0], t[1] + dy) for t in X0_TGTS]
    globals()["X0_TGT"] = X0_TGTS[0]
    globals()["FACE_MID"] = (FACE_MID[0], FACE_MID[1] + dy)
    for P in PRED:
        P["tip"] = (P["tip"][0], P["tip"][1] + dy)
    for D in DEC:
        D["cy"] += dy; D["out_cy"] += dy
        D["src"] = (D["src"][0], D["src"][1] + dy)
    # THE REWARD MODEL'S OUTPUT LABEL IS THE RIGHTMOST INK, now that the block's right edge is aligned
    # with the decoders': its arrow and "Similarity score" run past that edge and were clipped.
    _rm_x = (DEC_BOX[2] + 2.6 * ARROW_L + text_extent(RM_OUT, BOX_FS * 0.8)[0] + MARGIN) if RM_ON else 0.0
    return max([blk_x + blk_w, DEC_BOX[2] + MARGIN, AH["x1"] + MARGIN, _rm_x]
               + ([LD["x1"] + MARGIN] if LD_ON else [])
               + ([AS_BOX[2] + MIN_SEG + max(text_extent(nm, BB_FS * 0.85)[0] for nm in AS_AXES)
                   + MARGIN] if AS_BOX else [])) - x_min + 2 * MARGIN, (bot - top) + 2 * MARGIN, enc_x, blk_x


CANVAS_W, CANVAS_H, ENC_X, BLK_X = solve_layout()


# ----------------------------------------------------------------------------------------------------
# RENDER
# ----------------------------------------------------------------------------------------------------
def badge(ax, cx, cy, n):
    """A circled number, for the three heads in the order the paper introduces them."""
    ax.add_patch(Circle((cx, cy), BADGE_R, fc="white", ec=EDGE, lw=LW * 1.6, zorder=Z_TOKENIZER + 5))
    ax.text(cx, cy, str(n), ha="center", va="center", fontsize=BADGE_FS, color=EDGE,
            zorder=Z_TOKENIZER + 6)


def dist_glyph(ax, modes, box, z):
    """The implicit-distribution mark, in FIGURE coordinates. `box` is (x0, y0, x1, y1) with y0 the TOP
    (the axis is inverted), so the mark scales with the box instead of being a pasted bitmap.

    BOTH flow heads get the same KIND of glyph -- a 1-D density -- because both are the same kind of
    object: a rectified flow whose terminal law is sampled, never a point estimate. What differs is the
    SHAPE, one `modes` list each, so the two marks are not interchangeable at a glance and nobody reads
    them as the same distribution. Black, not the predicted-column pink: the mark is about the block it
    sits in, not about the red slice downstream.

    `modes` is [(centre, width, weight), ...] in a frame running -3.3 .. 3.3."""
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    u = np.linspace(-3.3, 3.3, 200)
    v = sum(wt * np.exp(-((u - c) ** 2) / (2 * sd ** 2)) for c, sd, wt in modes)
    v = v / v.max()
    px, py = x0 + (u + 3.3) / 6.6 * W, y1 - v * H * 0.92
    ax.fill_between(px, y1, py, fc="#e2e2e2", ec="none", zorder=z)
    ax.plot(px, py, color=EDGE, lw=LW * 1.3, zorder=z + 1, solid_capstyle="round")
    ax.plot([x0, x1], [y1, y1], color=EDGE, lw=LW * 0.9, zorder=z + 1, solid_capstyle="round")


def draw_box(ax, D, sx):
    """ONE transformer box: rounded outline, vertical title along the leading edge, the sublayer stack
    with a residual bypass arc around each sublayer, a brace down the right carrying the repeat count,
    and the trunk. `D["flow"]` is the only difference between the two boxes:
      +1  enters at the BOTTOM, stacks UP,   leaves to the RIGHT   (the space-time backbone)
      -1  enters at the LEFT,   stacks DOWN, leaves out the BOTTOM  (the rectified-flow head)"""
    f = D["flow"]
    x0, x1, y0, y1 = D["x0"] + sx, D["x1"] + sx, D["y0"], D["y1"]
    cx, arc_x = D["cx"] + sx, D["arc_x"] + sx
    bx0, bx1 = D["sub_x0"] + sx, D["sub_x1"] + sx
    ax.add_patch(PathPatch(rounded_polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], CORNER_R),
                           fc=ENC_FC, ec=BOX_EC, lw=ENC_LW, zorder=Z_TOKENIZER))
    # THE HEADS ARE BOLD, the summariser is not: bold marks the three things the paper trains, and the
    # summariser is a component inside the first of them.
    ax.text(0.5 * (x0 + x1), D["title_cy"], D["title"], ha="center", va="center",
            fontsize=BOX_FS, color="black", linespacing=1.3, zorder=Z_TOKENIZER + 4,
            fontweight="bold" if "head" in D["title"] else "normal")
    if D.get("badge"):
        badge(ax, x0 + BB_PAD + BADGE_R, y0 + BB_PAD + BADGE_R, D["badge"])
    if D.get("glyph"):
        gx1 = x1 - BB_PAD
        # IN LINE WITH THE REPEAT LABEL, not with the title: the glyph says what the head EMITS, and
        # that reads with the denoising loop rather than with the name.
        _gy = D.get("glyph_cy", D["title_cy"])
        dist_glyph(ax, D["glyph"], (gx1 - GLYPH_W, _gy - GLYPH_H / 2.0, gx1,
                                    _gy + GLYPH_H / 2.0), Z_TOKENIZER + 4)
    ys = [D["sub_y"] - f * i * (D["subh"] + D["subsep"]) for i in range(len(D["subs"]))]
    # the trunk spans the whole run, from where the signal enters to where it leaves
    trunk = (y1, D["out_y"]) if f > 0 else (D["bars"][-1][3], D["loop_y"])
    inner = []                                           # every line inside the box: ONE halo group
    if f < 0:                                            # the trunk is its own segment
        inner.append(dict(pts=[(cx, trunk[0]), (cx, trunk[1])]))
    for nm, y in zip(D["subs"], ys):
        ax.add_patch(PathPatch(rounded_polygon([(bx0, y), (bx1, y), (bx1, y + D["subh"]),
                                                (bx0, y + D["subh"])], CORNER_R),
                               fc="white", ec=BOX_EC, lw=ENC_LW * 0.6, zorder=Z_TOKENIZER + 2))
        ax.text((bx0 + bx1) / 2, y + D["subh"] / 2, nm, ha="center", va="center",
                fontsize=BB_FS, color="black", zorder=Z_TOKENIZER + 3)
        # RESIDUAL BYPASS: leaves the trunk BEFORE the sublayer, down the channel, rejoins AFTER it.
        # Drawn only where the model HAS one -- the action head's MLP is a plain nn.Sequential, so giving
        # it bypass arcs would be inventing skips that are not in the code.
        if D["residual"]:
            a = y + D["subh"] + D["subsep"] / 2 if f > 0 else y - D["subsep"] / 2   # BEFORE the sublayer
            b = y - D["subsep"] / 2 if f > 0 else y + D["subh"] + D["subsep"] / 2   # ...rejoins AFTER it
            inner.append(dict(pts=[(cx, a), (arc_x, a), (arc_x, b), (cx - ARROW_L * 0.5, b)],
                              tip=(cx, b), head_len=ARROW_L * 0.5, lw=LW * 1.6))
    # the repeat brace, down the RIGHT of the stack: the one brace not at the cascade angle, because it
    # groups two STACKED boxes (u vertical, offset sideways)
    xp, xmid = brace_between((bx1, min(ys)), (bx1, max(ys) + D["subh"]), BB_BRACE, u=(0.0, 1.0), off=(1.0, 0.0))
    inner.append(dict(pts=xp))
    ax.text(xmid[0] + BRACE_T, xmid[1], D["x4"], ha="left", va="center", fontsize=D["x4_fs"],
            color=EDGE, zorder=Z_TOKENIZER + 4)
    etip = (D["exit_tip"][0] + (sx if f > 0 else sx), D["exit_tip"][1])
    if f > 0:                                            # up from the bottom, then out to the RIGHT
        # h FANS OUT TO EVERY HEAD. The heads are stacked, so one h cannot be a single straight line any
        # more: it runs out to a shared vertical bus in the gap and splits there, one branch per head.
        # Drawn as ONE group (draw_arrows below) because the branches all touch at the split -- separate
        # calls would let the later halo punch a hole through the earlier line right at the junction.
        tips = [(t[0] + sx, t[1]) for t in D.get("fan") or [D["exit_tip"]]]
        xv = min(t[0] for t in tips) - ARROW_L - MIN_SEG   # the bus, just clear of the heads' left edges
        inner.append(dict(pts=[(cx, trunk[0]), (cx, D["out_y"]), (xv, D["out_y"])]))
        for t in tips:
            leg = [(xv, D["out_y"])] + ([] if abs(t[1] - D["out_y"]) < MIN_SEG else [(xv, t[1])])
            inner.append(dict(pts=leg + [(t[0] - ARROW_L, t[1])], tip=t))
        kind = f"bus + {len(tips)} branch" + ("es" if len(tips) > 1 else "")
    else:                                                # in at the LEFT, down, out the BOTTOM
        bars = [(b[0] + sx, b[1], b[2] + sx, b[3]) for b in D["bars"]]
        for (bx0, by0, bx1, by1), txt in zip(bars, D["bars_txt"]):
            ax.add_patch(PathPatch(rounded_polygon([(bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1)],
                                                   CORNER_R),
                                   fc="white", ec=BOX_EC, lw=ENC_LW * 0.6, zorder=Z_TOKENIZER + 2))
            ax.text(0.5 * (bx0 + bx1), 0.5 * (by0 + by1), txt, ha="center", va="center",
                    fontsize=BB_FS, color="black", zorder=Z_TOKENIZER + 3)
        edge = half_stroke(ENC_LW * 0.6)
        for lo, hi in zip(bars, bars[1:]):               # concat -> project, with the SHORT head
            inner.append(dict(pts=[(cx, lo[1]), (cx, hi[3] + edge + ARROW_L * 0.5)],
                              tip=(cx, hi[3] + edge), head_dir=(0.0, -1.0), head_len=ARROW_L * 0.5))
        for (tx0, ty0, tx1, ty1), txt in zip([(b[0] + sx, b[1], b[2] + sx, b[3]) for b in D["tail"]],
                                             D["tail_txt"]):
            ax.add_patch(PathPatch(rounded_polygon([(tx0, ty0), (tx1, ty0), (tx1, ty1), (tx0, ty1)],
                                                   CORNER_R),
                                   fc="white", ec=BOX_EC, lw=ENC_LW * 0.6, zorder=Z_TOKENIZER + 2))
            ax.text(0.5 * (tx0 + tx1), 0.5 * (ty0 + ty1), txt, ha="center", va="center",
                    fontsize=BB_FS, color="black", zorder=Z_TOKENIZER + 3)
        cat = bars[0]                                    # the concat box: h in on the left, x_tau on the right
        inner.append(dict(pts=[(x0, D["out_y"]), (cat[0], D["out_y"])]))
        # THE SAMPLING LOOP, kept INSIDE the box, in its own channel right of the "x N" brace.
        tap = D["loop_y"]                        # the loop CLOSES below everything the box stacks
        lx = D["loop_x"] + sx
        inner.append(dict(pts=[(cx, tap), (lx, tap), (lx, D["out_y"]),
                               (cat[2] + edge + ARROW_L * 0.5, D["out_y"])],
                          tip=(cat[2] + edge, D["out_y"]), head_dir=(-1.0, 0.0),
                          head_len=ARROW_L * 0.5))
        # ...and it is set on the SAME BASELINE as the sublayer brace's own count: they are two repeat
        # counts for two nested loops, so reading them as a pair is the point. Centring this one on the
        # loop channel instead left the two floating at unrelated heights.
        assert lx + MIN_SEG + text_extent(D["x4_steps"], D["x4_fs"])[0] <= x1 - MIN_SEG, (
            f"{D['title'].splitlines()[0]}: '{D['x4_steps']}' runs past the box edge -- narrow DT_LOOP")
        ax.text(lx + MIN_SEG, xmid[1], D["x4_steps"], ha="left", va="center",
                fontsize=D["x4_fs"], color=EDGE, zorder=Z_TOKENIZER + 5)   # same styling as the "x N" brace
        D["glyph_cy"] = xmid[1]                          # ...and the glyph lines up with it
        ax.text(0.5 * (cx + lx), tap + MIN_SEG, "denoising", ha="center", va="top",
                fontsize=BB_FS, color=EDGE, zorder=Z_TOKENIZER + 5)
        # x_0 leaves the BOTTOM of the box and descends into the top faces of the predicted slices. It
        # must ARRIVE ON THE DIAGONAL: an earlier version turned a single 45 that had to cover the whole
        # horizontal gap, which needs more vertical drop than there is, so the turn landed ABOVE the box
        # and the line left through the TOP. Hence the vertical trunk first, then the 45.
        dg = np.array([-1.0, 1.0]) / np.sqrt(2.0)        # down-left
        if D["exit"] == "x0":
            # x_0 leaves the box's bottom and ARRIVES ON THE DIAGONAL, into the middle of the top face,
            # stopping X0_CLEAR short along that same 45 so the head points into the face instead of
            # covering it. The direction is DERIVED from where the face sits relative to the trunk, so a
            # slice on either side works; with PRED_STEPS > 1 the extra slices branch off the same
            # vertical spine, each on its own 45.
            tgts = [(t[0] + sx, t[1]) for t in X0_TGTS]

            def _leg(t):
                """-> (branch y on the trunk, tip, head direction) for one target."""
                dxt = t[0] - cx
                if abs(dxt) < MIN_SEG:                   # directly under the trunk: a plain vertical
                    return t[1] - X0_CLEAR, (cx, t[1] - X0_CLEAR), (0.0, 1.0)
                g = np.array([1.0 if dxt > 0 else -1.0, 1.0]) / np.sqrt(2.0)
                tp = tuple(np.asarray(t) - g * X0_CLEAR)
                return tp[1] - abs(tp[0] - cx), tp, tuple(g)

            legs = [_leg(t) for t in tgts]
            assert min(y for y, _, _ in legs) > y1 + MIN_SEG, (
                f"x_0 would leave the box {y1 - min(y for y, _, _ in legs):.0f} u INSIDE it -- raise it")
            if len(legs) == 1:                           # ONE polyline, so its corner rounds like the rest
                y_b, tp, hd = legs[0]
                inner.append(dict(pts=[(cx, tap), (cx, y_b),
                                       tuple(np.asarray(tp) - np.asarray(hd) * ARROW_L)],
                                  tip=tp, head_dir=hd))
            else:
                inner.append(dict(pts=[(cx, tap), (cx, max(y for y, _, _ in legs))]))
                for y_b, tp, hd in legs:
                    inner.append(dict(pts=[(cx, y_b), tuple(np.asarray(tp) - np.asarray(hd) * ARROW_L)],
                                      tip=tp, head_dir=hd))
            # THE SAME WORD THE ACTION HEAD'S OUTPUT CARRIES: x_0 is a draw from this head's terminal
            # law, not its mean. Beside the vertical shaft rather than over it, since that shaft is
            # vertical and a label above it would sit on the box it just left.
            # ROTATED to run along the shaft it names: horizontal beside a vertical line reads as a
            # label for something else.
            # CENTRED ON ITS OWN BOX, offset clear of the shaft. rotation_mode="anchor" put the
            # baseline on the line itself, so the word straddled it whatever the offset.
            ax.text(cx + 2.2 * MIN_SEG + text_extent(SAMPLE_LAB, BOX_FS * 0.78)[1],
                    0.5 * (y1 + legs[0][0]), SAMPLE_LAB, ha="center", va="center", rotation=90,
                    fontsize=BOX_FS * 0.78, color=EDGE, zorder=Z_TOKENIZER + 5)
            ROUTES.append((D["title"].splitlines()[0], "x_0", f"45 arrival x {len(legs)}"))
            kind = "up-and-out"
        else:
            # NOTHING DOWNSTREAM YET: the action head's chunk is not consumed anywhere in this figure, so
            # it gets a labelled stub rather than an arrow pointing at empty canvas.
            # OUT THROUGH THE RIGHT EDGE, on the trunk's own line: the signal reaches the foot of the
            # stack and keeps going right, with the sampling loop tapping off it on the way. Down and out
            # of the bottom is where the grammar would put it, but the observation head is directly below.
            inner.append(dict(pts=[(cx, tap), (etip[0] - ARROW_L, etip[1])],
                              tip=etip, head_dir=(1.0, 0.0)))
            if D["out_lab"]:
                ax.text(etip[0] + MIN_SEG, etip[1], D["out_lab"], ha="left", va="center",
                        fontsize=ENC_FS * 0.8, color=EDGE, zorder=Z_TOKENIZER + 5)
            kind = "stub"
    # BELOW the sublayer boxes (+2) so the trunk passes BEHIND them rather than through their labels,
    # but as ONE group so the halos cannot chop each other.
    draw_arrows(ax, inner, Z_TOKENIZER + 1)
    ROUTES.append((D["title"].splitlines()[0], "box out", kind))


ROUTES = []                                              # (modality, kind, route form) -- reported below


def _draw_action_fan(ax, x0, y0, x1, y1):
    """`fan`: every draw, one row per stick axis, on ONE SHARED vertical scale -- the axes are the same
    physical quantity and their spreads are the comparison worth making, so per-row autoscaling would make
    a stick that never moved look as busy as one that swung. The recorded commands go on top in black, so
    the panel shows the prior's spread AND whether it covers what the pilot actually did."""
    r = float(max(np.abs(ACT_SMP).max(), np.abs(ACT_TRUE).max())) or 1.0
    t_s = np.linspace(x0, x1, ACT_SMP.shape[1])
    t_t = np.linspace(x0, x0 + (x1 - x0) * len(ACT_TRUE) / ACT_SMP.shape[1], len(ACT_TRUE))
    for j, nm in enumerate(AS_AXES):
        cy = y0 + AS_ROW / 2.0 + j * (AS_ROW + AS_SEP)
        sc = -(AS_ROW / 2.0) / r                         # y is inverted, so + stick reads UP
        ax.plot([x0, x1], [cy, cy], color="#999999", lw=LW * 0.7, zorder=Z_ARROW, solid_capstyle="butt")
        for k in range(ACT_SMP.shape[0]):                # every draw: the spread IS the thing being shown
            ax.plot(t_s, cy + sc * ACT_SMP[k, :, j], color=LAYERS[2]["ec"], lw=AS_LW, alpha=AS_ALPHA,
                    zorder=Z_ARROW + 1, solid_capstyle="round")
        ax.plot(t_t, cy + sc * ACT_TRUE[:, j], color=EDGE, lw=LW * 1.1, zorder=Z_ARROW + 2,
                solid_capstyle="round")
        ax.text(x1 + MIN_SEG, cy, nm, ha="left", va="center", fontsize=BB_FS * 0.85, color=EDGE,
                zorder=Z_ARROW + 2)
    ROUTES.append(("Action head", "out", f"fan: {ACT_SMP.shape[0]} draws x {len(AS_AXES)} axes"))


# The sampled chunk's arrow glyphs. `_arrows` is decorative in the input stream too (a seeded random
# compass per cell, not the command's direction), so the output tiles carry the same kind of glyph from
# their own seed: what makes a tile read as an ACTION here is the glyph, and dropping it would make the
# action output look like the proprio output.
# Same rule as the input stream: the glyph is the SIGN of the value in the tile it sits in, here the
# action chunk the head actually sampled -- not a random direction.
_smp_arrows = np.where(ACT_SMP[0, :AS_TILES] >= 0.0, 0.0, np.pi)


def _draw_action_tiles(ax, x0, y0, x1, y1):
    """`tiles`: ONE DRAW from the head, in the ordinary action notation -- the same cell renderer, the same
    lo/hi, the same cascade offsets the input streams and the other two outputs use. It is a sample, and
    the word on the arrow is what says so; see SAMPLE_LAB for why the distribution is not drawn."""
    for c in range(AS_TILES):
        _p_act_vec(ax, x0 + c * OFFSET_X, y0 + c * OFFSET_Y, CELL, _vh_a, ACT_SMP[0, c], 1.0,
                   Z_ARROW + 1 + c, ang=_smp_arrows[c], hatch=PRED_HATCH,
                   hatch_col=plt.get_cmap(ACT_OUT_CMAP)(0.85))
    ROUTES.append(("Action head", "out", f"tiles: 1 sampled chunk, {AS_TILES} steps"))


def draw_legend(ax):
    """WHAT THE HATCH MEANS, stated once. It marks four different things -- the predicted slice in the
    token column, both decoded outputs, and the sampled action chunk -- and a reader should not have to
    infer the convention separately at each of them. Plain white swatches, because the convention is
    about TEXTURE: colour already means modality everywhere in this figure, and a coloured swatch here
    would read as a fifth stream."""
    w_lab = max(text_extent(t, ENC_FS)[0] for t in (LEG_TRUE, LEG_PRED))
    # BOTTOM LEFT: the reward model now fills the lower right, and the streams having lifted leaves the
    # bottom-left corner empty.
    lx = MARGIN
    ly = CANVAS_H - MARGIN - 2 * LEG_S - LEG_SEP
    for k, (lab, hatch) in enumerate(((LEG_TRUE, None), (LEG_PRED, PRED_HATCH))):
        y = ly + k * (LEG_S + LEG_SEP)
        ax.add_patch(Rectangle((lx, y), LEG_S, LEG_S, fc="white", ec=EDGE, lw=LW * 1.6, hatch=hatch,
                               zorder=Z_TOKENIZER + 20))
        ax.text(lx + LEG_S + LEG_GAP, y + LEG_S / 2, lab, ha="left", va="center", fontsize=ENC_FS,
                color=EDGE, zorder=Z_TOKENIZER + 20)


def render(path, *, items=(), braces=False, tokenizers=False, blocks=False, backbone=False, dit=False,
           pred=False, decoders=False, group=False, action_head=False, legend=False,
           reward=False):
    """`items` = indices of LAYERS whose cascades to draw. Every component uses the SAME canvas, so the
    outputs stack exactly."""
    fig = plt.figure(figsize=(CANVAS_W / 100, CANVAS_H / 100), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, CANVAS_W); ax.set_ylim(0, CANVAS_H)
    ax.axis("off"); ax.invert_yaxis()
    ROUTES.clear()
    for k, L in enumerate(LAYERS):
        sx, sy, z0 = L["sx"], L["sy"], L["z"]
        if k in items:
            for i in range(L["n"]):                      # oldest first -> newest painted on top
                a_i = ALPHA_LO + (1.0 - ALPHA_LO) * i / max(1, L["n"] - 1)   # ramp spans the DRAWN items
                L["draw"](ax, i * OFFSET_X + sx, i * OFFSET_Y + sy, L["pw"], L["ph"], i, a_i, z0 + i)
            if L is LAYERS[-1]:
                for i, mark in ASTERISKS:
                    ax.text(i * OFFSET_X + sx + L["pw"] / 2, i * OFFSET_Y + sy + L["ph"] + BRACE_T,
                            mark, ha="center", va="top", fontsize=ENC_FS * 1.3, color=EDGE,
                            zorder=z0 + N + 2)
        if braces and L.get("time_seg"):
            (tx0, ty0), (tx1, ty1) = L["time_seg"]
            p0, p1 = (tx0 + sx, ty0 + sy), (tx1 + sx, ty1 + sy)
            u = np.array([OFFSET_X, OFFSET_Y], float); u /= np.linalg.norm(u)
            draw_arrow(ax, [p0, tuple(np.asarray(p1) - u * ARROW_L)], Z_ARROW, tip=p1, head_dir=tuple(u))
            ax.text(p1[0] + 2.0 * MIN_SEG, p1[1] + 0.6 * MIN_SEG, TIME_LAB, ha="left", va="center",
                    fontsize=ENC_FS * 1.15, color=EDGE, zorder=Z_ARROW + 1)
            ROUTES.append(("Time", "axis", "parallel to the cascade"))
        if braces:
            for bi, (i0, i1) in enumerate(L["braces"]):
                poly, mid = brace_segs(L["pw"], i0, i1)
                bp = [(q[0] + sx, q[1] + sy) for q in poly]
                ey = L["entry"][bi]                      # this stem's own entry height
                tip = ENC_X + sx - half_stroke(ENC_LW)   # FLUSH against the outline, not on top of it
                # horizontal -> 45 -> horizontal: the 45 absorbs the WHOLE drop, so there is no riser.
                # VISION IS THE EXCEPTION: its brace sits above the frames it marks and its tokenizer is
                # below and right, so a 45 long enough to cover the drop cuts straight across the cascade.
                # It keeps the right-angle riser, placed right of ALL content, which is what that rule was
                # for. No other stem is allowed one -- a riser is the fallback, not the default.
                _riser = k == 0
                rt, kind = route((mid[0] + sx, mid[1] + sy), (tip - ARROW_L, ey),
                                 (L["riser_x"][bi] + sx) if _riser else (tip - ARROW_L - MIN_SEG),
                                 diag_room=0.0 if _riser else ARM_DIAG)
                ROUTES.append((L["label"].splitlines()[0], "stem", kind))
                draw_arrow(ax, bp, z0 + N + 1.0)
                draw_arrow(ax, rt, z0 + N + 1.0, tip=(tip, ey))
        if tokenizers:
            pts = [(x + sx, y) for x, y in tokenizer_pts(L["enc_cy"], ENC_X)]   # cy is ABSOLUTE
            # ONE rounded path for fill and edge: no white outline, no corner mismatch.
            # ABOVE every arrow: the rule is that no feed may cross a tokenizer outline.
            ax.add_patch(PathPatch(rounded_polygon(pts, CORNER_R), fc=L["fc"], ec=L["ec"],
                                   lw=ENC_LW, zorder=Z_TOKENIZER + k))
            ax.text(ENC_X + ENC_W / 2 + sx, L["enc_cy"], L["label"], ha="center", va="center",
                    fontsize=ENC_FS, color="black", linespacing=1.3, zorder=Z_TOKENIZER + k + 0.5)
        if blocks:
            for j, (d, h, w) in enumerate(L["blks"]):
                f = np.array(L["feed"][j]) + np.array([sx, 0.0])
                t = np.array(L["tip"][j]) + np.array([sx, 0.0])        # the block's own anchor: unmoved
                if j:                                    # a continuation of the same stream: block, no arrow
                    iso_block(ax, tuple(t), d, h, w, L["blkz"][j], fc=L["blk"], ec=L["ec"])
                    continue
                tip = (t[0] - half_stroke(BLK_LW), t[1])               # FLUSH against the block outline
                shaft = (tip[0] - ARROW_L, tip[1])
                pts, kind = route(f, shaft, shaft[0] - MIN_SEG, diag_room=ARM_DIAG, slope=ARM_SLOPE)
                ROUTES.append((L["label"].splitlines()[0], "feed", kind))
                draw_arrow(ax, pts, Z_ARROW, tip=tip)
                iso_block(ax, tuple(t), d, h, w, L["blkz"][j], fc=L["blk"], ec=L["ec"])
    if group:                                            # the dashed wrapper: ONE box around the pair
        sx = LAYERS[0]["sx"]
        ax.add_patch(PathPatch(rounded_polygon([(LD["x0"] + sx, LD["y0"]), (LD["x1"] + sx, LD["y0"]),
                                                (LD["x1"] + sx, LD["y1"]), (LD["x0"] + sx, LD["y1"])],
                                               CORNER_R * 2),
                               fc="none", ec=EDGE, lw=ENC_LW * 0.6, ls=LD_DASH, zorder=Z_TOKENIZER - 1))
        ax.text(0.5 * (LD["x0"] + LD["x1"]) + sx, LD["label_cy"], LD_TITLE, ha="center", va="center",
                fontsize=LD_FS, color="black", zorder=Z_TOKENIZER - 1)
    if backbone:
        sx = LAYERS[0]["sx"]
        x0, x1, y0, y1 = BB["x0"] + sx, BB["x1"] + sx, BB["y0"], BB["y1"]
        # THE SLICE BRACE: same primitive as the cascades', lifted the other way, marking the 8 steps.
        draw_arrow(ax, [(q[0] + sx, q[1]) for q in BB["brace"]], Z_ARROW)
        # brace stem -> the space-time box's BOTTOM edge (its input, now that the pair sits above the
        # column): vertical up, across, vertical up. route() only runs left-to-right, so this one is the
        # explicit polyline -- one path, so draw_arrow still rounds every corner the same way.
        src = (BB["mid"][0] + sx, BB["mid"][1])
        tip = (BB["cx"] + sx, BB["y1"] + half_stroke(ENC_LW))
        # vertical -> 45 up-and-right -> vertical, the same grammar as every other riser in the figure
        # (route() transposed); the 45 is centred in the climb so both verticals are the same length.
        dx, ymid = tip[0] - src[0], 0.5 * (src[1] + tip[1])
        dv = STEM_SLOPE * abs(dx)                        # the diagonal's own rise, at STEM_SLOPE not 45
        assert src[1] - tip[1] > dv + 2 * MIN_SEG, "no room for the slice stem's diagonal"
        ROUTES.append(("Backbone", "slice", f"vertical + {np.degrees(np.arctan(STEM_SLOPE)):.0f} deg + vertical"))
        draw_arrow(ax, [src, (src[0], ymid + dv / 2.0), (tip[0], ymid - dv / 2.0),
                        (tip[0], tip[1] + ARROW_L)], Z_ARROW, tip=tip, head_dir=(0.0, -1.0))
        draw_box(ax, BB, sx)
    if pred:                                             # the predicted step, in red
        # SAME z as the flow head's inner lines, which is what makes x_0 JOIN it. At Z_ARROW the brace sat
        # far below x_0, so x_0's white halo punched a hole through it at the junction and the two read as
        # separate pieces. Equal z = both halos land under both lines, exactly as the cascade braces and
        # their stems already do.
        n_b = len(PRED_BLKS)
        for j, P in enumerate(PRED):
            step, row = divmod(j, n_b)                   # LATER steps on top, image row above proprio
            pal = LAYERS[row]                        # row 0 predicts VISION, row 1 predicts PROPRIO
            iso_block(ax, (P["tip"][0] + LAYERS[0]["sx"], P["tip"][1]), P["d"], P["h"], P["w"],
                      Z_BLOCK["vision"] + 50 + 10 * step + (n_b - 1 - row),
                      fc=pal["blk"], ec=pal["ec"], hatch=PRED_HATCH)   # slice: its own hue
    if decoders:
        sx = LAYERS[0]["sx"]
        for k, D in enumerate(DEC):
            pts = [(x + sx, y) for x, y in tokenizer_pts(D["cy"], DEC_X, flip=True)]
            ax.add_patch(PathPatch(rounded_polygon(pts, CORNER_R), fc=LAYERS[k]["fc"],
                                   ec=LAYERS[k]["ec"], lw=ENC_LW, zorder=Z_TOKENIZER))
            ax.text(DEC_X + ENC_W / 2 + sx, D["cy"], D["label"], ha="center", va="center",
                    fontsize=ENC_FS, color="black", linespacing=1.3, zorder=Z_TOKENIZER + 1)
            # red block -> decoder, and decoder -> the predicted item, both dead horizontal
            tip = (DEC_X + sx - half_stroke(ENC_LW), D["cy"])
            src = (D["src"][0] + sx, D["src"][1])        # leaves the red block at the BLOCK's height
            shaft = (tip[0] - ARROW_L, tip[1])
            # STAGGERED both sides so no two risers share a vertical, but in OPPOSITE orders: on the way
            # IN the arrows fan out from one face, on the way OUT they fan in to separate tiles.
            off_in, off_out = (k - 0.5) * RISER_STEP, (0.5 - k) * RISER_STEP
            pts, kind = route(src, shaft, shaft[0] - MIN_SEG + off_in, diag_room=ARM_DIAG,
                              slope=ARM_SLOPE)
            ROUTES.append((D["label"].splitlines()[0], "decode", kind))
            draw_arrow(ax, pts, Z_DECODE, tip=tip)       # UNDER the blocks: the tail hides inside the red
            tip2 = (OUT_X + sx, D["out_cy"])              # FLUSH: the item starts exactly at the tip
            o_src = (DEC_X + ENC_W + sx, D["cy"])
            shaft2 = (tip2[0] - ARROW_L, tip2[1])
            pts2, kind2 = route(o_src, shaft2, shaft2[0] - MIN_SEG + off_out, diag_room=ARM_DIAG)
            ROUTES.append((D["label"].splitlines()[0], "output", kind2))
            draw_arrow(ax, pts2, Z_ARROW, tip=tip2)
            # ONE TILE PER RED SLICE, cascading on the SAME offsets the input streams use -- the output
            # picks the diagonal back up where the input dropped it.
            n_out = PRED_STEPS if PRED_IMG is not None else 1
            for t_ in range(n_out):
                ox = OUT_X + sx + t_ * OFFSET_X
                oy = D["out_cy"] - D["oh"] / 2 + t_ * OFFSET_Y
                zt = Z_ARROW + 2 + t_                    # later steps paint over earlier, as the reds do
                if D["label"].startswith("Vision") and PRED_IMG is not None:
                    ax.imshow(PRED_IMG[t_], extent=(ox, ox + D["ow"], oy + D["oh"], oy), zorder=zt,
                              interpolation="bilinear")
                    # A HATCHED BAND, NOT A HATCHED IMAGE: a ring path (outer rectangle, inner rectangle
                    # reversed) carries the hatch, so the predicted frame is marked at its border and
                    # left legible in the middle.
                    bw = PRED_BAND_PX * D["ow"] / PRED_IMG[t_].shape[1]
                    W_, H_ = D["ow"], D["oh"]
                    for bx, by, bwd, bht in ((ox, oy, W_, bw), (ox, oy + H_ - bw, W_, bw),
                                             (ox, oy + bw, bw, H_ - 2 * bw),
                                             (ox + W_ - bw, oy + bw, bw, H_ - 2 * bw)):
                        ax.add_patch(Rectangle((bx, by), bwd, bht, fc="none", lw=0.0,
                                               ec=hatch_rgba(VIS_EC), hatch=PRED_HATCH,
                                               zorder=zt + 0.4))
                    ax.add_patch(Rectangle((ox, oy), D["ow"], D["oh"], fill=False, ec=VIS_EC,
                                           lw=LW * 2, zorder=zt + 0.5))
                elif PRED_OBS is not None:
                    _p_prop_vec(ax, ox, oy, D["ow"], D["oh"], PRED_OBS[t_], 1.0, zt,
                                hatch=PRED_HATCH,
                                hatch_col=plt.get_cmap(MOD_CMAP["proprio"])(0.85))
                else:
                    D["draw"](ax, ox, oy, D["ow"], D["oh"], PRED_AT, 1.0, zt)
    if dit:
        sx = LAYERS[0]["sx"]
        draw_box(ax, DT, sx)
        # (the `h` label that used to sit on the shared stem is gone: the text says what h is, and the
        #  figure was carrying a symbol nothing else in it defines)
    if action_head:
        # THE BOX. Its h arrives as one branch of the summariser's fan (draw_box, flow > 0), because that
        # fan is one connected set of lines and splitting it across two render calls would let one halo
        # punch a hole in the other at the junction.
        sx = LAYERS[0]["sx"]
        draw_box(ax, AH, sx)
        if AS_BOX is not None:
            box_ = (AS_BOX[0] + sx, AS_BOX[1], AS_BOX[2] + sx, AS_BOX[3])
            (_draw_action_tiles if AS_STYLE == "tiles" else _draw_action_fan)(ax, *box_)
            if AS_STYLE == "fan":
                ax.text(0.5 * (box_[0] + box_[2]), box_[3] + 2 * MIN_SEG,
                        r"$a_{t:t+K}\sim p(\cdot\mid h)$", ha="center", va="top", fontsize=BB_FS,
                        color=EDGE, zorder=Z_ARROW + 2)
            else:
                # the arrow carries the word instead: this is A DRAW, not the distribution
                # over the SHAFT: the arrow now runs flush into the first tile, so there is no gap
                # at its head to put a word in
                ax.text(0.5 * (AH["x1"] + AH["exit_tip"][0]) + sx, AH["exit_tip"][1] - MIN_SEG,
                        SAMPLE_LAB, ha="center", va="bottom", fontsize=BOX_FS * 0.78, color=EDGE,
                        zorder=Z_ARROW + 2)
    if reward and RM_ON:
        sx = LAYERS[0]["sx"]
        dy_ = LAYERS[0]["tip"][0][1] - LAYERS[0]["tip_abs"][0][1]      # solve_layout's own y offset
        # THE BRACE, AND WHY IT WAS SHORT. brace_between PROJECTS its two corners onto a line through
        # the lifted start point in direction `u`, and `u` defaults to the CASCADE ANGLE -- which is
        # right for the summariser's brace, because that one sits on an isometric edge of the block and
        # runs parallel to it. The bottom span here is horizontal, so projecting it onto a sloped line
        # shortened it by cos(angle) and slid the left end inward: the ends no longer matched the corners
        # they were meant to mark. Passing u=(1,0) makes the projection the identity in x, so the brace
        # runs from the ACTION (blue) block's left edge to the PROPRIOCEPTION (orange) block's right
        # edge exactly, which is the width of the token bag the reward reads. The lift is BRACE_M, the
        # same depth as the summariser's input brace.
        # FLUSH WITH THE BLUE CUBOID, and built exactly the way the summariser's input brace is built --
        # from the block's OWN bottom edge rather than from a bounding box. The action block's depth edge
        # runs at the cascade angle (ISO_C, ISO_S is OFFSET_X, OFFSET_Y normalised), which is why
        # brace_between's default direction is the right one here: same primitive, lift the other way.
        _O = np.asarray(LAYERS[-1]["blk_O"][0], float) + np.array([sx, dy_])
        _d = LAYERS[-1]["blks"][0][0]
        _vdu = np.array([-BLK_S * ISO_C, -BLK_S * ISO_S])
        poly, mid = brace_between(tuple(_O + _vdu * _d), tuple(_O), RM_GAP_BR)
        draw_arrow(ax, poly, Z_ARROW)
        # SAME RENDERING AS THE HEADS: outer box at ENC_LW with a BOX_FS title, internals at ENC_LW*0.6
        # with BB_FS text, which is what draw_box uses -- the reward model was drawn at its own weights
        # and read as a different kind of object.
        sub_h = text_extent(RM_LAT, BB_FS)[1] * 2.5
        ttl_h = text_extent(RM_TITLE, BOX_FS)[1]
        rh = 2 * RM_PAD + ttl_h + 2 * sub_h + RM_ROW_SEP
        # ALIGNED WITH THE TWO HEADS: same left and right edges as the observation and action head
        # boxes, so the three things the model runs read as one column of boxes.
        rx0, rx1 = AH["x0"] + sx, AH["x1"] + sx
        rw = rx1 - rx0
        ry0 = mid[1] + RM_GAP_BOX
        ax.add_patch(PathPatch(rounded_polygon([(rx0, ry0), (rx1, ry0), (rx1, ry0 + rh), (rx0, ry0 + rh)],
                                               CORNER_R), fc=ENC_FC, ec=BOX_EC, lw=ENC_LW,
                               zorder=Z_TOKENIZER))
        ax.text(0.5 * (rx0 + rx1), ry0 + RM_PAD * 0.45 + ttl_h / 2, RM_TITLE, ha="center", va="center",
                fontsize=BOX_FS, linespacing=1.3, zorder=Z_TOKENIZER + 4, fontweight="bold")
        badge(ax, rx0 + BB_PAD + BADGE_R, ry0 + BB_PAD + BADGE_R, 3)
        w_cos = text_extent(RM_COS, BB_FS)[0] + 1.7 * RM_PAD
        # THE BRANCH CELLS MUST FIT THEIR OWN TEXT. The box width is fixed (it matches the two heads), so
        # what gives is the gap before the cosine -- sizing the cells from the leftover instead clipped
        # "Project to f_t".
        w_enc = text_extent(RM_ENC, BB_FS)[0] + 1.3 * RM_PAD
        w_prj = text_extent(RM_TXT, BB_FS)[0] + 1.3 * RM_PAD
        w_br = w_enc + RM_ROW_SEP + max(w_prj, text_extent(RM_LAT, BB_FS)[0] + 1.3 * RM_PAD)
        cos_gap = max(2.2 * ARROW_L, rw - 2 * RM_PAD - w_br - w_cos)
        bx0 = rx0 + RM_PAD
        y_lat = ry0 + RM_PAD + ttl_h + sub_h / 2
        y_txt = y_lat + sub_h + RM_ROW_SEP
        _z_in = Z_TOKENIZER + 1.5          # above the box fill, or the arrows inside it are painted over

        def _cell(x, y, w, lab):
            ax.add_patch(PathPatch(rounded_polygon([(x, y - sub_h / 2), (x + w, y - sub_h / 2),
                                                    (x + w, y + sub_h / 2), (x, y + sub_h / 2)],
                                                   CORNER_R), fc="white", ec=BOX_EC, lw=ENC_LW * 0.6,
                                   zorder=Z_TOKENIZER + 2))
            ax.text(x + w / 2, y, lab, ha="center", va="center", fontsize=BB_FS,
                    zorder=Z_TOKENIZER + 3)
        # THE TEXT BRANCH IS TWO BLOCKS: a frozen sentence encoder, then the projection that is trained.
        # THE TWO PROJECTIONS ARE THE SAME BOX, one above the other: they do the same thing to two
        # different inputs, and drawing one full-width and one inset said otherwise.
        w_prj = max(w_prj, text_extent(RM_LAT, BB_FS)[0] + 1.3 * RM_PAD)
        px0 = bx0 + w_enc + RM_ROW_SEP
        _cell(px0, y_lat, w_prj, RM_LAT)
        _cell(bx0, y_txt, w_enc, RM_ENC)
        _cell(px0, y_txt, w_prj, RM_TXT)
        draw_arrow(ax, [(bx0 + w_enc, y_txt), (bx0 + w_enc + RM_ROW_SEP - ARROW_L * 0.7, y_txt)],
                   Z_TOKENIZER + 2.5, tip=(bx0 + w_enc + RM_ROW_SEP, y_txt), head_dir=(1.0, 0.0))
        cx0 = bx0 + w_br + cos_gap
        y_cos = 0.5 * (y_lat + y_txt)
        ax.add_patch(PathPatch(rounded_polygon([(cx0, y_cos - sub_h / 2), (cx0 + w_cos, y_cos - sub_h / 2),
                                                (cx0 + w_cos, y_cos + sub_h / 2), (cx0, y_cos + sub_h / 2)],
                                               CORNER_R), fc="white", ec=BOX_EC, lw=ENC_LW * 0.6,
                               zorder=Z_TOKENIZER + 2))
        ax.text(cx0 + w_cos / 2, y_cos, RM_COS, ha="center", va="center", fontsize=BB_FS,
                zorder=Z_TOKENIZER + 3)
        for yy in (y_lat, y_txt):
            draw_arrow(ax, [(px0 + w_prj, yy), (0.5 * (px0 + w_prj + cx0), yy),
                            (0.5 * (px0 + w_prj + cx0), y_cos), (cx0 - ARROW_L, y_cos)], _z_in,
                       tip=(cx0, y_cos), head_dir=(1.0, 0.0))
        # the brace stem comes down and turns into the latent branch
        # AT _z_in, NOT Z_ARROW: the last leg runs INSIDE the box, and the box fill sits far above
        # arrow depth, so at Z_ARROW the line vanished the moment it crossed the wall.
        draw_arrow(ax, [(mid[0], mid[1]), (mid[0], y_lat), (px0 - ARROW_L, y_lat)], _z_in,
                   tip=(px0, y_lat), head_dir=(1.0, 0.0))
        # ...and the request runs in from the SAME x the brace stem turns at, so the two inputs align
        lines = RM_REQ.split(chr(10))
        tw = max(text_extent(l, BB_FS * 1.5)[0] for l in lines)
        # THE REQUEST IS BRACED, like every other group of things in this figure, and the brace's stem
        # is the line that carries it into the text branch.
        _tw, _th = text_extent(RM_REQ, BB_FS * 2.0)
        _tx1 = mid[0] - RM_REQ_PAD
        ax.text(_tx1, y_txt, RM_REQ, ha="right", va="center", fontsize=BB_FS * 2.0,
                style="italic", color=EDGE, linespacing=1.25, zorder=Z_TOKENIZER + 1)
        rpoly, rmid = brace_between((_tx1 + MIN_SEG, y_txt - _th / 2 - BRACE_T),
                                    (_tx1 + MIN_SEG, y_txt + _th / 2 + BRACE_T),
                                    BRACE_M * 0.7, u=(0.0, 1.0), off=(1.0, 0.0))
        draw_arrow(ax, rpoly, Z_ARROW)
        draw_arrow(ax, [rmid, (bx0 - ARROW_L, y_txt)], _z_in, tip=(bx0, y_txt), head_dir=(1.0, 0.0))
        draw_arrow(ax, [(cx0 + w_cos, y_cos), (rx1 + 1.2 * ARROW_L, y_cos)], _z_in,
                   tip=(rx1 + 2.2 * ARROW_L, y_cos), head_dir=(1.0, 0.0))
        ax.text(rx1 + 2.6 * ARROW_L, y_cos, RM_OUT, ha="left", va="center", fontsize=BOX_FS * 0.8,
                zorder=Z_ARROW + 1)
        ROUTES.append(("Reward model", "brace", "action block left -> proprio block right"))
    if legend:
        draw_legend(ax)
    fig.savefig(path, transparent=True, dpi=DPI, pad_inches=0)
    plt.close(fig)
    print(f"  wrote {path.split('/')[-1]}")


import os                                                                          # noqa: E402
ELEM = f"{OUT}/elements"
os.makedirs(ELEM, exist_ok=True)
for i, name in enumerate(("vision", "proprio", "action")):
    render(f"{ELEM}/items_{name}.png", items=(i,))
render(f"{ELEM}/braces.png", braces=True)
render(f"{ELEM}/tokenizers.png", tokenizers=True)
render(f"{ELEM}/blocks.png", blocks=True)
render(f"{ELEM}/backbone.png", backbone=True)
render(f"{ELEM}/diffusion.png", dit=True)
render(f"{ELEM}/action_head.png", action_head=True)
render(f"{ELEM}/latent_dynamics.png", group=True)
render(f"{ELEM}/predicted.png", pred=True)
render(f"{ELEM}/decoders.png", decoders=True)
render(f"{ELEM}/legend.png", legend=True)
render(f"{ELEM}/reward.png", reward=True)
render(f"{OUT}/architecture.png", items=(0, 1, 2), braces=True, tokenizers=True, blocks=True,
       backbone=True, dit=True, pred=True, decoders=True, group=LD_ON, action_head=True,
       legend=True, reward=True)
print(f"  canvas {int(CANVAS_W * DPI / 100)} x {int(CANVAS_H * DPI / 100)} px, identical for every component")
print(f"\n  ONE SPACING, both axes: {_GAP:.1f} u ({_GAP * 3:.0f} px)")
print(f"    MEASURED as proprioception riser x - right edge of the last image item, then reused as the")
print(f"    vertical gap from each stream's lowest point to the stem line of the stream below it")
print("\n  STEM VERTICALITY: how far each stream's stem group sits from its tokenizer's centre")
for L, dv in zip(LAYERS, STEM_DY):
    print(f"    {L['label'].splitlines()[0]:<10} {dv:6.1f} u"
          + ("   horizontal" if dv < 1e-6 else ""))
print(f"\n  ARROWS: one router, three shapes. Stems get a plain rounded right-angle riser (it has to sit")
print(f"    right of every cascade); feeds spend all the room they have on a 45, so the descent starts early:")
for lab, kindof, form in ROUTES:
    print(f"    {lab:<10} {kindof:<5} {form}")
print(f"\n  SLICE STEM offset {_STEM_DX:.0f} u, climbed at {np.degrees(np.arctan(STEM_SLOPE)):.0f} deg "
      f"-> {STEM_SLOPE * _STEM_DX:.0f} u of rise (a 45 would have cost {_STEM_DX:.0f})")
print(f"\n  BOX PAIR   flow-head exit {X0_DIAG:.0f} u right of the slice it feeds, so x_0's last leg is a "
      f"{X0_DIAG:.0f} u 45   |   gap between them {DT['x0'] - BB['x1']:.0f} u   |   raised {_BOX_LIFT:.0f} u")
print(f"\n  FLUSH TIPS: stop short by half the stroke they must not cover -- "
      f"tokenizer {half_stroke(ENC_LW):.2f} u ({half_stroke(ENC_LW) * 3:.1f} px), "
      f"block {half_stroke(BLK_LW):.2f} u ({half_stroke(BLK_LW) * 3:.1f} px); heads are fill-only, so no "
      f"mitre runs past the tip")
print(f"\n  COLUMN PLACEMENT: pinned so the ACTION feed is dead horizontal -- {COL_LIFT:.0f} u above where "
      f"the bottom-flush rule would have put it")
print(f"\n  TOKEN BLOCKS: reached by 45-degree arrows, run = drop = {BLK_RUN:.0f} u")
for L in LAYERS:
    print(f"    {L['label'].splitlines()[0]:<15} " + ", ".join(
        f"{d}x{h}x{w} front-face midpoint at ({t[0]:.0f}, {t[1]:.0f})"
        for (d, h, w), t in zip(L["blks"], L["tip"])))
print("\n  BRACE / TOKENIZER SPACING (figure units; x3 for output px)")
for L in LAYERS:
    ys = [y + MARGIN - (L["sy"] - L["sy0"]) * 0 for y in L["stem_ys"]]
    print(f"    {L['label'].splitlines()[0]:<15} brace stem y = "
          + ", ".join(f"{y + L['sy'] - L['sy0']:.1f}" for y in L["stem_ys"])
          + f"   -> entry y = " + ", ".join(f"{y:.1f}" for y in L["entry"])
          + f"   tokenizer cy = {L['enc_cy']:.1f}")
for L in LAYERS:
    if L.get("span_note"):
        was, now = L["span_note"]
        print(f"    NOTE {L['label'].splitlines()[0]}: its two stems are {was:.1f} u apart at the braces, "
              f"but the {ENC_H:.0f}-tall input edge only fits {now:.1f} u, so the pair is COMPRESSED to "
              f"{now:.1f} u on the way in (still parallel, still symmetric).")
for a, b in zip(LAYERS, LAYERS[1:]):
    d_st = (b["stem_ys"][0] + b["sy"] - b["sy0"]) - (a["stem_ys"][0] + a["sy"] - a["sy0"])
    d_tk = b["enc_cy"] - a["enc_cy"]
    print(f"    {a['label'].splitlines()[0]} -> {b['label'].splitlines()[0]}: "
          f"brace gap {d_st:.1f} u ({d_st * 3:.0f} px) | tokenizer centre gap {d_tk:.1f} u "
          f"({d_tk * 3:.0f} px) = ENC_H {ENC_H:.0f} + ENC_SEP {ENC_SEP:.0f}")
