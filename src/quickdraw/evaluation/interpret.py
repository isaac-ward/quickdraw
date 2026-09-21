"""Helpers for eval_interpret: VLM clip labeling (OpenAI Responses API, mirrors the seamstress harness)
plus the analytic cross-check labels (speed/direction/color derived from the imagined proprio). The routine
in routines.py orchestrates; the API/label mechanics live here so they're testable + swappable per env."""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.error
import urllib.request

import numpy as np

_OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


# ------------------------- OpenAI Responses API (stdlib only, mirrors seamstress) -------------------------
def openai_api_key() -> str:
    """OPENAI_API_KEY from the env, falling back to a repo-root .env (KEY=VALUE lines)."""
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    here = os.path.abspath(os.path.dirname(__file__))
    for _ in range(6):                                   # walk up to the repo root looking for .env
        env_path = os.path.join(here, ".env")
        if os.path.exists(env_path):
            for line in open(env_path):
                if line.strip().startswith("OPENAI_API_KEY="):
                    return line.split("=", 1)[1].strip()
        here = os.path.dirname(here)
    raise RuntimeError("OPENAI_API_KEY not set (env or repo-root .env). eval_interpret needs it for VLM labeling.")


def encode_frame_to_data_url(frame_hwc_uint8: np.ndarray) -> str:
    """(H,W,3) uint8 -> base64 PNG data URL (the input_image payload)."""
    from PIL import Image
    frame = np.asarray(frame_hwc_uint8, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(frame, mode="RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def build_segment_schema(factors: dict, n_captions: int, H: int, max_segments: int) -> dict:
    """Strict json_schema for a SEGMENTED clip: the VLM splits the clip into runs and labels each run.

    WHY SEGMENTS RATHER THAN ONE LABEL PER CLIP. A clip label is broadcast to every latent of the clip, so
    if the drone faces one wall then turns to another, most of those latents carry a region the model was not
    looking at -- systematic mislabelling of the very plots we read as evidence. Segments make the label
    granularity match the point granularity.

    WHY SEGMENTS RATHER THAN ONE LABEL PER FRAME. The VLM is good at "where does this change" and much worse
    at being consistent across H independent judgements; asking for frames invites flicker.

    `max_segments` is enforced structurally by maxItems. A MINIMUM segment length cannot be expressed in a
    schema at all, so it is requested in the prompt and then ENFORCED in code by repair_segments().

    Factors marked `per_frame: true` are asked per segment; the rest stay clip-level."""
    per_seg = {f: fc for f, fc in factors.items() if fc.get("per_frame")}
    clip_lvl = {f: fc for f, fc in factors.items() if not fc.get("per_frame")}
    seg_props = {"start_frame": {"type": "integer"}, "end_frame": {"type": "integer"}}
    for name, fc in per_seg.items():
        seg_props[name] = {"type": "string", "enum": list(fc["buckets"])}
    if n_captions:
        seg_props["captions"] = {"type": "array", "items": {"type": "string"},
                                 "minItems": n_captions, "maxItems": n_captions}
    seg = {"type": "object", "properties": seg_props, "required": list(seg_props),
           "additionalProperties": False}
    props: dict = {"reasoning": {"type": "string"},
                   "segments": {"type": "array", "items": seg, "minItems": 1, "maxItems": max_segments}}
    for name, fc in clip_lvl.items():
        props[name] = {"type": "string", "enum": list(fc["buckets"])}
    props["self_reported_confidence"] = {"type": "number"}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def repair_segments(segs: list, H: int, min_frames: int) -> tuple[list, int]:
    """Make a VLM segment list into a valid, gapless, minimum-length cover of [0, H).

    Returns (segments, n_merged). The VLM is asked for at most N segments each at least `min_frames` long;
    the count is enforceable in the schema and the length is not, so short runs are MERGED here rather than
    dropped -- dropping would leave holes, and holes mean unlabelled latents. `n_merged` is the flicker
    count: how often the VLM claimed a change that did not last, which is exactly the instability that makes
    a clip-level label wrong in the first place."""
    segs = sorted(({**s} for s in segs), key=lambda s: int(s.get("start_frame", 0)))
    for s in segs:                                                   # clamp into range
        s["start_frame"] = max(0, min(H - 1, int(s.get("start_frame", 0))))
        s["end_frame"] = max(0, min(H - 1, int(s.get("end_frame", H - 1))))
    segs[0]["start_frame"] = 0                                       # cover the whole clip, no gaps
    for a_, b_ in zip(segs, segs[1:]):
        a_["end_frame"] = max(a_["start_frame"], b_["start_frame"] - 1)
    segs[-1]["end_frame"] = H - 1
    segs = [s for s in segs if s["end_frame"] >= s["start_frame"]]
    merged = 0
    while len(segs) > 1:                                             # absorb runs shorter than min_frames
        lens = [s["end_frame"] - s["start_frame"] + 1 for s in segs]
        k = min(range(len(segs)), key=lambda i: lens[i])
        if lens[k] >= min_frames:
            break
        j = k - 1 if k == len(segs) - 1 else (k + 1 if k == 0 else
                                              (k - 1 if lens[k - 1] >= lens[k + 1] else k + 1))
        lo, hi = min(j, k), max(j, k)
        segs[j]["start_frame"] = segs[lo]["start_frame"]              # the SURVIVOR keeps its own labels
        segs[j]["end_frame"] = segs[hi]["end_frame"]
        segs.pop(k)
        merged += 1
    return segs, merged


def expand_segments(segs: list, H: int, keys: list) -> dict:
    """segments -> {key: [value per frame]} for the per-segment keys (labels and caption lists alike)."""
    out = {k: [None] * H for k in keys}
    for s in segs:
        for t in range(s["start_frame"], s["end_frame"] + 1):
            for k in keys:
                out[k][t] = s.get(k)
    for k in keys:                                                   # belt and braces: no unlabelled frame
        last = next((v for v in out[k] if v is not None), None)
        for t in range(H):
            if out[k][t] is None:
                out[k][t] = last
            last = out[k][t]
    return out


def build_label_schema(factors: dict, n_captions: int = 0) -> dict:
    """Strict json_schema for the structured VLM output. A free-text `reasoning` field comes FIRST
    (reason-then-answer: describing what it sees before committing lifts accuracy), then one enum field per
    factor, then a confidence. n_captions>0 adds a `captions` array of exactly that many detailed free-form
    descriptions of the clip (for CLIP-style reward training — f_t learns to map real phrasings -> the region)."""
    props: dict = {"reasoning": {"type": "string"}}
    for name, fc in factors.items():
        props[name] = {"type": "string", "enum": list(fc["buckets"])}
    props["self_reported_confidence"] = {"type": "number"}
    if n_captions:
        props["captions"] = {"type": "array", "items": {"type": "string"},
                             "minItems": n_captions, "maxItems": n_captions}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def _post(api_key: str, body: dict, timeout_s: float = 120.0) -> dict:
    req = urllib.request.Request(
        _OPENAI_RESPONSES_URL, data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"OpenAI Responses API HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:500]}") from exc


def _extract_text(payload: dict) -> str:
    """Concatenate the output_text chunks from a Responses-API payload."""
    chunks = []
    for item in payload.get("output", []) or []:
        for ci in (item.get("content", []) if isinstance(item, dict) else []) or []:
            if isinstance(ci, dict) and str(ci.get("type", "")).lower() == "output_text" and isinstance(ci.get("text"), str):
                chunks.append(ci["text"])
    txt = "".join(chunks).strip()
    if not txt:
        raise RuntimeError(f"Responses payload had no output_text: {json.dumps(payload)[:800]}")
    return txt


def label_clip(*, api_key: str, model: str, prompt: str, schema: dict, frames_uint8, action_text: str,
               retries: int = 2) -> dict | None:
    """Label ONE clip. frames_uint8: list/array of (H,W,3) uint8 (already subsampled). Returns the parsed
    label dict, or None if the call/parse fails after retries (the routine drops Nones)."""
    content = [{"type": "input_text", "text": action_text}]
    for fr in frames_uint8:
        content.append({"type": "input_image", "image_url": encode_frame_to_data_url(fr), "detail": "high"})
    body = {"model": model, "instructions": prompt, "input": [{"role": "user", "content": content}],
            "text": {"format": {"type": "json_schema", "name": "clip_label", "strict": True, "schema": schema}}}
    for attempt in range(retries + 1):
        try:
            return json.loads(_extract_text(_post(api_key, body)))
        except Exception:
            if attempt == retries:
                return None
    return None


def build_action_text(actions: np.ndarray, axes: list) -> str:
    """The clip's commanded actions, written in the environment's own words.

    `axes` is conf/interpret/<env>.yaml's `action_axes`: one entry per RAW action axis, in order, each
    {name, positive, negative}. REQUIRED per environment, and deliberately so -- this text goes into a VLM
    prompt, and a number whose meaning is not stated is worse than no number. The previous version printed
    `(a[i,0], a[i,1])` while claiming `{a.shape[1]} dims`, so a 4-axis drone would have had two of its
    sticks shown, unnamed, and the other two silently dropped.

    UNDER data.action_aggregate=concat one stored action holds `subsample` raw commands laid out
    TIME-MAJOR (index = sub_step * n_axes + axis), so the sub-steps are folded back and averaged here --
    the VLM wants the commanded motion, not the 15 Hz detail inside one 0.27 s step.
    """
    a = np.asarray(actions, dtype=np.float32)
    n = len(axes)
    assert a.shape[-1] % n == 0, f"action width {a.shape[-1]} is not a multiple of {n} declared axes"
    a = a.reshape(a.shape[0], a.shape[-1] // n, n).mean(axis=1)      # fold concat sub-steps, keep sign
    idx = np.unique(np.linspace(0, len(a) - 1, min(10, len(a))).round().astype(int))
    legend = "; ".join(f"{ax['name']}: + is {ax['positive']}, - is {ax['negative']}" for ax in axes)
    head = "  step   " + "  ".join(f"{ax['name']:>9s}" for ax in axes)
    rows = "\n".join("  t=" + f"{int(i):<4d} " + "  ".join(f"{a[i, j]:+9.2f}" for j in range(n)) for i in idx)
    return (f"Commanded stick inputs over this clip, each in [-1, 1] ({legend}):\n{head}\n{rows}")



def build_action_prose(actions: np.ndarray, axes: list, scene: str = "", subject: str = "The view",
                       max_phases: int = 4, thresh: float = 0.2) -> str:
    """The commanded actions as a SENTENCE, for conditioning a video generator (build_action_text is the
    table form, for a VLM that can already see the clip).

    WHY THIS IS NOT JUST A LIST OF DIRECTIONS. An axis whose value is a STATE rather than a rate -- a
    gripper -- averages to nothing useful: reporting its per-phase mean produced "opens its gripper, then
    opens its gripper and lowers, then opens its gripper and rises", one fact repeated three times, while
    the thing that actually happened (a cube was grasped, lifted and placed) was never said at all. A
    grasp is an EVENT, and the manipulation is the whole content of the clip.

    So an axis may declare `role: gripper` in `action_axes`, and it is then read as a transition:
      open -> closed   "closes its gripper on the cube"
      closed -> open   "opens its gripper and releases the cube"
      closed throughout, while moving   "... while holding the cube", said ONCE per hold and not
                                        repeated on every phase it spans
    Everything else is a rate axis and keeps the direction-of-mean treatment, two per phase, ranked by
    distance from its own `neutral`.

    WHAT IT STILL CANNOT SAY: WHICH cube. The dataset records the arm (ee pose, joints, gripper) and not
    the objects, so there are no cube poses to read; naming one from the future frames would be feeding
    the model the answer. "the cube" is the honest limit until object state exists.

    Magnitude and sub-phase reversals are lost either way -- a stick that swings +1 then -1 inside one
    span averages to nothing, which is the argument for calling this per CHUNK rather than per rollout.
    """
    a = np.asarray(actions, dtype=np.float32)
    n = len(axes)
    assert a.shape[-1] % n == 0, f"action width {a.shape[-1]} is not a multiple of {n} declared axes"
    a = a.reshape(a.shape[0], a.shape[-1] // n, n).mean(axis=1)      # fold concat sub-steps, keep sign
    k = max(1, min(int(max_phases), len(a)))
    spans = np.array_split(np.arange(len(a)), k)
    neutral = np.array([float(ax.get("neutral", 0.0)) for ax in axes], dtype=np.float32)

    gi = next((j for j, ax in enumerate(axes) if str(ax.get("role", "")) == "gripper"), None)
    shut = None if gi is None else a[:, gi] > float(axes[gi].get("closed_above", 0.5))

    phrases, held_said = [], False
    for sp in spans:
        m = a[sp].mean(0) - neutral
        rate = [j for j in np.argsort(-np.abs(m)) if j != gi and abs(m[j]) >= thresh]
        moved = [axes[j]["positive"] if m[j] > 0 else axes[j]["negative"] for j in rate[:2]]

        grip, holding = [], ""
        if shut is not None:
            a0, a1 = bool(shut[sp[0]]), bool(shut[sp[-1]])
            if not a0 and a1:
                grip = ["closes its gripper on the cube"]
            elif a0 and not a1:
                grip = ["opens its gripper and releases the cube"]
            elif a1 and shut[sp].mean() > 0.5 and moved and not held_said:
                holding = " while holding the cube"       # once per hold; a 5 s grasp spans ~2 phases
            held_said = a1 and (held_said or bool(holding) or bool(grip))
            if not a1:
                held_said = False
        said = grip + ([" and ".join(moved)] if moved else [])
        phrases.append((", ".join(said) + holding) if said else "holds still")

    seq = [ph for i, ph in enumerate(phrases) if i == 0 or ph != phrases[i - 1]]
    motion = seq[0] if len(seq) == 1 else ", then ".join(seq)
    return f"{scene.strip()} {subject.strip()} {motion}.".strip()


# ------------------------- analytic labels (exact, from the imagined proprio) -------------------------
def analytic_scalar(kind: str, pro_phys: np.ndarray, R: float, dims=None, fc: dict | None = None) -> float:
    """One scalar per clip for an analytic factor; `bucketize` turns a set of them into labels.

    pro_phys is the clip's physical (denormalized) proprio, (H, obs_dim). The torus kinds below assume its
    first six dims are [xyz, velocity]; the obs_* kinds take the dims from the factor config instead and so
    work for any dataset whose layout is known."""
    if kind == "hue_at_position":                                 # hue at the agent's mid-clip ring angle (the surface it's on)
        pos = pro_phys[len(pro_phys) // 2, :3]
        return float((np.arctan2(pos[1], pos[0]) / (2 * np.pi)) % 1.0)
    if kind == "speed_quantile":                                  # mean |velocity| over the clip (proprio dims 3:6)
        return float(np.linalg.norm(pro_phys[:, 3:6], axis=1).mean())
    if kind == "z_band":                                          # ambient height z at mid-clip (z in [-r, +r])
        return float(pro_phys[len(pro_phys) // 2, 2])
    # ---- GENERIC, obs-index driven: works for any dataset whose observation layout is known, instead of a
    # new hardcoded kind per environment. The factor config names the DIMS, so the same three kinds cover
    # altitude, speed, climb rate, yaw rate... (starling-2's layout is data/rosbag.py STATE_COLUMNS).
    if kind == "obs_value":                                       # one dim at mid-clip (e.g. altitude = z)
        return float(pro_phys[len(pro_phys) // 2, dims[0]])
    if kind == "obs_norm":                                        # mean ||dims|| over the clip (e.g. speed)
        return float(np.linalg.norm(pro_phys[:, dims], axis=1).mean())
    if kind == "obs_abs":                                         # mean |dim| over the clip (e.g. |yaw rate|)
        return float(np.abs(pro_phys[:, dims]).mean())
    if kind == "obs_mean":                                        # SIGNED mean over the clip and the dims --
        return float(pro_phys[:, dims].mean())                    # a commanded stick, whose sign is the point
    if kind == "axis_dominant":
        # WHICH axis dominates, as an index into the factor's buckets (bucketize just looks it up).
        # Axes are compared in units of their OWN deadband, because the sticks are not used at comparable
        # scales -- an absolute argmax would let the most-used stick win almost every clip.
        # TWO SEPARATE TESTS, because one number cannot do both jobs. `deadbands` are absolute and answer
        # "is this stick being used at all"; `scales` are each axis's own p90 |clip mean| over the corpus and
        # answer "which is being used hardest RELATIVE TO HOW HARD IT EVER GETS PUSHED".
        # Using deadbands for BOTH was a real bug: the sticks are not used at comparable magnitudes (fore/aft
        # runs 0.5-0.9, yaw ~0.2), so dividing by a common ~0.1 made fore/aft score 5-9 against yaw's 2 and
        # translation won essentially every clip it was active in -- measured, lateral and fore/aft won 97-100%
        # of the time they were active while yaw won 5%, giving ONE `rotate right` clip in 1024.
        ac = fc["analytic"]
        means = [float(pro_phys[:, ax].mean()) for ax in ac["axes"]]
        if not any(abs(mu) > db for mu, db in zip(means, ac["deadbands"])):
            return 0.0                                            # nothing above its own deadband -> none
        sc = ac.get("scales") or ac["deadbands"]                  # scales absent -> old behaviour
        k = int(np.argmax([abs(mu) / s for mu, s in zip(means, sc)]))
        scores = means
        return float(1 + 2 * k + (1 if scores[k] > 0 else 0))     # [none, ax0-, ax0+, ax1-, ax1+, ...]
    raise ValueError(f"unknown analytic kind {kind!r}")


def bucketize(kind: str, scalars, fc: dict, r: float | None = None) -> list[str]:
    """Per-clip scalars -> bucket labels. hue_at_position maps each hue to its nearest color center (per-clip,
    independent). speed_quantile splits the scalar distribution at the configured quantile EDGES (self-calibrating
    thirds). z_band thresholds the ambient height against the torus tube radius `r`: |z| > (1-2*frac)*r is the
    outer top/bottom `frac` of the z-range [-r, +r]; everything else is middle."""
    scalars = np.asarray(scalars, dtype=float)
    buckets = list(fc["buckets"])
    if kind == "hue_at_position":
        centers = fc["analytic"]["hue_centers"]
        return [min(centers, key=lambda b: min(abs(h - centers[b]), 1.0 - abs(h - centers[b]))) for h in scalars]
    if kind == "speed_quantile":
        edges = np.quantile(scalars, fc["analytic"]["edges"])     # e.g. [q33, q66] -> 3 bands
        return [buckets[int(np.searchsorted(edges, s, side="right"))] for s in scalars]
    if kind == "z_band":
        thr = (1.0 - 2.0 * float(fc["analytic"].get("frac", 0.1))) * float(r)   # |z| above this -> top/bottom
        return ["top" if z > thr else "bottom" if z < -thr else "middle" for z in scalars]
    if kind == "axis_dominant":
        return [buckets[int(round(v))] for v in scalars]
    if kind in ("obs_value", "obs_norm", "obs_abs", "obs_mean"):
        # `thresholds` = ABSOLUTE cut points in physical units (metres, m/s, rad/s) -- use when the bands
        # mean something on their own ("above 2 m is high"). `edges` = QUANTILES of this run's own scalars,
        # self-calibrating thirds, for quantities whose scale is not known in advance. Absolute wins if both
        # are given, because a self-calibrating band cannot be compared across runs.
        cuts = fc["analytic"].get("thresholds")
        cuts = np.asarray(cuts, dtype=float) if cuts is not None else \
            np.quantile(scalars, fc["analytic"]["edges"])
        assert len(cuts) == len(buckets) - 1, (
            f"{kind}: {len(cuts)} cut points need {len(cuts) + 1} buckets, got {len(buckets)}")
        return [buckets[int(np.searchsorted(cuts, s, side="right"))] for s in scalars]
    raise ValueError(f"unknown analytic kind {kind!r}")


# ------------------------- plotting glue (categorical colors + legend) -------------------------
def point_colors(labels: list[str], fc: dict):
    """Per-point RGB array + legend list [(bucket, hexcolor)] for a factor, from its config color map."""
    import matplotlib.colors as mcolors
    cmap = fc["colors"]
    rgb = np.array([mcolors.to_rgb(cmap.get(l, "#cccccc")) for l in labels])
    return rgb, [(b, cmap[b]) for b in fc["buckets"]]


def confusion(analytic: list[str], vlm: list[str], buckets: list[str]) -> np.ndarray:
    """Count matrix (rows = analytic bucket, cols = VLM bucket) over paired labels."""
    ix = {b: i for i, b in enumerate(buckets)}
    m = np.zeros((len(buckets), len(buckets)), dtype=int)
    for a, v in zip(analytic, vlm):
        if a in ix and v in ix:
            m[ix[a], ix[v]] += 1
    return m
