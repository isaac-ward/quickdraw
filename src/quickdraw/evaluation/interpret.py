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


def build_label_schema(factors: dict) -> dict:
    """Strict json_schema for the structured VLM output. A free-text `reasoning` field comes FIRST
    (reason-then-answer: describing what it sees before committing lifts accuracy), then one enum field per
    factor, then a confidence."""
    props: dict = {"reasoning": {"type": "string"}}
    for name, fc in factors.items():
        props[name] = {"type": "string", "enum": list(fc["buckets"])}
    props["self_reported_confidence"] = {"type": "number"}
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


def build_action_text(actions: np.ndarray) -> str:
    """Compact text payload: the clip's action sequence (subsampled to <=10 rows), one (a0,a1) per line."""
    a = np.asarray(actions, dtype=np.float32)
    idx = np.unique(np.linspace(0, len(a) - 1, min(10, len(a))).round().astype(int))
    rows = "\n".join(f"  t={int(i):2d}: ({a[i, 0]:+.3f}, {a[i, 1]:+.3f})" for i in idx)
    return f"Agent action sequence over the clip (per step, {a.shape[1]} dims):\n{rows}"


# ------------------------- analytic labels (exact, from the imagined proprio) -------------------------
def analytic_scalar(kind: str, pro_phys: np.ndarray, R: float) -> float:
    """One scalar per clip for an analytic factor; `bucketize` turns a set of them into labels. pro_phys: (H,6)
    physical proprio [xyz, velocity]."""
    if kind == "hue_at_position":                                 # hue at the agent's mid-clip ring angle (the surface it's on)
        pos = pro_phys[len(pro_phys) // 2, :3]
        return float((np.arctan2(pos[1], pos[0]) / (2 * np.pi)) % 1.0)
    if kind == "speed_quantile":                                  # mean |velocity| over the clip (proprio dims 3:6)
        return float(np.linalg.norm(pro_phys[:, 3:6], axis=1).mean())
    raise ValueError(f"unknown analytic kind {kind!r}")


def bucketize(kind: str, scalars, fc: dict) -> list[str]:
    """Per-clip scalars -> bucket labels. hue_at_position maps each hue to its nearest color center (per-clip,
    independent). speed_quantile splits the scalar distribution at the configured quantile EDGES, so the bands
    are self-calibrating thirds of the observed data (no magic thresholds)."""
    scalars = np.asarray(scalars, dtype=float)
    buckets = list(fc["buckets"])
    if kind == "hue_at_position":
        centers = fc["analytic"]["hue_centers"]
        return [min(centers, key=lambda b: min(abs(h - centers[b]), 1.0 - abs(h - centers[b]))) for h in scalars]
    if kind == "speed_quantile":
        edges = np.quantile(scalars, fc["analytic"]["edges"])     # e.g. [q33, q66] -> 3 bands
        return [buckets[int(np.searchsorted(edges, s, side="right"))] for s in scalars]
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
