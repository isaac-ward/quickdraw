"""Pull the paper's numbers out of the run directories, once, so no table is typed by hand.

Every run writes `logs/metrics.jsonl` (one {step, tag, value} per line) and `logs/config.json`. This reads
the LAST value of each tag, which is the final-epoch reading, and exposes the few lookups the tables need.
Nothing here computes a metric -- if a number is not already in a run's metrics, it does not belong in a
table, because it was never produced by the pipeline that produced the rest.
"""
from __future__ import annotations

import json
import os


def metrics(run: str) -> dict:
    """{tag: last value} for one run directory."""
    p = os.path.join(run, "logs", "metrics.jsonl")
    out = {}
    if not os.path.exists(p):
        return out
    for line in open(p):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("tag") is not None:
            out[r["tag"]] = r["value"]
    return out


def config(run: str) -> dict:
    p = os.path.join(run, "logs", "config.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def suffix(d: dict, pat: str):
    """Last value whose tag ENDS WITH pat -- the horizon-indexed image/proprio curves."""
    v = [val for k, val in d.items() if k.endswith(pat)]
    return v[-1] if v else None


def contains(d: dict, pat: str):
    v = [val for k, val in d.items() if pat in k]
    return v[-1] if v else None


def fmt(x, p=3, dash="--"):
    return dash if x is None else f"{x:.{p}f}"
