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


_REDUCE = {"mean": lambda v: sum(v) / len(v), "min": min, "max": max,
           "first": lambda v: v[0], "last": lambda v: v[-1]}


def _pick(d: dict, pat: str, match, reduce):
    """One value for `pat`, or a LOUD failure if the pattern is ambiguous.

    Returning v[-1] on a multi-match is how Table III came to report rest AUC for dimension 15 and W_1 at
    whichever lead happened to be logged last: both patterns match many tags, both silently resolved to an
    arbitrary one, and the column header claimed an aggregate. A caller that matches more than one tag now
    has to say which reduction it means."""
    hits = {k: val for k, val in d.items() if match(k, pat)}
    if not hits:
        return None
    if len(hits) == 1:
        return next(iter(hits.values()))
    if reduce is None:
        raise KeyError(f"{pat!r} matches {len(hits)} tags -- pass reduce= one of {sorted(_REDUCE)}. "
                       f"Matched: {sorted(hits)[:6]}{' ...' if len(hits) > 6 else ''}")
    if reduce not in _REDUCE:
        raise KeyError(f"reduce={reduce!r} unknown; pick one of {sorted(_REDUCE)}")
    return _REDUCE[reduce](list(hits.values()))


def suffix(d: dict, pat: str, reduce: str | None = None):
    """The value whose tag ENDS WITH pat -- the horizon-indexed image/proprio curves."""
    return _pick(d, pat, lambda k, p: k.endswith(p), reduce)


def contains(d: dict, pat: str, reduce: str | None = None):
    """The value whose tag CONTAINS pat. Ambiguous patterns must name a reduction."""
    return _pick(d, pat, lambda k, p: p in k, reduce)


def fmt(x, p=3, dash="--"):
    return dash if x is None else f"{x:.{p}f}"
