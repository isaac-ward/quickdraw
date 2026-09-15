"""HUMAN-REVIEWED anomaly windows for the starling-2 OOD splits -- the ground truth.

The detectors in `scratch/analyse_ood_anomaly.py` are a REVIEW TOOL: they put a candidate span on a plot
and in a video so a person can check it. This file is the result of that review (2026-09-15), and it is
what any measurement should use. Two things it records, per episode:

    exclude   the episode carries no usable anomaly (the noodle never really entered, or the blower
              produced no motion above the hover noise), so it must not be scored.
    start/end the frame range over which the anomaly is actually present, at the dataset's 15 Hz.

WHY THIS LIVES IN THE REPO. It is dataset annotation, not run output, and `logs/` is gitignored -- putting
it there would lose it. It should eventually be pushed into the HF dataset's own metadata so someone who
pulls starling-2 gets the windows with the data; until then this module is the single source.

HOW THE REVIEW CHANGED THE DETECTORS' ANSWERS, kept because it says what the detectors are worth:
  noodle      12/12 detected, 10 kept. Two episodes dropped; four spans corrected -- three of them because
              the noodle was still in frame when the clip ended and the detector closed the span early.
  leafblower  8/12 detected, 7 kept. Five dropped (four of which the detector had already declined to
              flag, which is agreement, not failure); three spans corrected, all start times.
"""
from __future__ import annotations

FPS = 15.0

# episode -> None (exclude) or (start_frame, end_frame) inclusive-exclusive, at 15 Hz
WINDOWS: dict[str, dict[int, tuple[int, int] | None]] = {
    "eval_ood_noodle": {
        0: None,            # reviewed out
        1: (0, 66),
        2: (0, 72),
        3: (0, 68),         # detector ran to the end of the clip; review: finishes at 4.5 s
        4: None,            # reviewed out
        5: (82, 118),       # review: starts at 5.5 s (detector opened at 6.7 s)
        6: (68, 126),
        7: (34, 137),       # review: runs to the end of the video (detector closed at 6.5 s)
        8: (70, 146),       # review: runs to the end of the video (detector closed at 6.1 s)
        9: (4, 92),
        10: (0, 78),
        11: (0, 31),
    },
    "eval_ood_leafblower": {
        0: (30, 47),        # review: starts at 2.0 s (detector opened at 2.6 s)
        1: (99, 109),
        2: None,            # reviewed out
        3: None,            # reviewed out
        4: (52, 72),        # review: starts at 3.5 s (detector opened at 4.1 s)
        5: None,            # reviewed out
        6: (56, 63),
        7: None,            # reviewed out
        8: None,            # reviewed out
        9: (113, 121),
        10: (75, 98),       # review: 5.0-6.5 s (the detector had found only the first 0.5 s of the clip)
        11: (11, 18),
    },
}

# The memory splits carry no anomaly window, only an exclusion: eval_memory_backwall2 ep02's away-and-back
# returns at step 28 of 28, so the return cannot fall inside any rollout that leaves room for the context.
EXCLUDE_EPISODES: dict[str, list[int]] = {"eval_memory_backwall2": [2]}


def windows(split: str) -> dict[int, tuple[int, int] | None]:
    """The reviewed windows for a split, or {} when the split has none (train/val/memory)."""
    return WINDOWS.get(split, {})


def kept(split: str) -> list[int]:
    """Episode indices that survived review, in order. Every other episode must be dropped."""
    w = WINDOWS.get(split)
    if w is not None:
        return [i for i, v in sorted(w.items()) if v is not None]
    return [i for i in range(1000) if i not in EXCLUDE_EPISODES.get(split, [])]


def window_steps(split: str, episode: int, subsample: int) -> tuple[int, int] | None:
    """The window in MODEL steps at a given frame stride -- what a rollout is indexed by."""
    v = WINDOWS.get(split, {}).get(episode)
    return None if v is None else (v[0] // subsample, -(-v[1] // subsample))
