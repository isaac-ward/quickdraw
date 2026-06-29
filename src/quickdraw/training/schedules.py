"""Generic training schedules (linear ramps over epochs).

One small reusable component shared by every scheduled quantity — the p_tf teacher-forcing curriculum
and the physical-loss warmup — so they all use the same code and are logged together under `schedules/`.
"""

from __future__ import annotations


def linear_schedule(start: float, end: float, warmup_epochs: float, epoch: float) -> float:
    """Linear ramp from `start` (epoch 0) to `end` (reached at `warmup_epochs`, then held). A
    non-positive `warmup_epochs` means no ramp -> constant `end`."""
    if warmup_epochs <= 0:
        return end
    frac = min(1.0, max(0.0, epoch) / warmup_epochs)
    return start + (end - start) * frac
