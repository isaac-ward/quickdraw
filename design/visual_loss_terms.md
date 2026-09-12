# VisualLoss: one term list instead of two

**Status: PLAN. Do not land while `bs_deriv_w1` / `bs_deriv_w10` are in flight** — not because an
edit on disk touches them (Python does not reload), but because a crash-and-resume would pick up
new code mid-experiment, and the sweep's value is that the arms differ in exactly one number.

## The problem

`forward()` and `temporal()` each carry their own `if self.w_*:` chain. A term added to one and not
the other is silently absent from half the objective, and nothing fails.

Everything else is already shared: `self.visual` is ONE instance per modality
(`modalities.py:381`), reached by both `recon_loss` (`:467`) and `derivative_loss` (`:459`).

## The change

Each term becomes an object with two methods; the class keeps one assembly loop.

```python
class Term:
    name: str
    def weight(self, vl) -> float: ...
    def pointwise(self, p, t) -> Tensor: ...              # (M,H,W,C) x2
    def difference(self, p0, p1, g0, g1) -> Tensor | None # contiguous pairs; None = no temporal form
```

Terms take the difference THEMSELVES, so a feature term can difference EMBEDDINGS rather than embed
a difference — the distinction `temporal()` already makes, which must survive.

```python
def _assemble(self, mode, *args):
    total = zeros
    for t in self._terms:
        w = t.weight(self)
        if not w: continue
        part = t.pointwise(*args) if mode == "pointwise" else t.difference(*args)
        if part is not None: total = total + w * part
    return total
```

`forward()` = flatten -> record -> subsample -> `_assemble("pointwise", ...)`
`temporal()` = rank-5 assert -> per stride `_pairs` -> `_assemble("difference", ...)`

**Sampling stays outside the terms.** `_subsample` (randperm over flattened rows) and `_pairs`
(contiguous pairs) are genuinely different and both load-bearing — `_subsample` would give
`(b=3,t=17), (b=0,t=52)...`, useless for differencing.

Side effect worth having: `w_l2`'s current SILENT absence from `temporal()` becomes an explicit
`return None` with its reason attached (a temporal difference image is sparse; L2 lets the largest
change swamp the rest).

## How to not break it

**Write `smoke/visual_loss_parity.py` first, against UNCHANGED code.** Fixed seed, every weight
combination (each of l2/l1/lpips alone, the pairs, all three), `frames` on and off, rank-4 and
rank-5 input, fp32 and bf16 autocast, `strides=(1,)`. Commit the golden. Then refactor and assert
`torch.equal` — not `allclose`.

That is the entire safety argument. If the harness does not exist and pass before the refactor, the
refactor proves nothing.

## What this is NOT

Not a vehicle for adding a critic. That is an open question with its own investigation
(`design/` TBD) and nothing about it is decided. This change is worth making on its own: it removes
a footgun that exists today.

## The seam that stays: two LPIPS distances

`forward()` and `temporal()` compute different functions of the same VGG features, deliberately:

* `_lpips_term` calls **torchmetrics as a black box**, learned per-layer weights included.
* `temporal()` uses `_layer_features` — raw unit-normalised features, squared difference of
  differences — and deliberately does NOT apply those learned weights, which were fitted to human
  judgements of IMAGE similarity with nothing calibrating them for temporal DIFFERENCES.

They share the network and the `_sanitise` guard, not the distance. Unifying them means
reimplementing torchmetrics' internals, which would change the `visual_lpips` number 25+ historical
runs are ranked on — §24's whole comparison table is LPIPS values. Keep the seam; keep them as two
`Term` objects if that reads clearer.
