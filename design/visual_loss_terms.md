# VisualLoss: one term registry, two reductions

**Status: PLAN. No code written. Do not start while `bs_deriv_w1` / `bs_deriv_w10` are in flight.**

## 0. Why

Adding a term to `VisualLoss` today means editing it in **two places** — `forward()` and
`temporal()` each carry their own `if self.w_*:` chain. That is the footgun: a term added to one
and not the other is silently absent from half the objective, and nothing fails.

The goal is that **declaring a term once makes it apply everywhere** — the ordinary decode loss,
the roundtrip anchor, the derivative term, and every modality that owns a `VisualLoss`. That
property is already half-true: `self.visual` is ONE shared instance per modality
(`modalities.py:381`), reached by both `recon_loss` (`:467`) and `derivative_loss` (`:459`). Only
the term list is duplicated.

Immediate motivation: a **per-patch cosine critic** (P-DINO style, `design/` TBD) to target
patch-level semantic persistence over long rollouts. Measured on `bs_stride10`, the predicted
frame-to-frame change sits at `cos = 0.07` against truth — equivalent to a ~6-8 px displacement on
a 128-wide frame, where a 1 px error would score 0.85. Objects are being repainted in the wrong
places. LPIPS cannot see this well because it reduces with a spatial `.mean()` over a frame that is
84.6% static, diluting the moving region 5x (measured, scene_right at stride 10).

## 1. What must NOT change

Every number 25+ historical runs are ranked on. Concretely:

* `forward()` must stay **bit-identical** for every existing weight combination.
* `temporal()` must stay **bit-identical** at `derivative_strides=(1,)`.
* The torchmetrics LPIPS call in `_lpips_term` stays a **black box**. See §5.

## 2. The shape

Each term becomes an object with two methods, and the class holds a list of them:

```python
class Term(Protocol):
    name: str
    def weight(self, vl: "VisualLoss") -> float: ...
    # POINTWISE: compare two frame batches directly.        (M,H,W,C) x2 -> scalar
    def pointwise(self, p: Tensor, t: Tensor) -> Tensor: ...
    # DIFFERENCE: compare two temporal DIFFERENCES, given contiguous pairs.
    #   p0,p1 = predicted frames t and t+k; g0,g1 = the true ones.
    #   Terms take the difference THEMSELVES so a feature term can difference EMBEDDINGS
    #   rather than embed a difference -- the distinction visual_loss.temporal already makes
    #   and which must survive the refactor.
    def difference(self, p0, p1, g0, g1) -> Tensor | None: ...
```

`difference` returning `None` means "this term has no temporal form" — which is how `w_l2`'s
current silent absence from `temporal()` becomes **explicit and greppable** instead of an
undocumented divergence a reader has to notice.

`VisualLoss` then keeps ONE assembly method:

```python
def _assemble(self, mode, *args) -> Tensor:
    total = ...zeros
    for term in self._terms:
        w = term.weight(self)
        if not w: continue
        part = term.pointwise(*args) if mode == "pointwise" else term.difference(*args)
        if part is not None:
            total = total + w * part
    return total
```

* `forward()`  = `_flatten` -> `_record` -> `_subsample` -> `_assemble("pointwise", p, t)`
* `temporal()` = rank-5 assert -> per stride `_pairs` -> `_assemble("difference", p0,p1,g0,g1)`

**Sampling stays where it is, outside the terms.** `_subsample` (randperm over flattened rows) and
`_pairs` (contiguous pairs) are genuinely different and both are load-bearing — `_subsample` would
give `(b=3,t=17), (b=0,t=52)...`, useless for differencing. The terms should never see that choice.

## 3. Phases

**Phase 0 — golden parity harness, BEFORE any refactor.** `smoke/visual_loss_parity.py`:
capture `forward()` and `temporal()` outputs at fixed seed over a matrix of weight combinations
(each of l2/l1/lpips alone, all pairs, all three; `frames` on and off; rank-4 and rank-5 input;
`strides=(1,)`; bf16 autocast and fp32). Serialise to a golden `.pt` committed alongside. This is
the contract §1 asks for, and it must exist and pass against UNCHANGED code first, or it proves
nothing.

**Phase 1 — extract the three existing terms**, no behaviour change. L2 and L1 are trivial. LPIPS
wraps the existing `_lpips_term` for `pointwise` and the existing `_layer_features` path for
`difference`, called verbatim. Assert `torch.equal` against the Phase 0 golden — not `allclose`.

**Phase 2 — make the divergences explicit.** `L2Term.difference` returns `None` with the reason in
its docstring (a temporal difference image is sparse; L2 lets the largest change swamp the rest).
Add `_record` to the temporal path so `pop_diagnostics` covers the derivative site, which it
currently does not.

**Phase 3 — add the new critic, default weight 0.** One class, two methods. Prove
`w_patchcos=0` is byte-identical to the Phase 0 golden. Only then wire a config knob.

Phases 0-2 are a pure refactor and land together or not at all. Phase 3 is a separate commit.

## 4. Why not to start now

Running processes do not reload code, so `bs_deriv_w1` and `bs_deriv_w10` are safe from an edit on
disk. But a crash-and-resume would pick up new code MID-EXPERIMENT, and the whole point of the
sweep is that the two arms differ in exactly one number. Land it after the sweep reports.

## 5. The seam that stays: two LPIPS distances

`forward()` and `temporal()` compute genuinely different functions of the same VGG features, and
this refactor deliberately does NOT unify them.

* `_lpips_term` calls the **torchmetrics metric as a black box**, including its learned per-layer
  weights.
* `temporal()` uses `_layer_features` — raw unit-normalised features, then a squared difference of
  differences — and deliberately does NOT apply LPIPS's learned weights, because those were fitted
  to human judgements of IMAGE similarity and nothing calibrates them for the similarity of
  temporal DIFFERENCES.

They share the NETWORK (one process-level cache) and the `_sanitise` guard — not the distance.
Unifying them would mean reimplementing torchmetrics' internals, which would change the
`visual_lpips` number that every historical run is ranked on. The seam is a deliberate, documented
trade. Keep it, and keep the two LPIPS terms as two separate `Term` objects if that reads clearer.
