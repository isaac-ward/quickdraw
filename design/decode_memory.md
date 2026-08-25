# Decode memory: chunk + checkpoint the decoder forward

Status: **PLAN ONLY. No code changed.** 2026-08-25.

## The problem

`recon_frac=1.0` — the largest measured lever on long-horizon quality (OL LPIPS@+128 0.374 -> 0.278) — costs
**13.17 GB/sample**, so autobatch picks **batch 7** on a 93 GB budget. The card is already 99% full; the
issue is per-sample cost, not utilisation.

## Where the memory is (two independent derivations, agreeing to 2%)

| derivation | decode | non-decode |
|---|---|---|
| `recon_frac` 0.25 -> 1.0 at fixed `ae_bottleneck=16` (5.34 -> 12.93 GB/sample; +7.59 for 48 more frames) | 10.1 | 2.83 |
| `decode_base` 32 -> 64 (13.17 -> 23.50 GB/sample; decode ∝ base) | 10.33 | 2.84 |

**~78% of per-sample memory scales with the decoder** — but see the CRITICAL CORRECTION above: that 78% is
split across TWO decoder passes, not the one the first draft proposed to chunk.

Audit notes on these derivations: the fit slope IS the marginal per-sample slope (peak RESERVED at two batch
sizes, `setup.py:518-519`), and linearity in `decode_base` IS valid despite the U-Net pyramid, because every
level's channel count is `base * min(4, 2^i)` (`vision.py:386`) so every activation tensor is ∝ base — the
quadratic-in-base terms (conv weights, Adam states) land in the intercept, not the slope. BUT the two
derivations do not measure the same bucket: (a) includes the probe's roundtrip TARGET RE-ENCODE (the probe
omits `pre_z_targets`, `setup.py:473`) which scales with `recon_frac`; (b) does not, since `encode_base` is
fixed. They also differ in bottleneck config (a is `ae_bottleneck=16`, 3 U-Net levels; the headline 13.17 is
bottleneck 8, 4 levels) and in fragmentation regime (0% vs 3%). **The "agreeing to 2%" cross-check is
therefore weaker than it reads** — the mechanism is confirmed, the precision is not. Every other knob (`d`, `depth`, `window`, `num_tokens`,
`ae_bottleneck`, `flow_arch_depth`) lives inside the 2.84 GB and cannot buy much. This is why the decode is
the only lever worth pulling.

Measured alternatives, for the record: `data.F` 64->48 gives batch 9; `decode_base` 32->24 gives batch 8;
both together give 11 — and each costs something measured as mattering (trained horizon; decoder capacity).

## CRITICAL CORRECTION (adversarial audit, 2026-08-25): there are TWO decoder passes

The first version of this plan chunked `decode_loss` only. **Every run these numbers come from also pays an
equal-size SECOND decoder forward** — the roundtrip anchor (`latent_loss_weight: 10` in all of them):

    lit.py:167 -> recon_losses -> roundtrip_losses (multimodal.py:371) -> to_obs (:398, :326-334)
                -> Modality.decode (modalities.py:107-114, flattens (B,T) exactly like decode_loss)
                -> decode_head.sample -> _sample -> self.velocity(...)   <-- flow.py:131

For the mse/no_noise head this is **byte-for-byte the same `velocity(zeros, temb, cond)` call** as the loss
path at `flow.py:94`, so its activation footprint is equal. Its backward runs BEFORE `decode_loss`'s, so
both passes are alive at the peak. The 10.33 GB is therefore ~5.2 (loss) + ~5.2 (roundtrip), not 10.33 in
one place — and derivation (b) confirms it, since doubling `decode_base` doubles BOTH passes.

**Payoff, corrected:**

| version | per-sample | batch |
|---|---|---|
| today | 13.17 | 7 |
| chunk `decode_loss` only (the original plan) | ~8.0 + a batch-independent ~5.2 | **~10-11, NOT ~20** |
| **chunk BOTH decode sites** | ~2.84 + ~5.2 constant | **~29-30** |

So the original plan was worth less than claimed, and the CORRECTED plan is worth more. Both live decode
heads take `_sample`'s single-call path — image is `decode_kind: mse` -> no_noise (`flow.py:130-132`),
proprio is flow-`x0` with `steps=1` (`modalities.py:112`, first iteration at `flow.py:141`) — so routing
those two calls through the same helper covers it without touching the Euler loop.

**Unstated consequence, worth having:** with both sites chunked, per-sample memory stops depending on
`recon_frac` at all. The 1.0 lever becomes memory-free (compute-only).

## The mechanism

`decode_loss` (`modalities.py:116-121`) flattens `(B,T)` into ONE batch of `B*T` frames — 448 at batch 7,
F=64 — and calls `decode_head.loss` once. Autograd retains every intermediate activation for all 448 frames.
That is the 10 GB.

**Checkpointing alone does NOT fix this.** Wrapping the whole decode in `torch.utils.checkpoint` frees
activations at forward time and recomputes them during backward — at which point the recompute materialises
all 448 frames' activations at once while everything else is still alive. Peak unchanged. **Chunking is what
makes it work**; the checkpoint is what makes each chunk cheap.

(Ordering detail the first draft got wrong: decode is NOT last in the forward — `recon_losses` runs at
`lit.py:167`, before `loss_terms` at `:174`. The conclusion survives regardless: the decode conditions on
the predicted bag, so rollout activations cannot free until the decode backward completes, and the
roundtrip pass backwards before `decode_loss` — everything is co-resident at the peak.)

Eval's existing `decode_chunk` (`imagine_eval`) is NOT a port of this: it chunks `to_obs` under `no_grad`,
where there are no activations to save, so it never needed the checkpoint half.

## The design: chunk the FORWARD, keep ONE loss

The naive version chunks the loss and accumulates. Avoid it: `F.mse_loss` returns a MEAN, so mean-of-means
is wrong unless every chunk is equal-sized (the last never is), and the flow decoder returns TWO losses
(main + shortcut) both needing weighting. Easy to get subtly wrong, and hard to verify.

Instead chunk only the `velocity` forward and concatenate its outputs, then compute the loss once over the
full tensor. This works because the decoder's OUTPUTS are tiny (448 frames x 128x128x3 bf16 = 44 MB) while
its INTERMEDIATES are the 10 GB.

```python
# flow.py, on TransportHead
def _chunked_velocity(self, x, temb, cond, demb, chunk):
    if not chunk or x.shape[0] <= chunk:
        return self.velocity(x, temb, cond, demb)          # default path: bit-identical
    outs = []
    for i in range(0, x.shape[0], chunk):
        sl = slice(i, i + chunk)
        outs.append(torch.utils.checkpoint.checkpoint(
            self.velocity, x[sl], temb[sl], cond[sl],
            None if demb is None else demb[sl], use_reentrant=False))
    return torch.cat(outs, 0)
```

`TransportHead.loss` then calls `self._chunked_velocity(...)` in place of `self.velocity(...)`. No
accumulation, no weighting, no per-chunk bookkeeping.

## Scope: this is a DECODE-HEAD feature, not a model feature

`decode_loss` is called from `recon_losses`, which `lit.py` runs for EVERY model class — so mm_flow,
mm_lsar, mm_dsar and mm_categorical/gaussian all benefit identically.

| head | class | covered |
|---|---|---|
| proprio (vector) | `FlowField` (`modalities.py:145`) | yes — TransportHead |
| image, unet decode | `ImageUNetFlowHead` (`:183`) | yes |
| image, vit decode | `ImageFlowHead` (`:186`) | yes |
| pretrained TAESD | `PretrainedImageHead` (`:266`) | **yes** — it IS a TransportHead (`modalities.py:203`); the first draft said otherwise. Moot (pretrained is ruled out) but the table was wrong. |

**`FlowField` is ALSO the dynamics head** (`multimodal.py:925` — the first draft cited `:717`, wrong) **and
the action head** (`:962`, whenever `action_head_enabled`). Per-head opt-in is still the right design, but
the audit downgrades this from a correctness trap to a PERFORMANCE one: chunking on dim 0 splits the batch,
which is mathematically exact either way (velocity is per-element on dim 0; the transformer arch mixes only
over dim -2, `flow.py:239-241`). Default `chunk=0` on `TransportHead`, opt in per head.

## Plan

1. `flow.py::TransportHead.__init__` — accept `chunk: int = 0`, store it. Default 0 = off.
2. `flow.py::TransportHead` — add `_chunked_velocity` as above.
3. `flow.py::TransportHead.loss` — route the `velocity` call(s) through it. `_consistency` needs only its
   `v_2d` call (`:121`); the other two are under `no_grad` (`:116-119`) so checkpointing them buys nothing.
   Moot for the live config anyway (image=mse, proprio=x0 are single-call branches).
3b. **`flow.py::_sample` — route the no_noise call (`:131`) and the x0 first-step call (`:141`) through it
   too.** This is the roundtrip-anchor pass and it is HALF the decode memory (see CRITICAL CORRECTION).
   Without this the plan lands at batch ~10-11 instead of ~29-30.
4. `modalities.py` — thread `spec.decode_chunk_train` into the ImageUNetFlowHead / ImageFlowHead / FlowField
   DECODE-head constructors only. NOTE all three have CLOSED signatures (no `**kwargs`: `flow.py:196`,
   `:258`, `:313`), so each needs the parameter added explicitly — step 1 alone is not sufficient.
5. `ModalitySpec` — `decode_chunk_train: int = 0`.
6. `conf/model/mm_flow.yaml` — set it on the image modality.

## Verification (the point of this design: it is exactly checkable)

* `chunk=0` -> bit-identical loss, unchanged memory.
* `chunk=64` -> loss equal to the unchunked value **to floating-point tolerance** (same arithmetic, same
  order after `cat`), and measured peak memory lower.
* autobatch re-probes to a larger batch on the same config.

## Risks to settle before implementing

* **Non-determinism inside `velocity`: CLEARED by audit.** No dropout or RNG anywhere in `vision.py` or
  `flow.py`'s velocity paths; `tau`/`eps` are sampled outside in `loss`. Recompute is deterministic, and
  `preserve_rng_state=False` can be passed to skip the per-chunk RNG stash.
* **`cond` broadcasting: CLEARED.** Never broadcast — image `(M,T,d)` (`modalities.py:191-192`), proprio
  `(M,d)` (`:152-153`), and the U-Net takes M from `cond.shape[0]` (`vision.py:405`).
* **flex_attention/BlockMask: CLEARED.** Decode heads use plain SDPA (`vision.py:76,:96`; `flow.py:187`);
  flex lives only inside the compiled `_rollout_step`, which the decode never enters.
* **checkpoint x autocast** — needs `use_reentrant=False` to carry the autocast state into the recompute.
* **`torch.compile`** — no interaction expected; only `_rollout_step` is compiled, the decode is not.
* **Compute** — the decoder forward runs twice. The "~+13% overall" figure rests on "decode is ~40% of step
  time", which is **unverified by any log**, and correcting for the second decode pass applies the 1.33x to
  a larger share. Expect somewhat more than +13%, still second-order against ~4x fewer optimizer steps at
  batch ~29.
