# Up-backed flow decoder — design plan

**Goal.** Make `decode_kind=flow` available on the image head **without leaving the `up` decoder**, by
sharing one backend between the deterministic decoder (`up-mse`) and a generative denoiser (`flow`). Retire
the rank-512 `ConditionalUNet`/`ViT` image decoders. Add per-level token re-injection on **both** stacks,
implemented behind flags and covered by parity tests.

## Why

- The block-stack cubes fail on **identity** (recolour/merge, incl. stationary), the arm renders great. The
  flow-decode head has **never been run to completion on an image head** (longhand §8.12), and every reason it
  was deferred is a confound: it moved two variables (arch `up`→`unet` *and* head `mse`→`flow`), and the
  historical instability was `flow_hidden=512` (§23), not the head.
- "Flick flow on" today forces the decoder off `up` (mse-only) onto `ConditionalUNet`, whose token→pixel path
  is `cond.mean(1)` + `Linear(T·d→512)` — a rank-≤640 choke that *"annihilated token identity"* (decoders.py
  module docstring). So the only flow-capable decoder is the one that structurally destroys identity — a
  useless substrate for a cube-identity experiment.
- Fix: **one `UpBackend`** (the `up` readout + synthesis), used by both heads. `flow` adds an analysis path
  built on `up`'s principles. `up-mse` is untouched in behavior.

## Non-goals / honest scope

- This is **not** claimed to fix identity. bs_stride10 already uses `up` (the identity-preserving readout) and
  the cubes still fail, so identity is upstream (dynamics/binding, §8.26). This targets the **contact-blur /
  vanish-on-interaction** part (a generative decoder can render one sharp future instead of the conditional
  mean) and **closes the flow-decode confound** cleanly.
- A tie (flow ≈ mse) is a valid, wanted result: it exonerates the decoder and points squarely upstream.

## Scope: additive. The unet chain is foundational — do NOT delete it.

`vl128_blockstack_2cam → vl128 → vl64 → up64 → bsp32mse → mm_flow`. So `mm_flow`, `bsp32mse`, `up64` are
**direct ancestors** of the block-stack recipe (and of `vl128_phys`, the published owm-iss models). Their
`decode_arch=unet` *default* image modality is overridden to `up` by `vl128`, so the chain never uses unet —
but the files are load-bearing and must stay. `ConditionalUNet`/`ImageUNetFlowHead` also still back
`_oneoff_decoder_ab.py` (the up-vs-unet A/B harness) and `smoke/decode_recon.py`.

**Therefore this work is purely additive: keep unet as legacy, add `up-flow`, no global raise.** The
`unet-mse` footgun is prevented **structurally** — from a `decode_arch=up` config the only reachable heads are
`up-mse` and `up-flow`; you cannot land on `unet-mse`. Full unet retirement (rewriting base-recipe defaults,
migrating/deleting the Dreamer configs `mm_gaussian`/`mm_categorical`, reworking the A/B harness) is a
**separate, deliberate migration**, out of scope here.

## The reachable heads

Flow ⇒ must read the noised iterate `x_t` ⇒ needs an analysis path ⇒ *is* a U-Net. So there is no distinct
"up-flow": it **equals** "unet-flow with `up`'s readout." Reachable configs:

| config | class | analysis path? |
|---|---|---|
| `up` + `mse` | `TokenGridDecoder` | no (pure decoder) |
| `up` + `flow` | `UpFlowDecoder` (new) | yes (`_UpAnalysis`) |
| `up` + `mse` + analysis | impossible by construction | — |
| `unet`/`vit` | `ImageUNetFlowHead` / ViT | **kept, legacy** (foundational; block-stack never routes here) |

`unet-mse` is unreachable *from an `up` config*: the `up` mse head (`TokenGridDecoder`) never builds an
analysis path, and `up`+`flow` routes to `UpFlowDecoder`.

## Architecture

### `UpBackend(nn.Module)` — the shared `up` (readout + synthesis)
Holds exactly what `TokenGridDecoder` has today: `TokenPool` (→ `g`), `TokenGridReadout` (→ `r`), `to_ch`,
`mid` `_FiLMResBlock`, the `ups` `_FiLMResBlock` list, the `inject`/`xattn` (Features 2/3), and
`out_norm`/`out_conv`/`out_act`.

```
g_of(cond, temb, demb):                 # pooled global; time added only in flow
    g = TokenPool(cond)
    if temb is not None: g = g + t_proj(temb) + d_proj(demb)
    return g

synthesize(cond, g, seed_extra=None, skips=None):
    r = TokenGridReadout(cond)                       # full-rank per-cell seed
    h = to_ch(r)
    if seed_extra is not None: h = h + seed_extra    # analysis bottleneck (flow)
    h = mid(h, g)
    for i, up in enumerate(ups):
        h = interp(h, x2)                            # resize-conv (up path only)
        if skips is not None: h = h + skips[i]       # ADDITIVE zero-init skip (see below)
        if inject:     h = h + inject[i](resize(r, h))          # Feature 2 (per-level readout map)
        if i in xattn: h = xattn[i](h, cond)                    # Feature 3 (per-level re-attend)
        h = up(h, g)
    return out_act(out_conv(silu(out_norm(h))))
```

### `_UpAnalysis(nn.Module)` — the down path, mimicking `up`
New. Reads the noised iterate; conditions with the **same** mechanisms as the up path (FiLM-`g`, the resampled
readout map, low-res cross-attention), plus an **explicit positional grid** so position never comes from a
padding halo.

```
forward(x, g, cond, r):
    h = in_conv(x) + pos_grid                        # explicit position
    skips = []
    for i, down in enumerate(downs):
        h = down(h, g)                               # FiLM(g), every block  (always on)
        if inject_d:     h = h + inject_d[i](resize(r, h))     # per-level readout map  (flag, zero-init)
        if i in xattn_d: h = xattn_d[i](h, cond)              # per-level re-attend    (flag, zero-init)
        skips.append(skip_proj[i](h))                # ADDITIVE zero-init skip -> up path
        h = avgpool(h)
    return to_bott(h), skips
```

### The two heads
```
TokenGridDecoder(TransportHead):        # up-mse  (no_noise=True)
    back = UpBackend(spec)
    velocity(x,temb,cond,demb): return back.synthesize(cond, back.g_of(cond,None,None))   # x ignored

UpFlowDecoder(TransportHead):           # flow  (no_noise=False, param x0/v, steps>1)
    back = UpBackend(spec);  ana = _UpAnalysis(spec)
    velocity(x,temb,cond,demb):
        g = back.g_of(cond,temb,demb)
        a_bott, skips = ana(x, g, cond, back.readout(cond))
        return back.synthesize(cond, g, seed_extra=a_bott, skips=skips)
```

### Two load-bearing decisions
1. **Additive zero-init skips**, not concatenation. Concatenation changes the up-block input channels, so a
   concat-flow-head could never share conv shapes with `up-mse`. Additive skips through a **zero-init** conv
   keep the up-path shapes identical to `up-mse`, which gives: (a) at init the flow head computes **exactly**
   `up-mse` (clean parity), and (b) the flow head can **warm-start from an `up-mse` checkpoint** (load
   `back`, analysis is a no-op, train it in).
2. **Per-level re-injection on both stacks.** `up` already re-injects on the *up* path (SPADE rationale,
   decoders.py:204 — a single global `g` "washes away semantic information"). The **only new thing** is the
   *down*-path application of the same two modules, justified by the SD U-Net pattern (denoisers condition
   their down blocks too). All new outputs are zero-init, so parity holds until learned into.

## Config / dispatch (`modalities.py`, ImageModality build ~L411)

```
if decode_arch == "up":
    head = TokenGridDecoder(spec) if decode_kind == "mse" else UpFlowDecoder(spec)   # up-mse | up-flow
elif decode_arch == "unet":
    head = ImageUNetFlowHead(...)          # LEGACY, unchanged (mm_flow/bsp32mse defaults, Dreamer, harness)
elif decode_arch == "vit":
    head = ...                             # LEGACY, unchanged
else:
    raise ValueError(f"unknown decode_arch={decode_arch!r}")   # existing guard, unchanged
```
The `up`+`flow` branch is the only new route; the `unet`/`vit` branches are byte-for-byte as they are today.

New `ModalitySpec` flags (all default OFF/zero → bit-identical when unused):
- `decode_down_inject: bool = False` — Feature 2 on the analysis path.
- `decode_down_xattn_max_res: int = 0` — Feature 3 on the analysis path (resolution-gated, low-res only).
- flow controls reuse existing `decode_steps`, `decode_param`, `decode_stochastic`.

## Files
- **`models/decoders.py` → `models/decoders/` package** (the file is already 262 lines and would ~double):
  - `__init__.py` — re-exports `TokenGridDecoder`, `UpFlowDecoder`, `LevelCrossAttn`, `UpBackend` (keeps the 4
    existing `from quickdraw.models.decoders import ...` sites working) + the up-vs-unet rationale docstring.
  - `components.py` — `TokenGridReadout`, `TokenPool`, `LevelCrossAttn` (shared conditioning primitives, unchanged).
  - `backend.py` — `UpBackend`.
  - `analysis.py` — `_UpAnalysis`.
  - `up.py` — `TokenGridDecoder` (up-mse head), refactored to hold an `UpBackend` (forward bit-identical — see tests).
  - `up_flow.py` — `UpFlowDecoder` (flow head).
  - Internal imports shift one level (`from ..flow import TransportHead`, `from ..vision import ...`).
- `models/modalities.py`: add the `up`+`flow` → `UpFlowDecoder` branch (additive; `unet`/`vit` branches
  unchanged) + the two new down-path flags.
- `models/vision.py`, `models/flow.py`: **unchanged.** `ConditionalUNet`/`ImageUNetFlowHead` stay as legacy
  (foundational to the mm_flow/bsp32mse defaults + the `_oneoff_decoder_ab.py` harness + `smoke/decode_recon.py`).
  Full unet retirement is a separate future migration, out of scope.
- `smoke/up_flow_parity.py` (+ `up_flow_parity_golden.pt`): the parity/superset tests below.
- `conf/model/`: a `vl128_blockstack_flow` overlay = bs_stride10 recipe + `decode_kind=flow` on both image
  heads (mirrors how `vl128_phys` overlays `vl128`).

## Invariants (what the tests must lock)
1. **`up-mse` forward bit-identical** to pre-refactor, same seed/weights → `torch.equal`. (State-dict keys
   change under the `back.` nesting; harmless — no `up-mse` checkpoints on this box.)
2. **Superset:** `UpFlowDecoder` with analysis zero-init and both down-flags off, sharing `back` weights with
   a `TokenGridDecoder`, produces `torch.equal` output for any `x`. (Additive zero-init skips make this exact.)
3. **Warm-start:** loading `up-mse` weights into `UpFlowDecoder.back` reproduces invariant 2.
4. **Learnable:** with a down-flag on, output is still `torch.equal` at step 0 (zero-init) **but** the
   flag's parameters receive non-zero gradient on a backward — proving it's a superset that can be learned
   into, not dead.
5. **Dispatch:** `decode_arch=unet` raises; `mse` builds no analysis path; `flow` builds one.

## Test plan (the "testable")
`smoke/up_flow_parity.py`, in the `quickdraw` container, CPU, seeded:
- **T1 up-mse golden:** build `TokenGridDecoder`, forward a fixed input, assert `torch.equal` vs
  `up_flow_parity_golden.pt` (regenerated once, reviewed).
- **T2 superset:** build `UpFlowDecoder`, copy `back` weights from a `TokenGridDecoder`, zero-init analysis,
  flags off → assert `torch.equal` to the T1 output for random `x`, τ.
- **T3 warm-start:** `UpFlowDecoder.back.load_state_dict(up_mse.back.state_dict())` → T2 holds.
- **T4 learnable injection:** enable `decode_down_inject` and `decode_down_xattn_max_res`; assert step-0
  output unchanged (T2) AND `sum(p.grad.abs())>0` for those modules after a dummy backward.
- **T5 dispatch:** `decode_arch=unet` raises; `mse`→no `ana` attribute; `flow`→has `ana`.
- **T6 shape/mem:** one forward+backward at 96×128, 2 cams, `decode_steps=6`, to size the batch for the run
  (flow decoders are memory-heavy — §18: ~78% of per-sample memory is the decoder, two passes).

## Experiment plan (after the build lands)
All fork **bs_stride10** (already `flow_hidden=128`), **batch pinned** (kill the autobatch confound of
§8.17/§8.20), evals to **ep13+**, judged on **the MP4s at contact moments** + `OL − floor`, **warm-started
from an up-mse checkpoint** where available:
- **GPU 0 — baseline:** bs_stride10 (`up-mse`) — the reference (fire except cubes).
- **GPU 1 — flow, minimal:** `UpFlowDecoder`, analysis on, **down-injection OFF** — differs from `up`+analysis
  by nothing but the noise curriculum. The clean flow-vs-mse read.
- **Follow-up (next pair):** flow + **down per-level injection ON** — the A/B for whether re-injecting tokens
  through the analysis stack helps, exactly as it did on `up`'s synthesis stack.

## Risks / open questions
- **Skips vs token conditioning.** Additive `x`-skips carry the noised image's structure and compete with the
  token path; a lazy denoiser could lean on skips at low noise. Mitigation: skips are zero-init (must be
  learned in) and the full-rank readout is injected at the bottleneck and (optionally) every level.
- **Memory.** Flow + 2 cams + `decode_steps` is heavy; T6 sizes the pinned batch. Expect a smaller batch than
  the mse baseline — keep it matched across arms.
- **It may not help identity.** Expected. The value is (a) the contact-blur test and (b) closing the
  longest-standing decode confound at the correct base. Result feeds the object-centric decision either way.
