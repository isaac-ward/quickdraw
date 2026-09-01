# Two camera streams: a second image input AND a second prediction head

Status: **PLAN ONLY. No code changed.** 2026-09-01. Requested by the user; code survey by a subagent,
every claim below carries a `file:line` and was spot-checked against the source.

## What we are building

Two `ImageModality` entries in the `modalities` list, each with its own camera. Each gets its own encoder,
its own decode head, its own `VisualLoss` and its own roundtrip anchor. The dataset already has the data:
`isaac-ronald-ward/robocasa-scene4-4h` carries `robot0_agentview_left`, `robot0_agentview_right` and
`robot0_eye_in_hand` as of `25dfe59`.

### The latents ARE fused — this is not two independent world models

Verified in source, because it is the whole point of the exercise:

| stage | fused? | where |
|---|---|---|
| encoders | **no** — one per camera, 32 tokens each | `multimodal.py:322` |
| latent bag | concatenated, then LayerNormed as a whole | `multimodal.py:323-324` |
| **dynamics** | **YES — spatial attention over ALL 65 tokens** | `spacetime.py:74` (`s_attn` over the N-slot axis) |
| decoders | **no** — each reads only its own slice `bag[..., off:off+n, :]` | `multimodal.py:331` |
| roundtrip anchor | **no** — per-modality, `Dec1(Enc1(x1)) ~ x1` only | `multimodal.py:377-421` |

So the transition function is forced to model both views jointly (predicting camera 2's next tokens is
easier using camera 1's, and the transformer will find that), while each codec stays private. That is the
intended design. `modalities.py:6-8` already promises it: *"Later: image1, image2, ... with no code
change."*

NOT included, and worth naming so nobody assumes it: nothing forces camera 1's 32 tokens to CONTAIN
camera-2 information. Rendering one view from the OTHER view's tokens (which would force genuinely
view-agnostic 3-D state in the codec, not just in the dynamics) is a small addition on top of this plan,
not part of it.

## What needs NO change (survey conclusion, and the reason this is cheap)

* **`models/multimodal.py` and `models/modalities.py` — nothing.** `self.layout`
  (`multimodal.py:126-127`), `encode_state` (`:320-324`), `to_obs` (`:326-336`), `recon_losses`
  (`:338-375`), `roundtrip_losses` (`:377-421`), `arch_table`, `_imagine`/`imagine_eval` all iterate the
  layout. There is no `next((n for n,_ in layout if n != "proprio"))` idiom anywhere in either file.
  Backbone (`n_slots=self.n_input`), `FlowField(n_tokens=self.n_state)` and `dist_head.build` are all
  sized from the layout.
* **`training/lit.py` — nothing.** obs is built from the layout (`:108-111`); per-head losses
  (`:229-230` `train/loss/decode/<name>`), per-head val metrics (`:256-263` `val/metric/<name>/*`) and
  per-module grad norms (`:315-318` `decode_<mod>` / `encode_<mod>`) are all name-keyed. The
  `recon_frac` subset uses ONE shared frame index across heads (`:164-171`) — correct for two cameras.
* **autobatch — nothing.** `synth` builds one tensor per image spec (`setup.py:460-468`),
  `_resident_frame_bytes` sums over ALL image specs (`:415-428`), `probe_eval` passes every head
  (`:637`). It will size the batch for two decoders on its own and print the result (`[batch]` line,
  `005cf5b`).
* **eval metric keys — no collisions.** Already `<routine>/<head>/<leaf>` (`products.py:25-28`) and
  `{routine}/{head}/{stat}/@+{x}` (`openloop.py:140-151`).
* **`trainer.checkpoint_monitor` — already selectable by head.** `train_world_model.py:283-286` takes an
  explicit override; only its DEFAULT is first-image. With two heads pass e.g.
  `trainer.checkpoint_monitor=val/metric/cam_left/mse`. `checkpoint_mode` auto-derives min/max from the
  metric suffix (`:290-293`).

## THE CHECKLIST

### Phase 1 — training end to end (5 items, all mechanical)

- [ ] **1. `ModalitySpec` gains `cam: str | None = None`** — `models/modalities.py:29-151`.
      Today `_modality_specs` does `ModalitySpec(**kw)` (`setup.py:75`), so an unknown `cam:` key in the
      yaml raises `TypeError`. Resolution order: per-modality `cam` -> `cfg.data.cam` -> `"fpv"`, so every
      existing single-camera config keeps working untouched.
      Also add a **name-uniqueness assert** in `build_modalities` (`modalities.py:496-498`): it is a dict
      comprehension, so two modalities with the same name silently OVERWRITE each other today.
      *Verify:* `--cfg job --resolve` on a two-image yaml shows both `cam:` values.

- [ ] **2. `load_split_episodes_mm` serves several cameras** — `data/dataset.py:253-270`.
      Take `cams: list[tuple[str, size]]` (or a `{head: (cam, size)}` map) and return per-episode
      `(obs, act, frames_by_head)`. `_subsample_episodes` is **already generic** over extra streams
      (`dataset.py:175`: `tuple(x[ph:][:n*s:s] for x in ep[2:])`), so appending streams subsamples
      correctly with no change there. The `.npy` decode cache is already per-camera
      (`dataset.py:208`: `f"{cam}_{tag}.npy"`), so no cache-collision risk.
      *Verify:* load two cameras from the 3-cam dataset and assert the frame arrays differ and both
      align 1:1 with the obs rows (the existing assert at `:267` per stream).

- [ ] **3. `MMWindowLoader` holds one frame store and yields one key per head** —
      `data/dataset.py:273-310`. Currently a single `image_head: str | None`, a single `self.frames` GPU
      store, a single `e[2]` read (`:291`) and a single yield key (`:309`). `win_idx` (`:296`) is shared
      across streams and needs no duplication — the same window indices apply to every camera.
      *Verify:* one batch contains both head keys, both `(B,T,H,W,3)` float in [0,1], and the wrist
      camera has a visibly higher frame-to-frame delta than the fixed view.

- [ ] **4. `window_loaders` loops image specs** — `training/setup.py:800-818`. Replace
      `img = next((s for s in specs if s.kind == "image"), None)` (`:806`) with the full list, resolve
      each spec's camera per item 1, and pass all heads to `MMWindowLoader`.
      **This is the single blocker**: with it unfixed, a second image modality is BUILT in the model but
      never LOADED, and `lit.py:108-111` KeyErrors at step 0 — silent at config time, crash at run time.
      *Verify:* a two-image run reaches `[compile] first training batch` without a KeyError.

- [ ] **5. `conf/model/vl128_2cam.yaml`** — inherits `vl128`, restates the `modalities` list in full with
      three entries (proprio + two cameras). It MUST be a file: OmegaConf REPLACES lists rather than
      merging, and Hydra's `+` adds keys, not list ITEMS, so `model.modalities.2.name=...` cannot work.
      Pass `data.autobatch=true` here — let the probe find the batch rather than pinning 26, since the
      memory changes materially (below).
      *Verify:* `--cfg job --resolve` shows two image entries with distinct `name` and `cam`.

**After 1-5, training works.** Nothing in the model, the Lightning module or autobatch needs touching.

### Phase 2 — evaluation (needed before any result is trustworthy)

- [ ] **6. One per-head-aware loader helper**, then replace 8 identical call sites. Every one currently
      does `cam=cfg.data.get("cam")` plus `img_size = next(first image trunk)`:
      `routines.py:160-162, 331-333, 443-445, 498-500, 634-636, 714-716, 851-853, 1071-1088`, and
      `controller/run.py:53-66`. Change them once via a shared helper so they cannot drift apart.

- [ ] **7. Per-head ground truth in the two headline evals.** `eval_ood_horizon` and `eval_ae_floor`
      already LOOP `img_heads` (`routines.py:149-150, 173-174, 192-193, 264-272` and
      `:321, 341-342, 370-376, 395`) but load ONE camera and feed that same `im` array to EVERY head.
      They will run and **score head 2 against camera 1's frames** — a silently wrong number, which is
      worse than a crash. Fix `itrue` (`:173-174`), the re-ground contexts (`:192-193`), the filmstrip GT
      (`:270`) and `eval_ae_floor`'s `:342, :374`.

- [ ] **8. KeyError fixes** — these build an obs dict with only the FIRST image head and then call
      `encode_state`, which indexes every layout name: `evaluation/manifold.py:38-43, 127`;
      `routines.py:497-509` (denoising multistep); `routines.py:704-736, 746, 768` (denoising filmstrip —
      **small design decision**: one strip per head, or name the head to show); `routines.py:840-872`
      (interpret).

- [ ] **9. `eval_action_distribution`** — loops heads (`routines.py:1070, 1095-1096`) but loads one
      camera (`:1071, 1087-1088`): same wrong-frames-for-head-2 issue as item 7.

### Phase 3 — optional / deferred

- [ ] **10. `apply_size_preset`** — `setup.py:118` applies `model.size` presets to the FIRST image
      modality only. `vl128` does not use `model.size`; loop it or document the limitation.
- [ ] **11. Two-camera control rendering** — `controller/run.py:53-66`, `controller/mppi.py:148-149,
      225, 238-245, 280`. Recorded-env control is proprio-only (`mppi.py:441`), so this is inert today.
- [ ] **12. `logging/callback.py:295-297, 381-383`** — pretrained-TAESD affine path only; inert for the
      bespoke recipe.
- [ ] **13. One-off scripts** carrying the first-image idiom, non-blocking: `_oneoff_latent_rank.py:46`,
      `_oneoff_jacobian_probe.py:65`, `_oneoff_decode_kind.py:40`, `_oneoff_visual_terms.py:40`,
      `_oneoff_decoder_ab.py:61`, `_oneoff_latent_drift.py:57`, `_oneoff_colour_cast.py:36`,
      `scripts/bench_batch.py:19`, `scripts/ood_voe.py:69`.

## Memory: expect the batch to roughly halve, and let the probe decide

Measured baselines: unchunked `recon_frac=1.0` at `decode_base=64`/128px costs **13.17 GB/sample**, split
**~10.3 GB decoder** (two passes, ~5.2 each) **vs ~2.84 GB everything else** (`design/decode_memory.md:7-16,
43-47`). With `decode_chunk_train: 64` chunking both sites — which `vl128` sets — the decoder's cost moves
into a batch-INDEPENDENT constant, and the vl64 recipe as actually run measures **~3.5 GB/sample, batch 26**
with VGG-LPIPS at both sites.

A second image modality duplicates the conv encoder pass, both chunked decoder passes and the LPIPS term,
AND doubles the state bag (33 -> 65 tokens), which roughly doubles backbone/flow activations inside that
~2.84 GB non-decode bucket too. Estimate **~6-7 GB/sample -> batch ~12-14**, plus a second ~2.9 GB resident
frame store subtracted from the budget (`setup.py:394-432`).

An earlier estimate of "batch ~21" was too optimistic: it counted only the second encoder against the
per-sample slope and ignored the doubled bag. **Do not tune off either estimate** — the autobatch probe
already models two image specs faithfully (`setup.py:415-428, 460-468`) and prints the real number.

Eval memory roughly doubles too: `imagine_eval`'s `decode_chunk` decodes ALL heads per time chunk
(`multimodal.py:729-734`). It is batch-independent and only warned about (`setup.py:661-673`).

## Design decisions the user should make (not mine to assume)

1. **Camera binding**: an explicit per-modality `cam:` field (flexible, one new spec field) vs a
   convention that the modality NAME is the camera name (no new field, but forces ugly names like
   `robot0_eye_in_hand` into every metric key). The plan above assumes the field.
2. **Which head `best.ckpt` follows.** Already supported — pass
   `trainer.checkpoint_monitor=val/metric/<head>/mse`. The question is which, or whether to log a
   combined metric to monitor instead. Leaving the default means best.ckpt is blind to camera 2.
3. **Which camera the denoising filmstrip shows** (item 8), or one strip per head.

## Deliberately NOT in scope

**Input-only second camera** (a view the model sees but never renders). There is no config expression for
it today: every image modality gets a decode head and a weighted decode loss, so "input only" would mean
`weight: 0` + `latent_loss_weight: 0`, which is loss-free but STILL BUILDS AND RUNS the decoder — and the
decoder is the memory cost, so it buys nothing. A real input-only mode wants a spec flag that skips head
construction entirely. Not needed now (user, 2026-09-01); logged in `design/ideas.md`.
