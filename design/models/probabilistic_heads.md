# Probabilistic prediction heads (parametric distribution over the next latent)

A third **prediction mechanism** alongside Direct (LSAR/DSAR) and Flow/diffusion (`MultiModalFlow`). The
spine's per-timestep context parameterizes an **explicit distribution** over the next latent, which is
**sampled** and decoded. Categorical (DreamerV3), diagonal Gaussian, and low-rank/full MVN are concrete heads
that differ *only* in (a) output parameters, (b) sampler, (c) loss term. All are mutually exclusive with the
flow head. Related: [rssm.md](rssm.md) (the earlier RSSM sketch this supersedes), [flow_heads.md](flow_heads.md),
[collapse.md](collapse.md) (the swappable-policy pattern we mirror).

---

## 1. The idea, and why there is a *posterior* head

A probabilistic world model reasons about **two** distributions over each latent `z_t`:

- **Posterior** `q(z_t | o_t)` — "given the observation I actually saw, what is the latent?" Produced by the
  **encoder** from the real observation. It grounds the latent in data and is what the decoder reconstructs from.
- **Prior** `p(ẑ_t | h_t)` — "predict the latent from the past *before* seeing the observation." Produced by the
  **prediction head** from the transformer context `h_t` (past latents + actions). This is the dynamics /
  imagination distribution.

Training pulls the **prior toward the posterior** — `KL(posterior ‖ prior)` — which is *how the dynamics learns
to predict the next latent*. At rollout there are no future observations, so we sample the **prior** and carry it
forward (latent-space autoregression). The posterior therefore exists to give the prediction a **target
distribution** to match, and the decoder + reconstruction ground that posterior in reality.

**"Observation-only" posterior.** DreamerV3 conditions its posterior on *both* the recurrent state and the
observation, `q(z_t | h_t, o_t)` — a filtering correction. We use `q(z_t | o_t)` — the observation **only** —
because quickdraw's encoder (`encode_state`) is **per-frame and context-free**, a property the shared-encode
fast path in `lit._step` and the KV-cache prefill both rely on. This is the one intentional deviation from
DreamerV3; it matches transformer world models (TWM, STORM) and is what the README's "no separate recurrent
hidden state" bullet already promises. The transformer backbone *is* the sequence model (`h_t`); there is no GRU.

The posterior conditions on the **FUSED single-timestep observation** — *all observation modalities together*
(proprio **and** image), never image-only and never per-modality-isolated — via a **within-frame fusion** step (a
single spatial-attention block over the frame's tokens, or an MLP over the concatenated modality features) with
**no temporal context**. Within-frame fusion is still purely a function of `o_t`, so the per-frame/context-free
property (and thus the shared-encode + KV-cache paths) is preserved. The **action is not fused into the
posterior**: it is exogenous conditioning for the prior/dynamics, and `z_t` decodes back to observations, not
actions.

**Not every mode needs a posterior.** If the latent is a stochastic/discrete variable (categorical, Gaussian-KL),
you need a posterior to sample from and a KL to train the prior toward it. If the latent is **deterministic** and
only the *prediction* is a distribution (Gaussian-NLL, the Ward-2026 setup), there is **no posterior and no
bottleneck** — the prior is trained directly by the likelihood of the true next (deterministically encoded)
latent. The next section shows both side by side.

---

## 2. Parallel paths — the current flow head vs the three new heads

Same spine, same encoders, same decoders. Only the shaded middle (posterior → carry → prior → sample → loss)
differs. `e_t = ` continuous per-modality encoder features (today's `encode_state` output); `z_t = ` the bag
actually carried into the backbone; `h_t = _cond(backbone output)`; `sg = ` stop-gradient.

```
STAGE            A. FLOW (current)        B. CATEGORICAL           C. GAUSSIAN-KL           D. GAUSSIAN-NLL
                    MultiModalFlow           (DreamerV3)              (PlaNet/DreamerV1-2)     (Ward 2026)
────────────────────────────────────────────────────────────────────────────────────────────────────────────
encode o_t    →  e_t (point)             e_t                      e_t                      e_t
posterior     →  — (none; e_t IS z)      q=Cat(softmax(Wе)        q=N(μ_q,σ_q)=head(e)     — (none; deterministic
                                           ·0.99 + unimix·0.01)                               encoder: e_t IS z)
sample/carry  →  z_t = e_t               z_t = embed(            z_t = μ_q + σ_q⊙ε        z_t = e_t
                                           ST_onehot(q))            (reparam)
backbone      →  h_t = _cond(bkbn(z))    h_t                      h_t                      h_t
prior/predict →  v_θ(x_τ,τ,h_t)          p=Cat(softmax(Wh)        p=N(μ_p,σ_p)=head(h)     p=N(μ_p,σ_p)=head(h)
                   (velocity field)        ·0.99 + unimix·0.01)
sample next   →  ∫ ODE over K (or 1      ẑ = embed(sample(p))     ẑ = μ_p + σ_p⊙ε          ẑ = μ_p (committed)
                   shortcut) steps →ẑ      argmax if committed      μ_p if committed          / μ_p+σ_p⊙ε (stoch)
decode        →  to_obs(ẑ)               to_obs(ẑ)                to_obs(ẑ)                to_obs(ẑ)
────────────────────────────────────────────────────────────────────────────────────────────────────────────
LATENT IS     →  continuous, det.        DISCRETE, stochastic     continuous, stochastic   continuous, det.
POSTERIOR?    →  no                       YES                      YES                      NO
```

Read across a row to see what each head swaps. The whole left/right frame — encoders, `to_obs` decoders,
`recon_losses`, `roundtrip_losses`, the rollout machinery (`_rollout_from`, `_rollout_cached`/KV-cache,
`imagine_eval`) — is **identical and inherited** in every column.

### The one structural choice: discrete latent on a continuous bag (B, C)
For B/C the carried bag must be a *sample*. Rather than replace the token bag, a **bottleneck sits on top of it**:
`e_t → posterior params → sample → embed back to a d-dim token`. So the bag handed to the backbone and to
`to_obs` is still `(B,T,n_state,d)`, and every decoder / roundtrip / anchor path is unchanged. For categorical,
`codec/roundtrip_<mod>` *becomes* the Dreamer decoder reconstruction of the posterior sample, for free.

For D (Gaussian-NLL) there is **no bottleneck**: `e_t` is carried straight through (deterministic), and the
distribution appears only at the prior. This is exactly Ward-2026 (Cosmos latents are deterministic; the model
predicts a tensor of diagonal Gaussians over the next latent).

---

## 3. How each path affects the loss

All terms flow through the existing `loss_terms → (raw, weights)` + `recon_losses → (losses, weights)` contract
summed in `lit._step`; **`lit.py` needs no change**. `⟨decode⟩` = the per-modality `decode/<mod>` recon on the
rolled predictions (unchanged, kept in every mode); `⟨roundtrip⟩` = `codec/roundtrip_<mod>`.

**A. Flow (unchanged, for reference).** Regress the constant rectified-flow velocity of the straight path from
noise to the true residual `Δz = sg(z_{t+1}) − z_t`:
`L_flow = ‖ f_θ(x_τ, τ, h_t) − (ε − Δz) ‖²`  (+ optional shortcut consistency) + `⟨decode⟩` + `⟨roundtrip⟩`.

**B. Categorical (DreamerV3).** KL between posterior and prior, **balanced** (train the prior toward the
posterior harder than the reverse) with a **free-bits** floor, on the unimix-mixed categoricals; the KL of the
factorized joint is the sum over the G groups:
```
L_dyn = max(free_nats, Σ_g KL[ sg(q_g) ‖ p_g ])          weight β_dyn = 1.0
L_rep = max(free_nats, Σ_g KL[ q_g ‖ sg(p_g) ])          weight β_rep = 0.1
```
+ `⟨decode⟩` + `⟨roundtrip⟩`. No flow loss. Defaults: **1% unimix**, **free_nats = 1**, **β_dyn/β_rep = 1.0/0.1**
(DreamerV3 Nature / reference impl — note the Jan-2023 v1 used β_dyn≈0.5; `rssm.md` should be corrected).
Straight-through: forward = one-hot, backward = softmax grad (`onehot + p − sg(p)`).

**C. Gaussian-KL (PlaNet / DreamerV1–V2 lineage).** Identical structure to B, but closed-form Gaussian KLs and a
reparameterized carried sample: `L_dyn / L_rep = KL[N(μ_q,σ_q) ‖ N(μ_p,σ_p)]` with sg on opposite sides, same
`β_dyn/β_rep/free_nats`, + `⟨decode⟩` + `⟨roundtrip⟩`.

**D. Gaussian-NLL (Ward-2026).** No posterior, no KL(post‖prior). The prior is trained by the likelihood of the
true next (deterministically encoded) latent, plus Ward's stabilizers:
```
L_nll        = − log N( sg(e_{t+1}) ; μ_p, σ_p² )         weight 1.0
L_latent_mse = ‖ μ_p − sg(e_{t+1}) ‖²                     weight 2.0     (Ward's 2·L^z_recon)
L_kl_prior   = KL[ N(μ_p,σ_p²) ‖ N(0, I) ]               weight 0.05    (Ward's 1/20; keeps σ sane)
```
+ `⟨decode⟩`. Free bits do not apply (NLL is unbounded below). `σ = softplus(raw) + min_std` (`min_std = 0.1`).
The **mean σ over the bag is the uncertainty / conformal non-conformity score** (Ward's detector), exposed as an
optional `_unc` readout next to `_bag` in `_imagine`.

Unified view: `L_nll` is the degenerate `KL(Dirac posterior ‖ prior)`, so D is the same abstraction with a
point-mass posterior — which is why one interface covers all three.

---

## 4. The modular abstraction

New module `src/quickdraw/models/dist_heads.py`, a policy object the model *holds* (mirrors `collapse.py`):

```
class DistributionHead:
    needs_posterior: bool        # False -> encoder untouched (Gaussian-NLL / Dirac posterior)
    is_discrete: bool
    build(d, n_state) -> modules  # posterior head, prior head, embedding (registered on the MODEL)
    posterior(feats) -> Params    # per-token params from encoder features (Dirac for NLL mode)
    prior(cond)      -> Params    # per-token params from _cond(h)
    sample(params, deterministic) -> bag   # ST one-hot+embed | reparam | identity(mean)
    losses(post, prior) -> (raw, w)        # kl_dyn+kl_rep (unimix, balancing, free bits) | nll(+mse,+kl_prior)
    diagnostics(post, prior) -> dict       # entropies, free-bits floor fraction, sigma_mean, code perplexity
    uncertainty(prior_params) -> Tensor    # Ward score (mean sigma / mean prior entropy)

CategoricalHead(groups, classes, unimix, dyn_scale, rep_scale, free_nats, straight_through)
GaussianHead(loss={nll|kl}, min_std, max_std, dyn_scale, rep_scale, free_nats, kl_prior_scale, latent_mse_scale)
MVNHead(GaussianHead + rank)               # low-rank+diag; rank>=d -> full Cholesky
make_dist_head(cfg) -> DistributionHead
```

One nuance vs `CollapseStrategy`: heads own **parameters** (posterior/prior nets, embedding). Resolution:
`build()` returns modules the **model registers** under stable names (as LSAR registers its predictor), so
checkpoints serialize cleanly and `load_checkpoint`'s missing-parameter tripwire keeps working.

New model class `MultiModalDistribution(MultiModalSequenceModel)` in `multimodal.py` (append-only; **zero edits to
existing classes**): holds the head, overrides `encode_state` (features → optional posterior sample → embed),
`predict_next` (`head.sample(head.prior(h))`, deterministic when `not training and not stochastic_eval` — the
`MultiModalFlow` convention, incl. the `stochastic_eval` attribute `kvcache_report` pins), and `loss_terms` (one
teacher-forced parallel pass yielding posterior + prior params). Everything else (rollout, KV-cache, `_imagine`,
teacher-forcing, decoders) is inherited.

### Config
Selected by `model.name = mm_dist` + `model.dist_head ∈ {categorical, gaussian, mvn}`, new presets
`conf/model/mm_categorical.yaml` and `conf/model/mm_gaussian.yaml`:
```
name: mm_dist
dist_head: categorical
categorical: { groups: 16, classes: 16, unimix: 0.01, dyn_scale: 1.0, rep_scale: 0.1,
               free_nats: 1.0, straight_through: true }
# gaussian: { loss: nll, min_std: 0.1, dyn_scale: 1.0, rep_scale: 0.1, free_nats: 1.0,
#             kl_prior_scale: 0.05, latent_mse_scale: 2.0 }   # mvn adds rank: 8
```
Default sizing note: the bag has `n_state` tokens (1 proprio + 8 image in the bimodal default); `G=16×K=16` per
token → far above Dreamer's 32 categoricals per step, but it is a **hard quantization** of the continuous latent,
so the codec floor drops (measured in Phase 3). DreamerV3's canonical is 32 latents × d/16 classes (classic
32×32); 32×32 per token is the headroom setting.

### Dispatch guards (in `setup.build_model`, mirroring the existing `raise ValueError` style)
- `dist_head` set on a non-`mm_dist` name, or `diffusion.*` on `mm_dist` → raise (pick ONE prediction mechanism).
- `variations.contraction.weight > 0` with `mm_dist` → raise (sampling/argmax step has no double-backward
  Jacobian; same reason as the diffusion guard).
- Diffusion forcing (`df_scale > 0`) already raises for non-flow names; add `mm_dist` to the message.
- `categorical` + `collapse ∈ {ema, sigreg, vicreg}` → raise (EMA target contradicts the posterior-as-target
  design; variance/covariance regularizers on one-hot embeddings are meaningless). `gaussian-nll` (deterministic
  encoder, structurally LSAR-like) may compose `sigreg/vicreg` under the existing `latent_norm=none` rule — this
  is the README's "composable with the same collapse regularizers" configuration.

---

## 5. Blast radius, risks, phasing

**Byte-identical-off.** Every change is additive (new module, new class, new dispatch branch reached only by a new
`model.name`, new yaml). `mm_flow`/`mm_lsar`/`mm_dsar` are untouched. Verify by hashing an `mm_flow` state-dict +
one-step loss (fixed seed) before/after.

**Risks + mitigations.**
- **Codec-floor regression (the real one)** — quantizing the ~1024-float TAESD latent costs dB. Measure up front
  (`eval_ae_floor`, sweep `G×K`) vs the ~20.4 dB continuous floor; accept as a *relative* trade (Dreamer trades
  fidelity for a well-behaved prior — that's the point of the shoot-out).
- **Posterior collapse / KL→0** — free bits (1 nat) is DreamerV3's countermeasure; `codec/roundtrip` fails loudly
  if the posterior ignores the obs; log the free-bits floor-binding fraction.
- **KL spikes** — the 1% unimix makes zero-probability impossible (its stated purpose); existing grad-clip +
  non-finite-skip catch the rest.
- **Straight-through bias/instability** — small `β_rep`; a `detach_encoder` escape hatch; a `straight_through:
  false` ablation knob.
- **Gaussian σ pathologies** — `min_std` floor, optional `max_std`, `kl_prior_scale`, zero-init σ output.

**Phased plan (each gated by a check).**
1. `dist_heads.py` + unit smoke: unimix floor, ST identity + grad, `KL(q‖q)=0`, free-bits clamp, Gaussian KL vs
   `torch.distributions`, Dirac-posterior CE == NLL, MVN low-rank log-prob vs dense, deterministic==argmax/mean.
2. `MultiModalDistribution` wiring + **byte-identical-off** check + each dispatch guard fires.
3. **Codec-floor measurement** (categorical) — sweep `(G,K)`, pick default with eyes open.
4. Supervised overfit smoke (proprio-only `n_state=1` first, then bimodal): KL falls, floor engages late, entropy
   off both rails, recon drops, no non-finite skips.
5. KV-cache + rollout parity (committed mode `latent_max_abs_diff ≈ 0`; `imagine_eval` vs `imagine_shared(K=1)`).
6. Gaussian/MVN pass: NLL-mode σ-mean rises under OOD contexts (Ward sanity); KL-mode short run.
7. Full smoke training run vs the `mm_flow` baseline before any long run.
