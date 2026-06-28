# Collapse diagnostics — explained

Plain-language reference for the four `collapse/` diagnostics (logging spec in `logging.md`; why they
matter in `latent_space_autoregressor.md`). All four are computed from one batch of latents stacked
into a matrix **`Z ∈ ℝ^{N×dz}`** (`N` = batch × timesteps), for a **single run/model** — they aren't
about comparing models, they're four views on whether *this* model's latent is healthy.

Notation: `Z̄` is `Z` with the per-dim mean subtracted; the latent covariance is
`C = (1/(N−1))·Z̄ᵀZ̄ ∈ ℝ^{dz×dz}`; per-dim std `σⱼ = std(Z[:,j])`.

## 1. Effective rank — *how many latent dimensions actually carry information*

Take the eigenvalues `λ₁…λ_{dz}` of `C` and compute the **participation ratio**
`PR = (Σλᵢ)² / Σλᵢ²`. If all variance lives in one direction (full collapse), `PR ≈ 1`; if variance is
spread evenly over all dims, `PR ≈ dz`. It's the single cleanest collapse number — it directly answers
"of my `dz` dimensions, how many are real?" For the torus we'd hope for something ≥ 4 (the intrinsic
state `θ, φ, θ̇, φ̇`) and worry if it slides toward 1.

## 2. Per-dim std — *which individual dimensions are alive vs dead*

The standard deviation of each latent coordinate over the batch, `σⱼ`, plotted sorted. A healthy
latent shows a spread of non-trivial stds; a collapsing one shows many dimensions pinned near zero
(dead). This is the per-dimension, un-aggregated companion to effective rank — effective rank tells
you *how many* dims are dead, this tells you *which*. (A variance-floor reference line is drawn only
for a mechanism that defines one — VICReg's `γ`; other runs just get the bars.)

## 3. Off-diagonal correlation mass — *how redundant the dimensions are*

First the **correlation matrix**: normalize the covariance, `Corr_ij = C_ij / (σᵢ·σⱼ)` — so the
diagonal is all 1s and every off-diagonal entry is a correlation in `[−1, 1]`. The diagnostic is the
**mean absolute off-diagonal**, `mean_{i≠j} |Corr_ij|`. Near 0 means the dimensions are decorrelated —
each carries independent information, so the latent is used efficiently. High means dimensions
duplicate each other (two dims encoding the same thing), so the *effective* dimensionality is lower
than it looks. It's a soft early-warning: rising correlation is a precursor to a falling effective
rank. (VICReg's covariance term explicitly drives the off-diagonal of the *covariance* toward 0; we
log the scale-free *correlation* version because it's bounded and comparable regardless of latent
scale.)

## 4. `L_pred` magnitude — *is the prediction task trivially easy?*

Just the (standardized) latent prediction loss. On its own it's ambiguous — low `L_pred` is good if
the latent is rich, but **near-zero `L_pred` while obs-space error is high** is the smoking gun: the
model made prediction easy by making the latent degenerate, not by learning dynamics. So it's only
meaningful read *against* the obs metric and effective rank — which is why the panel shows them
together.

## The collapse signature

Across the panel, the alarm is: **`L_pred ≈ 0` ∧ obs-space error high ∧ effective rank low.** The obs
metric is the *downstream score* ("is it a good world model?"); the latent diagnostics are the
*is-it-actually-degenerate* check. A mechanism can pass the obs metric while quietly half-collapsed —
only the latent diagnostics catch that; conversely the latent can look healthy yet predict poorly —
the obs metric catches that. You need both.
