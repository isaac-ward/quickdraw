# Environment — Torus Manifold Benchmark

Toy environment for the world-model shoot-out. Long-horizon consistency = staying on the data
manifold. The manifold is a torus surface in 3D, so divergence is both visible and computable.
Built with PyTorch (torch-vectorized sim on GPU). Modalities: `observation_vector`,
`observation_image`, `action`.

## Manifold and state

Torus (major `R`, minor `r`, axis z), angles `θ` (toroidal), `φ` (poloidal):
```
x = (R + r·cosφ)·cosθ      y = (R + r·cosφ)·sinθ      z = r·sinφ
```
Hidden state `(θ, φ, θ̇, φ̇)`. A unit-mass particle is driven by action `a = (a_θ, a_φ)`, a
tangential thrust. Semi-implicit Euler with damping `γ`:
```
θ̇ ← θ̇ + dt·(a_θ − γ·θ̇);   θ ← (θ + dt·θ̇) mod 2π     (same for φ)
```
Defaults: `R=2, r=0.7, dt=1/30 (30 Hz), γ=0.1, a_max=2.0`. Every episode starts with nonzero speed.
The true state is always exactly on the torus.

## Observations

- **`observation_vector` ∈ ℝ⁶** = `[p; ṗ]`, where `ṗ = θ̇·∂p/∂θ + φ̇·∂p/∂φ` (tangent to the surface).
  Models predict in **ambient** ℝ⁶, never in angles — that is what lets a prediction leave the
  manifold and makes the error measurable.
- **`observation_image`** = egocentric render, `(3, 72, 128)`, ImageNet-normalized. Camera is a
  Darboux frame derived from state (no stored orientation):
  ```
  up = n̂(θ,φ);   forward = normalize(ṗ − ⟨ṗ,n̂⟩·n̂);   right = forward × up
  n̂ = (cosθ·cosφ, sinθ·cosφ, sinφ)
  ```
  At a mid-trajectory speed-zero reversal, hold the last heading and blend back over `‖ṗ‖∈[0,v_min]`.
  The surface carries a **colored rainbow gradient** (hue ← `θ`, modulated by `φ`) so that **every
  location and viewing angle produces a visually unique image** — no two views look alike, which
  removes the symmetry ambiguity of a plain torus and lets a single frame localize the camera. It
  also makes image prediction harder (a continuous color field to reproduce). The coloring is the
  visual-OOD knob.
- **`action` ∈ ℝ²** = `(a_θ, a_φ)`, clamped to `[−a_max, a_max]`.

## The three errors (single source of truth; reused by logging)

Compare predicted `p̂, ṗ̂` to the torus each step:
- **`manifold_distance_error`** = `|signed_dist(p̂)|`, `signed_dist(p) = √((√(x²+y²)−R)² + z²) − r`.
  How far off the surface (0 = on-manifold). Headline.
- **`pointwise_error`** = `‖p̂ − p‖`. Distance to the true point (on-manifold but wrong place counts).
- **`tangent_velocity_error`** = `|⟨ṗ̂, n̂(p̂)⟩|`. Velocity pointing off the surface; early warning.

## Why divergence happens (curvature)

To stay on a curved surface a model must learn second-order structure. A tangent-only step drifts
off by `≈ ½·κ·(‖ṗ‖·dt)²` — small per step, compounding over the horizon, larger for high curvature
(`κ ≈ 1/r`). A flat manifold has no second-order structure to miss, so it cannot discriminate
models. Thin tube + long horizon are the discriminative knobs.

## OOD axes (eval; vary one thing vs. train)

- **Visual**: same geometry/dynamics, different surface coloring.
- **Geometric**: different `(R, r)` → different curvature.
- **Dynamics**: different `γ`, `a_max`, action correlation.

## Surface progression (future-proofing)

Torus (vector) → torus (image) → saddle / height-field (occlusion-free, for the image stage) →
Lorenz attractor (chaos chapter where consistency provably ≠ pointwise accuracy). The
`observation_vector`/`observation_image`/`action` schema is the same one real robot data uses, so
models transfer without interface changes.

## Implementation

- `TorusEnv`: torch module. Steps `B` envs as tensors; actions from an OU process. Per-episode
  determinism via `torch.Generator` seeded `fold_in(base_seed, episode_id)`.
- Renderer: matplotlib 3D for trajectory plots/videos (see `logging.md`); a batched rasterizer
  (nvdiffrast) for `observation_image` at the image stage, encoded straight to MP4.
- Metric functions (`signed_dist`, `n̂`, phase drift) live here and are imported by training, eval,
  and logging — no reimplementation.
