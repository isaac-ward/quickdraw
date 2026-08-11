# Action conditioning — how the action reaches a prediction

Why this document exists: on 2026-08-11 the autoregressive rollouts did not move the arm, and
`grad/norm/act_enc` was **0.0010** against the flow's **0.5932** — the action pathway was receiving **0.17% of
the total gradient**. The action was very nearly decoration.

---

## 1. Explicit vs implicit — the thing that was actually wrong

`predict_next(h_state, prev_bag)` takes **no action argument**, and never has. So how did actions matter at all?

```python
_to_input(bag, act):  cat([bag, act_enc(act).unsqueeze(-2)], dim=-2)   # action = token #10 of 10
h_bag = backbone(...)                                                  # (B, T, n_input=10, d)
readout(h_bag, prev):  predict_next(h_bag[..., :n_state, :], prev)     # slot n_state SLICED OFF
```

Two distinct senses of "the action is in there":

- **IMPLICITLY (before 2026-08-11).** The action occupied slot 10, and within-step attention let the 9 state
  tokens *read* it. So action information appeared in `h_state` **only to the extent attention chose to put
  weight on that slot**. There is a complete path from input to output that never touches it — softmax over 10
  slots can suppress one to negligible weight, and nothing in the architecture or the loss forbids that. The
  action slot's OWN output was computed and then **thrown away** by the slice.
- **EXPLICITLY (now, `concat_action_embedding=true`).** That discarded slot is concatenated onto every state
  token, so the denoiser receives the action as its own dedicated 128 dims:

```python
def _cond(self, h_bag):
    h_state = h_bag[..., : self.n_state, :]
    if not self.concat_action_embedding: return h_state
    act = h_bag[..., self.n_state : self.n_state+1, :].expand_as(h_state)
    return torch.cat([h_state, act], dim=-1)          # (..., n_state, 2d)
```

It needed **no plumbing**: the action slot is already present in `h_bag` at all six `readout` call sites and in
`loss_terms`. Cost: **no new embedding parameters** (it is the tensor already being discarded); the flow's
`h_dim` doubles, `in_dim` 288 -> 416, flow params 0.0721M -> 0.0885M. It is also re-consulted at every one of
the `sampling_steps` ODE steps, instead of once per rollout step.

**Honest scope:** this is the action slot's **post-backbone** value, not the raw `act_enc(act)`. It is
contextualised (attention has mixed other tokens in), though a pre-norm residual stream preserves the slot's
own input strongly. A truly unmediated raw embedding would require threading actions into `predict_next`.

---

## 2. Why FiLM / adaLN was REJECTED

FiLM (`h <- gamma(a) * h + beta(a)`) makes the action multiplicative, so there is no path around it. It was
considered and dropped, for a better reason than "the token gets lost":

**Proprio also enters as a bag token and works fine.** So the token mechanism is not the problem. The
difference is that **proprio is also a TARGET** — it has a decode head, `decode/proprio`, whose gradient is
0.106-0.299. Actions have no head, no target, no reconstruction term: the only gradient `act_enc` can ever
receive is whatever flows back from "consulting the action helped predict the next state."

And DiT's adaLN-**zero** initialises to `gamma=1, beta=0` — *exactly* the ignore-the-conditioning state. That
works there because the loss **cannot be minimised** without the conditioning: you cannot denoise without
knowing the noise level, and you cannot draw a dog rather than a cat without reading the label. Here actions
are **optional** — a robot arm on a smooth trajectory has `z_{t+1} - z_t` largely predictable from recent
history, so **momentum explains most of the residual and the action adds marginal information**. Zero-init
would start at "ignore actions" and nothing would pull it off zero.

So FiLM makes actions *usable*; it does not make them *wanted*. That is why the incentive matters more than the
architecture here, and why this is textbook shortcut learning rather than a capacity limit.

---

## 3. Fourier features on the action (and vectors)

The flow head has **always** Fourier-encoded its diffusion time (`_temb`, 16 bands). The action — the input
with *more* structure and finer distinctions — got a raw `nn.Linear`. robocasa's action is 12-dim but
effectively ~4 (base 4 + mode 1 + EEF pos 3 + EEF rot 3 + gripper 1, most near-constant), and consecutive
actions differ slightly, so a linear map of near-collinear inputs yields near-collinear embeddings.

ONE implementation, `models/features.py`, used by every call site:

```python
fourier_freqs(n_freq, f_max=100)  -> 2*pi * logspace(1..f_max)     # matches the flow's historical ladder
fourier_dim(n_in, n_freq)         -> 2 * n_in * n_freq
fourier_features(x, freqs, squash) -> cat([sin, cos]) per element per band
```

`flow._time_features` is now a thin alias over it (`squash=None`, since tau is already in [0,1]) — verified
bit-identical to the previous inline expression.

**`squash` is REQUIRED for anything z-scored, and here is why.** `data/dataset.Normalizer` applies
`(x - mean)/std` per dim — so actions and proprio are **unbounded**, not in [0,1]. Worse, some dims are
near-constant: robocasa's action `std` floors around **1e-6**, so a 0.5 deviation on that dim normalises to
**5e5** (measured). A raw expansion would alias catastrophically — at the top band a z-score of 3 gives a phase
of 600*pi. `fourier_features` therefore does `clamp(x / squash, -1, 1)` first (default squash=4, i.e. 4 sigma),
which bounds the phase AND neutralises the 1e6 amplification.

**There is no min/max in `normalization_stats.json`** (only mean/std), so a true [0,1] min-max rescale would
need the stats regenerated. The 4-sigma clamp is the no-new-stats equivalent and is what is implemented.

Widths: a 12-dim action at 16 bands -> `12 + 2*12*16 = 396` input dims. 16-dim proprio -> `16 + 2*16*16 = 528`.

---

## 4. What is implemented, and what is NOT

**IMPLEMENTED**

| thing | knob | default |
|---|---|---|
| Action concatenated onto every state token for the denoiser | `model.diffusion.concat_action_embedding` | **true** |
| Fourier features on the ACTION encoder | `model.action_fourier_freqs` | **0 (off)** |
| Fourier features on VECTOR modality encoders | `model.modalities.<i>.fourier_freqs` | **0 (off)** |
| Shared expansion, one implementation | `models/features.py` | — |
| Stated in progress.log | `[action]` and `[fourier]` lines | always |

`progress.log` now prints, at startup:

```
[action] fourier=16 bands (raw 12 + 384 sin/cos, |x|<=4) | concat_to_denoiser=ON  <- the action token's
         backbone output is concatenated onto every state token, so the denoiser has a dedicated action
         channel it cannot route around
[fourier] proprio: 16 bands (raw 16 + 512 sin/cos)
```

and when conditioning is off it says so, with the reason:

```
[action] fourier=OFF | concat_to_denoiser=OFF  <- WARNING: readout() DISCARDS the action slot, so actions
         reach the prediction ONLY via attention onto 1 of the bag's slots — measured grad/norm/act_enc was
         0.17% of the total gradient
```

**NOT IMPLEMENTED** (discussed, deliberately not built)

- **FiLM / adaLN action conditioning** — rejected, see §2.
- **Raw (pre-backbone) action embedding into the denoiser** — the concat uses the post-backbone slot; an
  unmediated version needs actions threaded into `predict_next`.
- **Motion-weighted reconstruction** (reweight recon by `|o_t - o_{t-1}|`) — not wanted.
- **Inverse dynamics auxiliary head** (predict `a_t` from `(z_t, z_{t+1})`) — not wanted.
- **Action-sensitivity metric** (`||rollout(a) - rollout(a')||`) — designed, not built. This is the ONLY direct
  test of whether actions are used; the 0.17% gradient figure is suggestive, not conclusive.
- **min/max action stats for a true [0,1] rescale** — the 4-sigma clamp stands in.

## 5. Verification status

Verified: shapes (`act_enc` in 12 -> 396 at 16 bands; proprio 16 -> 528), `fourier_features` bit-identical to
the old `_time_features` on the scalar path, squash tames a 5e5 input, `h_dim` 288 -> 416, LSAR/DSAR untouched
(predictor in_features stays 128), off == bit-identical, and the progress.log lines render both ways.
Smokes: multimodal 10/10, multimodal_diffusion 9/9, modalities 9/9, kvcache 5/5, eval_products 21/21.

**NOT verified:** that any of this makes the arm move. An action-sensitivity check at random init shows no
difference between concat on and off (3.35 vs 3.28) because an untrained network is sensitive to everything.
Whether the model *uses* the channel once trained is an empirical question, and momentum may still make
ignoring actions the cheaper solution.
