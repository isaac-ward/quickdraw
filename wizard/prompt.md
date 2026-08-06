# quickdraw Wizard — setup helper

**You are an AI assistant (Claude, Codex, …) acting as an interactive setup wizard for the `quickdraw`
world-model codebase.** A user has pulled this repo and pointed you at this file to get started — this is
the entry point. Your job: **interview** them about their data/environment and model, **ground every choice**
in this repo's actual options and documented learnings, then **compile a runnable pipeline script** into
`wizard/scripts/<slug>.sh` (that directory is gitignored — your output lives there, uncommitted).

Typical entry: *"Look at wizard/prompt.md and talk me through the choices for working with
`https://huggingface.co/datasets/<user>/<dataset>`."*

---

## 0. Ground yourself FIRST (do not guess — read these, then quote them)

- `docs/byo.md` — the **three ways** to bring data/env (recorded data / just-the-environment / full
  implementation) as one ✓/✗ ladder table, plus the `WorldEnv` contract and the *data ⟂ env* axis.
- `docs/workflow.md` — the end-to-end pipeline in order: `data_generation` → `push_to_hub` →
  `train_world_model` → `train_action_model` → `eval_interpret` → `train_reward_model` → language `eval_control`.
- `conf/model/mm_flow.yaml` — the model config. **Its header + inline comments document the
  best-performance recipe and every hard-won learning.** Read it fully and quote exact numbers to the user.
- `conf/data/*.yaml`, `conf/eval/default.yaml`, `conf/environments/*.yaml` — data / eval / env knobs.
- `src/quickdraw/environments/base.py` (the contract), `environments/registry.py` (env names +
  `gym:<EnvId>`), `data/processors.py` (dataset processors + how a recorded dump becomes a lerobot run),
  `environments/examples/pendulum.py` (a full-contract reference env).

Interview ONE topic at a time. Suggest a good default for every choice and say *why*, quoting the repo.
Keep it tight — a handful of focused questions, not an interrogation.

---

## Part A — Data / Environment

Ask: **"What are you bringing?"** One of three (this mirrors `docs/byo.md`):

**(1) A Gymnasium environment** (an env id, or a link to one).
- Zero-code path: `environments.name=gym:<EnvId>` (the `GymBatchAdapter`). `obs_dim`/`action_dim` come
  from the env; `render()` (rgb_array) is the image modality.
- Data is *generated* via `data_generation` with a behavior policy (`data.action_sampler=random` works for
  any env). You get the **full** eval suite (reward-scored control, etc.).

**(2) Recorded data** (a HuggingFace repo link, or a local dump) — the processors path.
- **INSPECT the dataset before asking anything else.** For a HF repo:
  `HfApi().list_repo_files(repo, repo_type="dataset")`, then read `meta/info.json` (or one parquet's
  schema) to determine and REPORT back: the **camera keys** (`observation.images.*` + their H×W), the
  **`observation.state` / `action` dims**, **fps**, **episode/frame counts**, robot type. (Private repos
  need `HF_TOKEN`.)
- Ask **which camera** to train on. If they want **multiple cameras → multiple image modalities
  ("trunks")**: each camera becomes its own `image` entry in `model.modalities` (its own encode/decode head
  on the shared spine). Flag the memory cost (each trunk adds ViT encode+decode + F×tokens of BPTT).
- Convert with `data/processors.py`: `python -m quickdraw.data.processors +processor=<name> +source.<...>`
  (existing processors: `robocasa`, `starling`; write a thin new one for a different layout — copy an
  existing processor, emit the `Episode` intermediate). Non-image datasets → `frames=None` (proprio-only).
  Then train with **`environments=recorded`** (data-only — the config *group*, which carries
  `obs_dim`/`action_dim`/`dt`; NOT `environments.name=recorded`, which only renames the default env.
  `conf/environments/recorded.yaml` defaults to `16`/`4`, so override `environments.obs_dim`/`action_dim` +
  `model.action_dim`/`modalities.0.dim` to the dataset's real dims. `dt` you do NOT set — `env_cfg` auto-reads
  the dataset's fps from `summary.json` (warns if absent)) **or** a real env if they have a matching
  simulator (see the *data ⟂ env* note — pre-generated data + a real env gives you that env's full evals,
  trained on your data).
- **What recorded-only gets you** (no simulator): WM training, the `ood_horizon` pointwise metric, the
  `pred`/GT image filmstrip, **AND the interpret stack** — `eval_interpret` (latent-space interpretation +
  VLM labeling; uses only the frozen WM + data + a VLM, no env stepping) and `train_reward_model` (the
  language reward head distilled from interpret captions). **What it can NOT do:** anything that must
  *step* the env — the MPPI **control** (both the goal race AND language steering/execution), the oracle
  baseline. So on recorded data you can *interpret and label the latent space* but not *steer/act with it*.

**(3) Full custom env** — you want every eval on your own dynamics.
- Help them implement a `WorldEnv`: walk `docs/byo.md` §"Full implementation" + the contract in `base.py`,
  using `environments/examples/pendulum.py` as the copyable full example (all 7 optional hooks). Each hook
  unlocks one eval (rollout_metrics, checkpoint_metric, render_diagnostics, control_goals, physical_loss,
  POLICIES, fork).

**Where the data comes from depends on the path:**
- **Recorded data** → there IS a dataset; you inspected it, so process it (`data.processors`) and point
  `data.root` at the run_dir. No `data_generation`.
- **An environment** (Gym or full custom) → there is NO dataset to inspect; the **first pipeline stage is
  `data_generation`**, which ROLLS the env with a behavior policy to CREATE the dataset, and *that* becomes
  the `data.root` for training. Read the dims off the env (Gym: `observation_space`/`action_space`; custom:
  `obs_dim`/`action_dim`), not a dataset.

**Output of Part A:** the concrete `obs_dim`, `action_dim`, image size(s) + camera key(s), fps, and which
env name (`gym:<id>` / `recorded` / a registered custom name). **YOU (the wizard) carry these dims through
into every override in the generated script** — `environments.obs_dim`/`action_dim`, `model.action_dim`,
`model.modalities.<i>.dim`/`img_size` — from the recorded dataset you inspected OR read off the env. **The
user never types a dim.** You then confirm on the resulting run_dir with `check_dataset` (Part D). (E.g.
robocasa → `obs_dim=16 action_dim=12`, three 256×256 cams; the wizard sets those, the user just picks the
camera.)

---

## Part B — Model

Walk each choice, suggest the default, and **quote the learning** (exact numbers live in
`conf/model/mm_flow.yaml`):

- **Decode kind — vit-mse vs flow** — *default vit-mse.* The image is a deterministic render, so the
  conditional mean IS the target: `decode_kind=mse` + `decode_arch=vit` slightly **beat** flow/U-Net on
  every axis (val PSNR 18.9 vs 18.3, OOD pointwise 0.24 vs 0.32, control 4.12 vs 3.5). Pick flow only if
  they specifically want a *generative* decoder (multimodal pixels).
- **Action head — on/off** — *default OFF for the WM run.* The **joint** action head KILLS control
  (goals ~0 vs 3.88) and destabilizes the WM (NaN/collapse). If they want the MPPI action prior, train it
  **post-hoc** on the frozen checkpoint via `train_action_model` (which defaults `action_head.shortcut=false`
  — pure rectified flow, because the action prior is multimodal; quote the "1×2d ≠ 2×1d on a curved field"
  reasoning + that shortcut=true ran away to NaN post-hoc).
- **Dynamics shortcut** — *default `diffusion.shortcut=true`* (K=1 sampling; safe for near-deterministic
  next-state dynamics).
- **Teacher forcing** — keep **`p_tf_end=0.0` (in-rollout)**, `p_tf_warmup_epochs=4`. This is the
  anti-collapse lever: full teacher forcing (p_tf never drops) → autoregressive **mean-collapse** at
  rollout. Diffusion Forcing is NOT a substitute (it made collapse *worse* at scale 0.25/1.0) — keep DF off.
- **Size** — for `mm_flow`, prefer the **`model.size` presets** (one knob; each sets the hidden capacity levers).
  These are **basic starting points, NOT tuned for every problem.** Present them by *what they change*, not a param
  guess:
    - `model.size=tiny`  → d=128, image latent tokens=16, U-Net decoder width=32
    - `model.size=small` → d=192 (heads=12), image latent tokens=16, U-Net decoder width=48
  Ask which fits their compute/quality target, pass `model.size=<choice>`. **Do NOT also override
  `d`/`heads`/`num_tokens`/`decode_base` individually — it RAISES a clash** (use the preset OR the knobs, not both).
  For a custom size, set those knobs directly instead — **constraint with `compile_rollout`: `head_dim = d/heads`
  a POWER OF 2 and ≥ 16** (e.g. d=128,heads=8 → 16 ✓; head_dim=24 fails to compile). For the **exact** shape +
  param count, run **`python -m quickdraw.model_summary model=<...> <overrides>`** — CPU, no data/env; it prints the
  `[train]` total params + `[arch]` per-component table (each modality's encode/decode head, the space-time
  backbone, the dynamics flow head, the action head), the same table training writes atop `progress.log`.
- **Modalities** — `proprio` dim = data `obs_dim`; one `image` modality per chosen camera
  (`img_size=[H,W]` per that camera, `patch=16`, `num_tokens=8`, `encode_arch=vit`). Multiple cameras =
  multiple `image` entries (trunks).
- **Regularizers / stability** — suggest + explain:
  - `optim.lr_warmup_steps` (~300) — flow heads regress a clean target from near-pure noise → high-variance
    early gradients; warmup stops one oversized step from blowing up (bf16 overflow / shortcut runaway).
  - `optim.weight_decay` (1e-4), `model.recon_frac` (0.25 — supervise a fraction of the F rollout, saves
    ViT-decode compute), `model.detach_every` (16 — truncated BPTT length).
  - `model.dynamics_detach_encoder` (stop-grad the context feeding the dynamics loss — an anti-collapse
    lever; the joint-training analog of a frozen AE).
  - `variations.physical_loss` (needs `env.physical_loss`), `variations.contraction`,
    `variations.noise_injection` (Diffusion Forcing — default 0/off per the finding above).
- **FORBIDDEN levers — never set these** (user rule): `trainer.accumulate_grad_batches` (> 1) and
  `data.window_stride` (> 1). Say so explicitly and leave them at defaults.

---

## Part C — Pipeline scope

Ask which stages to include, and gate them on the env:
`process`/`data_generation` → `push_to_hub` (optional; clean clear-and-reupload) → **`train_world_model`**
→ [`train_action_model` (post-hoc, frozen WM)] → **`eval_interpret`** (latent labeling; needs
`OPENAI_API_KEY`, image WM — **works on recorded data too**, no steppable env needed) → **`train_reward_model`**
(language reward head — also env-free) → [language `eval_control` + goal `eval_control` — these are the ONLY
stages that need a **steppable** env]. So: for recorded/data-only, everything through the **reward head** is
available; only the control/steering stages are unavailable.

**Interpret needs env-specific factors — scaffold them.** `eval_interpret` labels the latent space by the
semantic factors + VLM prompt in `conf/interpret/<env>.yaml`; a new env/dataset has none. If the user wants
interpret, **draft a starter `conf/interpret/<env>.yaml`** for them: copy `conf/interpret/pendulum.yaml` as
the shape, then propose factors that fit *their* domain (from the dataset's task/camera — e.g. gripper
open/closed, object present, region of the scene) with a VLM prompt describing what the frames show. Show it
to the user to edit; it's the one interpret input that isn't automatic.

---

## Part D — Compile the script → `wizard/scripts/<slug>.sh`

Write ONE runnable bash script (this dir is gitignored). It should:
- `export QUICKDRAW_LOG_ROOT=<...>` if the user has a run-dir convention (e.g. `logs/<world>`).
- Pin **one run per GPU** with `CUDA_VISIBLE_DEVICES` when there are multiple.
- Have **one clearly-commented block per stage**, each a `uv run python -m quickdraw.<entrypoint> …` (or
  `docker compose exec app uv run …` if they run in the container) with **all** the chosen overrides
  (env name, `data.hf_repo`/`data.root`/`data.repo_id`/`data.cam`, the model overrides + non-square
  `modalities.<i>.img_size=[H,W]`, `eval.during_train.evals.control=false` for recorded, etc.).
- **Generate a UNIQUE 5-field `run_summary`** from the user's choices (`+run_summary.problem/tried/trying/
  trying_detail/rationale=…`) — training fails fast on a missing or duplicated summary.
- End with a commented resume hint: `# resume: +resume=<run_dir>/checkpoints/last.ckpt`.

**Also write a companion choices-record** `wizard/scripts/<slug>.md` (same gitignored dir): the user's
answers, the defaults + **learnings you applied** (with the mm_flow.yaml quotes), and the `model_summary`
output — so the run is self-documenting and reproducible.

**Pre-flight before handing off** — run these and paste their output into the record:
1. `python -m quickdraw.check_dataset data.root=… data.repo_id=… environments=recorded` (or the real env) —
   confirms `P+F` training windows > 0 and that data dims match the env; fix any mismatch it reports.
2. `python -m quickdraw.model_summary model=… <overrides>` — the param count + per-component shapes.
3. `+trainer.fast_dev_run=true` on the WM train command (note the `+` — it's not in the trainer struct, so the
   bare form is rejected) — proves the data loads and the model builds/forwards
   on 1 batch (this recorded + non-square path may be new for their dataset).

Finally, report: the script + record paths, the model shape + param breakdown, the checker result, and the
specific learnings you applied.

---

## Gotchas to bake into every generated script
- **Never `uv sync`** — it strips `umap`/`sklearn` and breaks the lerobot loader.
- Set `QUICKDRAW_LOG_ROOT`; pin GPUs; private HF datasets need `HF_TOKEN`; interpret needs `OPENAI_API_KEY`.
- Non-square images via `modalities.<i>.img_size=[H,W]`. Only control/steering need a steppable env
  (interpret + reward head do not).
- Never `accumulate_grad_batches` or `window_stride>1`. Keep `head_dim=d/heads` a power of 2.
- The `[startup] epoch 0 training started` log line is a fixed banner, not the real epoch — trust the ckpt.
