"""NVIDIA Cosmos Video2World, through this repo's evals.

Loaded from the HF hub via diffusers, which is already in the container -- no NVIDIA monorepo, none of
the transformer-engine/apex/megatron machinery that exists for training we are not doing.

    python -m quickdraw.eval_ood_horizon model=cosmos interpret=starling data=starling2 \
        environments=recorded data.subsample=4 data.action_aggregate=concat

Two diffusers pipelines answer to this adapter and their `__call__`s agree on every argument we pass,
so there is one code path and a `pipeline` knob:
    predict2  Cosmos2VideoToWorldPipeline  -- Predict2, the newer family, 2B is the smallest
    cosmos1   CosmosVideoToWorldPipeline   -- Cosmos-1.0, 7B

FOUR THINGS THIS ADAPTER RECONCILES, none of them bugs, all of them shaping what the number means:

  RESOLUTION. The pipeline's only hard constraint is that height and width divide by 16, and 112x192
  does (7x12). So Cosmos can run at OUR native size with no resampling in either direction -- clean for
  a comparison, but far outside what it was trained at (704x1280), so a bad output may say nothing about
  dynamics. `height`/`width` are knobs precisely so that can be MEASURED: run native, run at 704x1280
  and let the ABC resample the output back, compare.

  CLIP LENGTH. Video2World emits a fixed-length clip, not an arbitrary rollout, so a long horizon is
  CHAINED: generate, feed the tail of the output back as the next call's context, repeat. Still
  open-loop -- the model never sees ground truth again after the context, exactly like ours -- but the
  feedback is PIXELS, re-encoded by the VAE at each seam, where ours feeds back a latent and pays no
  round trip. Fewer, longer chunks mean fewer seams.

  CONTEXT GRANULARITY, AND IT BITES. Conditioning is pinned at LATENT frames, and the VAE compresses time
  (4x on Predict2, 8x on Cosmos-1.0) with latent 0 covering pixel frame 0 alone, so only a kt+1-frame
  context is held fixed; anything past that is redrawn by the denoiser. `_cond_len` snaps DOWN to kt+1 and
  drops exactly that many frames off the front of the output, so what we score is only ever prediction.
  The consequence is worth saying out loud, because it is a handicap and not a detail: at the default
  data.P=8, Predict2 gets 5 context frames and Cosmos-1.0 gets ONE, against our 8. Raise `data.P` (9, 17,
  33) if the comparison is meant to be on equal context; the adapter prints what it actually used.

  FRAME RATE. Our steps are every fourth frame of 15 Hz footage, ~3.75 Hz; Cosmos assumes smooth video
  and defaults to 16 (Predict2) or 30 (Cosmos-1.0). Consecutive frames of ours are a much bigger jump
  than it expects. `fps` is a knob for the same reason as the resolution.

The action channel is TEXT -- Video2World conditions on frames and a prompt -- so the ABC's text bridge
renders the recorded sticks as a SENTENCE (build_action_prose, `external.prompt_style=prose`, the
default here). The table form build_action_text produces is written for a VLM that can already see the
clip; a generator's T5 encoder only ever saw scene descriptions, and under classifier-free guidance an
out-of-distribution prompt embedding is not ignored but pushed toward. `prompt_style=table` switches back
if you want to measure that. Either way this row is NOT comparable to a per-step-action row: the model is
told the intent, and prose does not even carry the magnitude.

The NVIDIA guardrail (`cosmos_guardrail`) is a licence term, not an option: the pipeline refuses to run
without it. It screens the prompt and blurs faces in the output.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .external import ExternalWorldModel, register

PIPELINES = {"predict2": "Cosmos2VideoToWorldPipeline", "cosmos1": "CosmosVideoToWorldPipeline"}


@register("cosmos")
class CosmosVideo2World(ExternalWorldModel):
    heads = ("image",)
    img_size = (112, 192)
    action_mode = "text"
    text_style = "prose"          # a T5-conditioned generator, not a VLM reading a table -- see _actions

    DEFAULTS = dict(model_id="nvidia/Cosmos-Predict2-2B-Video2World", pipeline="predict2",
                    num_frames=93, num_inference_steps=35, guidance_scale=7.0, fps=16,
                    height=None, width=None, batch=1, seed=0, negative_prompt="",
                    prompt_style="prose", prompt_phases=4)

    def __init__(self, cfg=None):
        super().__init__(cfg)
        e = (cfg.get("external", None) if cfg is not None else None) or {}
        g = lambda k: (e.get(k, self.DEFAULTS[k]) if hasattr(e, "get") else getattr(e, k, self.DEFAULTS[k]))
        self.model_id = str(g("model_id"))
        self.pipeline = str(g("pipeline"))
        self.num_frames = int(g("num_frames"))        # frames per pipeline call, context included
        self.steps = int(g("num_inference_steps"))
        self.guidance = float(g("guidance_scale"))
        self.fps = int(g("fps"))
        self.batch = int(g("batch"))                  # episodes per pipeline call
        self.seed = int(g("seed"))
        self.negative = str(g("negative_prompt") or "") or None
        self.text_style = str(g("prompt_style"))
        self.prompt_phases = int(g("prompt_phases"))
        hs = [] if cfg is None else [m for m in (cfg.model.get("modalities", None) or [])
                                     if str(m.get("kind", "")) == "image"]
        if hs:
            self.heads = tuple(str(m["name"]) for m in hs)
        # RESOLUTION FOLLOWS THE DATA unless it is overridden. The model config's image modality already
        # states the frame size the dataset is decoded at, and running Cosmos anywhere else means the ABC
        # resamples -- fine, but a choice, so it should be one someone made. external.height/width are the
        # override, and they move together.
        hw = (hs[0].get("img_size", None) if hs else None) or self.img_size
        hw = (int(hw), int(hw)) if isinstance(hw, int) else (int(hw[0]), int(hw[1]))
        self.img_size = (int(g("height") or hw[0]), int(g("width") or hw[1]))
        assert self.pipeline in PIPELINES, f"external.pipeline must be one of {sorted(PIPELINES)}"
        assert self.img_size[0] % 16 == 0 and self.img_size[1] % 16 == 0, (
            f"Cosmos requires height and width divisible by 16; got {self.img_size}")
        self._pipe = None

    # ---- the pipeline, loaded on first use ------------------------------------------------------------
    def pipe(self):
        """Not in __init__: building the model must stay cheap enough that a config error, `--help`, or a
        routine the harness SKIPS does not first pull ~10 GB off the hub."""
        if self._pipe is None:
            import diffusers
            cls = getattr(diffusers, PIPELINES[self.pipeline])
            print(f"[cosmos] loading {self.model_id} via {cls.__name__} (bf16) ...", flush=True)
            p = cls.from_pretrained(self.model_id, torch_dtype=torch.bfloat16)
            p.to("cuda" if torch.cuda.is_available() else "cpu")
            p.set_progress_bar_config(disable=True)
            _patch_guardrail(getattr(p, "safety_checker", None))
            _pin_device(p)
            self._pipe = p
            print(f"[cosmos] ready: {self.img_size[0]}x{self.img_size[1]} num_frames={self.num_frames} "
                  f"steps={self.steps} guidance={self.guidance} fps={self.fps} batch={self.batch}",
                  flush=True)
        return self._pipe

    def _cond_len(self, p: int) -> int:
        """How many context frames to actually feed. Conditioning is held fixed at LATENT frames, and the
        VAE compresses time (4x on Predict2, 8x on Cosmos-1.0), so only a kt+1-frame context survives the
        denoiser untouched; anything else has its tail redrawn and would silently become part of what we
        score."""
        t = int(self.pipe().vae_scale_factor_temporal)
        c = min(p, self.num_frames - t)               # leave room for at least one new latent frame
        c = max(1, ((c - 1) // t) * t + 1)
        assert c < self.num_frames, (
            f"external.num_frames={self.num_frames} leaves no room for new frames after a "
            f"{c}-frame context; raise num_frames")
        return c

    # ---- the rollout ----------------------------------------------------------------------------------
    def _rollout(self, ctx: dict[str, Tensor], actions, horizon: int, heads) -> dict[str, Tensor]:
        assert len(heads) == 1, f"Cosmos predicts one video stream; asked for {heads}"
        head = heads[0]
        frames = ctx[head]                                        # (N,P,h,w,3) in [0,1] at img_size
        n, p = frames.shape[:2]
        prompts = actions if isinstance(actions, list) else [str(actions)] * n
        pipe, c = self.pipe(), self._cond_len(p)
        if not getattr(self, "_said", False):
            self._said = True
            print(f"[cosmos] context {c}/{p} frames ({self.num_frames - c} new per call, "
                  f"{-(-horizon // (self.num_frames - c))} calls for horizon {horizon})", flush=True)
        # PER-EPISODE HISTORY, seeded with the context: the next call always conditions on the last c
        # frames we have, and early on that is still partly ground truth. Keeping the history (rather than
        # only the generated tail) is what makes a short clip length safe -- with c close to num_frames a
        # single call does not yet produce c new frames to condition the next one on.
        hist = [list(frames[i, -c:].permute(0, 3, 1, 2).float().cpu()) for i in range(n)]
        gen: list[list[Tensor]] = [[] for _ in range(n)]

        while min(len(g) for g in gen) < horizon:
            for lo in range(0, n, self.batch):
                hi = min(lo + self.batch, n)
                cond = torch.stack([torch.stack(hist[i][-c:], 0) for i in range(lo, hi)], 0)
                # PROMPT THE CHUNK, NOT THE ROLLOUT. One sentence summarising 2048 steps says almost
                # nothing about the 88 this call generates -- and a stick that reverses inside the span
                # averages away entirely. `window` re-renders from the actions belonging to these frames;
                # an adapter given a plain list (a stub, say) still gets the whole-rollout string.
                g0 = len(gen[lo])                  # every episode advances in lockstep: same clip
                assert all(len(gen[i]) == g0 for i in range(lo, hi)), \
                    "episodes are out of step, so one prompt window cannot describe the whole batch"
                g1 = min(g0 + self.num_frames - c, horizon)
                chunk = [prompts.window(i, g0, g1) if hasattr(prompts, "window") else prompts[i]
                         for i in range(lo, hi)]
                self.prompt_log += [(i, g0, g1, chunk[i - lo]) for i in range(lo, hi)]
                out = pipe(video=list(cond), prompt=chunk,
                           negative_prompt=None if self.negative is None else [self.negative] * (hi - lo),
                           height=self.img_size[0], width=self.img_size[1], num_frames=self.num_frames,
                           num_inference_steps=self.steps, guidance_scale=self.guidance, fps=self.fps,
                           generator=[torch.Generator("cpu").manual_seed(self.seed + i)
                                      for i in range(lo, hi)],
                           output_type="pt").frames                # list of (F,3,H,W) in [0,1]
                for i in range(lo, hi):
                    # the first c frames are the conditioning echoed back; only what follows is prediction
                    added = list(out[i - lo][c:])
                    assert added, f"pipeline returned {len(out[i - lo])} frames for a {c}-frame context"
                    gen[i].extend(added)
                    hist[i].extend(added)

        y = torch.stack([torch.stack(g[:horizon], 0) for g in gen], 0)    # (N,horizon,3,H,W)
        return {head: y.permute(0, 1, 3, 4, 2).to(frames.device, frames.dtype)}


def _patch_guardrail(sc):
    """cosmos_guardrail 0.3.1 swapped Llama-Guard for Qwen3Guard, and its `device`/`dtype` properties now
    read attributes a plain nn.Module does not have -- so DiffusionPipeline.device, which diffusers 0.35
    calls on every registered component before the first denoising step, raises AttributeError and the
    pipeline never runs. Derive both from parameters instead.

    This fixes an attribute lookup broken by a version skew. The guardrail is untouched and fully active:
    the prompt is still screened and faces in the output are still blurred."""
    if sc is None or getattr(type(sc), "_qd_device_patched", False):
        return

    def first_param(self):
        for m in self.nn_models:
            for q in m.parameters():
                return q
        return torch.zeros(())

    cls = type(sc)
    cls.device = property(lambda self: first_param(self).device)
    cls.dtype = property(lambda self: first_param(self).dtype)
    cls._qd_device_patched = True


def _pin_device(pipe):
    """`DiffusionPipeline.device` returns the device of the alphabetically FIRST component, and with the
    guardrail registered that is `safety_checker` -- which the pipeline deliberately parks back on the CPU
    at the end of every call. So from the SECOND call on, the pipeline believes it is running on CPU, puts
    the token ids there, and dies against its own weights on the GPU. (The first call works, which is what
    makes this worth a comment: a one-shot script never sees it and an eval loop always does.)

    Report the denoiser's device instead, which is what every reader of `.device` actually means. Scoped to
    this instance via a throwaway subclass, so nothing else using diffusers is affected."""
    cls = type(pipe)
    if getattr(cls, "_qd_device_pinned", False):
        return
    pipe.__class__ = type(cls.__name__, (cls,), {
        "device": property(lambda self: next(self.transformer.parameters()).device),
        "_qd_device_pinned": True})
