"""Evaluate a model this repo did not train, through the evals this repo already has.

THE SEAM IS ONE METHOD. `eval_ood_horizon` -- the routine behind every open-loop number in the paper --
touches a model only through `layout` and `imagine_eval`; `eval_manifold` adds `forward`. Everything
else (LPIPS/SSIM/PSNR, the re-grounding modes, the filmstrips, the error-vs-step curves) is computed
from the returned tensors and is model-agnostic by construction. So an external model is not a port, it
is an adapter with one required method.

`imagine_eval` here is CONCRETE and validating. It checks the shapes going in, resamples between our
frame size and the model's native one, delegates to the abstract `_rollout`, then checks the shapes
coming back. Putting that here rather than in each adapter is the whole point: an adapter that returns
(N,H,3,h,w) instead of (N,H,h,w,3), or logits instead of [0,1], is told so immediately and by name,
rather than producing a plausible LPIPS that is silently measuring a transposed image.

An adapter declares what it CAN do and the harness skips what it cannot -- `latents` raises
NotImplementedError by default and `run_standalone` turns that into a logged skip, so a model that
predicts pixels only runs ood_horizon and quietly sits out manifold, instead of failing halfway or,
worse, producing a number that does not mean what the column says.

Register an adapter in EXTERNAL and select it with `model=<key>`; nothing downstream knows the
difference.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ExternalWorldModel(nn.Module, ABC):
    """Base for a model trained elsewhere. Subclass, set the class attributes, implement `_rollout`."""

    # ---- DECLARED, never inferred. The harness reads these to decide what it can ask for. --------------
    heads: tuple[str, ...] = ()          # image heads this model predicts, e.g. ("image",)
    img_size: tuple[int, int] = (112, 192)   # (H, W) the model natively runs at
    action_mode: str = "per_step"        # per_step | chunk | text | none
    action_chunk: int = 1                # frames per action, when action_mode == "chunk"
    text_style: str = "table"            # table | prose, when action_mode == "text" (see `_actions`)
    prompt_phases: int = 4               # spans a prose prompt is split into (prose only)
    is_external: bool = True             # run_standalone reads this to skip load_checkpoint

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
        # `model.render_size` — WHAT THIS MODEL GENERATES AT, which is not what the data is decoded at and
        # not what the metrics are computed at. The parallel is `environments.render_size` (what the SIM
        # renders at before the cache downsamples it): in both cases a producer has its own resolution and
        # ours is the target. `imagine_eval` resamples between the two, so the only rule is that the
        # aspect should match the data's or the resample distorts.
        #
        # It lives HERE and not in an adapter because every external model has a resolution it was trained
        # at, and running one at our frame size because that is what the dataset happens to ship is how
        # Cosmos spent a day producing saturated noise (wizard/records/cosmos.md, Finding 1b). It is an
        # EXTERNAL-model field: no quickdraw model has one, because our decoder's output shape IS
        # `modalities.img_size` — changing it is an architecture change, not a config knob.
        rs = cfg.model.get("render_size", None) if cfg is not None else None
        if rs is not None:
            self.img_size = (int(rs), int(rs)) if isinstance(rs, int) else (int(rs[0]), int(rs[1]))
        # `modalities` exists so routines that probe `m.modalities.values()` for an image size fall
        # through to their own default instead of raising AttributeError.
        self.modalities: dict = {}
        # WHAT THE MODEL WAS ACTUALLY TOLD. For a text-conditioned model the prompt IS the action channel,
        # and it is rendered on the fly -- so without this it is unrecoverable from the rollout it
        # produced. Adapters append (episode, lo, hi, text) per generated window; the eval routine drains
        # this next to the filmstrips and clears it. Empty for every model that takes actions as numbers.
        self.prompt_log: list[tuple[int, int, int, str]] = []
        # OUR frame size, set by imagine_eval before each _rollout. An adapter that generates far above
        # it (Cosmos renders 25x the pixels the metrics use) should downsample each chunk as it goes:
        # holding a long rollout at render resolution is what a 9-minute horizon cannot afford.
        self.out_hw: tuple[int, int] | None = None

    @property
    def layout(self):
        """(head, n_tokens) pairs, the same shape MultiModal exposes. No proprio: an external model that
        predicts a proprio vector should say so by adding it here, and the evals will then score it."""
        return [(h, None) for h in self.heads]

    # ---- the one method an adapter must write ---------------------------------------------------------
    @abstractmethod
    def _rollout(self, ctx: dict[str, Tensor], actions: Tensor, horizon: int,
                 heads: list[str]) -> dict[str, Tensor]:
        """ctx[head]: (N,P,h,w,3) in [0,1] at self.img_size. actions: whatever `_actions` produced.
        Return {head: (N,horizon,h,w,3)} in [0,1] at self.img_size."""

    def latents(self, obs, act) -> Tensor:
        """(B,T,n_state,d) carried latents, for eval_manifold. Override only if the model exposes them."""
        raise NotImplementedError(
            f"{type(self).__name__} does not expose latents, so manifold/interpret cannot run on it. "
            f"Override `latents` (and it will be used for both) if the model has a token bag.")

    def forward(self, obs, act):
        return self.latents(obs, act)

    # ---- action conditioning: the bridge between our per-step sticks and whatever the model eats -------
    def action_axes(self):
        """[{name, positive, negative}] from conf/interpret/<env>.yaml -- the environment's own words for
        its sticks, which is what turns a number into a sentence. Same source the VLM labelling uses."""
        ax = (self.cfg or {}) and self.cfg.get("interpret", None)
        ax = ax and ax.get("action_axes", None)
        if not ax:
            raise NotImplementedError(
                f"{type(self).__name__} conditions on text (action_mode='text'), which needs the "
                f"environment's action_axes to name the sticks -- pass `interpret=<env>` on the CLI "
                f"(e.g. interpret=starling), the same flag eval_steer and eval_interpret take.")
        from omegaconf import OmegaConf
        return OmegaConf.to_container(ax, resolve=True) if not isinstance(ax, list) else ax

    def _actions(self, actions: Tensor, horizon: int, norm=None):
        """Our per-step normalized sticks -> whatever this model eats. The returned object is passed
        straight to `_rollout`, so its type is the adapter's business: a tensor for per_step/chunk, a
        list of strings for text."""
        if self.action_mode == "per_step":
            return actions
        if self.action_mode == "chunk":
            # fold k per-step commands into one, the same mean-fold build_action_text and our own
            # action_aggregate=concat use. Requires the chunk to divide the sequence evenly.
            k = int(self.action_chunk)
            n, t, a = actions.shape
            assert t % k == 0, (f"action_chunk={k} does not divide the {t}-step action sequence; "
                                f"pick a chunk that divides P-1+horizon")
            return actions.reshape(n, t // k, k, a).mean(2)
        if self.action_mode == "text":
            # THE ACTIONS BECOME A SENTENCE, using the same function that captions clips for the VLM
            # labelling. They arrive NORMALIZED, so they are denormalized first -- a caption describing
            # standardized units would be describing nothing the pilot ever did.
            if norm is None:
                raise NotImplementedError(
                    f"{type(self).__name__} needs `norm` to denormalize actions before describing them; "
                    f"the caller did not pass one.")
            from ..evaluation.interpret import build_action_prose, build_action_text
            raw = norm.denorm_act(actions).float().cpu().numpy()      # (N, T, act_dim) in stick units
            ax = self.action_axes()
            if self.text_style == "table":
                # the VLM format: every number, named. Right when the reader can already see the clip.
                return [build_action_text(raw[i], ax) for i in range(raw.shape[0])]
            if self.text_style == "prose":
                # the GENERATOR format: a sentence. A video model's text encoder was trained on scene
                # descriptions, and under classifier-free guidance an out-of-distribution prompt embedding
                # is not ignored, it is pushed toward -- so a grid of floats is worse than no numbers.
                it = (self.cfg or {}) and self.cfg.get("interpret", None) or {}
                scene, subj = str(it.get("scene_prompt", "") or ""), str(it.get("prose_subject", "") or "The view")
                # `lead` context actions precede the first PREDICTED step, so window step g is raw index
                # lead+g. A chunked adapter asks for the window it is about to generate; everything else
                # just uses the list, which describes the whole rollout.
                lead = raw.shape[1] - horizon

                def render(i, lo, hi, _p=self.prompt_phases):
                    return build_action_prose(raw[i, lead + lo:lead + hi], ax, scene=scene, subject=subj,
                                              max_phases=_p)
                return TextActions([render(i, 0, horizon) for i in range(raw.shape[0])], render)
            raise ValueError(f"unknown text_style={self.text_style!r}; expected 'table' or 'prose'")
        if self.action_mode == "none":
            raise NotImplementedError(
                f"{type(self).__name__} takes no action input (action_mode='none'), so an "
                f"action-conditioned rollout is not defined for it. ood_horizon scores a prediction of "
                f"THIS recorded future, which requires the actions that produced it.")
        raise ValueError(f"unknown action_mode={self.action_mode!r}")

    # ---- resampling, so LPIPS is always computed at OUR frame size ------------------------------------
    @staticmethod
    def _resize(x: Tensor, hw: tuple[int, int]) -> Tensor:
        """(N,T,h,w,3) -> (N,T,H,W,3), bilinear. No-op when already the right size."""
        if tuple(x.shape[-3:-1]) == tuple(hw):
            return x
        n, t = x.shape[:2]
        y = x.reshape(n * t, *x.shape[2:]).permute(0, 3, 1, 2)          # -> (N*T,3,h,w)
        y = F.interpolate(y, size=hw, mode="bilinear", align_corners=False)
        return y.permute(0, 2, 3, 1).reshape(n, t, *hw, 3)

    # ---- the validating wrapper the routines actually call --------------------------------------------
    @torch.no_grad()
    def imagine_eval(self, ctx_obs: dict, actions: Tensor, horizon: int, heads=None,
                     decode_chunk=None, norm=None, return_bag: bool = False, **_ignored):
        want = [h for h in (heads or self.heads) if h in self.heads]
        missing = [h for h in (heads or []) if h not in self.heads and h != "proprio"]
        assert not missing, (f"{type(self).__name__} was asked for head(s) {missing}, but declares "
                             f"{list(self.heads)}. Fix the config's modality list or the adapter.")
        assert want, f"no head to roll: asked for {heads}, model has {list(self.heads)}"

        native = dict(self._check(ctx_obs, want, "context"))
        n, p = next(iter(native.values())).shape[:2]
        assert actions.shape[0] == n and actions.ndim == 3, (
            f"actions must be (N,P-1+horizon,act_dim) with N={n}; got {tuple(actions.shape)}")
        assert actions.shape[1] == p - 1 + horizon, (
            f"actions has {actions.shape[1]} steps; P={p} and horizon={horizon} require {p - 1 + horizon}")
        native = {h: self._resize(v, self.img_size) for h, v in native.items()}

        ours = tuple(int(v) for v in next(iter(ctx_obs.values())).shape[-3:-1])   # OUR frame size
        self.out_hw = ours          # a chunked adapter may downsample to this as it generates
        out = self._rollout(native, self._actions(actions, horizon, norm=norm), horizon, want)

        assert isinstance(out, dict), f"_rollout must return a dict, got {type(out).__name__}"
        res = {}
        for h in want:
            assert h in out, f"_rollout returned {sorted(out)} but was asked for {h}"
            v = out[h]
            assert v.ndim == 5 and v.shape[-1] == 3, (
                f"{h}: expected (N,horizon,h,w,3) channels-LAST, got {tuple(v.shape)}")
            assert v.shape[:2] == (n, horizon), (
                f"{h}: expected N={n} horizon={horizon}, got {tuple(v.shape[:2])}")
            lo, hi = float(v.min()), float(v.max())
            assert -0.01 <= lo and hi <= 1.01, (
                f"{h}: frames must be in [0,1], got [{lo:.3f},{hi:.3f}] -- "
                f"divide by 255, or apply the model's own output activation")
            res[h] = self._resize(v.clamp(0, 1).float(), tuple(ours))
        if return_bag:
            res["_bag"] = None                  # latent-space curves are skipped for models without a bag
        return res

    def _check(self, ctx_obs: dict, want, what: str):
        for h in want:
            assert h in ctx_obs, f"{what} has no entry for head {h!r}; got {sorted(ctx_obs)}"
            v = ctx_obs[h]
            assert v.ndim == 5 and v.shape[-1] == 3, (
                f"{what}[{h}]: expected (N,P,h,w,3) channels-LAST, got {tuple(v.shape)}")
            yield h, v


class TextActions(list):
    """The per-episode prompts, as a plain `list[str]` -- plus `window(i, lo, hi)` for an adapter that
    generates the rollout in CHUNKS and should prompt each chunk with the actions belonging to it.

    A list subclass rather than a new type on purpose: every existing adapter indexes and slices this
    exactly as before, and only the ones that need windows have to know windows exist."""

    def __init__(self, whole, render):
        super().__init__(whole)
        self._render = render

    def window(self, i: int, lo: int, hi: int) -> str:
        return self._render(i, lo, hi)


EXTERNAL: dict = {}


def register(key: str):
    def deco(cls):
        EXTERNAL[key] = cls
        return cls
    return deco


def _load_builtin():
    """Import the shipped adapters for their @register side effect. Kept lazy and forgiving: an adapter
    whose third-party dependency is not installed must not stop the others from being selectable."""
    for mod in ("external_stub", "external_cosmos"):
        try:
            __import__(f"{__package__}.{mod}", fromlist=["_"])
        except ImportError:
            pass


_load_builtin()
