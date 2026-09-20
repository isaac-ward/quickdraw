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
    is_external: bool = True             # run_standalone reads this to skip load_checkpoint

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
        # `modalities` exists so routines that probe `m.modalities.values()` for an image size fall
        # through to their own default instead of raising AttributeError.
        self.modalities: dict = {}

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
            from ..evaluation.interpret import build_action_text
            raw = norm.denorm_act(actions).float().cpu().numpy()      # (N, T, act_dim) in stick units
            return [build_action_text(raw[i], self.action_axes()) for i in range(raw.shape[0])]
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

        out = self._rollout(native, self._actions(actions, horizon, norm=norm), horizon, want)

        assert isinstance(out, dict), f"_rollout must return a dict, got {type(out).__name__}"
        ours = next(iter(ctx_obs.values())).shape[-3:-1]                 # OUR frame size, from the context
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
