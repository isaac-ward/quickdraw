"""A do-nothing external model, so the harness can be tested before any third-party weights exist.

`RepeatLastFrame` predicts the last context frame for every step of the horizon. It is the honest floor
for an open-loop video prediction: a model that has learned nothing about dynamics but everything about
the fact that consecutive frames look alike. Its LPIPS should be clearly WORSE than the trained model's
and clearly BETTER than noise, which is exactly the signal needed to prove the harness is measuring the
model rather than measuring itself.

    python -m quickdraw.eval_ood_horizon model=stub data=starling2 environments=recorded \
        data.subsample=4 data.action_aggregate=concat
"""
from __future__ import annotations

from torch import Tensor

from .external import ExternalWorldModel, register


@register("stub")
class RepeatLastFrame(ExternalWorldModel):
    heads = ("image",)
    img_size = (112, 192)
    action_mode = "per_step"        # it ignores them, but it CAN be handed them: the rollout is still
    #                                 action-conditioned in shape, so ood_horizon is well-defined

    def __init__(self, cfg=None):
        super().__init__(cfg)
        if cfg is not None:
            hs = [m for m in (cfg.model.get("modalities", None) or []) if str(m.get("kind", "")) == "image"]
            if hs:
                self.heads = tuple(str(m["name"]) for m in hs)
                sz = hs[0].get("img_size", None)
                if sz is not None:
                    self.img_size = (int(sz[0]), int(sz[1])) if not isinstance(sz, int) else (int(sz), int(sz))

    def _rollout(self, ctx: dict[str, Tensor], actions, horizon: int, heads) -> dict[str, Tensor]:
        # ctx[h] is (N,P,h,w,3); take the last context frame and hold it for the whole horizon.
        return {h: ctx[h][:, -1:].expand(-1, horizon, -1, -1, -1).contiguous() for h in heads}


@register("stub_text")
class RepeatLastFrameFromText(RepeatLastFrame):
    """The text bridge, end to end, on a model whose behaviour is known.

    Identical to RepeatLastFrame except that it declares action_mode='text', so the harness renders the
    recorded actions into the environment's own words before handing them over. It prints the caption it
    was given and then ignores it -- the point is to read what a text-conditioned model would actually
    receive, before one exists to receive it.
    """
    action_mode = "text"

    def _rollout(self, ctx, actions, horizon: int, heads):
        first = actions[0] if isinstance(actions, list) else str(actions)
        print(f"[stub_text] conditioned on {len(actions) if isinstance(actions, list) else 1} caption(s); "
              f"the first reads:\n{first}", flush=True)
        return super()._rollout(ctx, actions, horizon, heads)
