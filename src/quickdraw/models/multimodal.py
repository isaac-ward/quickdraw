"""Multimodal world-model spine (design/models/vision.md).

The carried state is a per-step TOKEN BAG (B, T, n_state, d): each enabled modality contributes
fixed token(s) (proprio:1, image:num_tokens) via the registry. Action is its OWN input-only token, so the
per-step bag fed to the backbone is [state tokens ++ action token] = n_input = n_state+1. The backbone is
the factorized space-time transformer; `readout` predicts the NEXT state bag from the state-token context
(the action token is never decoded/predicted). `to_obs` decodes each modality's slice back to its obs.

Proprio-only is the degenerate case (n_state=1, n_input=2) — the same spine, no special-casing. The
prediction mechanism in token space is model-specific (`predict_next`): LSAR = per-token MLP residual;
diffusion (P4) = a DiT denoiser. Shared rollout/forward/loss live here so every model inherits them."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import Tensor

from .flow import FlowField
from .modalities import ModalitySpec, build_modalities
from .spacetime import SpaceTimeTransformer
from .transformer import pad_block_mask
from .collapse import CollapseStrategy, Reconstruction


def _ln(x: Tensor) -> Tensor:
    """Per-token (last-dim) non-affine LayerNorm — scale-invariant carried-token normalization."""
    return F.layer_norm(x, (x.shape[-1],))


def _mlp(i: int, o: int, h: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


def _pred_loss(pred: Tensor, target: Tensor, metric: str) -> Tensor:
    """Latent-prediction loss (mirrors models/lsar.py.pred_loss): stop-grad the target; mse / BYOL
    normed_mse / cosine per the collapse strategy's pred_metric."""
    target = target.detach()
    if metric == "mse":
        return F.mse_loss(pred, target)
    if metric == "normed_mse":
        return F.mse_loss(pred, _ln(target))
    return F.mse_loss(_ln(pred), _ln(target))   # cosine: standardize both


class MultiModalSequenceModel(nn.Module):
    """Token-bag backbone + the ONE autoregressive rollout. Subclasses implement `predict_next` (and may
    add model-specific loss terms via `loss_terms`)."""

    use_kv_cache = True   # temporal KV-cache for inference rollouts (imagine_eval / imagine_shared/MPPI). Never
    #                       used in training (needs the full backprop graph); toggled off for the parity A/B.

    def __init__(self, specs: list[ModalitySpec], *, d: int, depth: int, heads: int, window: int,
                 mlp_ratio: float, rope_theta: float, action_dim: int, grad_checkpoint: bool = False,
                 compile_rollout: bool = False):
        super().__init__()
        self.grad_checkpoint = bool(grad_checkpoint)   # checkpoint each rollout-step backbone forward (train only)
        # OPT-IN (default off): torch.compile(step, mode="default") the per-step AR compute (backbone + readout)
        # to fuse the step (incl. the FlexAttention kernel, which runs UNFUSED in the eager rollout) and collapse
        # the dispatch-bound serial F-loop (design/rollout_throughput.md Phase 1B.1). NOT mode="reduce-overhead"
        # (CUDA graphs): that is fundamentally incompatible with the retained-BPTT rollout — see _compiled_step.
        # Applied ONLY in the steady p_tf==0 regime; eager everywhere else. Held in a 1-elem list so the
        # OptimizedModule wrapper is NOT registered as a submodule of self (which wraps self -> recursion).
        self.compile_rollout = bool(compile_rollout)
        self._compiled_step_holder: list = []
        self.modalities = build_modalities(specs, d)
        self.layout = [(m.name, m.n_tokens) for m in self.modalities.values()]  # bag order + slices
        self.n_state = sum(n for _, n in self.layout)
        self.n_input = self.n_state + 1                       # + action token
        self.d, self.window = d, window
        self.act_enc = _mlp(action_dim, d, d)                 # action -> 1 token
        self.backbone = SpaceTimeTransformer(d, depth, heads, window, mlp_ratio,
                                             n_slots=self.n_input, rope_theta=rope_theta)
        self.latent_norm = True   # LN the carried token bag (scale-free); LSAR turns it OFF for variance-based regs

    def arch_table(self) -> list[tuple[str, str, int]]:
        """Rows (component, shape transform, #params) describing the token-bag dataflow — printed at the top
        of progress.log. Shows how each modality becomes token(s), how the bag is fused by the backbone
        (spatial within-step + temporal across-step), and the per-token prediction head."""
        def npar(mod): return sum(p.numel() for p in mod.parameters()) if mod is not None else 0

        rows = []
        # encoders (obs -> tokens) + the action encoder
        for name, ntok in self.layout:
            mod = self.modalities[name]
            is_img = hasattr(mod, "ae")
            ins = f"(B,T,{mod.ae.cfg.img_size},{mod.ae.cfg.img_size},3)" if is_img else f"(B,T,{mod.dim})"
            enc = mod.ae if is_img else mod.enc     # image AE (encoder-only when decode_kind=flow); proprio enc MLP
            earch = getattr(mod, "encode_arch", "vit") if is_img else "mlp"   # vit|conv for image; mlp for vector
            rows.append((f"{name} encoder ({earch})", f"{ins} -> (B,T,{ntok},{self.d})", npar(enc)))
        rows.append(("action_enc", f"(B,T,{self.act_enc[0].in_features}) -> (B,T,1,{self.d})", npar(self.act_enc)))
        # backbone: fuse the token bag over space (within-step) + time (causal)
        rows.append(("space-time backbone", f"(B,T,{self.n_input},{self.d}) -> same "
                     f"[spatial {self.n_input} tok/step + temporal causal]", npar(self.backbone)))
        rows.append(("token_bag (per step)", f"{self.n_state} state + 1 action = (B,T,{self.n_input},{self.d})", 0))
        if getattr(self, "df_scale", 0.0) > 0.0 and getattr(self, "df_level_emb", None) is not None:
            rows.append(("diffusion_forcing level_emb", f"level -> (..,{self.d}) added to state tokens", npar(self.df_level_emb)))
        # dynamics head (state tokens -> NEXT state) — in the dataflow this runs BEFORE decode, so list it here
        dyn = getattr(self, "flow", None) or getattr(self, "predictor", None)
        lbl = "flow (rectified, per-token)" if hasattr(self, "flow") else "predictor (MLP residual)"
        rows.append((f"predict_next: {lbl}", f"(B,T,{self.n_state},{self.d}) -> same", npar(dyn)))
        # OPTIONAL action-flow prior: learned p(next action | context h), trained jointly, used LATER as the MPPI
        # proposal (never fed back into the WM). Only present when action_head.enabled -> show it so the head is visible.
        if getattr(self, "action_head_enabled", False) and getattr(self, "action_flow", None) is not None:
            rows.append(("action_flow (learned action prior)",
                         f"context h -> (B,T,{self.act_enc[0].in_features}) action dist", npar(self.action_flow)))
        if getattr(self, "predictor_q", None) is not None:
            rows.append(("predictor_q (BYOL online)", f"(B,T,{self.n_state},{self.d}) -> same", npar(self.predictor_q)))
        # decode heads (predicted tokens -> obs)
        for name, ntok in self.layout:
            mod = self.modalities[name]
            is_img = hasattr(mod, "ae")
            dk = getattr(mod, "decode_kind", "mse")
            dh = getattr(mod, "decode_head", None)
            ins = f"(B,T,{mod.ae.cfg.img_size},{mod.ae.cfg.img_size},3)" if is_img else f"(B,T,{mod.dim})"
            arch = getattr(mod, "decode_arch", "mlp")   # image: vit|unet; vector: mlp. UNIFIED net; kind = flow|mse(no-noise)
            net = f"{arch} {'flow' if dk == 'flow' else 'mse/no-noise'}"
            rows.append((f"{name} decode ({net})", f"(B,T,{ntok},{self.d}) -> {ins}", npar(dh)))
        return rows

    # ---- modality <-> token bag ----
    def encode_state(self, obs: dict[str, Tensor]) -> Tensor:        # {name:(B,T,*)} -> (B,T,n_state,d)
        toks = [self.modalities[name].encode(obs[name]) for name, _ in self.layout]
        bag = torch.cat(toks, dim=-2)
        return _ln(bag) if self.latent_norm else bag

    def to_obs(self, bag: Tensor, heads=None) -> dict[str, Tensor]:  # (B,*,n_state,d) -> {name:(B,*,*)}
        out, off = {}, 0
        for name, n in self.layout:
            if heads is None or name in heads:                       # partial decode (e.g. proprio-only long rollouts)
                out[name] = self.modalities[name].decode(bag[..., off:off + n, :])
            off += n
        return out

    def recon_losses(self, bag: Tensor, targets: dict[str, Tensor]) -> dict[str, Tensor]:
        """Per-modality DECODE loss of the predicted token bag vs clean target obs. Keys: `<name>` for mse
        decoders (bit-identical to before) or `flow/<name>` (+ `shortcut/<name>`) for flow decoders. The
        per-key weight is the modality weight (key.split('/')[-1] -> name)."""
        out, off = {}, 0
        for name, n in self.layout:
            mod = self.modalities[name]
            main, sc = mod.decode_loss(bag[..., off:off + n, :], targets[name])
            if mod.decode_kind == "flow":
                out[f"flow/{name}"] = main
                if sc is not None:
                    out[f"shortcut/{name}"] = sc
            else:
                out[name] = main
            off += n
        return out

    def _add_level_emb(self, bag: Tensor, levels) -> Tensor:
        """Diffusion-forcing hook: add a per-state-token noise-LEVEL embedding to the bag before fusion.
        Base = no-op (bit-identical for DSAR/LSAR and for the flow model with DF off)."""
        return bag

    def _to_input(self, bag: Tensor, act: Tensor, levels=None) -> Tensor:  # (B,T,n_state,d),(B,T,2)->(B,T,n_input,d)
        bag = self._add_level_emb(bag, levels)                 # DF: condition the backbone on context noise levels
        return torch.cat([bag, self.act_enc(act).unsqueeze(-2)], dim=-2)

    def physical_state(self, bag: Tensor):
        """Proprio 6-vec for the physical-loss variation, decoded with a FROZEN decoder (grad flows to the
        latent, not the decoder weights — like LSAR). Returns None if there is no proprio head."""
        from torch.func import functional_call
        off = 0
        for name, n in self.layout:
            if name == "proprio":
                mod = self.modalities["proprio"]
                head = mod.decode_head                                  # unified head; decode = velocity(x=0, tau=1, cond)
                cond = bag[..., off:off + n, :][..., 0, :]              # the proprio token (...,d)
                x0 = cond.new_zeros(cond.shape[:-1] + (mod.dim,))
                temb = head._temb(cond.new_ones(cond.shape[:-1] + (1,)))
                pb = {k: v.detach() for k, v in head.named_parameters()}   # FROZEN decoder (grad -> latent only)
                pb.update({k: b.detach() for k, b in head.named_buffers()})
                return functional_call(head, pb, (x0, temb, cond, None))   # head.forward == velocity -> (...,6)
            off += n
        return None

    # ---- model-specific token-space prediction (subclass) ----
    def predict_next(self, h_state: Tensor, prev_bag: Tensor) -> Tensor:
        """h_state: backbone context at the state-token slots (B,*,n_state,d); prev_bag: same shape.
        Returns the next state bag (B,*,n_state,d)."""
        raise NotImplementedError

    def readout(self, h_bag: Tensor, prev_bag: Tensor) -> Tensor:
        return self.predict_next(h_bag[..., : self.n_state, :], prev_bag)

    def carry_transform(self, bag: Tensor) -> Tensor:
        """What gets fed back into the rollout. Latent models (LSAR/diffusion) carry the predicted latent
        bag as-is (identity). Data-space (DSAR) overrides to re-encode the DECODED obs — so the rollout
        carries the observation, compounding error in data space (the defining DSAR property)."""
        return bag

    def _rollout_step(self, s_win: Tensor, a_win: Tensor, bm, prev_bag: Tensor) -> Tensor:
        """One rollout advance as backbone(+readout): (s_win (B,W,n_state,d), a_win (B,W,2), bm, prev_bag
        (B,n_state,d)) -> next state bag (B,n_state,d). This is the unit wrapped by torch.compile (mode="default")
        when compile_rollout is on. Bit-identical to the inline eager body; factored out so the compiled and eager
        paths run the SAME code (detach/teacher-forcing/carry stay OUTSIDE, in the Python loop, per detach-segment)."""
        h_last = self.backbone(self._to_input(s_win, a_win), temporal_block_mask=bm)[:, -1]
        return self.readout(h_last, prev_bag)

    def _compiled_step(self):
        """Lazily build (once) the compiled per-step fn. Each distinct pad -> a distinct temporal_block_mask ->
        one compiled variant (the ~25 pad shapes at P=8/W=32 are cached under train.py's cache_size_limit=256).

        mode="default" (inductor fusion), NOT "reduce-overhead" (CUDA graphs), even though Phase 1B.1 proposed
        the latter: reduce-overhead's per-replay STATIC memory pool is fundamentally incompatible with the
        retained-BPTT training rollout — it clobbers the saved-for-backward activations of earlier F-steps and
        raises "accessing tensor output of CUDAGraphs that has been overwritten"; its cudagraph fast-path is also
        disabled outright ("pending, uninvoked backwards"). mode="default" still fuses the per-step backbone (incl.
        the FlexAttention kernel, which runs UNFUSED in the eager rollout) + flow readout, collapsing the
        dispatch-bound serial loop: parity-verified, ~6x faster on the joint mm_flow step (design/rollout_throughput.md)."""
        if not self._compiled_step_holder:
            self._compiled_step_holder.append(torch.compile(self._rollout_step))
        return self._compiled_step_holder[0]

    def one_step_states(self, bag_win: Tensor, act_win: Tensor, attn_eager: bool = False) -> Tensor:
        """One advance of the rollout as a pure bag->bag map (token-bag analogue of SequenceWorldModel):
        (bag_win (B,W,n_state,d), act_win (B,W,2)) -> next bag (B,n_state,d) at the LAST position. Backbone-
        only, so it's identical for every MM model; attn_eager routes through the double-backprop-able
        sdpa(MATH) path used by the contraction penalty's Jacobian power-iteration."""
        h = self.backbone(self._to_input(bag_win, act_win), attn_eager=attn_eager)
        return self.readout(h[:, -1], bag_win[:, -1])

    # ---- teacher-forced parallel forward ----
    def forward(self, obs: dict[str, Tensor], act: Tensor) -> Tensor:
        s = self.encode_state(obs)
        h = self.backbone(self._to_input(s, act))
        return self.readout(h, s)

    # ---- shared autoregressive rollout (token-bag analogue of SequenceWorldModel._rollout) ----
    def _rollout(self, ctx_obs: dict[str, Tensor], actions: Tensor, horizon: int, p_tf: float,
                 true_future: dict[str, Tensor] | None, detach_every: int, use_cache: bool = False) -> Tensor:
        bag_buf = list(self.encode_state(ctx_obs).unbind(dim=1))     # P bags of (B,n_state,d)
        tf_future = self.encode_state(true_future) if true_future is not None else None
        return self._rollout_from(bag_buf, actions, horizon, p_tf, tf_future, detach_every, use_cache=use_cache)

    def _rollout_from(self, bag_buf, actions: Tensor, horizon: int, p_tf: float,
                      tf_future: Tensor | None, detach_every: int, use_cache: bool = False) -> Tensor:
        """Rollout from a PRE-ENCODED context (list of P bags). Lets callers encode the context once and
        roll many action variants from it (MPPI: encode the image context once, share across K candidates).
        use_cache: temporal KV-cache path (inference only) — see `_rollout_cached`."""
        if use_cache:
            assert p_tf == 0.0 and tf_future is None, "KV-cache rollout is inference-only (no teacher forcing)"
            return self._rollout_cached(list(bag_buf), actions, horizon)
        W = self.window
        bag_buf = list(bag_buf)
        B = bag_buf[0].shape[0]
        # Compiled per-step path is opt-in AND only for the steady AR regime (p_tf==0, training). Teacher-forcing
        # (p_tf>0, warmup) is a data-dependent branch -> stays eager; the graph captures the p_tf==0 step only.
        compiled = self.compile_rollout and p_tf == 0.0 and self.training
        step_fn = self._compiled_step() if compiled else self._rollout_step
        preds = []
        for h in range(horizon):
            Lh = len(bag_buf)
            real = min(Lh, W)
            pad = W - real
            s_win = torch.stack(bag_buf[-real:], dim=1)              # (B,real,n_state,d)
            a_win = actions[:, Lh - real:Lh]                        # (B,real,2)
            if pad:
                s_win = F.pad(s_win, (0, 0, 0, 0, pad, 0))          # pad the TIME axis at front
                a_win = F.pad(a_win, (0, 0, pad, 0))
            bm = pad_block_mask(W, pad, s_win.device)               # temporal causal + drop padded steps
            if compiled:
                s_pred = step_fn(s_win, a_win, bm, bag_buf[-1])     # backbone+readout as one fused compiled step
            elif self.grad_checkpoint and self.training:
                # recompute this step's backbone forward during backward instead of storing its activations
                # -> AR memory ∝ detach_every, not F (design/accelerations.md Exp 6). bm (a BlockMask, not a
                # tensor) is bound as a default arg so the backward-time recompute uses THIS step's mask.
                x = self._to_input(s_win, a_win)                    # (B,W,n_input,d)
                h_last = torch.utils.checkpoint.checkpoint(
                    lambda x_, _bm=bm: self.backbone(x_, temporal_block_mask=_bm)[:, -1],
                    x, use_reentrant=False)                         # (B,n_input,d)
                s_pred = self.readout(h_last, bag_buf[-1])          # (B,n_state,d)
            else:
                s_pred = self._rollout_step(s_win, a_win, bm, bag_buf[-1])  # (B,n_state,d)
            preds.append(s_pred)                                   # raw prediction -> loss/decode
            carried = self.carry_transform(s_pred)                 # data-space re-encode for DSAR; identity else
            if tf_future is not None and p_tf > 0.0:
                tf = (torch.rand(B, 1, 1, device=s_pred.device) < p_tf).float()
                s_feed = tf * tf_future[:, h] + (1.0 - tf) * carried
            else:
                s_feed = carried
            if detach_every and ((h + 1) % detach_every == 0):
                s_feed = s_feed.detach()
            bag_buf.append(s_feed)
        return torch.stack(preds, dim=1)                            # (B,horizon,n_state,d)

    def _rollout_cached(self, bag_buf, actions: Tensor, horizon: int) -> Tensor:
        """Autoregressive rollout with a temporal KV-cache (INFERENCE only — no grad, no teacher forcing).
        Numerically matches `_rollout_from` (p_tf=0) but advances the backbone INCREMENTALLY: the parallel
        path re-runs the full W-window backbone every step (O(horizon*W)); here each step is one cached
        forward (O(horizon)). Every step feeds S=1; global-index RoPE + a window-length ring buffer reproduce
        the sliding-window attention exactly, so the only difference from the parallel path is fp arithmetic
        (SDPA vs FlexAttention). Not for training: the cache holds no autograd graph across steps."""
        P = len(bag_buf)
        cache = self.backbone.make_cache()

        def step(bag, idx):                                          # bag:(B,n_state,d) at global index idx
            x = self._to_input(bag.unsqueeze(1), actions[:, idx:idx + 1]).squeeze(1)   # (B,n_input,d)
            return self.backbone.forward_cached(x, cache, idx)                         # (B,n_input,d)

        for i in range(P - 1):                                       # prefill context [0..P-2] (populate cache)
            step(bag_buf[i], i)
        preds = []
        for h in range(horizon):
            idx = P - 1 + h                                          # read out at the last fed position
            h_last = step(bag_buf[idx], idx)                         # (B,n_input,d)
            s_pred = self.readout(h_last, bag_buf[idx])              # (B,n_state,d)
            preds.append(s_pred)
            bag_buf.append(self.carry_transform(s_pred))
        return torch.stack(preds, dim=1)                             # (B,horizon,n_state,d)

    def rollout_train(self, ctx_obs, actions, true_future: dict, p_tf: float, detach_every: int = 8) -> Tensor:
        horizon = next(iter(true_future.values())).shape[1]
        return self._rollout(ctx_obs, actions, horizon, p_tf, true_future, detach_every)

    @torch.no_grad()
    def imagine_shared(self, ctx_obs: dict, actions: Tensor, horizon: int, K: int, heads=None,
                       return_bag: bool = False) -> dict[str, Tensor]:
        """MPPI helper: encode B contexts ONCE (the expensive image encode), expand to B*K, then roll K
        action variants per context. ctx_obs: (B,P,*); actions: (B*K, P-1+horizon, 2). Decodes only `heads`
        (e.g. ['proprio'] for scoring). Avoids re-encoding the image context per candidate.
        return_bag=True also returns the rolled latent token bag under key `_bag` ((B*K,H,n_state,d)) — the
        object a language reward scores directly (decode-free)."""
        with torch.autocast(device_type=actions.device.type, dtype=torch.bfloat16, enabled=actions.is_cuda):
            bags = self.encode_state(ctx_obs)                                    # (B,P,n_state,d) — one encode
            buf = [b.repeat_interleave(K, dim=0) for b in bags.unbind(1)]        # each (B*K,n_state,d)
            bag = self._rollout_from(buf, actions, horizon, 0.0, None, 0, use_cache=self.use_kv_cache)
            out = self.to_obs(bag, heads=heads)
        out = {k: v.float() for k, v in out.items()}
        if return_bag:
            out["_bag"] = bag.float()
        return out

    @torch.no_grad()
    def imagine_eval(self, ctx_obs: dict, actions: Tensor, horizon: int, heads=None,
                     use_cache: bool | None = None) -> dict[str, Tensor]:
        """`heads` limits which modalities are decoded (e.g. ['proprio'] for cheap long-horizon rollouts —
        the full latent bag, including image tokens, still rolls forward; we just skip decoding images).
        use_cache: temporal KV-cache (default self.use_kv_cache; pass False for the parity A/B)."""
        uc = self.use_kv_cache if use_cache is None else use_cache
        with torch.autocast(device_type=actions.device.type, dtype=torch.bfloat16, enabled=actions.is_cuda):
            bag = self._rollout(ctx_obs, actions, horizon, 0.0, None, 0, use_cache=uc)
            out = self.to_obs(bag, heads=heads)
        return {k: v.float() for k, v in out.items()}

    @torch.no_grad()
    def kvcache_report(self, ctx_obs: dict, actions: Tensor, horizon: int, heads=None) -> dict[str, float]:
        """One-time A/B: run the SAME rollout with and without the KV-cache, CUDA-timed (1 warmup + 1 timed
        each), to log the realized wall-clock speedup to kvcache/. Cheap — a few extra rollouts, called once
        per run. Also reports the max abs latent divergence as a correctness sanity signal."""
        import time
        dev = actions.device
        sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)

        def timed(uc):
            self.imagine_eval(ctx_obs, actions, horizon, heads=heads, use_cache=uc)     # warmup (compile/alloc)
            sync(); t0 = time.perf_counter()
            self.imagine_eval(ctx_obs, actions, horizon, heads=heads, use_cache=uc)
            sync(); return time.perf_counter() - t0

        t_cached, t_uncached = timed(True), timed(False)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=actions.is_cuda):
            bc = self._rollout(ctx_obs, actions, horizon, 0.0, None, 0, use_cache=True)
            bu = self._rollout(ctx_obs, actions, horizon, 0.0, None, 0, use_cache=False)
        return {"kvcache/speedup": t_uncached / max(t_cached, 1e-9),
                "kvcache/ms_cached": t_cached * 1e3, "kvcache/ms_uncached": t_uncached * 1e3,
                "kvcache/horizon": float(horizon),
                "kvcache/latent_max_abs_diff": float((bc - bu).abs().max())}

    def loss_terms(self, pred_bag, future_obs, obs, p_tf, act_seq=None):
        return {}, {}

    def on_optimizer_step(self) -> None:            # parity with SequenceWorldModel (EMA hook; no-op here)
        pass


class MultiModalLSAR(MultiModalSequenceModel):
    """Latent-space AR over the token bag: predict the next bag with a per-token MLP residual (+ LN);
    pred_latent = MSE to the encoded true-next bag (Reconstruction collapse: obs heads ground the encoder)."""

    def __init__(self, specs, *, d, depth, heads, window, mlp_ratio, rope_theta, action_dim,
                 grad_checkpoint: bool = False, compile_rollout: bool = False,
                 pred_hidden: int = 0, lambda_pred_latent: float = 1.0,
                 collapse: CollapseStrategy | None = None, lambda_reg: float = 1.0, expander_dim: int = 256):
        super().__init__(specs, d=d, depth=depth, heads=heads, window=window, mlp_ratio=mlp_ratio,
                         rope_theta=rope_theta, action_dim=action_dim, grad_checkpoint=grad_checkpoint,
                         compile_rollout=compile_rollout)
        h = pred_hidden or d
        self.predictor = _mlp(d, d, h)                          # per-token residual predictor
        self.lambda_pred_latent = lambda_pred_latent
        self.lambda_reg = lambda_reg
        self.lambda_pred_obs = 1.0
        # collapse strategy = the SAME abstraction as vector LSAR (recon/ema/naked/vicreg/sigreg). It's a
        # POLICY object here: MM keeps its own multi-encoder EMA/encode mechanics but reads the strategy's
        # flags + reg_loss + pred_metric. Reconstruction (obs grounds the encoder) is the default.
        self.collapse = collapse or Reconstruction()
        self.latent_norm = not self.collapse.has_reg           # OFF for vicreg/sigreg (var/cov fight LN)
        self.pred_obs_in_loss = self.collapse.obs_grounds_encoder
        self.predictor_q = None                                # BYOL online-only predictor q (asymmetry)
        if self.collapse.needs_predictor:
            self.predictor_q = _mlp(d, d, h)
        self.ema_modalities = None                             # I-JEPA/BYOL: frozen slow copy of the encoders
        if self.collapse.needs_ema:
            import copy
            self.ema_modalities = copy.deepcopy(self.modalities)
            for p in self.ema_modalities.parameters():
                p.requires_grad_(False)
            self.ema_tau = float(getattr(self.collapse, "tau", 0.996))
        self.expander = None                                   # VICReg expander (only if strategy asks; none do)
        if getattr(self.collapse, "needs_expander", False):
            self.expander = _mlp(d, expander_dim, expander_dim)

    def predict_next(self, h_state: Tensor, prev_bag: Tensor) -> Tensor:
        bag = prev_bag + self.predictor(h_state)
        return _ln(bag) if self.latent_norm else bag

    def _encode_ema(self, obs) -> Tensor:
        toks = [self.ema_modalities[name].encode(obs[name]) for name, _ in self.layout]
        bag = torch.cat(toks, dim=-2)
        return _ln(bag) if self.latent_norm else bag

    def loss_terms(self, pred_bag, future_obs, obs, p_tf, act_seq=None):
        fut = {k: future_obs[k] for k, _ in self.layout}
        target = (self._encode_ema(fut) if self.collapse.needs_ema else self.encode_state(fut)).detach()
        online = self.predictor_q(pred_bag) if self.predictor_q is not None else pred_bag  # BYOL: q(online)
        raw = {"pred_latent": _pred_loss(online, target, self.collapse.pred_metric)}
        w = {"pred_latent": self.lambda_pred_latent * self.collapse.lambda_pred}
        if self.collapse.has_reg:                              # SIGReg/VICReg on the (un-LN'd) token latents
            z = self.encode_state(obs).reshape(-1, self.d)
            if self.expander is not None:
                z = self.expander(z)
            raw["reg"], w["reg"] = self.collapse.reg_loss(z), self.lambda_reg
        return raw, w

    @torch.no_grad()
    def on_optimizer_step(self) -> None:
        if self.ema_modalities is not None:                    # EMA target encoders <- online encoders
            for pe, p in zip(self.ema_modalities.parameters(), self.modalities.parameters()):
                pe.mul_(self.ema_tau).add_(p.detach(), alpha=1.0 - self.ema_tau)
            for be, b in zip(self.ema_modalities.buffers(), self.modalities.buffers()):
                be.copy_(b)

    @torch.no_grad()
    def collapse_diagnostics(self, obs) -> dict:
        from .collapse import latent_diagnostics
        return latent_diagnostics(self.encode_state(obs).reshape(-1, self.d).float())


class MultiModalDSAR(MultiModalLSAR):
    """Data-space AR over the token bag: same delta predictor as LSAR, but the carried state is RE-ENCODED
    from the decoded obs each rollout step (carry_transform), so error compounds in data space. No
    pred_latent term — the per-head obs reconstruction is the only loss (as vector DSAR)."""

    def carry_transform(self, bag: Tensor) -> Tensor:
        return self.encode_state(self.to_obs(bag))    # decode -> obs -> re-encode (data-space feedback)

    def loss_terms(self, pred_bag, future_obs, obs, p_tf, act_seq=None):
        return {}, {}


class MultiModalFlow(MultiModalSequenceModel):
    """Latent flow-matching over the token bag. `predict_next` denoises the next-bag RESIDUAL with the
    shared FlowField (rectified flow / shortcut), applied PER TOKEN (the token is a leading dim, and the
    per-token context h already carries cross-token structure from the space-time backbone). Mirrors
    models/diffusion.py's teacher-forced flow loss, generalized to the bag. `predict_next` samples
    (deterministic ε=0 at eval unless stochastic_eval — the committed prediction)."""

    def __init__(self, specs, *, d, depth, heads, window, mlp_ratio, rope_theta, action_dim,
                 grad_checkpoint: bool = False, compile_rollout: bool = False,
                 sampling_steps: int = 6, shortcut: bool = False, predict: str = "residual",
                 stochastic_eval: bool = False, time_sampling: str = "uniform", flow_hidden: int = 0,
                 lambda_flow: float = 1.0, lambda_consistency: float = 1.0,
                 df_scale: float = 0.0, df_granularity: str = "timestep",
                 action_head_enabled: bool = False, action_head_weight: float = 1.0,
                 action_head_shortcut: bool = True, action_head_detach_gradient: bool = False,
                 dynamics_detach_encoder: bool = False):
        super().__init__(specs, d=d, depth=depth, heads=heads, window=window, mlp_ratio=mlp_ratio,
                         rope_theta=rope_theta, action_dim=action_dim, grad_checkpoint=grad_checkpoint,
                         compile_rollout=compile_rollout)
        assert predict in ("residual", "absolute")
        self.predict_residual = predict == "residual"
        self.sampling_steps = int(sampling_steps)
        self.stochastic_eval = bool(stochastic_eval)
        self.time_sampling = time_sampling
        self.lambda_flow, self.lambda_consistency = lambda_flow, lambda_consistency
        self.flow = FlowField(d, h_dim=d, hidden=(flow_hidden or d), cond="concat", shortcut=shortcut)
        self.pred_obs_in_loss = True
        self.lambda_pred_obs = 1.0
        # ---- diffusion forcing (opt-in; df_scale=0 -> everything below is inert / bit-identical) ----
        # noise the ENCODED context tokens at independent per-timestep levels (train only), and CONDITION the
        # backbone on those levels via a learned level embedding (added to the state tokens in _to_input). At
        # inference / rollout, levels default to 0 (clean) -> emb(0). See design/models/flow_heads.md sec 8.
        import math as _math
        self.df_scale = float(df_scale)
        self.df_granularity = str(df_granularity)
        if self.df_scale > 0.0:
            if self.df_granularity != "timestep":
                raise NotImplementedError(f"diffusion_forcing granularity={self.df_granularity!r} not implemented "
                                          "(only 'timestep' for now; 'modality' is a future option).")
            nfreq = 16
            self.register_buffer("_df_freqs", 2.0 * _math.pi * torch.logspace(0.0, 2.0, nfreq), persistent=False)
            self.df_level_emb = nn.Sequential(nn.Linear(2 * nfreq, d), nn.GELU(), nn.Linear(d, d))
        # ---- action-distribution head (opt-in; a learned behavior/play PRIOR for MPPI, never fed back into
        # the WM). Same FlowField class as the dynamics head: predicts the NEXT action a[t] (dz=action_dim)
        # from the PREVIOUS-step pooled backbone context h[t-1] (leak-free — it never sees a[t]). detach_gradient
        # controls whether its gradient reshapes the WM trunk. See design/models/flow_heads.md.
        self.action_head_enabled = bool(action_head_enabled)
        self.action_head_weight = float(action_head_weight)
        self.action_head_detach_gradient = bool(action_head_detach_gradient)
        # dynamics_detach_encoder: stop-grad the ENCODED context feeding the DYNAMICS (flow/latent) loss, so the
        # dynamics gradient cannot reshape the encoder. The encoder is then trained ONLY by the decode/recon loss
        # (which has no collapse shortcut) — the joint-training equivalent of IWS's frozen AE. See loss_terms.
        self.dynamics_detach_encoder = bool(dynamics_detach_encoder)
        if self.action_head_enabled:
            self.action_flow = FlowField(action_dim, h_dim=d, hidden=(flow_hidden or d), cond="concat",
                                         shortcut=action_head_shortcut)

    def _add_level_emb(self, bag: Tensor, levels) -> Tensor:
        """Add a per-timestep noise-level embedding to the state tokens (DF). levels: (...,1) in [0,1] over the
        bag's leading (position) dims, or None -> 0 (clean, the inference/rollout default). Inert when df_scale=0."""
        if self.df_scale <= 0.0:
            return bag
        if levels is None:
            levels = bag.new_zeros(bag.shape[:-2] + (1,))       # level 0 (clean) — matches inference
        feats = torch.cat([(levels * self._df_freqs).sin(), (levels * self._df_freqs).cos()], dim=-1)
        return bag + self.df_level_emb(feats).unsqueeze(-2)     # (...,1,d) broadcast over the n_state tokens

    def predict_next(self, h_state: Tensor, prev_bag: Tensor) -> Tensor:
        det = (not self.training) and (not self.stochastic_eval)
        out = self.flow.sample(h_state, steps=self.sampling_steps, deterministic=det)   # per-token over the bag
        return _ln(prev_bag + out) if self.predict_residual else _ln(out)

    def loss_terms(self, pred_bag, future_obs, obs, p_tf, act_seq=None):
        """Teacher-forced rectified-flow loss over the bag (mirrors models/diffusion.py)."""
        assert act_seq is not None
        z = self.encode_state(obs)                              # (B,L,n_state,d)
        L = z.shape[1]
        s = z[:, :-1]                                           # contexts (B,L-1,n_state,d)
        if self.dynamics_detach_encoder:                        # stop-grad: dynamics loss won't reshape the encoder
            s = s.detach()                                      #   (encoder trained only by decode/recon; anti-collapse)
        levels = None
        if self.df_scale > 0.0 and self.training:               # diffusion forcing: noise the context + tell the backbone
            levels = torch.rand(s.shape[:-2] + (1,), device=s.device, dtype=s.dtype) * self.df_scale  # (B,L-1,1)
            eps = torch.randn_like(s)
            lv = levels.unsqueeze(-2)                            # (B,L-1,1,1) broadcast over n_state,d
            s = _ln((1.0 - lv) * s + lv * eps)                  # noised context (renormalized on the sphere)
        h = self.backbone(self._to_input(s, act_seq[:, :L - 1], levels=levels))
        h_state = h[..., : self.n_state, :]                     # (B,L-1,n_state,d)
        target = (z[:, 1:] - z[:, :-1]).detach() if self.predict_residual else z[:, 1:].detach()  # target off CLEAN z
        l_flow, l_cons = self.flow.loss(h_state, target, time_sampling=self.time_sampling)
        raw, w = {"flow/latent": l_flow}, {"flow/latent": self.lambda_flow}   # dynamics flow (was "flow")
        if l_cons is not None:
            raw["shortcut/latent"], w["shortcut/latent"] = l_cons, self.lambda_consistency   # was "flow_consistency"
        if self.action_head_enabled and L >= 3:
            # action-flow PRIOR: predict a[t] from the PREVIOUS-step pooled context h[t-1] (leak-free — h[t-1]
            # never attended to a[t]). Pool the backbone context over the bag's tokens -> one vector per step.
            # detach_gradient=True -> detach so the action task does NOT reshape the WM trunk.
            h_ctx = h.mean(dim=-2)                              # (B, L-1, d): per-step context (all input tokens)
            cond = h_ctx[:, :-1]                                # h[t-1], aligned to predict a[t] for t=1..L-2
            if self.action_head_detach_gradient:
                cond = cond.detach()
            a_target = act_seq[:, 1:L - 1].detach()             # normalized a[1..L-2] (never the a[t] in cond)
            l_aflow, l_acons = self.action_flow.loss(cond, a_target, time_sampling=self.time_sampling)
            raw["flow/action"], w["flow/action"] = l_aflow, self.action_head_weight
            if l_acons is not None:
                raw["shortcut/action"], w["shortcut/action"] = l_acons, self.action_head_weight
        return raw, w

    def action_context(self, obs, act_seq) -> Tensor:
        """Per-step pooled backbone context for the action prior. Returns (B, L-1, d): entry k is h[k], the
        leak-free context (state[<=k] + a[<k]) from which the NEXT action a[k+1] is predicted. Mirrors the
        h computation in loss_terms (clean context; DF noise is train-only). For eval_action_distribution / MPPI."""
        z = self.encode_state(obs)                               # (B, L, n_state, d)
        L = z.shape[1]
        h = self.backbone(self._to_input(z[:, :-1], act_seq[:, :L - 1]))   # (B, L-1, n_input, d)
        return h.mean(dim=-2)                                    # (B, L-1, d) pooled per step

    def sample_action(self, h_ctx: Tensor, *, deterministic: bool = False, eps: Tensor | None = None) -> Tensor:
        """Sample from the learned action PRIOR given a per-step context vector `h_ctx` (..., d) — the pooled
        backbone context h[t-1]. Returns NORMALIZED actions (..., action_dim); the caller denorms. For the
        eval_action_distribution routine + the MPPI proposal. Requires action_head_enabled."""
        return self.action_flow.sample(h_ctx, steps=self.sampling_steps, deterministic=deterministic, eps=eps)
