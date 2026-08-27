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


LATENT_NORMS = ("layernorm", "affine", "none")


def _ln(x: Tensor) -> Tensor:
    """Per-token (last-dim) non-affine LayerNorm — scale-invariant carried-token normalization."""
    return F.layer_norm(x, (x.shape[-1],))


def resolve_latent_norm(v) -> str:
    """`model.latent_norm` -> one of LATENT_NORMS. Accepts the legacy bool (true -> layernorm, false -> none).

      layernorm  per-token non-affine LN on the BAG, at encode and after every dynamics step. Scale-free and
                 stable, but NOT invertible: it discards each token's mean+std (2 scalars/token), measured at
                 -3.51 dB of reconstruction ceiling on robocasa 128px + frozen TAESD.
      affine     fixed per-CHANNEL scale+shift on the AE LATENT (calibrated once from the training set), with
                 the exact inverse before decode. Same unit-variance input for the dynamics, INVERTIBLE, so it
                 costs 0 dB. The bag itself is then left alone. This is the Stable-Diffusion `scaling_factor`
                 idea (a latent std) generalized per channel.
      none       no normalization anywhere (research escape hatch; LSAR's variance regularizers force this)."""
    if isinstance(v, bool):
        return "layernorm" if v else "none"
    s = str(v).strip().lower()
    if s not in LATENT_NORMS:
        raise ValueError(f"model.latent_norm must be one of {LATENT_NORMS} (or a bool for the legacy form), got {v!r}")
    return s


def _mlp(i: int, o: int, h: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class FourierMLP(nn.Module):
    """[raw x | fourier_features(x)] -> MLP -> out. n_freq=0 makes it EXACTLY the plain `_mlp(i, o, h)` it
    replaces (same submodule name `net`, same shapes, same init order), so it is bit-identical when off.

    Used for the ACTION encoder and for vector-modality encoders: both take z-scored, unbounded inputs whose
    small differences matter, and a raw linear map of near-collinear inputs discards exactly that. `squash`
    bounds the input first -- required, see features.fourier_features."""

    def __init__(self, i: int, o: int, h: int, n_freq: int = 0, squash: float = 4.0,
                 input_squash: str = "none"):
        super().__init__()
        from .features import fourier_dim, fourier_freqs
        if input_squash not in ("none", "symlog"):
            raise ValueError(f"input_squash must be 'none' or 'symlog', got {input_squash!r}")
        self.n_freq, self.squash, self.in_raw = int(n_freq), float(squash), int(i)
        self.input_squash = input_squash
        in_dim = i + (fourier_dim(i, n_freq) if n_freq > 0 else 0)
        if n_freq > 0:
            self.register_buffer("freqs", fourier_freqs(n_freq), persistent=False)
        self.net = _mlp(in_dim, o, h)

    def forward(self, x: Tensor) -> Tensor:
        if self.input_squash == "symlog":
            from .features import symlog
            x = symlog(x)                    # BEFORE both paths: the raw copy AND the fourier expansion see it,
            #                                  so the fourier clamp below becomes nearly inert rather than doing
            #                                  the bounding by itself (and losing everything past the threshold).
        if self.n_freq > 0:
            from .features import fourier_features
            x = torch.cat([x, fourier_features(x, self.freqs, squash=self.squash)], dim=-1)
        return self.net(x)


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
                 compile_rollout: bool = False, latent_norm: str | bool = "affine",
                 action_fourier_freqs: int = 0, action_squash: str = "none",
                 relative_position: bool = False, position_idx=None, relative_scale=None,
                 dynamics_prior=None):
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
        # RELATIVE POSITION ENCODING (design/collapse.md floor lever): re-express the proprio POSITION channels
        # relative to the window's first step, `p~ = (p - anchor)/s_rel`, so the codec represents a small
        # within-window displacement (~8.6x smaller than absolute) instead of a ±300 value -> lower raw floor.
        # Applied at the obs<->bag boundary only (encode_state input / to_obs+recon output); the latent rollout
        # is untouched, and every EXTERNAL caller passes/receives ABSOLUTE obs (the anchor is a transient arg,
        # never stored). Default OFF = bit-identical. position_idx are the proprio dims that are world position.
        self.relative_position = bool(relative_position)
        self._pos_idx = list(int(i) for i in position_idx) if (relative_position and position_idx is not None) else None
        if self._pos_idx is not None:
            sc = relative_scale if relative_scale is not None else 1.0
            sc = torch.as_tensor(sc, dtype=torch.float32)
            if sc.ndim == 0:
                sc = sc.repeat(len(self._pos_idx))
            assert len(sc) == len(self._pos_idx), f"relative_scale {sc.shape} != position_idx {len(self._pos_idx)}"
            self.register_buffer("_rel_scale", sc)   # NORMALIZED-space per-dim std of the within-window displacement
        # OWM PHYSICS PRIOR (RecordedEnv hook; default None -> BYTE-IDENTICAL). When a dynamics_prior callable is
        # given, proprio prediction = physics(prev_obs, action) + residual_head(predicted proprio token). This adds
        # one zero-init head, driven by physics_proprio_chained (train, via the decode/proprio prior path or the
        # legacy physics/proprio term) and _physics_proprio_rollout (eval/MPPI). residual=0 at init -> prediction
        # == the verified physics law, so the action-response is correct by construction.
        self.dynamics_prior = dynamics_prior
        if dynamics_prior is not None:
            import torch.nn as _nn
            self._proprio_dim = int(self.modalities["proprio"].dim)
            _off = 0
            for _nm, _nt in self.layout:
                if _nm == "proprio":
                    self._proprio_off = _off
                    break
                _off += _nt
            self.residual_head = _nn.Sequential(_nn.Linear(d, d), _nn.SiLU(), _nn.Linear(d, self._proprio_dim))
            _nn.init.zeros_(self.residual_head[-1].weight)
            _nn.init.zeros_(self.residual_head[-1].bias)     # zero-init output -> pred == physics at init
        # UNIFIED decode/physics (design/models/probabilistic_heads.md): when the proprio modality declares a
        # prior (identity|physics) the physics chain IS the proprio decode -> `decode/proprio` (no separate
        # `physics/proprio` bolt-on, no `codec/roundtrip_proprio`). prior=="none" with dynamics_prior set is the
        # LEGACY path (physics/proprio + roundtrip, byte-identical). A residual head can't do a context-free
        # autoencode, so round-trip is illegal in prior mode: fail loudly rather than silently anchor nothing.
        _pm = self.modalities["proprio"] if "proprio" in self.modalities else None
        self._proprio_prior = str(getattr(_pm, "prior", "none") or "none")
        if self._proprio_prior != "none" and float(getattr(self.modalities["proprio"], "latent_loss_weight", 0.0) or 0.0) > 0.0:
            raise ValueError(
                f"proprio modality has prior={self._proprio_prior!r} AND latent_loss_weight>0 (round-trip). A "
                f"prior-mode head predicts a RESIDUAL on prev/physics, so it cannot do the context-free "
                f"encode->decode round-trip the codec anchor needs. Set modalities.<proprio>.latent_loss_weight=0 "
                f"(the encoder is grounded by the decode/proprio dynamics loss instead), or prior=none to keep the "
                f"round-trip.")
        # action -> 1 token. action_fourier_freqs>0 prepends sin/cos features so SMALL action differences are
        # linearly separable (robocasa's 12-dim action is effectively ~4 dims and consecutive actions differ
        # slightly). 0 = off = bit-identical to a plain _mlp.
        self.act_enc = FourierMLP(action_dim, d, d, n_freq=int(action_fourier_freqs),
                                  input_squash=str(action_squash))
        self.backbone = SpaceTimeTransformer(d, depth, heads, window, mlp_ratio,
                                             n_slots=self.n_input, rope_theta=rope_theta)
        # How the latent is made scale-free for the dynamics — see resolve_latent_norm for the three options.
        # `self.latent_norm` stays a BOOL meaning "LN the carried bag", because that is what the five bag-level
        # _ln sites gate on. It MUST be honoured at EVERY point the bag is produced — encode_state AND
        # predict_next AND the DF noised context — or the encoder and the dynamics emit bags in two different
        # spaces (bug, 2026-08-09: predict_next LN'd unconditionally).
        self.concat_action_embedding = False    # Flow subclass opts in; LSAR/DSAR predictors expect width d
        self.use_action_slot = False            #   same -- see _cond
        self.latent_norm_type = resolve_latent_norm(latent_norm)
        self.latent_norm = self.latent_norm_type == "layernorm"
        if self.latent_norm_type == "affine":       # normalization moves OFF the bag and ONTO the AE latent
            _capable = [n for n, md in self.modalities.items() if hasattr(md, "enable_latent_affine")]
            # affine presupposes a FIXED latent. A pretrained trunk that is still TRAINING does not have one:
            # its per-channel statistics drift away from the one-shot fit-start calibration, and the decode-side
            # inverse then stops being an inverse. Same premise failure as a bespoke encoder, so same answer.
            _thawed = [n for n in _capable if not getattr(self.modalities[n], "taesd_frozen", True)]
            if _thawed:
                raise ValueError(
                    f"model.latent_norm='affine' requires a FROZEN pretrained trunk, but {_thawed} have "
                    f"freeze=false. affine calibrates per-channel statistics ONCE at fit start and inverts them "
                    f"exactly before decode; a trunk that keeps training drifts away from those numbers, so the "
                    f"'inverse' silently stops inverting. Set modalities.<i>.freeze=true, or use "
                    f"model.latent_norm=layernorm which re-normalizes every forward and needs no fixed stats."
                )
            for _n in _capable:
                self.modalities[_n].enable_latent_affine()
            # FAIL LOUDLY rather than silently degrade to `none`. `affine` is implemented on the PRETRAINED
            # trunk only, because it presupposes a FIXED latent whose per-channel statistics can be calibrated
            # once from the data. A bespoke/learned AE has no such fixed latent -- its encoder drifts its own
            # scale during training, so a one-shot calibration is meaningless. Without this check an
            # affine + pretrained=false run got NO normalization anywhere (no bag LN because the type is not
            # layernorm, and no modality affine because the trunk cannot do it) -- and since affine became the
            # DEFAULT on 2026-08-10, that silently applied to every bespoke run.
            if not _capable:
                # RAISE, no fallback (user, 2026-08-11). A config must MEAN what it says: a run that silently
                # trains under a different normalization than the one requested is unattributable afterwards.
                # affine is implemented on the PRETRAINED trunk only -- it calibrates fixed per-channel stats of
                # a FIXED latent, and a learned encoder has none (it drifts its own scale as it trains), so a
                # one-shot calibration at fit start is meaningless. Be explicit at the call site instead.
                raise ValueError(
                    "model.latent_norm='affine' but NO modality supports it (it needs a PRETRAINED image trunk: "
                    "a learned/bespoke encoder has no fixed latent whose per-channel statistics can be "
                    "calibrated once). Set model.latent_norm=layernorm EXPLICITLY for a bespoke AE "
                    "(modalities.<i>.pretrained=false), or modalities.<i>.pretrained=true to use the frozen "
                    "trunk affine was built for. There is deliberately NO fallback."
                )

    def arch_table(self) -> list[tuple[str, str, int]]:
        """Rows (component, shape transform, #params) describing the token-bag dataflow — printed at the top
        of progress.log. Shows how each modality becomes token(s), how the bag is fused by the backbone
        (spatial within-step + temporal across-step), and the per-token prediction head."""
        # `mod is not None` is not enough: the PRETRAINED image modality's `.ae` is an _AEHolder, a plain
        # non-Module shim that only carries `.cfg` (it has no `.parameters()` by design). Guard on the method.
        def npar(mod): return sum(p.numel() for p in mod.parameters()) if hasattr(mod, "parameters") else 0

        rows = []
        # encoders (obs -> tokens) + the action encoder
        for name, ntok in self.layout:
            mod = self.modalities[name]
            is_img = hasattr(mod, "ae")
            ins = f"(B,T,{mod.ae.cfg.img_size},{mod.ae.cfg.img_size},3)" if is_img else f"(B,T,{mod.dim})"
            enc = mod.ae if is_img else mod.enc     # image AE (encoder-only when decode_kind=flow); proprio enc MLP
            if is_img and hasattr(mod, "down_adapter"):   # pretrained: .ae is the param-less shim -> count the
                enc = mod.down_adapter                    # REAL encode path (frozen TAESD reported separately)
            earch = getattr(mod, "encode_arch", "vit") if is_img else "mlp"   # vit|conv for image; mlp for vector
            if is_img and hasattr(mod, "taesd"):
                earch = f"taesd{'-frozen' if getattr(mod, 'taesd_frozen', False) else ''}+adapter"
            rows.append((f"{name} encoder ({earch})", f"{ins} -> (B,T,{ntok},{self.d})", npar(enc)))
        rows.append(("action_enc", f"(B,T,{self.act_enc.in_raw}) -> (B,T,1,{self.d})", npar(self.act_enc)))
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
                         f"context h -> (B,T,{self.act_enc.in_raw}) action dist", npar(self.action_flow)))
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

    # ---- relative-position encoding (pure helpers; no-op when off) ----
    def _rel_on(self) -> bool:
        return self.relative_position and self._pos_idx is not None

    def rel_anchor(self, obs) -> Tensor:
        """Per-window anchor = the proprio POSITION at the FIRST timestep (in normalized space). obs may be the
        obs dict or a proprio tensor (B,T,dim). Returns (B,1,len(pos_idx)), broadcastable over the time axis."""
        p = obs["proprio"] if isinstance(obs, dict) else obs
        return p[..., :1, self._pos_idx]

    def relativize(self, obs: dict, anchor: Tensor) -> dict:
        """obs dict -> obs dict with proprio position channels re-expressed as (p - anchor)/s_rel. No-op if off.
        Does NOT mutate the input. `anchor` is (B,1,len(pos_idx)); broadcasts over T."""
        if not self._rel_on():
            return obs
        p = obs["proprio"]
        rel = p.clone()
        # .to(rel.dtype): _rel_scale is fp32 so the RHS promotes to fp32; under bf16 autocast the destination is
        # bf16 and index_put requires matching dtypes. Cast back so it works in fp32 AND autocast.
        rel[..., self._pos_idx] = ((p[..., self._pos_idx] - anchor) / self._rel_scale).to(rel.dtype)
        return {**obs, "proprio": rel}

    def absolutize_proprio(self, proprio: Tensor, anchor: Tensor) -> Tensor:
        """Inverse of relativize on a proprio tensor (B,T,dim): position channels -> value*s_rel + anchor.
        No-op if off. Used so decoded/reconstructed proprio comes back to ABSOLUTE for loss/report.
        fp32 (issue #14): the absolute position is a small displacement ON TOP OF a ~hundreds-of-metres anchor.
        Storing it in bf16 (~3 sig-figs) quantizes it to ~1 m, which INFLATES the reported position error AND
        caps the proprio decode loss at a ~1 m floor. Compute + return the position channels in fp32."""
        if not self._rel_on():
            return proprio
        out = proprio.float()
        out[..., self._pos_idx] = proprio[..., self._pos_idx].float() * self._rel_scale + anchor.float()
        return out

    # ---- modality <-> token bag ----
    # `anchor` (relative-position encoding): the per-window position anchor (see rel_anchor). When given (and the
    # feature is on), encode relativizes the input position channels and to_obs de-relativizes the decoded ones,
    # so the CODEC works in the small within-window frame while every caller stays in ABSOLUTE obs. None = off.
    def encode_state(self, obs: dict[str, Tensor], anchor: Tensor | None = None) -> Tensor:  # -> (B,T,n_state,d)
        if anchor is not None:
            obs = self.relativize(obs, anchor)                       # no-op if _rel_on() is False
        toks = [self.modalities[name].encode(obs[name]) for name, _ in self.layout]
        bag = torch.cat(toks, dim=-2)
        return _ln(bag) if self.latent_norm else bag

    def to_obs(self, bag: Tensor, heads=None, anchor: Tensor | None = None, commit: bool = False) -> dict[str, Tensor]:
        """`commit=True` -> deterministic decode even under decode_stochastic (see Modality.decode)."""
        out, off = {}, 0
        for name, n in self.layout:
            if heads is None or name in heads:                       # partial decode (e.g. proprio-only long rollouts)
                out[name] = self.modalities[name].decode(bag[..., off:off + n, :], commit=commit)
            off += n
        if anchor is not None and "proprio" in out:
            out["proprio"] = self.absolutize_proprio(out["proprio"], anchor)   # back to ABSOLUTE (no-op if off)
        return out

    def recon_losses(self, bag: Tensor, targets: dict[str, Tensor],
                     pre_z_targets: Tensor | None = None, anchor: Tensor | None = None,
                     prior: dict | None = None) -> dict[str, Tensor]:
        """Per-modality DECODE loss of the predicted token bag vs clean target obs.

        Returns (losses, weights) -- the SAME contract as loss_terms, so no caller infers a weight by parsing
        a key. Keys are ROLE-first and never encode the decode parameterization: `decode/<name>` for every
        modality (mse or flow), `decode/<name>_shortcut` for flow decoders with self-consistency, and
        `codec/roundtrip_<name>` from roundtrip_losses. All values are RAW; weights are applied at the sum.

        `prior` (unified decode/physics): when a modality declares a decode PRIOR its `decode/<name>` is the
        prior-anchored CHAINED prediction (env_fn(prev,act)+head(token), obs-space AR) rather than the stateless
        per-frame decode -- so the old `physics/proprio` bolt-on IS this decode term. It runs on the FULL rollout
        (a chain can't be frame-subsampled) so the caller passes it separately from the recon_frac-subset `bag`.
        Bundle: {preds(B,F,n_state,d full), prev0_abs, true_future_abs, act_raw, p_tf, norm, target(normalized)}."""
        # decode_loss decodes the predicted bag (in the RELATIVE frame when the anchor is on) WITHOUT going
        # through to_obs, so it never de-relativizes -> its targets must be relativized to match. (roundtrip
        # below goes through to_obs, which DOES de-relativize, so it keeps the ABSOLUTE targets.)
        dtgt = self.relativize(targets, anchor) if anchor is not None else targets
        out, wts, off = {}, {}, 0
        for name, n in self.layout:
            mod = self.modalities[name]
            if name == "proprio" and getattr(self, "_proprio_prior", "none") != "none" and prior is not None:
                # PRIOR MODE: decode/proprio = the chained prior-anchored prediction (== the old physics/proprio).
                phys_abs = self.physics_proprio_chained(prior["preds"], prior["prev0_abs"],
                                                        prior["true_future_abs"], prior["act_raw"], prior["p_tf"])
                out["decode/proprio"] = F.mse_loss(prior["norm"].norm_obs(phys_abs), prior["target"])
                wts["decode/proprio"] = float(mod.weight)
                off += n
                continue
            main, sc = mod.decode_loss(bag[..., off:off + n, :], dtgt[name])
            out[f"decode/{name}"], wts[f"decode/{name}"] = main, float(mod.weight)
            if sc is not None:                                  # flow decoders only
                out[f"decode/{name}_shortcut"], wts[f"decode/{name}_shortcut"] = sc, float(mod.weight)
            off += n
        rt, rtw = self.roundtrip_losses(targets, pre_z=pre_z_targets, anchor=anchor)   # codec/roundtrip_<name>
        out.update(rt); wts.update(rtw)
        return out, wts

    def roundtrip_losses(self, targets: dict[str, Tensor], pre_z: Tensor | None = None,
                         anchor: Tensor | None = None) -> tuple[dict, dict]:
        """ENCODE->DECODE round-trip loss for pretrained-AE trunks (#12), through THE MODEL'S OWN path.

        Deliberately uses encode_state()/to_obs() rather than the modality's encode/decode: whatever the
        dynamics actually consumes — LayerNormed bag or not (see self.latent_norm) — is what gets supervised,
        with no duplicated normalisation logic to drift. The earlier modality-level version measured
        up(down(g)) with NO LayerNorm and so reported a perfect 2.5e-5 while the real path was 3.5 dB worse.

        Note decode_loss only ever trains the decoder on the dynamics' PREDICTED bag; nothing else asks the
        encode->decode path to reconstruct, which is how the first adapter reached only ~10.5 dB."""
        wts = {n: (getattr(self.modalities[n], "latent_loss_weight", 0.0) or 0.0) for n, _ in self.layout}
        # Any modality with latent_loss_weight>0 gets the encode->decode anchor -- NOT just pretrained/taesd
        # trunks. A TRAINABLE (bespoke) encoder needs Dec(Enc(x))->x too, or it collapses (design/collapse.md);
        # `to_obs`/`encode_state` below are already modality-agnostic, this was only gated off.
        heads = [n for n, w in wts.items() if w > 0]
        if not heads:
            return {}, {}      # NOTE the TUPLE: recon_losses unpacks (losses, weights). A bare {} here broke
            #                    every bespoke-AE run (no pretrained trunk -> no heads) with
            #                    "ValueError: not enough values to unpack (expected 2, got 0)" -- missed when
            #                    this function changed contract, because nothing tested a non-pretrained trunk.
        bag = self.encode_state(targets, anchor) if pre_z is None else pre_z   # REAL encode (LN incl.; relativizes
        #                          internally when anchor on). pre_z = the SAME encode from the shared-encode fast
        #                          path (_step), sliced -- bit-identical at noise_std=0 (design/accelerations P4).
        # commit=True: the anchor measures the CODEC, which is deterministic. Scoring a stochastic sample here
        # would train its variance to zero (MSE of a sample = bias^2 + variance) at weight 10 -- see decode().
        rec = self.to_obs(bag, heads=heads, anchor=anchor, commit=True)   # REAL decode, de-relativized -> ABSOLUTE
        # RAW mse + its weight, so the logged series is comparable across runs that sweep latent_loss_weight
        # (every sibling term is logged raw and weighted at the sum). Returning it pre-scaled made the codec
        # panel rescale while the decode panels did not.
        return ({f"codec/roundtrip_{n}": F.mse_loss(rec[n], targets[n]) for n in heads},
                {f"codec/roundtrip_{n}": wts[n] for n in heads})

    def _add_level_emb(self, bag: Tensor, levels) -> Tensor:
        """Diffusion-forcing hook: add a per-state-token noise-LEVEL embedding to the bag before fusion.
        Base = no-op (bit-identical for DSAR/LSAR and for the flow model with DF off)."""
        return bag

    def _to_input(self, bag: Tensor, act: Tensor, levels=None) -> Tensor:  # (B,T,n_state,d),(B,T,2)->(B,T,n_input,d)
        bag = self._add_level_emb(bag, levels)                 # DF: condition the backbone on context noise levels
        return torch.cat([bag, self.act_enc(act).unsqueeze(-2)], dim=-2)

    def physics_proprio_chained(self, pred_bag: Tensor, prev0_abs: Tensor, true_future_abs: Tensor,
                                act_raw: Tensor, p_tf: float) -> Tensor:
        """p_tf-respecting chained physics rollout, ABSOLUTE units, (B,F,obs_dim). Each step feeds the PREVIOUS
        step's own prediction forward with probability (1-p_tf) instead of the true prev — so the residual head
        trains on the SAME compounding-error regime the eval rollout (imagine_eval) uses, closing the
        teacher-forced-train / autoregressive-eval exposure-bias gap. At p_tf>=1 this reduces exactly to the
        fully teacher-forced physics_proprio loss; at p_tf<=0 it is the eval rollout `_physics_proprio_rollout`
        (see its INVARIANT note — this is the scheduled-sampling GENERALIZATION of that same physics formula).
        pred_bag (B,F,n_state,d); prev0_abs (B,obs_dim) true ctx-last; true_future_abs (B,F,obs_dim); act_raw (B,F,act_dim)."""
        steps = pred_bag.shape[1]
        prev = prev0_abs.float()
        outs = []
        for t in range(steps):
            token = pred_bag[:, t, self._proprio_off, :]                   # (B,d)
            cur = self.dynamics_prior(prev, act_raw[:, t].float()) + self.residual_head(token).float()
            outs.append(cur)
            true_t = true_future_abs[:, t].float()
            if p_tf >= 1.0:
                prev = true_t                                             # full teacher forcing (== physics_proprio)
            elif p_tf <= 0.0:
                prev = cur                                                # fully autoregressive (eval regime)
            else:                                                         # scheduled sampling, per-element coin
                tf = (torch.rand(cur.shape[0], 1, device=cur.device) < p_tf).float()
                prev = tf * true_t + (1.0 - tf) * cur
        return torch.stack(outs, dim=1)                                   # (B,F,obs_dim) ABSOLUTE

    def physical_state(self, bag: Tensor, anchor: Tensor | None = None):
        """Proprio physical readout for the physical-loss variation, decoded with a FROZEN decoder (grad flows
        to the latent, not the decoder weights — like LSAR). Returns None if there is no proprio head.
        `anchor`: de-relativize to ABSOLUTE units (v = dp/dt continuity only holds in absolute — relativization
        scales position but not velocity), so the physics residual is computed on real physical quantities."""
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
                out = functional_call(head, pb, (x0, temb, cond, None))    # head.forward == velocity -> (...,dim)
                return self.absolutize_proprio(out, anchor) if anchor is not None else out
            off += n
        return None

    # ---- model-specific token-space prediction (subclass) ----
    def predict_next(self, h_state: Tensor, prev_bag: Tensor) -> Tensor:
        """h_state: backbone context at the state-token slots (B,*,n_state,d); prev_bag: same shape.
        Returns the next state bag (B,*,n_state,d)."""
        raise NotImplementedError

    def _cond(self, h_bag: Tensor, act: Tensor | None = None) -> Tensor:
        """Backbone output (+ the raw action) -> the per-token conditioning the dynamics consumes.

        Up to three channels per state token, concatenated on the feature axis:
          h_state   h_bag[..., :n_state, :]                   the state slots (always)
          slot      h_bag[..., n_state, :]  broadcast         the ACTION token's own backbone output --
                    (use_action_slot)                         the action CONTEXTUALISED by the current state
          raw       act_enc(act)            broadcast         the RAW pre-backbone embedding
                    (concat_action_embedding)

        WHY BOTH, and why the slot is no longer discarded. readout() used to slice slot n_state off entirely, so
        the action's only influence was whatever attention weight the state tokens chose to give 1 of 10 slots --
        measured grad/norm/act_enc 0.17% of the total gradient at ep0. Computing that slot through the whole
        backbone and then throwing it away is a configuration NOBODY in the literature uses: TWM (2303.07109)
        reads out ONLY through action positions, IRIS (2209.00588) generates the next frame conditioned on the
        action token, and DiT's own ablation (2212.09748) found append-a-token-then-remove-it the WORST
        conditioning scheme it tested. The two channels carry DIFFERENT information and are kept for different
        reasons: the slot is action x state but is suppressible (the backbone's residual branches could cancel
        it); the raw embedding never passes through the backbone so it cannot be cancelled, but it is the same
        vector for every token and knows nothing about the current state.

        act is threaded from every call site rather than stashed on self, so the compiled and grad-checkpointed
        rollouts see the same pure function. If a caller omits it, the raw channel is skipped -- which would
        change the conditioning WIDTH, so callers must pass it whenever concat_action_embedding is on."""
        h_state = h_bag[..., : self.n_state, :]
        parts = [h_state]
        if getattr(self, "use_action_slot", False):
            parts.append(h_bag[..., self.n_state : self.n_state + 1, :].expand_as(h_state))
        if getattr(self, "concat_action_embedding", False) and act is not None:
            parts.append(self.act_enc(act).unsqueeze(-2).expand_as(h_state))
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

    def readout(self, h_bag: Tensor, prev_bag: Tensor, act: Tensor | None = None) -> Tensor:
        return self.predict_next(self._cond(h_bag, act), prev_bag)

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
        return self.readout(h_last, prev_bag, a_win[:, -1])

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
        return self.readout(h[:, -1], bag_win[:, -1], act_win[:, -1])

    # ---- teacher-forced parallel forward ----
    def forward(self, obs: dict[str, Tensor], act: Tensor, anchor: Tensor | None = None) -> Tensor:
        s = self.encode_state(obs, anchor)
        h = self.backbone(self._to_input(s, act))
        return self.readout(h, s, act)

    # ---- shared autoregressive rollout (token-bag analogue of SequenceWorldModel._rollout) ----
    def _rollout(self, ctx_obs: dict[str, Tensor], actions: Tensor, horizon: int, p_tf: float,
                 true_future: dict[str, Tensor] | None, detach_every: int, use_cache: bool = False,
                 anchor: Tensor | None = None, return_feeds: bool = False) -> Tensor:
        bag_buf = list(self.encode_state(ctx_obs, anchor).unbind(dim=1))     # P bags of (B,n_state,d)
        # Only encode the true future when teacher forcing can actually USE it (p_tf>0). At p_tf==0 — the
        # steady AR regime (most of training) AND all of val — tf_future is never read (see the `p_tf > 0.0`
        # guard at the mix below), so encoding F frames + retaining their adapter graph is pure waste.
        # SAME anchor as the context (the window's first-step position) so ctx and future share the frame.
        tf_future = self.encode_state(true_future, anchor) if (true_future is not None and p_tf > 0.0) else None
        return self._rollout_from(bag_buf, actions, horizon, p_tf, tf_future, detach_every, use_cache=use_cache,
                                  return_feeds=return_feeds)

    def _rollout_from(self, bag_buf, actions: Tensor, horizon: int, p_tf: float,
                      tf_future: Tensor | None, detach_every: int, use_cache: bool = False,
                      return_feeds: bool = False) -> Tensor:
        """Rollout from a PRE-ENCODED context (list of P bags). Lets callers encode the context once and
        roll many action variants from it (MPPI: encode the image context once, share across K candidates).
        use_cache: temporal KV-cache path (inference only) — see `_rollout_cached`."""
        if use_cache:
            assert p_tf == 0.0 and tf_future is None, "KV-cache rollout is inference-only (no teacher forcing)"
            return self._rollout_cached(list(bag_buf), actions, horizon)
        W = self.window
        bag_buf = list(bag_buf)
        P0 = len(bag_buf)          # return_feeds: bag_buf[P0+k] is what the step predicting frame k+1 STOOD ON
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
                s_pred = self.readout(h_last, bag_buf[-1], a_win[:, -1])   # (B,n_state,d)
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
        preds = torch.stack(preds, dim=1)                           # (B,horizon,n_state,d)
        if not return_feeds:
            return preds
        # FEEDS = the p_tf mix each step actually stood on, so a dynamics loss conditioned on them inherits
        # the p_tf schedule instead of being pinned teacher-forced (design/flow.md). ATTACHED on purpose: a
        # detached context would only ever teach a one-step correction and would never train COMPOUNDING,
        # which is the whole defect. The graph is already retained for the decode loss, so the marginal cost
        # is ~zero -- autograd accumulates into nodes that backward already traverses. detach_every still
        # truncates the reach, exactly as it does for the decode loss.
        return preds, torch.stack(bag_buf[P0:], dim=1)              # (B,horizon,n_state,d)

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
            s_pred = self.readout(h_last, bag_buf[idx], actions[:, idx])   # (B,n_state,d)
            preds.append(s_pred)
            bag_buf.append(self.carry_transform(s_pred))
        return torch.stack(preds, dim=1)                             # (B,horizon,n_state,d)

    def rollout_train(self, ctx_obs, actions, true_future: dict, p_tf: float, detach_every: int = 8,
                      precomputed_ctx: Tensor | None = None, anchor: Tensor | None = None,
                      return_feeds: bool = False) -> Tensor:
        """return_feeds -> (preds, feeds): feeds[k] is what the step predicting future frame k+1 STOOD ON
        (the p_tf teacher-forcing mix). Used by dynamics_follows_p_tf; see design/flow.md."""
        horizon = next(iter(true_future.values())).shape[1]
        if precomputed_ctx is not None:                       # shared-encode fast path (_step, p_tf==0 only):
            assert p_tf == 0.0, "precomputed_ctx is the p_tf==0 shared encode (no teacher forcing)"  # ctx already
            return self._rollout_from(list(precomputed_ctx.unbind(dim=1)),           # LN'd + relativized already
                                      actions, horizon, 0.0, None, detach_every,      # (encoded with anchor in _step)
                                      return_feeds=return_feeds)
        return self._rollout(ctx_obs, actions, horizon, p_tf, true_future, detach_every, anchor=anchor,
                             return_feeds=return_feeds)

    def _physics_proprio_rollout(self, bag: Tensor, prev0_abs: Tensor, actions: Tensor,
                                 horizon: int, P: int, norm) -> Tensor:
        """Canonical fully-autoregressive physics proprio rollout (shared by imagine_eval + imagine_shared):
        obs[t] = dynamics_prior(obs[t-1], act[t]) + residual_head(latent[t]), chained on its OWN prediction,
        obs[-1] = last true context. fp32 (issue #14). Returns NORMALIZED (B,horizon,obs_dim) matching the decode
        it replaces. `actions` NORMALIZED (B,P-1+horizon,act_dim); the action for step t is index P-1+t.

        INVARIANT — one rollout, one physics, two entry points. There is now a SINGLE inference rollout
        implementation, `_imagine`; imagine_eval and imagine_shared are thin specializations of it (K=1 vs
        encode-once K fan-out), so they are equivalent BY CONSTRUCTION for matched inputs (K=1) — not by two
        code paths kept in sync. Keep it that way:
          * imagine_eval   == _imagine(K=1).
          * imagine_shared == _imagine(K>=1) (fan out K action variants from one encode).
          * physics_proprio_chained (training loss) GENERALIZES the physics here to p_tf scheduled sampling;
                            THIS helper is its p_tf<=0 branch (chain on own prediction, no true-prev coin).
        Do NOT re-inline the physics per call site (that, plus imagine_shared skipping the rel-position anchor,
        is exactly how it silently rolled the black-box / wrong-frame dynamics before this was unified). If you
        change the physics, change it HERE; if you touch _imagine, keep the imagine_eval-vs-imagine_shared(K=1)
        obs parity (parity test)."""
        acts_raw = norm.denorm_act(actions).float()
        prev = prev0_abs.float()
        traj = []
        for t in range(horizon):
            token = bag[:, t, self._proprio_off, :].float()   # bag is bf16 (autocast); residual_head is fp32 and
            #                                                    this block runs OUTSIDE autocast (physics is fp32,
            #                                                    issue #14), so match the Linear's dtype explicitly.
            cur = self.dynamics_prior(prev, acts_raw[:, P - 1 + t]) + self.residual_head(token).float()
            traj.append(cur)
            prev = cur
        return norm.norm_obs(torch.stack(traj, dim=1))

    @torch.no_grad()
    def _imagine(self, ctx_obs: dict, actions: Tensor, horizon: int, K: int = 1, heads=None,
                 use_cache: bool | None = None, decode_chunk: int | None = None, norm=None,
                 return_bag: bool = False) -> dict[str, Tensor]:
        """THE single inference rollout. `imagine_eval` (K=1) and `imagine_shared` (K>=1, encode-once fan-out)
        are thin specializations that just call this — there is ONE rollout path, so a change here reaches both.

        Steps: encode the context ONCE, threading the relative-position anchor (rel_anchor) so a rel-position
        model gets the small within-window frame it was TRAINED on — NOT threading this was the bug that made
        imagine_shared roll absolute positions through a relative encoder. Then optionally fan out to B*K action
        variants (K>1: MPPI, encode-once shared across candidates), roll the latent spine (_rollout_from, p_tf=0,
        no teacher forcing), decode `heads` (optionally chunked over time so long-horizon image decode doesn't
        OOM), and — owm hook + norm given — replace the proprio decode with the physics rollout
        (_physics_proprio_rollout; fp32, issue #14). Without dynamics_prior or without norm the proprio stays the
        black-box decode. actions NORMALIZED (B*K, P-1+horizon, act_dim); action for step t is index P-1+t.
        return_bag also returns the rolled latent bag under `_bag` (a language reward scores it decode-free)."""
        uc = self.use_kv_cache if use_cache is None else use_cache
        anchor = self.rel_anchor(ctx_obs) if self._rel_on() else None
        with torch.autocast(device_type=actions.device.type, dtype=torch.bfloat16, enabled=actions.is_cuda):
            bags = self.encode_state(ctx_obs, anchor)                            # (B,P,n_state,d) — ONE encode
            if K != 1:
                bags = bags.repeat_interleave(K, dim=0)                          # (B*K,P,...) fan out variants
            bag = self._rollout_from(list(bags.unbind(dim=1)), actions, horizon, 0.0, None, 0, use_cache=uc)
            anc = anchor if (anchor is None or K == 1) else anchor.repeat_interleave(K, dim=0)  # match B*K for decode
            if decode_chunk and bag.ndim >= 2 and bag.shape[1] > decode_chunk:   # chunk decode over the time axis
                parts = [self.to_obs(bag[:, s:s + decode_chunk], heads=heads, anchor=anc)
                         for s in range(0, bag.shape[1], decode_chunk)]
                out = {k: torch.cat([p[k] for p in parts], dim=1) for k in parts[0]}
            else:
                out = self.to_obs(bag, heads=heads, anchor=anc)
        out = {k: v.float() for k, v in out.items()}                            # physics runs fp32, OUTSIDE autocast
        if self.dynamics_prior is not None and norm is not None:
            P = ctx_obs["proprio"].shape[1]
            prev0 = norm.denorm_obs(ctx_obs["proprio"][:, -1])                  # (B,obs_dim) last context, ABSOLUTE
            if K != 1:
                prev0 = prev0.repeat_interleave(K, dim=0)                       # per-context -> B*K
            out["proprio"] = self._physics_proprio_rollout(bag, prev0, actions, horizon, P, norm)
        if return_bag:
            out["_bag"] = bag.float()
        return out

    def imagine_eval(self, ctx_obs: dict, actions: Tensor, horizon: int, heads=None,
                     use_cache: bool | None = None, decode_chunk: int | None = None, norm=None,
                     return_bag: bool = False) -> dict[str, Tensor]:
        """Eval/single-sequence rollout = `_imagine` with K=1. `heads` limits which modalities decode (e.g.
        ['proprio'] for cheap long-horizon rollouts). `decode_chunk` bounds the image decoder's peak memory over
        long horizons. `norm` activates the physics-anchored proprio rollout (owm)."""
        return self._imagine(ctx_obs, actions, horizon, K=1, heads=heads,
                             use_cache=use_cache, decode_chunk=decode_chunk, norm=norm,
                             return_bag=return_bag)     # `_bag` = the rolled latents, for latent-space curves

    def imagine_shared(self, ctx_obs: dict, actions: Tensor, horizon: int, K: int, heads=None,
                       return_bag: bool = False, norm=None) -> dict[str, Tensor]:
        """MPPI rollout = `_imagine` with encode-once K fan-out: encode B contexts ONCE (the expensive image
        encode), expand to B*K, roll K action variants per context. ctx_obs (B,P,*); actions (B*K,P-1+H,act_dim).
        return_bag also returns the rolled latent bag under `_bag`. `norm` MUST be passed for MPPI, else it scores
        the mislearned black-box dynamics instead of the physics rollout."""
        return self._imagine(ctx_obs, actions, horizon, K=K, heads=heads,
                             use_cache=self.use_kv_cache, norm=norm, return_bag=return_bag)

    @torch.no_grad()
    def kvcache_report(self, ctx_obs: dict, actions: Tensor, horizon: int, heads=None) -> dict[str, float]:
        """One-time A/B: run the SAME rollout with and without the KV-cache, CUDA-timed (1 warmup + 1 timed
        each), to log the realized wall-clock speedup to kvcache/. Cheap — a few extra rollouts, called once
        per run. Also reports the max abs latent divergence as a correctness sanity signal."""
        import time
        dev = actions.device
        sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)
        # The divergence number below is a CORRECTNESS signal (cached rollout == uncached rollout), which is
        # only defined for a deterministic readout. stochastic_eval defaults TRUE since 2026-08-10, so without
        # this pin the two rollouts draw different eps and the metric reads ~4 on every run -- indistinguishable
        # from a genuinely broken cache. Restored in the finally below.
        _se = getattr(self, "stochastic_eval", False)
        self.stochastic_eval = False

        def timed(uc):
            self.imagine_eval(ctx_obs, actions, horizon, heads=heads, use_cache=uc)     # warmup (compile/alloc)
            sync(); t0 = time.perf_counter()
            self.imagine_eval(ctx_obs, actions, horizon, heads=heads, use_cache=uc)
            sync(); return time.perf_counter() - t0

        try:
            t_cached, t_uncached = timed(True), timed(False)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=actions.is_cuda):
                bc = self._rollout(ctx_obs, actions, horizon, 0.0, None, 0, use_cache=True)
                bu = self._rollout(ctx_obs, actions, horizon, 0.0, None, 0, use_cache=False)
        finally:
            self.stochastic_eval = _se
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
                 grad_checkpoint: bool = False, compile_rollout: bool = False, latent_norm: bool = True, action_fourier_freqs: int = 0, action_squash: str = "none",
                 pred_hidden: int = 0, lambda_pred_latent: float = 1.0,
                 collapse: CollapseStrategy | None = None, lambda_reg: float = 1.0, expander_dim: int = 256, **kw):
        super().__init__(specs, d=d, depth=depth, heads=heads, window=window, mlp_ratio=mlp_ratio,
                         rope_theta=rope_theta, action_dim=action_dim, grad_checkpoint=grad_checkpoint,
                         compile_rollout=compile_rollout, latent_norm=latent_norm,
                         action_fourier_freqs=action_fourier_freqs, action_squash=action_squash, **kw)
        h = pred_hidden or d
        self.predictor = _mlp(d, d, h)                          # per-token residual predictor
        self.lambda_pred_latent = lambda_pred_latent
        self.lambda_reg = lambda_reg
        self.lambda_pred_obs = 1.0
        # collapse strategy = the SAME abstraction as vector LSAR (recon/ema/naked/vicreg/sigreg). It's a
        # POLICY object here: MM keeps its own multi-encoder EMA/encode mechanics but reads the strategy's
        # flags + reg_loss + pred_metric. Reconstruction (obs grounds the encoder) is the default.
        self.collapse = collapse or Reconstruction()
        if self.collapse.has_reg and self.latent_norm_type != "none":
            # RAISE, do not silently force (user, 2026-08-11). vicreg/sigreg regularize the latent's variance
            # and covariance directly, and normalizing the carried bag fights them -- LN pins per-token variance
            # to 1, which is exactly the statistic the reg term controls. Silently overriding meant the run
            # trained under 'none' while its config said something else.
            raise ValueError(
                f"collapse strategy {type(self.collapse).__name__} regularizes latent variance/covariance "
                f"(has_reg=True), which is incompatible with model.latent_norm='{self.latent_norm_type}': "
                f"normalizing the bag pins the very statistic the regularizer controls. Set "
                f"model.latent_norm=none EXPLICITLY for vicreg/sigreg, or use a collapse strategy with no "
                f"variance term (reconstruction/ema/naked)."
            )
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
                 latent_norm: str | bool = "affine",   # was `bool = True` -> silently gave LAYERNORM on a direct
                 #                                       construct, contradicting the LOCKED affine default
                 action_fourier_freqs: int = 0, action_squash: str = "none",
                 sampling_steps: int = 6, shortcut: bool = False, predict: str = "residual",
                 stochastic_eval: bool = True, time_sampling: str = "uniform", flow_hidden: int = 0,
                 flow_arch: str = "mlp", flow_arch_depth: int = 2, flow_arch_heads: int = 4,
                 concat_action_embedding: bool = True,
                 lambda_flow: float = 1.0, lambda_consistency: float = 1.0,
                 df_scale: float = 0.0, df_granularity: str = "timestep",
                 action_head_enabled: bool = False, action_head_weight: float = 1.0,
                 action_head_shortcut: bool = True, action_head_detach_gradient: bool = False,
                 dynamics_detach_encoder: bool = False, p_tf_dynamics: float | None = 1.0, **kw):
        super().__init__(specs, d=d, depth=depth, heads=heads, window=window, mlp_ratio=mlp_ratio,
                         rope_theta=rope_theta, action_dim=action_dim, grad_checkpoint=grad_checkpoint,
                         compile_rollout=compile_rollout, latent_norm=latent_norm,
                         action_fourier_freqs=action_fourier_freqs, action_squash=action_squash, **kw)
        assert predict in ("residual", "absolute")
        self.predict_residual = predict == "residual"
        self.sampling_steps = int(sampling_steps)
        self.stochastic_eval = bool(stochastic_eval)
        self.time_sampling = time_sampling
        self.lambda_flow, self.lambda_consistency = lambda_flow, lambda_consistency
        # flow_arch="transformer" makes the denoiser token-mixing -> a JOINT over the bag instead of a product
        # of per-token marginals. n_state (NOT n_input) is the token axis predict_next denoises.
        # concat_action_embedding: give the denoiser the action token's own backbone output on a dedicated
        # channel (see _cond). Doubles the conditioning width, so the FlowField's h_dim doubles with it.
        self.concat_action_embedding = bool(concat_action_embedding)
        # The action slot's backbone output is ALWAYS used now (user, 2026-08-12) -- never sliced off. Not a
        # config knob: computing it and discarding it was the one configuration with no precedent.
        self.use_action_slot = True
        _hd = d * (1 + int(self.use_action_slot) + int(self.concat_action_embedding))
        self.flow = FlowField(d, h_dim=_hd, hidden=(flow_hidden or d), cond="concat", shortcut=shortcut,
                              arch=flow_arch, n_tokens=self.n_state, depth=flow_arch_depth, heads=flow_arch_heads)
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
        # p_tf_dynamics: the probability that the DYNAMICS loss conditions on the TRUTH (clean encoder
        # latents) rather than on the rollout's own feeds. Exactly the same quantity p_tf denotes, for a
        # different consumer -- p_tf is the ROLLOUT's (and so the decode loss's), this is the dynamics loss's.
        #   1.0  = always clean. The historical behaviour, and the DEFECT: p_tf ramped 1 -> 0, the decode loss
        #          duly became autoregressive, and the transition function stayed teacher-forced for the whole
        #          run, never once asked to step from a latent it produced itself.
        #   None = follow p_tf (so 0 after warmup). Measured: latent_cos@+32 2.1x better than the control for
        #          five epochs, then a rollout-Jacobian blow-up -- grad/norm/flow 0.42 -> 1.3e7, ae_floor
        #          19.1 -> 11.4, latent_cos NEGATIVE. Full substitution is too strong.
        #   0<q<1 = the middle ground the dose-response curve points at: diffusion forcing corrupts the same
        #          context isotropically at 0.1 strength and is both stable AND the best result measured, while
        #          full substitution collapses. This is a STRENGTH knob, which p_tf cannot supply -- p_tf is
        #          shared with the decode loss, and that loss NEEDS 0 (it is the only autoregressive gradient
        #          in the model). Replaces the old `dynamics_follows_p_tf` bool: false == 1.0, true == None.
        # design/flow.md, record section 18.
        self.p_tf_dynamics = None if p_tf_dynamics is None else float(p_tf_dynamics)
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
        out = prev_bag + out if self.predict_residual else out
        return _ln(out) if self.latent_norm else out

    def loss_terms(self, pred_bag, future_obs, obs, p_tf, act_seq=None, pre_z: Tensor | None = None,
                   anchor: Tensor | None = None, feeds: Tensor | None = None):
        """Teacher-forced rectified-flow loss over the bag (mirrors models/diffusion.py)."""
        assert act_seq is not None
        z = self.encode_state(obs, anchor) if pre_z is None else pre_z  # (B,L,n_state,d); pre_z = shared-encode (already relativized)
        L = z.shape[1]
        s = z[:, :-1]                                           # contexts (B,L-1,n_state,d)
        # q = probability of conditioning on TRUTH. None -> track the rollout's own p_tf.
        q_dyn = p_tf if self.p_tf_dynamics is None else self.p_tf_dynamics
        if q_dyn < 1.0 and feeds is not None:
            # Condition on what each rollout step actually STOOD ON, so the p_tf schedule reaches this loss.
            # feeds[k] fed the step predicting future frame k+1, i.e. parallel-pass position P+k -> alignment
            # is by CONSTRUCTION, not shape inference. feeds[F-1] fed no step and is dropped. The first L-F
            # positions are the given context frames the model never predicted, so they stay clean.
            # torch.cat, NEVER in-place: `s` is a VIEW of `z`, and writing through it would corrupt `z` and
            # with it the target below. See design/flow.md.
            F_ = feeds.shape[1]
            if L - F_ >= 1 and F_ >= 2:
                tail = feeds[:, :F_ - 1]
                if q_dyn > 0.0:
                    # per (sample, position) Bernoulli, mirroring how p_tf mixes inside the rollout
                    # (_rollout_from). Only drawn when 0 < q < 1, so q == 1.0 and q == 0.0 leave the RNG
                    # stream untouched and stay bit-identical to not having this feature.
                    keep_true = torch.rand(tail.shape[:2] + (1, 1), device=tail.device,
                                           dtype=tail.dtype) < q_dyn
                    tail = torch.where(keep_true, z[:, L - F_:-1], tail)
                s = torch.cat([z[:, :L - F_], tail], dim=1)
        if self.dynamics_detach_encoder:                        # stop-grad: dynamics loss won't reshape the encoder
            s = s.detach()                                      #   (encoder trained only by decode/recon; anti-collapse)
        s_ref = s      # the contexts the target is measured FROM. Snapshotted BEFORE the DF block below
        #                overwrites `s` with its noised version -- otherwise the residual target would absorb
        #                the diffusion-forcing noise, silently redefining DF. Aliases z[:, :-1] when off.
        levels = None
        if self.df_scale > 0.0 and self.training:               # diffusion forcing: noise the context + tell the backbone
            levels = torch.rand(s.shape[:-2] + (1,), device=s.device, dtype=s.dtype) * self.df_scale  # (B,L-1,1)
            eps = torch.randn_like(s)
            lv = levels.unsqueeze(-2)                            # (B,L-1,1,1) broadcast over n_state,d
            s = (1.0 - lv) * s + lv * eps                        # noised context
            if self.latent_norm:
                s = _ln(s)                                       # renormalized back onto the sphere
        h = self.backbone(self._to_input(s, act_seq[:, :L - 1], levels=levels))
        h_state = self._cond(h, act_seq[:, :L - 1])            # (B,L-1,n_state,d) or 2d if concat_action
        # The residual target is defined RELATIVE TO THE CONTEXT it will be added to (`predict_next` does
        # `next = prev + d`), so it must follow `s_ref`. With dynamics_follows_p_tf off this is byte-for-byte
        # the previous expression; on, it becomes `z[t+1] - what_the_step_stood_on` -- the correction back
        # onto the true trajectory. Absolute mode never referenced the context and is unchanged.
        target = (z[:, 1:] - s_ref).detach() if self.predict_residual else z[:, 1:].detach()  # off CLEAN z
        l_flow, l_cons = self.flow.loss(h_state, target, time_sampling=self.time_sampling)
        raw, w = {"dynamics/latent": l_flow}, {"dynamics/latent": self.lambda_flow}   # the transition function
        if l_cons is not None:
            raw["dynamics/latent_shortcut"], w["dynamics/latent_shortcut"] = l_cons, self.lambda_consistency
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
            raw["action/flow"], w["action/flow"] = l_aflow, self.action_head_weight
            if l_acons is not None:
                raw["action/shortcut"], w["action/shortcut"] = l_acons, self.action_head_weight
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


class MultiModalDistribution(MultiModalSequenceModel):
    """Parametric-distribution prediction over the next latent (design/models/probabilistic_heads.md). The
    spine's per-step context parameterizes an EXPLICIT distribution (categorical / diagonal Gaussian) via a
    swappable DistributionHead; the prior is sampled and carried forward (latent-space AR). When the head
    needs a posterior (categorical, Gaussian-KL), encode_state fuses the frame's observation tokens into a
    posterior, samples it, and embeds the sample back to a d-dim bag -- so every decoder / roundtrip / rollout
    path is inherited UNCHANGED (the bag stays (B,T,n_state,d)). Gaussian-NLL (Ward-2026) has no posterior:
    the continuous encoder features are carried straight through and the distribution appears only at the
    prior. Mutually exclusive with the flow head."""

    def __init__(self, specs, *, d, depth, heads, window, mlp_ratio, rope_theta, action_dim,
                 grad_checkpoint: bool = False, compile_rollout: bool = False,
                 latent_norm: str | bool = "affine", action_fourier_freqs: int = 0, action_squash: str = "none",
                 head=None, stochastic_eval: bool = True, **kw):
        super().__init__(specs, d=d, depth=depth, heads=heads, window=window, mlp_ratio=mlp_ratio,
                         rope_theta=rope_theta, action_dim=action_dim, grad_checkpoint=grad_checkpoint,
                         compile_rollout=compile_rollout, latent_norm=latent_norm,
                         action_fourier_freqs=action_fourier_freqs, action_squash=action_squash, **kw)
        assert head is not None, "MultiModalDistribution requires a DistributionHead (build via make_dist_head)"
        self.dist_head = head
        self.dist_head.build(d, self.n_state)              # creates the head's nets; params serialize dist_head.*
        # committed vs stochastic prediction at eval (the MultiModalFlow convention kvcache_report pins).
        self.stochastic_eval = bool(stochastic_eval)
        self.pred_obs_in_loss = True                       # decode grounds the encoder (as Flow); never a detached probe
        self.lambda_pred_obs = 1.0

    def _features(self, obs, anchor=None) -> Tensor:
        """The CONTINUOUS per-modality encoder features e_t (the base encode_state), BEFORE the posterior
        bottleneck. loss_terms needs these for the deterministic target / posterior input, so it must bypass
        the sampling override below."""
        return MultiModalSequenceModel.encode_state(self, obs, anchor)

    def encode_state(self, obs, anchor=None) -> Tensor:
        e = self._features(obs, anchor)
        if not self.dist_head.needs_posterior:
            return e                                       # Gaussian-NLL: deterministic passthrough (e_t IS z_t)
        det = (not self.training) and (not self.stochastic_eval)
        bag = self.dist_head.sample(self.dist_head.posterior(e), deterministic=det)
        return _ln(bag) if self.latent_norm else bag

    def predict_next(self, h_state: Tensor, prev_bag: Tensor) -> Tensor:
        det = (not self.training) and (not self.stochastic_eval)
        bag = self.dist_head.sample(self.dist_head.prior(h_state), deterministic=det)   # absolute, not a residual
        return _ln(bag) if self.latent_norm else bag

    def loss_terms(self, pred_bag, future_obs, obs, p_tf, act_seq=None, pre_z: Tensor | None = None,
                   anchor: Tensor | None = None):
        """One teacher-forced parallel pass: encode the full window to continuous features, run the backbone
        on the (posterior-sampled or deterministic) context, and score the prior that predicts frame t+1
        against the posterior/target at t+1. Fully teacher-forced (Dreamer's dynamics KL is on the observed
        sequence); the p_tf ramp affects only the rolled decode losses (recon_losses), as for Flow."""
        assert act_seq is not None
        e = self._features(obs, anchor)                    # (B,L,n_state,d) continuous encoder features
        L = e.shape[1]
        if self.dist_head.needs_posterior:
            post_all = self.dist_head.posterior(e)         # per-frame posterior params (all L frames)
            z = self.dist_head.sample(post_all, deterministic=False)
            z = _ln(z) if self.latent_norm else z
        else:
            post_all, z = None, e
        h = self.backbone(self._to_input(z[:, :L - 1], act_seq[:, :L - 1]))
        prior = self.dist_head.prior(self._cond(h, act_seq[:, :L - 1]))                # predicts frames [1:]
        if self.dist_head.needs_posterior:
            post_tgt = {k: v[:, 1:] for k, v in post_all.items()}                      # posterior of true-next
            return self.dist_head.losses(post_tgt, prior, target=None)
        return self.dist_head.losses(None, prior, target=e[:, 1:])                     # NLL target = e_{t+1}

    @torch.no_grad()
    def collapse_diagnostics(self, obs) -> dict:
        from .collapse import latent_diagnostics
        return latent_diagnostics(self.encode_state(obs).reshape(-1, self.d).float())
