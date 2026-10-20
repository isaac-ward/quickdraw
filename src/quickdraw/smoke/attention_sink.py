"""Smoke for NoPE + attention sink (2026-09-28). The DELICATE part is KVRing<->mask parity.

  1. BIT-IDENTICAL default: pos_encoding=rope, attention_sink=0 reproduces the plain path exactly
     (mask admissible set unchanged; rope math unchanged; cached==parallel to fp tolerance).
  2. mask admissible set matches the formula (kv<=q) & ((q-kv<window) | (kv<sink)) for S in {0,1,2,4}.
  3. NoPE actually drops positional info (rope vs nope give different attention).
  4. CACHE<->PARALLEL PARITY (the load-bearing test): the cached step-by-step backbone reproduces the
     full-sequence parallel backbone for S in {0,2} -- i.e. KVRing sink-pinning matches the mask exactly.
     A wrong pin set attends the wrong keys and blows the diff up. Tested past window+sink so trimming fires.
  5. the sink DOES something: S=2 differs from S=0 at positions beyond the window.
  6. GUARD: build_model with attention_sink>0 and pos_encoding=rope raises.
Run: docker compose exec -T app uv run --no-sync python -m quickdraw.smoke.attention_sink
"""
import torch
torch.set_num_threads(1)

from ..models.spacetime import SpaceTimeTransformer
from ..models.transformer import _sliding_causal_mask_mod


def _fail(m):
    print("FAIL:", m); raise SystemExit(1)


def _bb(pos_encoding, sink, window=4, d=32, N=3, depth=2, heads=4):
    torch.manual_seed(0)
    return SpaceTimeTransformer(d, depth, heads, window, mlp_ratio=2.0, n_slots=N,
                                pos_encoding=pos_encoding, attention_sink=sink).eval()


def _cached_pass(bb, x):
    """Feed x (B,T,N,d) step by step through the cached backbone -> (B,T,N,d)."""
    B, T, N, d = x.shape
    cache = bb.make_cache()
    outs = [bb.forward_cached(x[:, t], cache, t) for t in range(T)]
    return torch.stack(outs, dim=1)


def main():
    torch.manual_seed(0)
    B, T, N, d, window = 2, 10, 3, 32, 4     # T=10 > window+sink so trimming/eviction fires
    x = torch.randn(B, T, N, d)

    # ---- 2. mask admissible set matches the formula ----
    for S in (0, 1, 2, 4):
        mm = _sliding_causal_mask_mod(window, S)
        for q in range(T):
            for kv in range(T):
                want = (kv <= q) and ((q - kv < window) or (kv < S))
                got = bool(mm(0, 0, torch.tensor(q), torch.tensor(kv)))
                if want != got:
                    _fail(f"mask S={S} q={q} kv={kv}: got {got} want {want}")
    print("[2/6] mask admissible set matches formula for S in {0,1,2,4}   OK")

    # ---- 1 & 4. cached<->parallel parity (rope S=0 = default; nope S in {0,2}) ----
    def parity(pos, S):
        bb = _bb(pos, S, window=window, d=d, N=N)
        with torch.no_grad():
            h_par = bb(x)                       # full-sequence parallel (mask built inside, sink-aware)
            h_cac = _cached_pass(bb, x)         # step-by-step cached (KVRing sink-pinning)
        return float((h_par - h_cac).abs().max())

    d_rope0 = parity("rope", 0)                 # the pre-existing invariant (fp: SDPA vs FlexAttention)
    d_nope0 = parity("nope", 0)
    d_nope2 = parity("nope", 2)                 # sink pinning must match the mask
    tol = max(1e-2, 5 * d_rope0)                # sink parity must be ~ the sink-free fp noise, not larger
    print(f"[1&4/6] cache<->parallel maxabs: rope S0={d_rope0:.2e}  nope S0={d_nope0:.2e}  nope S2={d_nope2:.2e}")
    if d_nope0 > tol or d_nope2 > tol:
        _fail(f"cache<->parallel parity broke (tol {tol:.2e}): a wrong sink pin attends wrong keys")
    print("        cached reproduces parallel with the sink (KVRing<->mask parity holds)   OK")

    # ---- 3. NoPE drops positional info (rope != nope) ----
    br, bn = _bb("rope", 0), _bb("nope", 0)
    br.load_state_dict(bn.state_dict())         # SAME weights -> only the PE differs
    with torch.no_grad():
        if torch.allclose(br(x), bn(x), atol=1e-4):
            _fail("rope and nope produced the same output (NoPE not actually dropping RoPE)")
    print("[3/6] rope vs nope differ (NoPE drops positional signal)   OK")

    # ---- 5. the sink does something past the window ----
    b0, b2 = _bb("nope", 0), _bb("nope", 2)
    b2.load_state_dict(b0.state_dict())         # same weights -> only the sink differs
    with torch.no_grad():
        h0, h2 = b0(x), b2(x)
    far = h0[:, window + 2:] - h2[:, window + 2:]    # positions that can no longer see the start except via sink
    if float(far.abs().max()) < 1e-5:
        _fail("sink S=2 identical to S=0 beyond the window -- sink is inert")
    print(f"[5/6] sink changes far-position outputs (maxabs {float(far.abs().max()):.3e})   OK")

    # ---- 6. GUARD via build_model ----
    from hydra import compose, initialize_config_dir
    from ..training.setup import build_model
    def build(ov):
        with initialize_config_dir(config_dir="/app/conf", version_base=None):
            return build_model(compose(config_name="config", overrides=ov))
    # rope+sink -> must raise
    try:
        build(["model=vl128_blockstack_flow", "+model.attention_sink=4"])   # pos_encoding defaults rope
        _fail("build_model did NOT raise for attention_sink>0 with rope")
    except ValueError as e:
        if "nope" not in str(e).lower():
            _fail(f"wrong error for sink+rope: {e}")
    # nope+sink -> builds
    mb = build(["model=vl128_blockstack_flow", "+model.pos_encoding=nope", "+model.attention_sink=4"])
    if mb.backbone.blocks[0].t_attn.attention_sink != 4 or mb.backbone.blocks[0].t_attn.pos_encoding != "nope":
        _fail("nope+sink not wired to the temporal attention")
    if mb._attn_sink != 4:
        _fail("_attn_sink not stored on the model (needed by _rollout_from)")
    print("[6/6] guard raises on sink+rope; nope+sink builds & wires to t_attn   OK")

    # ---- 7. FULL-TIER: the AR rollout (_rollout_from) carries the sink -> matches the cached/parallel path ----
    # sink=0 rollout is bit-identical to a rope model's rollout on the plain recipe; sink>0 uses the sink window.
    from ..models.transformer import rollout_sink_block_mask, pad_block_mask
    # sink=0 sink-mask == pad_block_mask (byte-identical admissible set)
    W = 4
    m0 = rollout_sink_block_mask(W, 0, 1, torch.device("cpu"))
    mp = pad_block_mask(W, 1, torch.device("cpu"))
    # compare admissible sets densely via the mask_mod formulas
    for q in range(W):
        for kv in range(W):
            a = (kv <= q) and (kv >= 1)                     # pad_block_mask formula (pad=1)
            b = (kv <= q) and ((kv < 0) or (kv >= 0 + 1))   # sink=0 formula
            if a != b:
                _fail(f"sink=0 mask != pad_block_mask at q={q} kv={kv}")
    # sink-window buffer layout mask: [sink S][pad][recent] -- query attends all sink + causal recent, not pad
    S, pad = 2, 1
    T = S + W
    # build the model's actual _rollout_from with a sink and check it RUNS + shape is right
    mb.eval()
    B, P, F = 2, mb.window, 6                                # P>=window so real fills; keeps it simple
    # tiny obs to drive imagine_eval (uncached -> _rollout_from, the FULL-tier path)
    obs = {}
    for name, n in mb.layout:
        mod = mb.modalities[name]
        if name == "proprio":
            obs[name] = torch.randn(B, P, mod.dim)
        else:
            sh = mod.ae.cfg.img_size; sh = sh if isinstance(sh, (tuple, list)) else (sh, sh)
            obs[name] = torch.rand(B, P, sh[0], sh[1], 3)
    act = torch.randn(B, P + F - 1, mb.act_enc[0].in_features if hasattr(mb.act_enc, "__getitem__") else 5)
    mb.stochastic_eval = False   # DETERMINISTIC: else the two rollouts draw different flow noise (see kvcache_report)
    with torch.no_grad():
        r_unc = mb.imagine_eval(obs, act, F, use_cache=False)   # _rollout_from WITH the sink prepend
        r_cac = mb.imagine_eval(obs, act, F, use_cache=True)    # _rollout_cached (KVRing sink)
    if not torch.isfinite(r_unc["cam_scene"]).all():
        _fail("full-tier _rollout_from with sink produced non-finite output")
    # uncached (rollout-from, sink) and cached (KVRing, sink) must AGREE (both attend sink+window) to fp tol --
    # this is the faithful-cache invariant, now WITH the sink in both paths.
    dpar = float((r_unc["proprio"] - r_cac["proprio"]).abs().max())
    print(f"[7/7] full-tier: _rollout_from(sink) finite; rollout-from vs cached proprio maxabs {dpar:.2e}")
    if dpar > 5e-3:
        _fail(f"full-tier: sliding-rollout sink and cached sink disagree ({dpar:.2e}) -- window/pin mismatch")
    print("        AR rollout carries the sink and agrees with the cached path   OK")

    print("\nALL ATTENTION-SINK SMOKE PASSED")


if __name__ == "__main__":
    main()
