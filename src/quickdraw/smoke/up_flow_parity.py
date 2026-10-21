"""Parity/superset smoke for the up-backed decoders (design/up_flow_decoder.md).

  T1  up-mse bit-identical to the frozen golden (proves the package refactor changed nothing).
  T2  superset: UpFlowDecoder with zero-init analysis == TokenGridDecoder (shared backend), any temb.
  T3  warm-start: loading an up-mse backend into UpFlowDecoder reproduces T2.
  T4  learnable: with down inject/xattn ON, step-0 output is still == up-mse, but those params get gradient.
  T6  shape/run: one forward+backward at the block-stack geometry.
(T5 dispatch lives in the model build path; see modalities.py.)

Run: docker compose exec -T app uv run --no-sync python -m quickdraw.smoke.up_flow_parity
"""

import os

import torch

torch.set_num_threads(1)   # bit-exact CPU reductions across processes (matches the golden's generation)

from ..models.decoders import TokenGridDecoder, UpFlowDecoder
from ..models.vision import VisionAEConfig

GOLDEN = os.path.join(os.path.dirname(__file__), "up_flow_parity_golden.pt")


def _fail(msg):
    print("FAIL:", msg); raise SystemExit(1)


def main():
    G = torch.load(GOLDEN, weights_only=False)
    cfg_kw, base, cond, gold = G["cfg"], G["base"], G["cond"], G["out"]
    M = cond.shape[0]
    H, W = cfg_kw["img_size"]
    mk_cfg = lambda: VisionAEConfig(**cfg_kw)

    # T1 -----------------------------------------------------------------
    torch.manual_seed(0)
    up = TokenGridDecoder(mk_cfg(), base=base, chunk=0, inject=False, xattn_max_res=0, out_act="none").eval()
    with torch.no_grad():
        o = up.velocity(cond=cond)
    if not torch.equal(o, gold):
        _fail(f"T1 up-mse != golden (max|diff| {float((o-gold).abs().max())})")
    print("T1 up-mse bit-identical to golden           OK")

    # T2 superset --------------------------------------------------------
    fl = UpFlowDecoder(mk_cfg(), base=base, chunk=0, down_inject=False, down_xattn_max_res=0).eval()
    fl.back.load_state_dict(up.back.state_dict())                    # share the backend
    x = torch.randn(M, H, W, 3)
    temb = torch.randn(M, 32)
    with torch.no_grad():
        of = fl.velocity(x, temb, cond)
    if not torch.equal(of, o):
        _fail(f"T2 flow(zero-init) != up-mse (max|diff| {float((of-o).abs().max())})")
    print("T2 flow superset == up-mse (any temb)       OK")

    # T3 warm-start ------------------------------------------------------
    fl2 = UpFlowDecoder(mk_cfg(), base=base, chunk=0).eval()
    sd = fl2.state_dict(); sd.update({f"back.{k}": v for k, v in up.back.state_dict().items()})
    fl2.load_state_dict(sd)
    with torch.no_grad():
        of2 = fl2.velocity(x, temb, cond)
    if not torch.equal(of2, o):
        _fail(f"T3 warm-start != up-mse (max|diff| {float((of2-o).abs().max())})")
    print("T3 warm-start from up-mse == up-mse         OK")

    # T4 learnable-but-parity-safe --------------------------------------
    flx = UpFlowDecoder(mk_cfg(), base=base, chunk=0, down_inject=True, down_xattn_max_res=max(H, W))
    flx.back.load_state_dict(up.back.state_dict()); flx.eval()
    with torch.no_grad():
        ox = flx.velocity(x, temb, cond)
    if not torch.equal(ox, o):
        _fail(f"T4 injected flow != up-mse at init (max|diff| {float((ox-o).abs().max())})")
    flx.train()
    flx.velocity(x, temb, cond).sum().backward()
    gnorm = lambda mod: sum(float(p.grad.abs().sum()) for p in mod.parameters() if p.grad is not None)
    gi = gnorm(flx.ana.inject_d) + gnorm(flx.ana.xattn_d) + gnorm(flx.ana.to_bott) + gnorm(flx.ana.skip_proj)
    if not (gi > 0):
        _fail("T4 down inject/xattn/skip received NO gradient (dead, not learnable)")
    print(f"T4 injection parity-safe + learnable        OK  (grad {gi:.3g})")

    # T6 shape / run -----------------------------------------------------
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    flm = UpFlowDecoder(mk_cfg(), base=base, chunk=0, down_inject=True, down_xattn_max_res=max(H, W)).to(dev).train()
    xr = torch.randn(M, H, W, 3, device=dev); tr = torch.randn(M, 32, device=dev)
    out = flm.velocity(xr, tr, cond.to(dev))
    out.sum().backward()
    if tuple(out.shape) != (M, H, W, 3):
        _fail(f"T6 bad output shape {tuple(out.shape)}")
    print(f"T6 flow fwd+bwd on {dev} -> {tuple(out.shape)}      OK")

    print("\nALL PARITY TESTS PASSED")


if __name__ == "__main__":
    main()
