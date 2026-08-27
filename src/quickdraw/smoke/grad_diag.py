"""Smoke: cudnn.benchmark is set, and sampled gradient diagnostics preserve the non-finite guard.

WHY IT EXISTS. `configure_gradient_clipping` does two jobs: it GATES the optimizer step (skip on any
non-finite grad — a single inf otherwise makes norm-clipping NaN every gradient and poisons AdamW forever)
and it LOGS diagnostics. The second half is ~5 passes over all grad tensors plus 2 forced GPU syncs, every
step, for numbers read once per epoch, so it is now sampled every `grad_diag_every` steps. The properties
that must NOT regress:
  1. grad/norm_preclip is logged EVERY step (headline early warning keeps full resolution).
  2. grad/nonfinite_skipped is logged EVERY step.
  3. A non-finite gradient computes the FULL diagnostic regardless of sampling phase, AND still zeroes the
     grads so the step is a no-op.
  4. The per-module norms are read PRE-clip (clipping turns one inf into all-NaN).
Run: uv run python -m quickdraw.smoke.grad_diag
"""
import torch
import torch.nn as nn

OK = [0, 0]
def check(name, cond, extra=""):
    OK[1] += 1; OK[0] += bool(cond)
    print(f"[{'OK' if cond else 'FAIL'}] {name}" + (f" — {extra}" if extra else ""))

check("cudnn.benchmark is set by the training entrypoint",
      "torch.backends.cudnn.benchmark = True" in open("/app/src/quickdraw/train_world_model.py").read())

from quickdraw.training.lit import LitWorldModel

class _M(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.backbone = nn.Linear(4, 4)
        self.model.flow = nn.Linear(4, 4)

class Probe(LitWorldModel):
    def __init__(self, every):
        nn.Module.__init__(self)
        self.model = _M().model
        self.grad_diag_every = every
        self._logged = []
        self._gs = 0
        self._clipped = 0
    @property
    def global_step(self): return self._gs
    def log(self, name, value, **kw): self._logged.append(name)
    def clip_gradients(self, *a, **kw): self._clipped += 1

def run(every, step, make_inf=False):
    p = Probe(every); p._gs = step
    for prm in p.parameters():
        prm.grad = torch.ones_like(prm)
    if make_inf:
        next(iter(p.parameters())).grad[0] = float("inf")
    p.configure_gradient_clipping(None, gradient_clip_val=1.0)
    return p

# 1/2. per-step keys always present, on a NON-sampled step
p = run(every=25, step=7)
check("grad/norm_preclip logged on a NON-sampled step", "grad/norm_preclip" in p._logged)
check("grad/nonfinite_skipped logged on a NON-sampled step", "grad/nonfinite_skipped" in p._logged)
check("per-module norms SKIPPED on a non-sampled step",
      not any(k.startswith("grad/norm/") for k in p._logged), f"{len(p._logged)} keys")
check("nan/inf counts SKIPPED on a non-sampled step",
      "grad/num_nans" not in p._logged and "grad/num_infs" not in p._logged)
check("clipping still happened on a non-sampled step", p._clipped == 1)

# sampled step -> full diagnostic
p = run(every=25, step=25)
check("per-module norms PRESENT on a sampled step",
      any(k.startswith("grad/norm/") for k in p._logged),
      ",".join(sorted(k for k in p._logged if k.startswith("grad/norm/"))))
check("nan/inf + postclip PRESENT on a sampled step",
      {"grad/num_nans", "grad/num_infs", "grad/norm_postclip"} <= set(p._logged))

# 3. non-finite on a NON-sampled step -> full diagnostic anyway, grads zeroed, NO clip
p = run(every=25, step=7, make_inf=True)
check("non-finite on a non-sampled step still logs per-module norms",
      any(k.startswith("grad/norm/") for k in p._logged))
check("non-finite on a non-sampled step still logs the COUNTS (read pre-clip)",
      {"grad/num_nans", "grad/num_infs"} <= set(p._logged))
check("non-finite -> grads ZEROED (optimizer step is a no-op)",
      all(float(prm.grad.abs().sum()) == 0.0 for prm in p.parameters()))
check("non-finite -> clip_gradients NOT called", p._clipped == 0)

# every=1 reproduces the historical behaviour
p = run(every=1, step=7)
check("grad_diag_every=1 logs everything (historical behaviour)",
      {"grad/num_nans", "grad/num_infs", "grad/norm_postclip"} <= set(p._logged)
      and any(k.startswith("grad/norm/") for k in p._logged))

print(f"\n{'ALL OK' if OK[0] == OK[1] else 'FAILURES'} ({OK[0]}/{OK[1]})")
raise SystemExit(0 if OK[0] == OK[1] else 1)
