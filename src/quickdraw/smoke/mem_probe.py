"""Peak-memory probe for one full-BPTT rollout step (the production peak regime; the rollout is eager
in train.py, so eager here is faithful). Reports per-run peak GB and the 3-per-GPU estimate.
  uv run python -m quickdraw.smoke.mem_probe [model=...] [+collapse=...]
"""
import os
import sys

import torch
from hydra import compose, initialize_config_dir

from quickdraw.training.lit import LitWorldModel
from quickdraw.training.setup import build_model, env_cfg, normalizer

CONF = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "conf"))
DATA = "logs/data_generation_2026_06_24_06_57_53_regen"
DEV = "cuda"

ov = [f"data.root={DATA}", "model.detach_every=0"] + sys.argv[1:]
with initialize_config_dir(config_dir=CONF, version_base=None):
    cfg = compose(config_name="config", overrides=ov)

norm = normalizer(cfg)
model = build_model(cfg).to(DEV)
e = env_cfg(cfg)
lit = LitWorldModel(model, norm, e.R, e.r, e.init_speed, cfg.data.P, cfg.data.F,
                    0.0, 0.0, 0, cfg.optim.lr, cfg.optim.weight_decay, 0,
                    env_name=str(cfg.environments.get("name", "torus_world"))).to(DEV)  # p_tf=0 -> full rollout, detach_every=0
lit.log = lambda *a, **k: None
B, P, F = cfg.data.batch, cfg.data.P, cfg.data.F
batch = {"obs_seq": torch.randn(B, P + F, 6, device=DEV), "act_seq": torch.randn(B, P + F, 2, device=DEV)}
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
for _ in range(3):  # warm up allocator to steady peak
    opt.zero_grad()
    loss = lit._step(batch, "train")
    loss.backward()
    opt.step()
torch.cuda.synchronize()
peak = torch.cuda.max_memory_allocated() / 1e9
name = f"{cfg.model.get('name','?')}/{cfg.get('collapse',{}).get('name','-')}"
print(f"[{name}] full-BPTT p_tf=0 batch={B} F={F}: per-run peak={peak:.2f} GB | x3={3*peak:.1f} GB "
      f"(+~1GB/proc context+dataset -> ~{3*peak+3:.1f} GB on a 95 GB GPU)")
