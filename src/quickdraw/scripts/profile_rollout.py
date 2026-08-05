"""Phase-0 profiling harness (design/rollout_throughput.md). Runs the REAL AR training step
(model.rollout_train at p_tf=0) on the joint mm_flow config + synthetic-but-faithful batch, and reports:
(1) is FlexAttention a FUSED Triton kernel or an eager bmm+softmax reference?  (2) where the per-batch time
goes (CUDA kernel table)  (3) s/batch at a given batch.  Gitignored (wizard/scripts/*). Run in-container:
  /app/.venv/bin/python /app/wizard/scripts/profile_rollout.py <run_dir> <batch> <mode:bench|profile>
"""
import os, sys, time, torch
from omegaconf import OmegaConf
from quickdraw.training.setup import build_model

RUN = sys.argv[1] if len(sys.argv) > 1 else "logs/train_world_model_2026_08_04_18_15_38_jointaction_detachT_resume"
B = int(sys.argv[2]) if len(sys.argv) > 2 else 32
MODE = sys.argv[3] if len(sys.argv) > 3 else "profile"
cfg = OmegaConf.load(os.path.join(RUN, "checkpoints", "config.resolved.yaml"))
P, F = int(cfg.data.P), int(cfg.data.F)
L = P + F
dev = "cuda"
torch.manual_seed(0)
model = build_model(cfg).to(dev).train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
pdim = int(cfg.model.modalities[0].dim)
print(f"[cfg] run={os.path.basename(RUN)} d={cfg.model.d} depth={cfg.model.depth} heads={cfg.model.heads} "
      f"window={cfg.model.window} | P={P} F={F} L={L} batch={B} proprio={pdim} action={cfg.model.action_dim} "
      f"| params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)


def mkbatch(b):
    obs = {"proprio": torch.randn(b, L, pdim, device=dev),
           "image": torch.rand(b, L, 128, 128, 3, device=dev)}
    act = torch.randn(b, L - 1, int(cfg.model.action_dim), device=dev)
    return {k: v[:, :P] for k, v in obs.items()}, act, {k: v[:, P:] for k, v in obs.items()}


def step(b):
    ctx, act, fut = mkbatch(b)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        preds = model.rollout_train(ctx, act, fut, 0.0, int(cfg.model.get("detach_every", 16)))
        loss = preds.float().pow(2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    return float(loss)


for _ in range(2):
    step(B)
torch.cuda.synchronize()
t = time.perf_counter()
for _ in range(5):
    step(B)
torch.cuda.synchronize()
print(f"[bench] batch={B}  s/batch={(time.perf_counter() - t) / 5:.3f}  "
      f"peak_mem={torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)

if MODE == "profile":
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        step(B); torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30), flush=True)
    ka = prof.key_averages()
    def has(sub): return [e.key for e in ka if sub in e.key.lower()][:6]
    print("[fusion] flex/triton attn kernels:", has("flex") + has("triton") + has("sdpa_flex"), flush=True)
    print("[fusion] eager sdpa/bmm/softmax  :", has("scaled_dot") + has("bmm") + has("softmax") + has("baddbmm"), flush=True)
