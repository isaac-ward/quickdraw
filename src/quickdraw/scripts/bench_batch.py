"""Throughput vs batch: does a bigger batch raise samples/sec, or are we FLOP-saturated?
Replicates the epoch-0 training step (parallel forward m(obs,act) + decode + backward + AdamW), bf16-mixed
autocast to match the trainer. Reports samples/sec + peak GB at several batch sizes for ONE MM model.
"""
import os, sys, time, torch
import quickdraw
from hydra import compose, initialize_config_dir
from quickdraw.training.setup import build_model

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mm_lsar_ema"
BATCHES = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["96", "192", "288"])]
conf_dir = os.path.abspath(os.path.join(os.path.dirname(quickdraw.__file__), "..", "..", "conf"))
dev = torch.device("cuda")

with initialize_config_dir(config_dir=conf_dir, version_base=None):
    cfg = compose(config_name="config", overrides=[f"model={MODEL}", "data.F=24"])
P, L = int(cfg.data.P), int(cfg.data.P) + 24
m = build_model(cfg).to(dev).train()
img_head = next(n for n, _ in m.layout if n != "proprio")
img_size = m.modalities[img_head].ae.cfg.img_size
opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
print(f"model={MODEL}  params={sum(p.numel() for p in m.parameters())/1e6:.1f}M  L={L}  img={img_size}  heads={[n for n,_ in m.layout]}", flush=True)

def make(B):
    return ({"proprio": torch.randn(B, L, 6, device=dev),
             img_head: torch.rand(B, L, img_size, img_size, 3, device=dev)},
            torch.randn(B, L, 2, device=dev))

def one_step(obs, act):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        preds = m({k: v[:, :-1] for k, v in obs.items()}, act[:, :-1])[:, P - 1:]
        dec = m.to_obs(preds)
        loss = sum(x.float().pow(2).mean() for x in dec.values())
    loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)

print(f"{'batch':>6} {'s/step':>9} {'samples/s':>11} {'peak_GB':>9}", flush=True)
for B in BATCHES:
    try:
        torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
        obs, act = make(B)
        for _ in range(3):  # warmup
            one_step(obs, act)
        torch.cuda.synchronize(); t0 = time.perf_counter(); iters = 8
        for _ in range(iters):
            one_step(obs, act)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"{B:>6} {dt:>9.4f} {B/dt:>11.1f} {peak:>9.1f}", flush=True)
        del obs, act
    except RuntimeError as e:
        print(f"{B:>6}  OOM/err: {str(e)[:60]}", flush=True)
        torch.cuda.empty_cache()
