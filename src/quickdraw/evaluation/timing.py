"""How long does a model take, and where does the time go -- comparably, across models.

TWO PRODUCTS, because they answer different questions.

`Timer` instruments the REAL eval opportunistically: what did this run cost, and how much of it was the
model versus the metrics. Every routine that wraps its calls gets it, so our flow models, the Dreamer
checkpoints and any external adapter are all covered with no opt-in.

The `timing` ROUTINE is the one to compare. A real eval's wall-clock is confounded by n_ep, by the
horizon, by which heads decoded and by whether LPIPS ran, so two models measured on two different evals
are not comparable. The routine pins all of it -- one episode, one image head, no metrics, a fixed
horizon, one warmup and three timed repeats -- and reports the median.

TWO THINGS THAT MAKE TIMING FICTION IF YOU SKIP THEM, both of which the existing kvcache_report already
respects and which are the reason this is a shared helper rather than a perf_counter at each site:
  * CUDA IS ASYNCHRONOUS. A perf_counter around a launch measures the launch, not the work. Every phase
    synchronizes on both sides.
  * THE FIRST CALL IS NOT LIKE THE OTHERS. torch.compile, cuDNN autotune and lazy weight loads all land
    on it. The benchmark discards a warmup; the instrumented path reports `calls` so a one-shot phase is
    readable as such.

Model time and metric time are reported SEPARATELY. LPIPS is not free, and a model with a cheap rollout
and an expensive decode otherwise reads as a slow model.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

import torch


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Timer:
    """Accumulates per-phase wall-clock and peak memory. `phases` is free-form: a model that cannot
    separate dynamics from decode simply never opens those phases, and reports the total it can."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.total: dict[str, float] = {}
        self.calls: dict[str, int] = {}
        self.peak: dict[str, float] = {}

    @contextmanager
    def phase(self, name: str):
        if not self.enabled:
            yield self
            return
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        _sync()
        t0 = time.perf_counter()
        try:
            yield self
        finally:
            _sync()
            dt = time.perf_counter() - t0
            self.total[name] = self.total.get(name, 0.0) + dt
            self.calls[name] = self.calls.get(name, 0) + 1
            gb = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0
            self.peak[name] = max(self.peak.get(name, 0.0), gb)

    def emit(self, writer, tag: str, step: int, *, horizon=None, n_ep=None, n_heads=None):
        """Normalised so models with different batch shapes are comparable. per_frame is the one to read
        across models: a rollout of 8 episodes x 128 steps is 1024 frames however it was batched."""
        for name, tot in sorted(self.total.items()):
            c = max(1, self.calls[name])
            writer.scalar(f"{tag}/timing/{name}/s_total", tot, step)
            writer.scalar(f"{tag}/timing/{name}/ms_per_call", 1e3 * tot / c, step)
            writer.scalar(f"{tag}/timing/{name}/calls", float(c), step)
            writer.scalar(f"{tag}/timing/{name}/peak_gb", self.peak.get(name, 0.0), step)
            if horizon:
                writer.scalar(f"{tag}/timing/{name}/ms_per_step", 1e3 * tot / (c * horizon), step)
                frames = horizon * max(1, n_ep or 1) * max(1, n_heads or 1)
                writer.scalar(f"{tag}/timing/{name}/ms_per_frame", 1e3 * tot / (c * frames), step)
                writer.scalar(f"{tag}/timing/{name}/frames_per_s", (c * frames) / max(tot, 1e-9), step)

    def summary(self) -> str:
        return " | ".join(f"{n} {self.total[n]:.2f}s x{self.calls[n]} peak {self.peak.get(n, 0):.1f}GB"
                          for n in sorted(self.total))


def eval_timing(cfg, model, norm, ecfg, writer, device, step=0):
    """THE COMPARABLE NUMBER. A fixed workload every model runs identically, so two models can be put
    side by side without the comparison being an artefact of how many episodes each eval happened to
    use. One context, one image head, a fixed horizon, no metrics, one warmup then three timed repeats;
    the median is reported, and the spread with it so a noisy card is visible rather than averaged away.

    Deliberately NOT derived from eval.horizon or eval.horizon_n_episodes: those are tuned per run, and
    a benchmark that moves with them measures the config."""
    import numpy as np
    import torch

    from ..data.dataset import load_split_episodes_mm
    from ..training.setup import image_head_cams, image_head_sizes, resolve_data_root

    m = getattr(model, "_orig_mod", model)
    img_heads = [n for n, _ in m.layout if n != "proprio"]
    has_pro = any(n == "proprio" for n, _ in m.layout)
    H = int(cfg.eval.get("timing_horizon", 128))
    REPEATS = int(cfg.eval.get("timing_repeats", 3))
    P = int(cfg.data.P)
    head = img_heads[:1]                                   # ONE head: decode cost scales with them, and
    #                                                        a fair comparison fixes how many there are
    heads = (["proprio"] if has_pro else []) + head

    eps = load_split_episodes_mm(resolve_data_root(cfg), "val",
                                 img_size=image_head_sizes(cfg) or 128,
                                 cam=image_head_cams(cfg) or cfg.data.get("cam", "fpv"),
                                 repo_id=cfg.data.get("repo_id", "torus"))
    o, a, fr = eps[0]
    if len(o) < P + H + 1:
        H = max(8, len(o) - P - 1)
    ctx = {h: torch.from_numpy(fr[h][:P]).float().div(255.0)[None].to(device) for h in head}
    if has_pro:
        ctx["proprio"] = norm.norm_obs(torch.from_numpy(o[:P])).float()[None].to(device)
    idx = np.clip(np.arange(0, P - 1 + H), 0, len(a) - 1)
    acts = norm.norm_act(torch.from_numpy(a[idx])).float()[None].to(device)

    def once():
        t = Timer()
        with t.phase("rollout"):
            model.imagine_eval(ctx, acts, H, heads=heads,
                               decode_chunk=int(cfg.eval.get("decode_chunk", 64) or 0) or None, norm=norm)
        return t.total["rollout"], t.peak["rollout"]

    once()                                                 # WARMUP, discarded: compile/autotune/lazy load
    runs = [once() for _ in range(REPEATS)]
    ts = sorted(r[0] for r in runs)
    med, peak = ts[len(ts) // 2], max(r[1] for r in runs)
    out = {"eval_timing/rollout_s_median": med,
           "eval_timing/rollout_s_min": ts[0], "eval_timing/rollout_s_max": ts[-1],
           "eval_timing/ms_per_step": 1e3 * med / H,
           "eval_timing/frames_per_s": H / max(med, 1e-9),
           "eval_timing/peak_gb": peak,
           "eval_timing/horizon": float(H), "eval_timing/repeats": float(REPEATS)}
    for k, v in out.items():
        writer.scalar(k, v, step)
    print(f"[eval_timing @ep{step}] fixed workload: 1 episode x H={H} x 1 image head, no metrics | "
          f"median {med:.3f}s (min {ts[0]:.3f} max {ts[-1]:.3f}) | {1e3 * med / H:.1f} ms/step | "
          f"{H / max(med, 1e-9):.1f} frames/s | peak {peak:.2f} GB", flush=True)
    return out
