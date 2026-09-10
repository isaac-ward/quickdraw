"""Unified run writer: log once, land in BOTH the local run folder and Weights & Biases.

Design goal (no divergence): all logging code calls ONE object, `RunWriter`, with four verbs —
`scalar`, `figure`, `video`, `config`. `RunWriter` just fans each call out to a list of backends.
The ONLY place local-vs-wandb behaviour differs is the backend's write method itself; there is no
duplicated "save a png AND also wandb.Image it" logic anywhere else in the codebase.

Both backends are driven with the SAME (tag, payload, step), so the local mirror and the wandb run
have identical names, identical artifacts, and identical step alignment (step = epoch).
"""

from __future__ import annotations

import json
import os

from . import viz


class _Backend:
    """Interface every backend implements. Tags may contain '/', used as a path/namespace."""

    def scalar(self, tag: str, value: float, step: int): ...
    def figure(self, tag: str, fig, step: int): ...
    def video(self, tag: str, frames, fps: int, step: int): ...  # frames: (T,H,W,3) uint8
    def scene(self, tag: str, scene: dict, step: int): ...  # 3D scene geometry -> JSON (Blender, local only)
    def config(self, cfg: dict): ...
    def finalize(self): ...


class LocalBackend(_Backend):
    """Local clone of the wandb run, in `<run_dir>/logs/` (the place to browse what's on wandb):
    scalars -> logs/metrics.jsonl, config -> logs/config.json, media -> logs/epoch_<i>/<tag>.{png,mp4}
    (epoch first, then the tag path, then the artifact)."""

    def __init__(self, run_dir: str):
        self.dir = os.path.join(run_dir, "logs")
        os.makedirs(self.dir, exist_ok=True)
        self._scalars = os.path.join(self.dir, "metrics.jsonl")

    def _path(self, tag: str, step: int, ext: str) -> str:
        p = os.path.join(self.dir, f"epoch_{step:04d}", tag) + f".{ext}"
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def scalar(self, tag, value, step):
        with open(self._scalars, "a") as f:
            f.write(json.dumps({"step": step, "tag": tag, "value": value}) + "\n")

    def figure(self, tag, fig, step):
        fig.savefig(self._path(tag, step, "png"), bbox_inches="tight", dpi=viz.DPI)

    def video(self, tag, frames, fps, step):
        viz.save_mp4(self._path(tag, step, "mp4"), frames, fps)

    def scene(self, tag, scene, step):
        # Plain-language 3D scene geometry next to the media (e.g. epoch_0030/eval_flow/denoising_multistep/example_0.json)
        # so it can be reconstructed in Blender later. numpy arrays/scalars -> nested lists/floats.
        import numpy as np

        def cvt(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            if isinstance(o, dict):
                return {k: cvt(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [cvt(v) for v in o]
            return o

        with open(self._path(tag, step, "json"), "w") as f:
            json.dump(cvt(scene), f)

    def config(self, cfg):
        with open(os.path.join(self.dir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)


class WandbBackend(_Backend):
    """Same verbs, written to a wandb run. Imports wandb lazily so local-only runs need no wandb."""

    def __init__(self, run):
        self.run = run

    def scalar(self, tag, value, step):
        self.run.log({tag: value}, step=step)

    def figure(self, tag, fig, step):
        import wandb

        self.run.log({tag: wandb.Image(fig)}, step=step)

    def video(self, tag, frames, fps, step):
        # encode to an mp4 ourselves (imageio/ffmpeg) and hand wandb the FILE — wandb.Video only needs
        # moviepy when passed raw arrays, which isn't installed; a path is uploaded as-is.
        # mkstemp -> a UNIQUE path: all concurrent shoot-out runs share the container's /tmp, so a fixed
        # name (per tag+step) raced across runs and silently corrupted some videos. Remove after upload.
        import os
        import tempfile

        import wandb

        from . import viz

        fd, path = tempfile.mkstemp(prefix="wandbvid_", suffix=".mp4")
        os.close(fd)
        try:
            viz.save_mp4(path, frames, fps)
            self.run.log({tag: wandb.Video(path)}, step=step)  # wandb copies the file in synchronously
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def config(self, cfg):
        self.run.config.update(cfg, allow_val_change=True)

    def finalize(self):
        self.run.finish()


class RunWriter:
    """Fan-out over backends. This is the single place that knows there is more than one sink."""

    def __init__(self, run_dir: str, backends: list[_Backend], playback_fps: float | None = None):
        self.dir = run_dir  # local mirror dir, so callers can drop summary.json next to the media
        self.backends = backends
        self.playback_fps = playback_fps  # container rate for video (viz.pace); None -> the true rate

    def scalar(self, tag, value, step):
        for b in self.backends:
            b.scalar(tag, float(value), step)

    def scalars(self, d: dict, step: int):
        for tag, value in d.items():
            self.scalar(tag, value, step)

    def figure(self, tag, fig, step):
        for b in self.backends:
            b.figure(tag, fig, step)

    def video(self, tag, frames, fps, step):
        # Retime ONCE here, not per backend, so the local mirror and the wandb run are identical
        # (this module's no-divergence rule). `fps` from callers is the frames' TRUE sample rate --
        # for a rollout that is the model's STEP rate, not the dataset's capture rate.
        frames, fps = viz.pace(frames, fps, self.playback_fps)
        for b in self.backends:
            b.video(tag, frames, fps, step)

    def scene(self, tag, scene: dict, step):
        for b in self.backends:
            b.scene(tag, scene, step)

    def array(self, tag, step, **arrays):
        """Save raw numpy arrays next to the media at logs/epoch_<step>/<tag>.npz (LOCAL only — not sent to
        wandb). For keeping the exact pixels behind a rendered product (e.g. filmstrip pred/GT frames) so
        sharpness/quality can be judged later at native resolution instead of from a downscaled figure."""
        import numpy as np
        p = os.path.join(self.dir, f"epoch_{step:04d}", tag) + ".npz"
        os.makedirs(os.path.dirname(p), exist_ok=True)
        np.savez_compressed(p, **arrays)

    def config(self, cfg: dict):
        for b in self.backends:
            b.config(cfg)

    def finalize(self):
        for b in self.backends:
            b.finalize()


def make_writer(run_dir: str, cfg, job_type: str) -> RunWriter:
    """Local backend always; wandb backend too unless it can't init (offline / no key)."""
    local = LocalBackend(run_dir)
    backends: list[_Backend] = [local]
    try:
        import wandb

        run = wandb.init(project=cfg.logging.project, group=cfg.logging.group, job_type=job_type,
                         name=os.path.basename(run_dir), dir=run_dir)
        backends.append(WandbBackend(run))
    except Exception as e:  # offline / missing key -> local-only, don't crash the run
        print(f"[writer] wandb disabled ({type(e).__name__}: {e}); logging locally only")
    # Playback rate is a property of the ENVIRONMENT (it knows what its frames mean), so it is read
    # from environments.preview_fps; absent -> viz.PREVIEW_FPS, explicit null -> the true rate.
    env = cfg.get("environments", None)
    pf = env.get("preview_fps", viz.PREVIEW_FPS) if env is not None else viz.PREVIEW_FPS
    return RunWriter(local.dir, backends, playback_fps=pf)  # .dir = the local clone (<run_dir>/logs)
