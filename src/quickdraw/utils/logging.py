"""Run-folder logging utility (ported from seamstress custom_logging).

Run dirs are named `<step>_<timestamp>_<experiment>`: the pipeline step
(`generate_data` | `train` | `eval`) as the prefix, the experiment summary as the suffix.
"""

from __future__ import annotations

import datetime as _dt
import os


def get_timestamp() -> str:
    return _dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


def make_run_dir(step: str, experiment: str, root: str | None = None) -> str:
    # create only the run dir; each step makes the subdirs it actually uses (no empty holdovers).
    # root defaults to $QUICKDRAW_LOG_ROOT (else "logs") so a batch of runs can be grouped in a subfolder.
    root = root or os.environ.get("QUICKDRAW_LOG_ROOT", "logs")
    path = os.path.join(root, f"{step}_{get_timestamp()}_{experiment}")
    os.makedirs(path, exist_ok=True)
    return path
