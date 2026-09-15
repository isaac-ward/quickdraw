"""Open-loop prediction error on the named held-out splits (the OOD and memory campaigns).

Runs `eval_ood_horizon` once per split with the SAME metric code that produces the headline val numbers,
so the OOD/memory figures are comparable to them by construction. Everything downloads from the Hub, so a
fresh clone needs no local data:

    python -m quickdraw.eval_splits \\
        checkpoint=<run_dir_or_ckpt> data=starling2 environments=recorded \\
        data.subsample=4 data.action_aggregate=concat \\
        'eval.splits=[val,eval_ood_noodle,eval_ood_leafblower,eval_memory_backwall1,eval_memory_backwall2]'

`eval.horizon_n_episodes` lifts the 8-episode image-head cap, which exists for val's 1785-step episodes
and is wrong for these ~31-step ones: set it to 12 to score every episode in each split.
"""

from __future__ import annotations

import hydra

from .evaluation.standalone import run_standalone


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg):
    run_standalone(cfg, ["held_out_splits"], label="splits")


if __name__ == "__main__":
    main()
