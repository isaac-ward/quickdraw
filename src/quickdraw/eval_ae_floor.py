"""Standalone AE-floor eval (issue #12 Phase-1 gate). `python -m quickdraw.eval_ae_floor checkpoint=...`

Modes (dispatched by `+ae_floor.*` overrides):
  default:                          run the eval_ae_floor routine on a checkpoint (bespoke-AE encode->decode floor).
  +ae_floor.taesd=true:             RAW pretrained-TAESD domain-shift probe on real frames (NO checkpoint needed).
                                    +ae_floor.sizes=[128,256] +ae_floor.n_frames=32 to configure.
  +ae_floor.num_tokens_sweep=[8,16,32]: run eval_ae_floor at each num_tokens (NEEDS a TRAINED ckpt at that width;
                                    see evaluation/ae_floor.num_tokens_floor_note — untrained = garbage, not run).
"""

from __future__ import annotations

import hydra

from .evaluation.ae_floor import num_tokens_floor_note, raw_taesd_floor
from .evaluation.standalone import run_standalone


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg):
    af = {}
    try:
        af = cfg.get("ae_floor", {}) or {}
    except Exception:
        af = {}
    if af.get("taesd"):                                   # raw-TAESD domain-shift probe — no model/checkpoint
        sizes = tuple(int(s) for s in af.get("sizes", [128]))
        raw_taesd_floor(cfg, sizes=sizes, n_frames=int(af.get("n_frames", 32)))
        return
    sweep = af.get("num_tokens_sweep", None)
    if sweep:                                             # bespoke-AE floor at each width (needs trained ckpts)
        num_tokens_floor_note()
        from omegaconf import OmegaConf, open_dict
        for nt in [int(x) for x in sweep]:
            with open_dict(cfg):
                cfg.model.modalities[1].num_tokens = nt   # image modality width (index 1 by convention)
            run_standalone(cfg, "ae_floor", label=f"ae_floor_nt{nt}")
        return
    run_standalone(cfg, "ae_floor")                       # default: the bespoke-AE floor on this checkpoint


if __name__ == "__main__":
    main()
