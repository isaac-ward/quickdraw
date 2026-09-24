"""Smoke for the general obs_fields mechanism (2026-09-24): the obs is a CONCATENATION of listed stored
features, in order; default ("observation_vector",) is bit-identical; obs_keep + normalization apply POST-concat.
Run: docker compose exec -T app uv run --no-sync python -m quickdraw.smoke.obs_fields
"""
import json
import os
import tempfile

import numpy as np
import torch
from omegaconf import OmegaConf

from ..data.dataset import DataConfig, Normalizer, _read_obs_fields


def _fail(m):
    print("FAIL:", m); raise SystemExit(1)


def main():
    # 1. DataConfig.from_cfg reads obs_fields; default is the single base field.
    d0 = DataConfig.from_cfg(OmegaConf.create({"data": {}}))
    if d0.obs_fields != ("observation_vector",):
        _fail(f"default obs_fields = {d0.obs_fields}, expected ('observation_vector',)")
    d1 = DataConfig.from_cfg(OmegaConf.create({"data": {"obs_fields": ["observation_vector", "observation.objects.cube"]}}))
    if d1.obs_fields != ("observation_vector", "observation.objects.cube"):
        _fail(f"obs_fields not read: {d1.obs_fields}")
    print(f"from_cfg obs_fields default {d0.obs_fields}, custom {d1.obs_fields}   OK")

    # 2. _read_obs_fields concatenates in order; single field == the bare stack (bit-identical).
    N = 7
    hf = {"observation_vector": [np.arange(17, dtype=np.float32) + i for i in range(N)],
          "observation.objects.cube": [np.arange(32, dtype=np.float32) + 100 + i for i in range(N)]}
    base = _read_obs_fields(hf, ("observation_vector",))
    if base.shape != (N, 17) or not np.array_equal(base, np.stack(hf["observation_vector"])):
        _fail("single-field read not bit-identical to np.stack")
    cat = _read_obs_fields(hf, ("observation_vector", "observation.objects.cube"))
    if cat.shape != (N, 49) or not np.array_equal(cat[:, :17], base) or not np.array_equal(cat[:, 17:], np.stack(hf["observation.objects.cube"])):
        _fail("concat wrong shape/order")
    print(f"_read_obs_fields: base {base.shape}, concat {cat.shape} (order preserved)   OK")

    # 3. missing field -> clear KeyError.
    try:
        _read_obs_fields(hf, ("observation_vector", "nope")); _fail("missing field did not raise")
    except KeyError:
        print("missing field raises KeyError   OK")

    # 4. Normalizer.from_file assembles per-field stats in obs_fields order; default == observation_vector alone.
    with tempfile.TemporaryDirectory() as td:
        stats = {"action": {"mean": [0.0] * 5, "std": [1.0] * 5},
                 "observation_vector": {"mean": [1.0] * 17, "std": [2.0] * 17},
                 "observation.objects.cube": {"mean": [3.0] * 32, "std": [4.0] * 32}}
        json.dump(stats, open(os.path.join(td, "normalization_stats.json"), "w"))
        nb = Normalizer.from_file(td, obs_keep=None)                                   # default single field
        if nb.o_mean.shape[0] != 17 or float(nb.o_std[0]) != 2.0:
            _fail("default Normalizer not observation_vector-only")
        nc = Normalizer.from_file(td, obs_keep=None, obs_fields=("observation_vector", "observation.objects.cube"))
        if nc.o_mean.shape[0] != 49 or float(nc.o_mean[17]) != 3.0 or float(nc.o_std[17]) != 4.0:
            _fail(f"per-field concat wrong: mean[17]={float(nc.o_mean[17])} std[17]={float(nc.o_std[17])}")
        # obs_keep POST-concat: keep one dim from each field
        nk = Normalizer.from_file(td, obs_keep=(0, 20), obs_fields=("observation_vector", "observation.objects.cube"))
        if nk.o_mean.tolist() != [1.0, 3.0] or nk.o_std.tolist() != [2.0, 4.0]:
            _fail(f"obs_keep post-concat wrong: {nk.o_mean.tolist()} {nk.o_std.tolist()}")
    print("Normalizer per-field assembly + obs_keep POST-concat   OK")

    print("\nALL OBS_FIELDS SMOKE PASSED")


if __name__ == "__main__":
    main()
