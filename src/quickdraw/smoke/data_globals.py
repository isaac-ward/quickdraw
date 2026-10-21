"""Smoke for DataConfig: the data-loading config is an EXPLICIT object (2026-09-21, replaced module globals).

The whole point of DataConfig is that a loader CANNOT be called without it (keyword-only, no default), so the
eval_checkpoint-class bug -- an entrypoint silently loading at subsample=1 because it forgot a setter -- is a
loud TypeError instead. This checks: from_cfg reads every key; validation rejects bad values; the loaders
reject a missing dcfg; and the old globals/setters are truly gone.
Run: docker compose exec -T app uv run --no-sync python -m quickdraw.smoke.data_globals
"""
import inspect

from omegaconf import OmegaConf

from ..data import dataset as ds
from ..data.dataset import DataConfig


def _fail(m):
    print("FAIL:", m); raise SystemExit(1)


def main():
    # 1. from_cfg reads every key off cfg.data.
    cfg = OmegaConf.create({"data": {"subsample": 10, "action_aggregate": "mean",
                                     "subsample_all_phases": True, "obs_keep": [0, 1, 2, 7]}})
    d = DataConfig.from_cfg(cfg)
    got = (d.subsample, d.action_aggregate, d.subsample_all_phases, d.obs_keep)
    want = (10, "mean", True, (0, 1, 2, 7))
    if got != want:
        _fail(f"from_cfg(full) = {got}, expected {want}")
    print(f"from_cfg(full) -> {got}   OK")

    # 2. missing keys fall to the documented defaults, in ONE place.
    d0 = DataConfig.from_cfg(OmegaConf.create({"data": {}}))
    if (d0.subsample, d0.action_aggregate, d0.subsample_all_phases, d0.obs_keep) != (1, "sum", False, None):
        _fail(f"from_cfg(empty) = {d0}, expected defaults (1,'sum',False,None)")
    print("from_cfg(empty) -> defaults   OK")

    # 3. validation rejects nonsense (a wrong value is caught at construction, not silently loaded).
    for bad in (dict(subsample=0), dict(action_aggregate="bogus")):
        try:
            DataConfig(**bad); _fail(f"DataConfig({bad}) did not raise")
        except ValueError:
            pass
    print("validation rejects subsample<1 and unknown aggregate   OK")

    # 4. THE core guarantee: the loaders REQUIRE dcfg keyword-only with NO default -> a missed config is a
    #    loud error, never a silent subsample=1. (Checked on the signature so it needs no dataset on disk.)
    for fn in (ds.load_split_episodes, ds.load_split_episodes_mm):
        p = inspect.signature(fn).parameters.get("dcfg")
        if p is None or p.kind != inspect.Parameter.KEYWORD_ONLY or p.default is not inspect.Parameter.empty:
            _fail(f"{fn.__name__} must take dcfg as a keyword-only arg with NO default (got {p})")
    print("both loaders require keyword-only dcfg with no default   OK")

    # 5. THE AMBIENT STATE IS GONE FROM THE LOADING PATH. (Revised in the 2026-10-20 longhand merge.)
    #    This used to assert the setters did not EXIST. They do again -- ~35 frozen paper_specific/
    #    scripts set their knobs imperatively and then load, and rewriting analysis nobody can re-run
    #    was the worse risk -- but they are shims over ONE DataConfig that callers fetch explicitly via
    #    DataConfig.legacy(). What must stay true is not their absence but that they CANNOT leak into
    #    the explicit path, which (with check 4) is what makes a silent wrong-rate load impossible.
    for gone in ("apply_data_globals", "_SUBSAMPLE", "_OBS_KEEP", "_TRAIN_EPISODES", "_TRAIN_STEPS"):
        if hasattr(ds, gone):
            _fail(f"dataset.{gone} still exists -- that ambient global was supposed to go")
    ds.set_subsample(7)
    if DataConfig.legacy().subsample != 7:
        _fail("set_subsample did not reach DataConfig.legacy()")
    if DataConfig.from_cfg(OmegaConf.create({"data": {}})).subsample != 1:
        _fail("the legacy shim LEAKED into from_cfg -- that is the silent-wrong-rate bug returning")
    ds.set_subsample(1)
    print("legacy shims feed only DataConfig.legacy(), never from_cfg   OK")

    # 6. main's data-volume fields survived the merge onto DataConfig.
    dv = DataConfig.from_cfg(OmegaConf.create({"data": {"train_episodes": 240, "train_steps": 173000}}))
    if (dv.train_episodes, dv.train_steps) != (240, 173000):
        _fail(f"train_episodes/train_steps = {(dv.train_episodes, dv.train_steps)}, expected (240, 173000)")
    if (DataConfig().train_episodes, DataConfig().train_steps) != (None, None):
        _fail("train_episodes/train_steps must default to None (= uncapped)")
    print("train_episodes/train_steps carried on DataConfig   OK")

    print("\nALL DATACONFIG SMOKE PASSED")


if __name__ == "__main__":
    main()
