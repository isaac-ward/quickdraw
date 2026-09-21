"""Smoke for apply_data_globals: ONE call sets ALL FOUR process-wide data globals from a cfg.

This is the guard against the eval_checkpoint class of bug -- an entrypoint that loads data but forgets a
process-global and silently runs at a default (subsample=1, action_aggregate='sum', ...).
Run: docker compose exec -T app uv run --no-sync python -m quickdraw.smoke.data_globals
"""
from omegaconf import OmegaConf

from ..data import dataset as ds


def _fail(m):
    print("FAIL:", m); raise SystemExit(1)


def main():
    # 1. a full cfg -> every global reflects it (not the defaults).
    cfg = OmegaConf.create({"data": {"subsample": 10, "action_aggregate": "mean",
                                     "subsample_all_phases": True, "obs_keep": [0, 1, 2, 7]}})
    ds.apply_data_globals(cfg)
    got = (ds.get_subsample(), ds.get_action_aggregate(), ds._SUBSAMPLE_ALL_PHASES, ds.get_obs_keep())
    want = (10, "mean", True, [0, 1, 2, 7])
    if got != want:
        _fail(f"apply_data_globals(full cfg) set {got}, expected {want}")
    print(f"full cfg -> {got}   OK")

    # 2. idempotent: the SAME cfg re-applied does not raise and holds the values.
    ds.apply_data_globals(cfg)
    if (ds.get_subsample(), ds.get_action_aggregate()) != (10, "mean"):
        _fail("re-applying the same cfg changed the globals")
    print("idempotent re-apply   OK")

    # 3. missing data keys fall to the documented defaults (fresh-process semantics).
    #    (safe to change here: no episodes were loaded, so the _USED guards are not armed.)
    ds.apply_data_globals(OmegaConf.create({"data": {}}))
    got2 = (ds.get_subsample(), ds.get_action_aggregate(), ds._SUBSAMPLE_ALL_PHASES, ds.get_obs_keep())
    if got2 != (1, "sum", False, None):
        _fail(f"apply_data_globals(empty) -> {got2}, expected defaults (1, 'sum', False, None)")
    print(f"empty data -> {got2}   OK")

    print("\nALL DATA-GLOBALS SMOKE PASSED")


if __name__ == "__main__":
    main()
