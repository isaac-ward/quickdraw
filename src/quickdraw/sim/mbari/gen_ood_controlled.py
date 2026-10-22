"""Controlled OOD episodes: a clean window, then a single anomaly that ARRIVES and stays.

WHY A SECOND OOD SET. The v7 OOD splits answer "is this episode anomalous" but they cannot answer
"WHEN, and WHERE in the frame", because the anomaly is present from step 0 and the vehicle is flying
a random walk that sweeps the scene unpredictably. Every episode here has the same structure:

    steps    0 - 299   (10 s)  CLEAN. The anomaly does not exist yet.
    steps  300 - 1199  (30 s)  PRESENT. It is in frame, continuously, until the end.

so the surprise signal can be read against a known onset -- low before 300, localised after -- and a
per-episode score has an uncontaminated baseline inside the SAME episode.

THE CAMERA FACES THE CIRCLE CENTRE, which is the whole reason this works. On a tangential heading the
vehicle yaws continuously and a stationary object sweeps through the view and straight out again; you
get a glimpse, not a 30 s window. Facing the centre makes the object's bearing constant, so it sits in
frame for the whole lap (`trajectories.policy_circle(face="centre")`).

THE BUOY CONDITION NEEDED A NEW OBJECT. `buoy_purple` recolours a fixture that is in the scene from
step 0 -- it can never "arrive". So `tank.mjcf(buoy_ood=True)` adds a purple buoy as a MOCAP body that
is parked below the floor and raised at the onset. That also makes it a cleaner anomaly: a novel object
appearing, rather than a known object changing colour.

THE THRUSTER CONDITION HAS NO OBJECT, so its anomaly is switched on in TIME instead -- `glitch_from`
fires the 30% action glitch at the same step 300. Every condition therefore shares one onset.

COUNTS. 30 per OOD condition, and 60 nominal split into 30 CALIBRATION + 30 TEST, so a conformal
threshold can be fitted on held-out nominals and the false-positive rate measured on nominals that
fitting never saw. The current 3-way classifier is leave-one-out nearest-centroid with no held-out
calibration at all; this set is what makes split-conformal possible.

    python -m quickdraw.sim.mbari.gen_ood_controlled --out logs/mbari/data/ood_controlled
"""
from __future__ import annotations

import argparse
import ast
import os
import pathlib

import numpy as np

FPS = 30
STEPS = 1200                 # 40 s
ONSET = 300                  # 10 s clean, then 30 s with the anomaly
GLITCH_P = 0.30              # same probabilistic glitch the v7 thruster split uses
HIDDEN_Z = -40.0             # parking depth, well below the tank floor
# RADIUS FLOOR, AND IT IS NOT COSMETIC. MuJoCo's near clipping plane here is
# vis.map.znear * stat.extent = 0.01 * 51.69 = 0.517 m, and the fpv camera sits 0.5 m forward of
# the body centre -- so on a radius-1.0 circle the camera is 0.59 m from an object at the centre,
# whose near surface at 0.25 m is INSIDE the near plane and the whole sphere is clipped away.
# The first attempt generated exactly that and produced 30 s of "anomaly present" with nothing
# visible in a single frame. Camera-to-object is radius - 0.5, so 2.2 m puts it at 1.7 m, well
# clear, and 3.4 m keeps it large enough to localise.
RADIUS_RANGE = (2.2, 3.4)
COUNTS = {"nominal_calib": 30, "nominal_test": 30,
          "ood_fish": 30, "ood_buoy": 30, "ood_thruster": 30}
SEED0 = {"nominal_calib": 300000, "nominal_test": 400000,
         "ood_fish": 500000, "ood_buoy": 600000, "ood_thruster": 700000}


def _model_for(split):
    """Scene per condition. Nominal and thruster share the plain tank -- the thruster anomaly is in
    the actions, not the scene, which is exactly why it is the hard one to localise in pixels."""
    from .tank import build
    import mujoco
    if split == "ood_fish":
        p, _ = build(textures=False, name="tank_ctl_fish.xml", n_fish=1)
    elif split == "ood_buoy":
        p, _ = build(textures=False, name="tank_ctl_buoyood.xml", buoy_ood=True)
    else:
        p, _ = build(textures=False, name="tank_ctl_nom.xml")
    return mujoco.MjModel.from_xml_path(str(p))


def _placer(model, split, centre):
    """extra(model, data, k): park the anomaly below the floor, raise it at ONSET.

    The object sits AT the circle centre, so the inward-facing camera holds it in frame for the whole
    remaining lap. It is given a slow spin rather than left frozen -- a statue reads as scenery."""
    import mujoco
    name = {"ood_fish": "fish0", "ood_buoy": "buoy_ood"}.get(split)
    if name is None:
        return None
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        return None
    mid = int(model.body_mocapid[bid])

    def extra(_m, d, k):
        if k < ONSET:
            d.mocap_pos[mid] = np.array([0.0, 0.0, HIDDEN_Z])
            return
        d.mocap_pos[mid] = np.asarray(centre, np.float64)
        th = 0.35 * (k - ONSET) / FPS                      # slow yaw so it is not a statue
        d.mocap_quat[mid] = np.array([np.cos(th / 2), 0.0, 0.0, np.sin(th / 2)])
    return extra


def generate(out, splits=tuple(COUNTS), size=(144, 192), log=print):
    from . import record, trajectories as tj
    from .water import UnderwaterRenderer

    root = pathlib.Path(out)
    for split in splits:
        n_want = COUNTS[split]
        d = root / split
        d.mkdir(parents=True, exist_ok=True)
        model = _model_for(split)
        rend = UnderwaterRenderer(model, size[0], size[1], cams=record.CAMS)
        made, tries, seed = 0, 0, SEED0[split]
        while made < n_want and tries < n_want * 8:
            tries += 1
            seed += 1
            rng = np.random.default_rng(seed)
            spec = tj.fit_circle(rng, STEPS / FPS, 1.0 / FPS)
            if spec is None:
                continue
            if not (RADIUS_RANGE[0] <= spec["rad"] <= RADIUS_RANGE[1]):
                continue                                   # see RADIUS_RANGE: near-plane clipping
            extra = _placer(model, split, spec["centre"])
            r = record.rollout(model, "circle", seed, steps=STEPS, size=size, renderer=rend,
                               glitch_p=(GLITCH_P if split == "ood_thruster" else 0.0),
                               glitch_from=ONSET, extra=extra, face="centre")
            if r is None or len(r["obs"]) < STEPS:          # a short episode breaks the onset contract
                continue
            meta = dict(r["meta"], split=split, onset=ONSET, controlled=True,
                        centre=[float(x) for x in spec["centre"]], radius=float(spec["rad"]),
                        cam_to_object=float(spec["rad"] - 0.5),
                        speed=float(spec["speed"]))
            tmp = d / f".tmp.ep_{seed}.npz"
            np.savez_compressed(tmp, obs=r["obs"], act=r["act"], meta=str(meta),
                                **{f"frames_{c}": v for c, v in r["frames"].items()})
            os.replace(tmp, d / f"ep_{seed}.npz")
            made += 1
            if made % 5 == 0 or made == n_want:
                log(f"  {split:15s} {made}/{n_want}  (tries {tries})")
        if made < n_want:
            log(f"  WARNING {split}: only {made}/{n_want} after {tries} tries")
    log(f"wrote {root}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="/app/logs/mbari/data/ood_controlled")
    ap.add_argument("--splits", default=",".join(COUNTS))
    a = ap.parse_args()
    generate(a.out, splits=tuple(x for x in a.splits.split(",") if x))


if __name__ == "__main__":
    main()
