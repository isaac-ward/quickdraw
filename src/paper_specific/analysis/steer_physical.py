"""What the plan DID, in metres and degrees -- not what its sticks said.

Why replace the stick readout. "Separation" was the mean commanded stick of one request minus its opposite,
and it has three defects: it is in arbitrary joystick units so no value is interpretable as good or bad; it
skips the world model entirely, scoring the planner's INPUT rather than the trajectory it produced; and it
exists only for the 8 directional requests, leaving 18 of 26 unmeasured. Steering is a claim about where the
drone went, so measure where it went.

Every plan folder already holds the imagined proprio, so this needs no new rollouts. From it:

    yaw        net heading change over the plan, DEGREES (unwrapped from the quaternion)
    altitude   net change in z, METRES
    forward    displacement along the START heading, METRES
    lateral    displacement across the start heading, METRES

CALIBRATED, NOT ASSUMED. Frame conventions and stick signs are exactly where this project has been bitten
before, so the mapping from "which request wants which physical axis, in which direction" is measured on
the RECORDED flights: correlate each commanded stick with each physical readout over real segments, and
take the sign the data shows. The recorded per-segment distribution also supplies the SCALE -- "rotated 47
degrees" only means something next to what a pilot does in the same 34 seconds.

    python scratch/steer_physical.py <steer_run> [<steer_run> ...]
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import yaml

AXES = [a["name"] for a in yaml.safe_load(open("conf/interpret/starling.yaml"))["action_axes"]]
NA = len(AXES)
# request -> the physical quantity it names, and the sign it wants. The SIGN is verified against recorded
# data by calibrate() below and this table is rewritten if the data disagrees.
WANTS = {"rotate left": ("yaw", +1), "rotate right": ("yaw", -1),
         "climb": ("altitude", +1), "descend": ("altitude", -1),
         "fly forward": ("forward", +1), "fly backward": ("forward", -1),
         "strafe right": ("lateral", +1), "strafe left": ("lateral", -1)}


MOTION_REQUESTS = tuple(WANTS)                  # the 8 with an exact physical readout
# THE OPPOSING REQUEST ON EACH AXIS, which is what makes a negative class available at all: "rotate left"
# and "rotate right" name the same axis with opposite signs, so a context asked for one is a negative for
# the other.
OPPOSING = {q: next(p for p in WANTS if p != q and WANTS[p][0] == WANTS[q][0]) for q in WANTS}
REST_FRAC = 0.05                                # a request counts as met past 5% of a pilot's own motion


def weighted_accuracy(hit_rate: float, false_positive_rate: float) -> float:
    """Balanced accuracy in percent: the mean of the true-positive and true-negative rates, so the
    no-skill value is 50 under ANY class imbalance. One definition, used for both blocks of the steering
    table and for the selection-rule sweep -- it lived in two places and the sign of the negative class
    is subtle enough that the first version returned exactly 50.0 for every arm (summed over an opposing
    pair, tpr_q + 1 - tpr_q' cancels to 1 identically)."""
    return 100.0 * 0.5 * (hit_rate + 1.0 - false_positive_rate)


def motion_weighted_accuracy(phys: dict, pilot: dict) -> float | None:
    """Mean weighted accuracy over the 8 motion primitives for ONE arm.

    `phys` is {request: (mean, n, per-context values)} as tables._phys returns it, storing motion*sgn for
    the request it was asked under. So for the OPPOSING request, motion in THIS request's direction past
    the threshold is a stored value below -threshold -- not above +threshold, which is that request's own
    hit rate and cancels to 0.5 when summed over the pair."""
    import numpy as np
    w = []
    for q in MOTION_REQUESTS:
        thr = REST_FRAC * pilot[WANTS[q][0]]
        pos, neg = phys.get(q), phys.get(OPPOSING[q])
        if not pos or not neg:
            continue
        tpr = float(np.mean([v > thr for v in pos[2]]))
        fpr = float(np.mean([v < -thr for v in neg[2]]))
        w.append(weighted_accuracy(tpr, fpr))
    return float(np.mean(w)) if w else None


def yaw_of(q):                                  # (T,4) qx,qy,qz,qw -> (T,) radians
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2 * (w * z + x * y), 1.0 - 2 * (y * y + z * z))


def physical(pro: np.ndarray) -> dict:
    """(T,16) raw proprio -> the four physical readouts over the whole span.

    forward/lateral are PATH-INTEGRATED in the instantaneous body frame, not projected onto the starting
    heading. That distinction is not pedantry: a pilot turns ~470 degrees in one of these 34 s segments, so
    the start heading describes the drone for a fraction of the span, and projecting net displacement onto
    it measured nothing -- calibration showed the fore/aft stick correlating -0.07 with "forward" under
    that definition, which is how the bug was caught. Rotating each step's displacement by that step's own
    heading and summing gives the distance travelled nose-first, which is what "fly forward" means."""
    p, q = pro[:, 0:3], pro[:, 6:10]
    yaw = np.unwrap(yaw_of(q))
    d = np.diff(p, axis=0)                                  # (T-1,3) per-step displacement
    c, s_ = np.cos(yaw[:-1]), np.sin(yaw[:-1])              # the heading each step was flown at
    # THE 180 DEGREE HEADING OFFSET. The quaternion's yaw points opposite the camera's forward, so the
    # body-frame projection comes out negated. Not assumed -- identified: without the flip the calibration
    # reads "fore/aft + = forward", contradicting the optical-flow finding that - = forward, AND
    # "lateral + = strafe left", contradicting conf/interpret/starling.yaml. A 180 degree offset negates
    # forward and lateral while leaving yaw DIFFERENCES and altitude untouched, and flipping makes all
    # four axes agree with both independently-established conventions at once. |corr| on the diagonal is
    # 0.99 / 1.00 / 0.99, so this is a sign convention, not a weak inference.
    return {"yaw": float(np.degrees(yaw[-1] - yaw[0])),
            "altitude": float(p[-1, 2] - p[0, 2]),
            "forward": -float(np.sum(c * d[:, 0] + s_ * d[:, 1])),
            "lateral": -float(np.sum(-s_ * d[:, 0] + c * d[:, 1]))}


def calibrate(span: int = 128):
    """Measure, on REAL flights, which stick drives which physical readout and with what sign + scale."""
    from omegaconf import OmegaConf
    from quickdraw.data.dataset import load_split_episodes_mm, set_action_aggregate, set_subsample
    from quickdraw.training.setup import image_head_cams, image_head_sizes, resolve_data_root
    cfg = OmegaConf.create(json.load(open(
        "logs/paper_icra_2027/model_backups/train_action_2026_09_14_04_41_17_s2_ah_chunk32_full/logs/config.json")))
    set_subsample(4); set_action_aggregate("concat")
    eps = load_split_episodes_mm(resolve_data_root(cfg), "val", img_size=image_head_sizes(cfg),
                                 cam=image_head_cams(cfg), repo_id="starling-2")
    rows, sticks = [], []
    for o, a, _ in eps:
        for i in range(0, len(o) - span, span // 2):
            rows.append(physical(o[i:i + span]))
            sticks.append(a[i:i + span].reshape(span, -1, NA).mean(axis=1).mean(axis=0))
    sticks = np.stack(sticks)
    keys = ["yaw", "altitude", "forward", "lateral"]
    M = np.stack([[r[k] for k in keys] for r in rows])
    print(f"\n  CALIBRATION on {len(M)} recorded {span}-step segments ({span * 4 / 15:.0f}s each)")
    print(f"  {'stick':10s} " + "  ".join(f"{k:>10s}" for k in keys) + "   <- corr(stick, physical)")
    for j, nm in enumerate(AXES):
        c = [float(np.corrcoef(sticks[:, j], M[:, i])[0, 1]) for i in range(len(keys))]
        print(f"  {nm:10s} " + "  ".join(f"{v:>+10.2f}" for v in c))
    print(f"  {'|value|':10s} " + "  ".join(f"{np.abs(M[:, i]).mean():>10.1f}" for i in range(len(keys)))
          + "   <- what a PILOT does per segment (deg, m, m, m)")
    return {k: float(np.abs(M[:, i]).mean()) for i, k in enumerate(keys)}, M


def main(*runs: str) -> int:
    scale, _ = calibrate()
    for run in runs:
        rows, obj = {}, None
        for f in sorted(glob.glob(os.path.join(run, "logs", "epoch_*", "eval_steer", "plans", "*", "*",
                                               "plan.json"))):
            d = json.load(open(f))
            obj = f"{d.get('objective')}/{d.get('proposal', '').split('(')[0]}"
            rows.setdefault(d["request"], []).append(
                physical(np.load(os.path.join(os.path.dirname(f), "proprio.npy"))))
        print(f"\n=== {os.path.basename(run)}   {obj}   {len(rows)} requests")
        print(f"  {'request':30s} {'wants':>18s} {'achieved':>10s} {'pilot':>8s}  {'% of pilot':>10s}  ok")
        print("  " + "-" * 92)
        hits = []
        for q in sorted(rows):
            v = {k: float(np.mean([r[k] for r in rows[q]])) for k in ("yaw", "altitude", "forward", "lateral")}
            if q not in WANTS:
                continue
            key, sgn = WANTS[q]
            got = v[key] * sgn                       # positive = went the way it was asked
            ok = got > 0.05 * scale[key]             # a real move, not a rounding sign
            hits.append(ok)
            unit = "deg" if key == "yaw" else "m"
            print(f"  {q:30s} {key + ' ' + ('+' if sgn > 0 else '-'):>18s} {got:>+7.2f}{unit:<3s} "
                  f"{scale[key]:>7.2f}  {100 * got / scale[key]:>9.0f}%  {'YES' if ok else 'no'}")
        if hits:
            print(f"  -> {sum(hits)}/{len(hits)} direction requests moved the drone the way they asked")
    print("\n  `achieved` is the IMAGINED trajectory's net motion along the axis the request names, signed so")
    print("  positive = obeyed. `pilot` is what a real flight covers in the same 34 s, so the percentage")
    print("  says whether the plan moved meaningfully or merely in the right direction by a hair.")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
