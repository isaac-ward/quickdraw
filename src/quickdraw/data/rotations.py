"""Continuous 6D rotation encoding for bimanual TCP poses (Zhou et al., CVPR 2019).

Every representation of SO(3) in four or fewer dimensions is discontinuous somewhere, so a network
asked to regress one has to learn a function with a jump in it. The two we get handed here both
carry that defect:

  * `observation.state` stores TCP orientation as roll/pitch/yaw degrees. Euler wraps at +-180 and
    degenerates at pitch=+-90. On `swoosh-data/lego_assemblies` that is not hypothetical -- roll
    wraps 702 (right) / 802 (left) times and yaw 48 / 30 times across 341,494 frames, because pitch
    reaches +-78deg and pushes roll/yaw toward the gimbal singularity.
  * `action` stores it as a quaternion. Quaternions double-cover SO(3) (q and -q are the same
    rotation), so the sign is arbitrary. The recorded stream happens to be nearly continuous -- only
    29 sign flips per arm -- but the rotations span essentially all of SO(3) (max geodesic distance
    from the mean is 172deg right / 180deg left; 5-6.5% of frames lie beyond 90deg), so no single
    hemisphere covers the data and the ambiguity cannot be canonicalised away.

The 6D encoding is the first two columns of the rotation matrix. It is continuous everywhere, has no
singularity, is invariant to quaternion sign by construction, and -- unlike a raw quaternion with an
arbitrary sign -- can be averaged and interpolated, which matters if the action model predicts chunks.
Decoding is Gram-Schmidt: normalise the first column, remove its projection from the second, and take
the cross product for the third.

Encoded layouts (per arm: xyz, then 6D, then the joints/gripper the source already provides):

    state   28 -> 34    [tcp_xyz(3) 6d(6) j1..j7(7) grip(1)] x {right, left}
    action  16 -> 20    [xyz(3) 6d(6) grip(1)] x {right, left}
"""

from __future__ import annotations

import numpy as np

# xArm's SDK reports TCP orientation as RPY, i.e. extrinsic x-y-z (R = Rz(yaw) @ Ry(pitch) @ Rx(roll)).
# scipy spells extrinsic in lowercase. NOTE: this could not be confirmed against `action` on
# lego_assemblies, because that column is not in the state's frame -- see `docs/` / the dataset
# discussion. Override if a future dataset says otherwise.
EULER_SEQ = "xyz"

STATE_DIM, STATE_DIM_6D = 28, 34
ACTION_DIM, ACTION_DIM_6D = 16, 20


def matrix_to_6d(m: np.ndarray) -> np.ndarray:
    """(..., 3, 3) rotation matrices -> (..., 6): the first two COLUMNS, stacked."""
    return np.concatenate([m[..., :, 0], m[..., :, 1]], axis=-1)


def sixd_to_matrix(d: np.ndarray) -> np.ndarray:
    """(..., 6) -> (..., 3, 3) via Gram-Schmidt. Accepts any non-degenerate 6-vector."""
    a, b = d[..., :3], d[..., 3:]
    c1 = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b - (c1 * b).sum(-1, keepdims=True) * c1          # orthogonalise against c1
    c2 = b / np.linalg.norm(b, axis=-1, keepdims=True)
    c3 = np.cross(c1, c2)
    return np.stack([c1, c2, c3], axis=-1)                 # columns, matching matrix_to_6d


def quat_to_6d(q: np.ndarray) -> np.ndarray:
    """(..., 4) xyzw quaternions -> (..., 6). Sign-invariant: q and -q give the same output."""
    from scipy.spatial.transform import Rotation as R
    return matrix_to_6d(R.from_quat(q.reshape(-1, 4)).as_matrix()).reshape(*q.shape[:-1], 6)


def sixd_to_quat(d: np.ndarray) -> np.ndarray:
    """(..., 6) -> (..., 4) xyzw quaternions."""
    from scipy.spatial.transform import Rotation as R
    m = sixd_to_matrix(d.reshape(-1, 6))
    return R.from_matrix(m).as_quat().reshape(*d.shape[:-1], 4)


def euler_to_6d(rpy_deg: np.ndarray, seq: str = EULER_SEQ) -> np.ndarray:
    """(..., 3) roll/pitch/yaw DEGREES -> (..., 6). Removes wraparound and gimbal degeneracy."""
    from scipy.spatial.transform import Rotation as R
    m = R.from_euler(seq, rpy_deg.reshape(-1, 3), degrees=True).as_matrix()
    return matrix_to_6d(m).reshape(*rpy_deg.shape[:-1], 6)


def sixd_to_euler(d: np.ndarray, seq: str = EULER_SEQ) -> np.ndarray:
    """(..., 6) -> (..., 3) roll/pitch/yaw DEGREES (wrapped back into scipy's output ranges)."""
    from scipy.spatial.transform import Rotation as R
    m = sixd_to_matrix(d.reshape(-1, 6))
    return R.from_matrix(m).as_euler(seq, degrees=True).reshape(*d.shape[:-1], 3)


def encode_state(s: np.ndarray, seq: str = EULER_SEQ) -> np.ndarray:
    """(T, 28) xarm7_bimanual state -> (T, 34), rpy replaced in place by 6D.

    Per arm in: tcp_xyz(3) rpy(3) j1..j7(7) grip(1) = 14.  Out: tcp_xyz(3) 6d(6) j1..j7(7) grip(1) = 17."""
    s = np.asarray(s, dtype=np.float64)
    if s.shape[-1] != STATE_DIM:
        raise ValueError(f"expected (..., {STATE_DIM}) state, got {s.shape}")
    out = []
    for off in (0, 14):                                    # right arm at 0, left arm at 14
        out += [s[..., off:off + 3], euler_to_6d(s[..., off + 3:off + 6], seq), s[..., off + 6:off + 14]]
    return np.concatenate(out, axis=-1).astype(np.float32)


def decode_state(s6: np.ndarray, seq: str = EULER_SEQ) -> np.ndarray:
    """(T, 34) -> (T, 28). Inverse of `encode_state` up to Euler wrapping (same ROTATION, maybe not
    the same rpy triple: -180 and +180 both decode to one of the two)."""
    s6 = np.asarray(s6, dtype=np.float64)
    if s6.shape[-1] != STATE_DIM_6D:
        raise ValueError(f"expected (..., {STATE_DIM_6D}) state, got {s6.shape}")
    out = []
    for off in (0, 17):
        out += [s6[..., off:off + 3], sixd_to_euler(s6[..., off + 3:off + 9], seq), s6[..., off + 9:off + 17]]
    return np.concatenate(out, axis=-1).astype(np.float32)


def encode_action(a: np.ndarray) -> np.ndarray:
    """(T, 16) bimanual action -> (T, 20), quaternion replaced in place by 6D.

    Per arm in: xyz(3) quat_xyzw(4) grip(1) = 8.  Out: xyz(3) 6d(6) grip(1) = 10."""
    a = np.asarray(a, dtype=np.float64)
    if a.shape[-1] != ACTION_DIM:
        raise ValueError(f"expected (..., {ACTION_DIM}) action, got {a.shape}")
    out = []
    for off in (0, 8):
        out += [a[..., off:off + 3], quat_to_6d(a[..., off + 3:off + 7]), a[..., off + 7:off + 8]]
    return np.concatenate(out, axis=-1).astype(np.float32)


def decode_action(a6: np.ndarray) -> np.ndarray:
    """(T, 20) -> (T, 16). Inverse of `encode_action` up to quaternion SIGN (same rotation)."""
    a6 = np.asarray(a6, dtype=np.float64)
    if a6.shape[-1] != ACTION_DIM_6D:
        raise ValueError(f"expected (..., {ACTION_DIM_6D}) action, got {a6.shape}")
    out = []
    for off in (0, 10):
        out += [a6[..., off:off + 3], sixd_to_quat(a6[..., off + 3:off + 9]), a6[..., off + 9:off + 10]]
    return np.concatenate(out, axis=-1).astype(np.float32)
