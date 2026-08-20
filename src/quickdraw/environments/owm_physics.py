"""OWM-ISS chaser translational-dynamics prior — ENVIRONMENT-SPECIFIC (owm-iss only), not a general model lever.

VERIFIED against recorded coop data (2026-08-20), in the OBS (LVLH) frame the WM operates in:
    a = R(q) · (F / mass)            q = obs quaternion [w,x,y,z], body->obs, +sign
    Δv = a · raw_dt                  raw_dt = 0.05 s ; action force is the SUMMED force over the subsample window
    v' = v + Δv
    p' = p + v' · dt_eff             dt_eff = subsample · raw_dt (0.25 s at subsample 5)
On subsampled coop data: cosine(Δv_pred, Δv_obs) = 0.996, magnitude ratio 1.00. mass = 12000 kg.

Intended use (staged): RecordedEnv exposes this as `dynamics_prior(prev_obs, action)` gated by
`environments.dynamics_prior=true`; the WM then predicts a RESIDUAL on top (next = prior + residual), making the
action->motion response correct by construction. Wiring into the model is a separate, supervised step.
Run `python -m quickdraw.environments.owm_physics` for the standalone data-verification self-test.
"""
import torch

MASS = 12000.0
RAW_DT = 0.05


def quat_rotate_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by quaternion q=[w,x,y,z] (body->world): v' = q ⊗ v ⊗ q*  (vectorized, batched)."""
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-9)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # rotation matrix rows applied to v (standard [w,x,y,z] -> R body->world)
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    rx = (1 - 2 * (y * y + z * z)) * vx + 2 * (x * y - w * z) * vy + 2 * (x * z + w * y) * vz
    ry = 2 * (x * y + w * z) * vx + (1 - 2 * (x * x + z * z)) * vy + 2 * (y * z - w * x) * vz
    rz = 2 * (x * z - w * y) * vx + 2 * (y * z + w * x) * vy + (1 - 2 * (x * x + y * y)) * vz
    return torch.stack([rx, ry, rz], dim=-1)


def dynamics_prior(prev_obs: torch.Tensor, action: torch.Tensor, pos_idx, vel_idx, quat_idx,
                   mass: float = MASS, raw_dt: float = RAW_DT, dt_eff: float = 0.25) -> torch.Tensor:
    """Physics next-state (pos/vel updated; quat/bodyrate copied — WM residual handles their evolution).
    prev_obs (...,obs_dim) ABSOLUTE; action (...,act_dim) raw force in N (action[...,0:3])."""
    pos_idx, vel_idx, quat_idx = list(pos_idx), list(vel_idx), list(quat_idx)
    q = prev_obs[..., quat_idx]
    F = action[..., 0:3]
    dv = quat_rotate_wxyz(q, F / mass) * raw_dt
    v_new = prev_obs[..., vel_idx] + dv
    p_new = prev_obs[..., pos_idx] + v_new * dt_eff
    out = prev_obs.clone()
    out[..., vel_idx] = v_new
    out[..., pos_idx] = p_new
    return out


if __name__ == "__main__":  # standalone data-verification self-test on subsampled coop data
    import glob, numpy as np, pyarrow.parquet as pq
    FILES = sorted(glob.glob("logs/recorded_hf/owm-iss-numerical-v1-coop-goal-dt50ms-500k/**/data/chunk-*/file-*.parquet", recursive=True))
    K = 5
    # ego-13 obs layout after obs_keep[2..14]: pos 0:3, vel 3:6, quat 6:10 (wxyz), bodyrate 10:13
    POS, VEL, QUAT = [0, 1, 2], [3, 4, 5], [6, 7, 8, 9]
    cos, rat = [], []
    for f in FILES[:2]:
        t = pq.read_table(f, columns=["observation_vector", "action", "episode_index"])
        ei = t.column("episode_index").to_numpy(); ov = t.column("observation_vector"); ac = t.column("action")
        for e in np.unique(ei)[:20]:
            idx = np.where(ei == e)[0]
            o = np.stack([ov[int(i)].as_py() for i in idx])[:, 2:15]      # ego-13
            a = np.stack([ac[int(i)].as_py() for i in idx])
            T = len(o) - (len(o) % K)
            for s in range(0, T - K, K):
                prev = torch.tensor(o[s], dtype=torch.float32)
                act = torch.tensor(a[s:s + K, 0:3].sum(0), dtype=torch.float32)   # SUMMED force
                nxt = dynamics_prior(prev[None], torch.cat([act, torch.zeros(3)])[None], POS, VEL, QUAT)[0]
                dv_pred = (nxt[VEL] - prev[VEL]).numpy()
                dv_obs = o[s + K, 3:6] - o[s, 3:6]
                if np.linalg.norm(dv_pred) > 0.02:
                    cos.append(float(np.dot(dv_pred, dv_obs) / (np.linalg.norm(dv_pred) * np.linalg.norm(dv_obs) + 1e-9)))
                    rat.append(float(np.linalg.norm(dv_pred) / (np.linalg.norm(dv_obs) + 1e-9)))
    print(f"self-test on {len(cos)} steps: mean cos {np.mean(cos):.3f}  median magratio {np.median(rat):.2f}  (expect ~0.99 / ~1.0)")
