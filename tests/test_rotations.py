"""Smoke test for the 6D rotation encoding against real lego_assemblies frames.

    python tests/test_rotations.py [DATA_ROOT]

Checks the two properties the encoding exists for -- sign-invariance (q and -q encode identically)
and continuity (the Lipschitz bound ||d6D|| <= 2*sqrt(2)*sin(theta/2), which Euler violates by up to
359.8deg for a 0.38deg real rotation) -- plus round-trip fidelity and Gram-Schmidt robustness."""
import glob, os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
DATA = sys.argv[1] if len(sys.argv) > 1 else "/home/isaac/data/lego_assemblies"
from scipy.spatial.transform import Rotation as R
from quickdraw.data.rotations import (encode_state, decode_state, encode_action, decode_action,
                                      quat_to_6d, sixd_to_quat, sixd_to_matrix, EULER_SEQ)

files = sorted(glob.glob(os.path.join(DATA, "data", "chunk-000", "*.parquet")))
if not files: sys.exit(f"no parquets under {DATA}")
S,A=[],[]
for f in files[::6]:                       # every 6th episode -> 13 episodes, all sessions represented
    t=pd.read_parquet(f,columns=["observation.state","action"])
    S.append(np.stack(t["observation.state"].to_numpy())); A.append(np.stack(t["action"].to_numpy()))
S,A=np.concatenate(S).astype(np.float64),np.concatenate(A).astype(np.float64)
print(f"smoke set: {len(S)} frames / {len(files[::6])} episodes\n")
ok=True
def chk(name, val, tol):
    global ok
    p = val <= tol; ok &= p
    print(f"  [{'PASS' if p else 'FAIL'}] {name:52s} {val:.3e}  (tol {tol:.0e})")

print("1. shapes")
s6, a6 = encode_state(S), encode_action(A)
chk("state 28 -> 34", abs(s6.shape[1]-34), 0); chk("action 16 -> 20", abs(a6.shape[1]-20), 0)

print("\n2. round-trip preserves the ROTATION (geodesic angle, degrees)")
Sd, Ad = decode_state(s6), decode_action(a6)
for arm,off in (("right",3),("left",17)):
    e0=R.from_euler(EULER_SEQ,S[:,off:off+3],degrees=True); e1=R.from_euler(EULER_SEQ,Sd[:,off:off+3],degrees=True)
    chk(f"state {arm} max geodesic err (deg)", np.degrees((e0.inv()*e1).magnitude()).max(), 1e-4)
for arm,off in (("right",3),("left",11)):
    q0=R.from_quat(A[:,off:off+4]); q1=R.from_quat(Ad[:,off:off+4])
    chk(f"action {arm} max geodesic err (deg)", np.degrees((q0.inv()*q1).magnitude()).max(), 1e-4)

print("\n3. non-rotation channels pass through exactly")
chk("state xyz+joints+grip max abs err", np.abs(Sd[:,[*range(0,3),*range(6,17),*range(20,28)]]
                                                - S[:,[*range(0,3),*range(6,17),*range(20,28)]]).max(), 1e-3)
chk("action xyz+grip max abs err", np.abs(Ad[:,[0,1,2,7,8,9,10,15]]-A[:,[0,1,2,7,8,9,10,15]]).max(), 1e-3)

print("\n4. THE POINT: sign-invariance -- q and -q must encode identically")
q=A[:,3:7]
chk("max |6d(q) - 6d(-q)|", np.abs(quat_to_6d(q)-quat_to_6d(-q)).max(), 1e-12)

def jumps(x, thr):    # per-episode, so episode joins are not counted
    n=0; j=0
    for f in files[::6]:
        T=len(pd.read_parquet(f,columns=["action"])); seg=x[j:j+T]; j+=T
        n+=int((np.linalg.norm(np.diff(seg,axis=0),axis=1)>thr).sum())
    return n

print("\n5. THE POINT: continuity -- the Lipschitz bound ||d6D|| <= 2*sqrt(2)*sin(theta/2)")
# ||R1-R2||_F = 2*sqrt(2)*sin(theta/2) exactly; 6D is two of the three columns, so it can only ever
# be SMALLER. If this bound holds for every pair, the encoding provably cannot jump unless the
# underlying rotation jumped -- no thresholds, no tuning. Euler obeys no such bound: that is the bug.
i=0; worst6d=0.0; worst_eul=(0.0,0.0); N=0; big=0
for f in files[::6]:
    T=len(pd.read_parquet(f,columns=["action"]))
    se,ae=S[i:i+T],A[i:i+T]; s6e,a6e=s6[i:i+T],a6[i:i+T]; i+=T; N+=T-1
    for rot, enc in ((R.from_euler(EULER_SEQ,se[:,3:6],degrees=True), s6e[:,3:9]),
                     (R.from_quat(ae[:,3:7]),                        a6e[:,3:9])):
        th=(rot[:-1].inv()*rot[1:]).magnitude()                     # radians
        bound=2*np.sqrt(2)*np.sin(th/2)
        worst6d=max(worst6d, float((np.linalg.norm(np.diff(enc,axis=0),axis=1)-bound).max()))
    # Euler's counterexample: largest rpy step that corresponds to a NEGLIGIBLE real rotation
    th_s=np.degrees((R.from_euler(EULER_SEQ,se[:-1,3:6],degrees=True).inv()
                     *R.from_euler(EULER_SEQ,se[1:,3:6],degrees=True)).magnitude())
    drpy=np.linalg.norm(np.diff(se[:,3:6],axis=0),axis=1)
    m=th_s<1.0                                                       # rotation essentially still
    big+=int((drpy[m]>10).sum())
    if m.any() and drpy[m].max()>worst_eul[0]: worst_eul=(float(drpy[m].max()), float(th_s[m][drpy[m].argmax()]))
print(f"       {N} consecutive pairs")
print(f"       6D   : worst violation of the bound = {worst6d:.3e}  (<=0 means it always holds)")
print(f"       Euler: {big} steps where rpy moves >10deg while the rotation moves <1deg")
print(f"              worst case: rpy jumps {worst_eul[0]:.1f}deg for a real rotation of {worst_eul[1]:.4f}deg")
chk("6D obeys the Lipschitz bound everywhere", max(worst6d,0.0), 1e-6)

print("\n6. decoded 6D is a valid rotation matrix (orthonormal, det=+1)")
m=sixd_to_matrix(a6[:,3:9].astype(np.float64))
chk("max |M^T M - I|", np.abs(np.einsum('nji,njk->nik',m,m)-np.eye(3)).max(), 1e-9)
chk("max |det - 1|", np.abs(np.linalg.det(m)-1).max(), 1e-9)

print("\n7. Gram-Schmidt is robust to un-normalised / non-orthogonal input")
rng=np.random.default_rng(0); d=a6[:2000,3:9].astype(np.float64)
noisy=d*rng.uniform(0.3,3.0,(2000,1))+rng.normal(0,0.05,d.shape)
m2=sixd_to_matrix(noisy)
chk("noisy input still orthonormal", np.abs(np.einsum('nji,njk->nik',m2,m2)-np.eye(3)).max(), 1e-9)

print("\n" + ("ALL CHECKS PASSED" if ok else "*** FAILURES ABOVE ***"))
sys.exit(0 if ok else 1)
