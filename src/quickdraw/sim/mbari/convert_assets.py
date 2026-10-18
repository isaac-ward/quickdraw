"""glTF -> OBJ for MuJoCo, which reads obj/stl/msh and not gltf.

Each scene is merged to a single mesh per material group, recentred on its own origin and reported in
metres so the MJCF can scale it deliberately rather than by trial and error."""
import pathlib
import numpy as np
import trimesh

SRC = pathlib.Path("/app/assets/mbari")
for name in ("mola", "dock", "diver"):
    g = trimesh.load(SRC / name / "scene.gltf", force="scene")
    m = g.to_mesh() if hasattr(g, "to_mesh") else g
    ext = m.bounds[1] - m.bounds[0]
    print(f"  {name:6s} {len(m.faces):7d} faces  bbox {np.round(ext, 3)}  "
          f"longest {ext.max():.3f} units")
    m.apply_translation(-m.bounds.mean(axis=0))          # recentre on the geometric centre
    out = SRC / f"{name}.obj"
    m.export(out)
    print(f"         -> {out.name}  {out.stat().st_size/1e6:.1f} MB")
