"""glTF -> OBJ for MuJoCo, which reads obj/stl/msh and not gltf.

Lives under src/ because that is a MOUNTED, TRACKED path: the repo root is not bind-mounted into the
container (see docker-compose volumes), so assets anywhere else exist only in the container's writable
layer and vanish with it. These meshes ARE the environment definition, so they belong in the package.

Each scene is merged to one mesh, recentred on its geometric centre -- so the MJCF sets pose and scale
explicitly instead of inheriting whatever origin the artist used -- and written into its OWN directory.
Per-model directories matter: trimesh writes a `material.mtl` beside the .obj, and exporting three
models into one directory silently leaves all three .obj files pointing at the last model's material.
"""
import pathlib

import numpy as np
import trimesh

HERE = pathlib.Path(__file__).parent / "assets"


def convert(names=("mola", "dock", "diver"), src=HERE, log=print):
    out = {}
    for name in names:
        g = trimesh.load(src / name / "scene.gltf", force="scene")
        m = g.to_mesh() if hasattr(g, "to_mesh") else g
        ext = m.bounds[1] - m.bounds[0]
        m.apply_translation(-m.bounds.mean(axis=0))
        d = src / name
        m.export(d / f"{name}.obj")
        out[name] = {"faces": int(len(m.faces)), "bbox": [round(float(v), 3) for v in ext]}
        log(f"  {name:6s} {len(m.faces):7d} faces  bbox {np.round(ext, 3)}  -> {name}/{name}.obj")
    return out


if __name__ == "__main__":
    convert()
