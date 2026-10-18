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


# DECIMATION TARGETS. The raw MOLA is 84,619 faces and the diver 56,454, and MuJoCo rebuilds the whole
# scene on every `update_scene` -- which made a 144x192 render take 15 ms and put the full dataset at
# 7.4 hours of pure rendering. At 144x192 the AUV spans a couple of hundred pixels; a few thousand
# faces is already more detail than the camera can resolve.
FACES = {"mola": 4000, "dock": 460, "diver": 3000}


def convert(names=("mola", "dock", "diver"), src=HERE, log=print, decimate=True):
    out = {}
    for name in names:
        g = trimesh.load(src / name / "scene.gltf", force="scene")
        m = g.to_mesh() if hasattr(g, "to_mesh") else g
        ext = m.bounds[1] - m.bounds[0]
        m.apply_translation(-m.bounds.mean(axis=0))
        raw = len(m.faces)
        tgt = FACES.get(name, raw)
        if decimate and raw > tgt * 1.1:
            m = m.simplify_quadric_decimation(face_count=tgt)
        d = src / name
        m.export(d / f"{name}.obj")
        out[name] = {"faces": int(len(m.faces)), "raw_faces": int(raw),
                     "bbox": [round(float(v), 3) for v in ext]}
        log(f"  {name:6s} {raw:7d} -> {len(m.faces):6d} faces  bbox {np.round(ext, 3)}")
    return out


if __name__ == "__main__":
    convert()
