"""glTF -> per-colour-group OBJ for MuJoCo, which reads obj/stl/msh and not gltf.

TWO THINGS THE FIRST VERSION GOT WRONG, both visible only once the mesh was actually rendered and
looked at:

  IT MERGED EVERY PART INTO ONE MESH AND PAINTED IT ONE COLOUR. The MOLA model is not textured -- it
  is 14 parts carrying six `baseColorFactor` materials: a yellow hull, a BLACK frame that is 32,939 of
  its 84,619 faces, cyan accents and grey fittings. Flattening that to a single yellow geom threw away
  most of what the vehicle looks like. Each colour group is now its own OBJ, its own MuJoCo mesh and
  its own material, so the AUV renders in its real colours.

  IT DECIMATED 84,619 -> 5,586 FACES, which destroyed it. Rendered side by side, the full mesh is a
  cylindrical hull in a frame with corner thrusters and the 5.6k version is a scatter of struts. The
  budget is now 28,000 total, split across groups in proportion to their size with a floor so small
  parts survive -- a 5x speedup over the raw mesh that still looks like the vehicle.

DECIMATION IS PER CONNECTED COMPONENT, not per group, and that distinction is the whole thing.
Quadric decimation on a mesh made of many small disconnected pieces -- six thruster pods merged into
one "yellow" group, say -- spends the budget on whichever piece has the most triangles and collapses
the rest to nothing. Rendered group by group, the hull group came back as a scatter of fragments
while the single-surface frame survived intact. Splitting into components first and giving each its
own floor keeps every piece.
"""
import pathlib
from collections import defaultdict

import numpy as np
import trimesh

HERE = pathlib.Path(__file__).parent / "assets"
BUDGET = {"mola": 34000, "dock": 460, "diver": 14000}
MIN_FACES = 300


def _colour_groups(scene):
    """{rgba: [transformed Trimesh]} from the scene graph, transforms applied."""
    groups = defaultdict(list)
    for node in scene.graph.nodes_geometry:
        T, gname = scene.graph[node]
        geo = scene.geometry[gname]
        if not isinstance(geo, trimesh.Trimesh):
            continue                                   # Path3D curves etc. are not renderable
        mat = getattr(geo.visual, "material", None)
        f = getattr(mat, "baseColorFactor", None)
        if f is None:
            f = getattr(mat, "main_color", None)
        if f is None:
            rgba = (0.75, 0.75, 0.75, 1.0)
        else:
            arr = np.asarray(f, float).ravel()[:4]
            if arr.max() > 1.5:
                arr = arr / 255.0
            rgba = tuple(float(round(v, 3)) for v in arr)
        m = geo.copy()
        m.apply_transform(T)
        groups[rgba].append(m)
    return groups


def _decimate_components(mesh, ratio):
    """Decimate each connected component to `ratio` of its own size, with a floor, then recombine."""
    if ratio >= 0.999:
        return mesh
    try:
        parts = mesh.split(only_watertight=False)
    except Exception:                                  # noqa: BLE001 - split is best-effort
        parts = [mesh]
    if len(parts) <= 1:
        tgt = max(MIN_FACES, int(len(mesh.faces) * ratio))
        return mesh.simplify_quadric_decimation(face_count=tgt) if len(mesh.faces) > tgt * 1.1 else mesh
    out = []
    for p in parts:
        tgt = max(MIN_FACES, int(len(p.faces) * ratio))
        out.append(p.simplify_quadric_decimation(face_count=tgt) if len(p.faces) > tgt * 1.1 else p)
    return trimesh.util.concatenate(out)


def convert(names=("mola", "dock", "diver"), src=HERE, log=print):
    """Write <name>/<name>_g<k>.obj per colour group. Returns {name: [(file, rgba, faces)]}."""
    out = {}
    for name in names:
        scene = trimesh.load(src / name / "scene.gltf", force="scene")
        groups = _colour_groups(scene)
        merged = {rgba: trimesh.util.concatenate(ms) for rgba, ms in groups.items()}
        total = sum(len(m.faces) for m in merged.values())
        centre = np.mean([m.bounds.mean(axis=0) for m in merged.values()], axis=0)
        budget = BUDGET.get(name, total)
        ratio = min(1.0, budget / max(1, total))
        rows = []
        for k, (rgba, m) in enumerate(sorted(merged.items(), key=lambda kv: -len(kv[1].faces))):
            m = m.copy()
            m.apply_translation(-centre)               # ONE shared centre: per-group centring would
            #                                            scatter the parts off their own assembly
            m = _decimate_components(m, ratio)
            p = src / name / f"{name}_g{k}.obj"
            m.export(p)
            # STRIP THE MTL. trimesh writes a material file beside the .obj and MuJoCo's OBJ loader
            # honours it OVER the <material> named in the MJCF -- which is why every part rendered
            # white-silver no matter what rgba the geom asked for. Colour belongs to the MJCF here,
            # one material per colour group, so the sidecar is removed and the obj's mtllib line with
            # it. (The glTF remains the source of truth for what those colours are.)
            mtl = p.with_suffix(".obj.mtl")
            for cand in (mtl, p.parent / "material.mtl", p.with_name(p.stem + ".mtl")):
                if cand.exists():
                    cand.unlink()
            txt = [ln for ln in p.read_text().splitlines()
                   if not ln.startswith(("mtllib", "usemtl"))]
            p.write_text("\n".join(txt) + "\n")
            rows.append((p.name, rgba, int(len(m.faces))))
        out[name] = rows
        kept = sum(r[2] for r in rows)
        log(f"  {name:6s} {total:7d} -> {kept:6d} faces in {len(rows)} colour groups")
        for fn, rgba, nf in rows:
            log(f"      {fn:16s} rgba {rgba}  {nf:6d} faces")
    return out


if __name__ == "__main__":
    convert()
