"""Extract the SISL quadrotor mesh from cave-maps.blend to .npz, for the cave control videos.

Reproduces the assets used by `drone_render.py`:
    <prospector>/.../cave_wm/assets/drone_body.npz   (non-logo panels: verts, faces, per-vertex paint colors)
    <prospector>/.../cave_wm/assets/drone_logo.npz   (SISL-textured panels: verts, faces, UVs -> UV-textured crisp)
    <prospector>/.../cave_wm/assets/drone_blade0.npz (one propeller; drone_render places 4 at the arm tips)

WHY this is a direct numpy extraction and not a glTF/OBJ export: bpy's exporters produce empty/corrupt geometry
headless on this box, and `open_mainfile` on the 4.5 GB blend OOMs -- so we lazily `libraries.load` only the drone
objects and read vertices/faces/colors/UVs straight out. The 4 `blade0..3` objects in the blend are identical
geometry stacked at the origin (identity transforms), so only blade0 is kept; `drone_render` instances it at the 4
motor arm-tips it detects from the body. The SISL logo panel(s) carry UVs into droneBody_diffuse_SISL_Stanford.png
and are kept SEPARATE so drone_render can UV-TEXTURE them crisply (per-vertex sampling at drone scale cannot resolve
the text); the remaining panels get their Principled base color per vertex.

RUN (needs bpy; use an ephemeral uv env so the shared venv is untouched -- NEVER `uv sync` here):
    uv run --no-project --with bpy --python 3.11 python -m quickdraw.sim.cave.extract_drone_mesh
(run from the quickdraw repo root with PYTHONPATH=src, or `python src/quickdraw/sim/cave/extract_drone_mesh.py`).
"""
import os

import numpy as np

BLEND = "/home/ubuntu/user_irw/prospector/src/assets/blender/cave-maps.blend"
OUTDIR = "/home/ubuntu/user_irw/prospector/src/prospector/cave_wm/assets"
PANEL_NAMES = ["default"] + [f"default.{i:03d}" for i in range(1, 20)]
BLADE_NAMES = ["blade0"]                                     # all 4 blades are identical; placed 4x in drone_render


def main():
    import bpy
    with bpy.data.libraries.load(BLEND, link=False) as (src, dst):
        dst.objects = [n for n in (PANEL_NAMES + BLADE_NAMES) if n in src.objects]
    objmap = {o.name: o for o in dst.objects if o is not None}

    def base_color(o):
        for ms in o.material_slots:
            m = ms.material
            if m and m.use_nodes:
                for nd in m.node_tree.nodes:
                    if nd.type == 'BSDF_PRINCIPLED':
                        c = nd.inputs['Base Color'].default_value
                        return np.array([c[0], c[1], c[2]], np.float32)
        return np.array([0.75, 0.75, 0.78], np.float32)

    def uses_sisl(o):
        for ms in o.material_slots:
            m = ms.material
            if m and m.use_nodes:
                for nd in m.node_tree.nodes:
                    if nd.type == 'TEX_IMAGE' and nd.image and 'SISL' in nd.image.name:
                        return True
        return False

    def mesh_arrays(o, want_uv):
        """verts (V,3), faces (F,3), and either per-vertex colors (V,3) or per-vertex UVs (V,2)."""
        m = o.data
        m.calc_loop_triangles()
        V = np.array([list(v.co) for v in m.vertices], np.float32)
        F = np.array([list(t.vertices) for t in m.loop_triangles], np.int32)
        if not want_uv:
            C = np.tile(base_color(o), (len(V), 1))
            return V, F, C
        uv = np.zeros((len(V), 2), np.float32)                       # UV is per-loop; collapse to per-vertex
        if m.uv_layers.active is not None:
            uvl = m.uv_layers.active.data
            for poly in m.polygons:
                for li in poly.loop_indices:
                    uv[m.loops[li].vertex_index] = uvl[li].uv
        return V, F, uv

    def collect(names, want_uv, path):
        Vs, Fs, Xs, off = [], [], [], 0
        for nm in names:
            o = objmap.get(nm)
            if o is None or o.type != 'MESH':
                continue
            V, F, X = mesh_arrays(o, want_uv)
            Vs.append(V); Fs.append(F + off); Xs.append(X); off += len(V)
        if not Vs:
            print("NONE for", os.path.basename(path), flush=True); return
        V = np.concatenate(Vs, 0); F = np.concatenate(Fs, 0); X = np.concatenate(Xs, 0)
        np.savez(path, verts=V, faces=F, **({"uv": X} if want_uv else {"colors": X}))
        print("SAVED", os.path.basename(path), "verts", len(V), "uv" if want_uv else "colors", flush=True)

    logo = [n for n in PANEL_NAMES if n in objmap and objmap[n].type == 'MESH' and uses_sisl(objmap[n])]
    body = [n for n in PANEL_NAMES if n not in logo]
    print("logo panels:", logo, flush=True)
    os.makedirs(OUTDIR, exist_ok=True)
    collect(body, False, os.path.join(OUTDIR, "drone_body.npz"))
    collect(logo, True, os.path.join(OUTDIR, "drone_logo.npz"))
    collect(BLADE_NAMES, False, os.path.join(OUTDIR, "drone_blade0.npz"))


if __name__ == "__main__":
    main()
