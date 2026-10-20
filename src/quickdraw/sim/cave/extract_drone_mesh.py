"""Extract the SISL quadrotor mesh from cave-maps.blend to .npz (body + one blade), for the cave control videos.

Reproduces the assets used by `drone_render.py`:
    <prospector>/src/prospector/cave_wm/assets/drone_body.npz   (verts, faces, per-vertex colors incl. SISL logo)
    <prospector>/src/prospector/cave_wm/assets/drone_blade0.npz (one propeller; drone_render places 4 at the arm tips)

WHY this is a direct numpy extraction and not a glTF/OBJ export: bpy's exporters produce empty/corrupt geometry
headless on this box, and `open_mainfile` on the 4.5 GB blend OOMs -- so we lazily `libraries.load` only the drone
objects and read vertices/faces/colors straight out. The 4 `blade0..3` objects in the blend are identical geometry
stacked at the origin (identity transforms), so only blade0 is kept; `drone_render` instances it at the 4 motor
arm-tips it detects from the body. Colors are the Principled base color per material, with the SISL logo panel
sampled per-vertex through its UVs against the logo texture (approximate; a crisp UV-texture path is a TODO).

RUN (needs bpy; use an ephemeral uv env so the shared venv is untouched -- NEVER `uv sync` here):
    uv run --no-project --with bpy --python 3.11 python -m quickdraw.sim.cave.extract_drone_mesh
(run from the quickdraw repo root with PYTHONPATH=src, or `python src/quickdraw/sim/cave/extract_drone_mesh.py`).
"""
import os

import numpy as np

BLEND = "/home/ubuntu/user_irw/prospector/src/assets/blender/cave-maps.blend"
OUTDIR = "/home/ubuntu/user_irw/prospector/src/prospector/cave_wm/assets"
TEX = "/home/ubuntu/user_irw/prospector/src/assets/textures/droneBody_diffuse_SISL_Stanford.png"
BODY_NAMES = ["default"] + [f"default.{i:03d}" for i in range(1, 20)]
BLADE_NAMES = ["blade0"]                                     # all 4 blades are identical; placed 4x in drone_render


def main():
    import bpy
    with bpy.data.libraries.load(BLEND, link=False) as (src, dst):
        dst.objects = [n for n in (BODY_NAMES + BLADE_NAMES) if n in src.objects]
    objmap = {o.name: o for o in dst.objects if o is not None}
    img = bpy.data.images.load(TEX)
    W, H = img.size
    PIX = np.array(img.pixels[:], np.float32).reshape(H, W, 4)[:, :, :3]      # row 0 = bottom

    def base_color(o):
        for ms in o.material_slots:
            m = ms.material
            if m and m.use_nodes:
                for nd in m.node_tree.nodes:
                    if nd.type == 'BSDF_PRINCIPLED':
                        c = nd.inputs['Base Color'].default_value
                        return np.array([c[0], c[1], c[2]], np.float32)
        return np.array([0.75, 0.75, 0.78], np.float32)

    def uses_tex(o):
        for ms in o.material_slots:
            m = ms.material
            if m and m.use_nodes:
                for nd in m.node_tree.nodes:
                    if nd.type == 'TEX_IMAGE' and nd.image and 'SISL' in nd.image.name:
                        return True
        return False

    def mesh_arrays(o):
        m = o.data
        m.calc_loop_triangles()
        V = np.array([list(v.co) for v in m.vertices], np.float32)
        F = np.array([list(t.vertices) for t in m.loop_triangles], np.int32)
        C = np.tile(base_color(o), (len(V), 1))
        if uses_tex(o) and m.uv_layers.active is not None:                    # sample the logo per vertex via UV
            uvl = m.uv_layers.active.data
            for poly in m.polygons:
                for li in poly.loop_indices:
                    vi = m.loops[li].vertex_index
                    u, v = uvl[li].uv
                    x = int((u % 1.0) * (W - 1)); y = int((v % 1.0) * (H - 1))
                    C[vi] = PIX[y, x]
        return V, F, C

    def collect(names, path):
        Vs, Fs, Cs, off = [], [], [], 0
        for nm in names:
            o = objmap.get(nm)
            if o is None or o.type != 'MESH':
                continue
            V, F, C = mesh_arrays(o)
            Vs.append(V); Fs.append(F + off); Cs.append(C); off += len(V)
        V = np.concatenate(Vs, 0); F = np.concatenate(Fs, 0); C = np.concatenate(Cs, 0)
        np.savez(path, verts=V, faces=F, colors=C)
        print("SAVED", os.path.basename(path), "verts", len(V), flush=True)

    os.makedirs(OUTDIR, exist_ok=True)
    collect(BODY_NAMES, os.path.join(OUTDIR, "drone_body.npz"))
    collect(BLADE_NAMES, os.path.join(OUTDIR, "drone_blade0.npz"))


if __name__ == "__main__":
    main()
