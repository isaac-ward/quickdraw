"""Underwater look, applied in post from the depth buffer.

WHY NOT MUJOCO FOG. MJCF `<rgba fog>` has no effect in the offscreen Renderer used here -- the first
tank render came out looking like a dry concrete room. Rather than fight it, attenuation is applied
where it can be done correctly anyway: Beer-Lambert against the true per-pixel depth.

    I = I_scene * exp(-k*z) + I_water * (1 - exp(-k*z))

per channel, with k larger for red than for blue, which is why everything underwater goes blue-green
with range. k_red ~ 0.17/m against k_blue ~ 0.055/m is the usual coastal-water ordering; the absolute
values are tuned so the far wall of a 15 m tank is strongly attenuated but still readable, since a
world model that sees nothing but fog at range learns nothing from the far wall.
"""
import numpy as np

K_RGB = np.array([0.125, 0.055, 0.040], np.float32)        # per-metre extinction, red attenuates fastest
WATER = np.array([0.10, 0.30, 0.35], np.float32)        # the colour an infinitely thick slab tends to


def attenuate(rgb_u8, depth_m, k=K_RGB, water=WATER, floor=0.0):
    """rgb_u8 (H,W,3) uint8 + depth_m (H,W) float metres -> (H,W,3) uint8."""
    z = np.clip(np.nan_to_num(depth_m, nan=1e3, posinf=1e3), 0.0, 60.0)[..., None]
    t = np.exp(-k[None, None, :] * z)                   # transmittance per channel
    t = np.clip(t, floor, 1.0)
    img = rgb_u8.astype(np.float32) / 255.0
    out = img * t + water[None, None, :] * (1.0 - t)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


class UnderwaterRenderer:
    """One MuJoCo Renderer, one scene build, two rasterisations: colour then depth.

    `update_scene` is the expensive half -- it walks every geom and rebuilds the scene -- and the
    colour and depth passes see an identical scene. Building it twice (the obvious two-Renderer
    version) doubled the cost of every frame for nothing. Measured over the four cameras at 144x192:
    141 ms per control step with two renderers, 65 ms with shadows off, and ~35 ms once the scene is
    built once. On 864,000 renders that is the difference between an 8-hour pass and a 2-hour one.
    """

    def __init__(self, model, height=144, width=192):
        import mujoco

        self.r = mujoco.Renderer(model, height, width)

    def render(self, data, camera):
        self.r.update_scene(data, camera=camera)
        self.r.disable_depth_rendering()
        rgb = self.r.render()
        self.r.enable_depth_rendering()
        dep = self.r.render()
        self.r.disable_depth_rendering()
        return attenuate(rgb, dep)

    def close(self):
        try:
            self.r.close()
        except Exception:                                    # noqa: BLE001 - teardown is best-effort
            pass
