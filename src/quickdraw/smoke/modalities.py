"""Greenlight smoke for the modality registry (models/modalities.py). Checks that proprio (vector) and
image modalities encode obs -> fixed token bags and decode back to the right shapes, for both (B,) and
(B,T) leading dims, and that toggling a modality out of the config drops it. Run:
  uv run python -m quickdraw.smoke.modalities
"""
import torch

from quickdraw.models.modalities import ModalitySpec, build_modalities

DEV = "cuda" if torch.cuda.is_available() else "cpu"
R = []


def check(name, cond, extra=""):
    R.append(bool(cond))
    print(f"[{'OK' if cond else 'FAIL'}] {name}{(' — ' + extra) if extra else ''}")


def main():
    torch.manual_seed(0)
    d, B, T = 256, 2, 4
    specs = [ModalitySpec("proprio", "vector", dim=6),
             ModalitySpec("image", "image", img_size=128, patch=16, num_tokens=8)]
    mods = build_modalities(specs, d).to(DEV)
    check("registry order preserved (proprio, image)", list(mods.keys()) == ["proprio", "image"])
    check("token counts (proprio=1, image=8)", mods["proprio"].n_tokens == 1 and mods["image"].n_tokens == 8)

    proprio = torch.randn(B, T, 6, device=DEV)
    image = torch.rand(B, T, 128, 128, 3, device=DEV)
    # (B,T) leading dims
    zp, zi = mods["proprio"].encode(proprio), mods["image"].encode(image)
    check("proprio encode (B,T,1,d)", zp.shape == (B, T, 1, d), str(tuple(zp.shape)))
    check("image encode (B,T,8,d)", zi.shape == (B, T, 8, d), str(tuple(zi.shape)))
    rp, ri = mods["proprio"].decode(zp), mods["image"].decode(zi)
    check("proprio decode (B,T,6)", rp.shape == (B, T, 6), str(tuple(rp.shape)))
    check("image decode (B,T,128,128,3)", ri.shape == (B, T, 128, 128, 3), str(tuple(ri.shape)))
    # (B,) leading dim (single step)
    zp1 = mods["proprio"].encode(proprio[:, 0])
    zi1 = mods["image"].encode(image[:, 0])
    check("single-step proprio encode (B,1,d)", zp1.shape == (B, 1, d))
    check("single-step image encode (B,8,d)", zi1.shape == (B, 8, d))

    # ablation: drop the image modality via config -> registry has only proprio, total bag = 1 token
    only_proprio = build_modalities([ModalitySpec("proprio", "vector", dim=6)], d).to(DEV)
    total = sum(m.n_tokens for m in only_proprio.values())
    check("disable image in config -> proprio-only (1 state token)", list(only_proprio.keys()) == ["proprio"] and total == 1)

    print(f"\n{'ALL OK' if all(R) else 'SOME FAILED'} ({sum(R)}/{len(R)})")
    import sys
    sys.exit(0 if all(R) else 1)


if __name__ == "__main__":
    main()
