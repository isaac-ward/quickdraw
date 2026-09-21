"""Smoke for the Diffusion-Forcing rollout (df_rollout_level): holding the fed-back past at level ε.
  - config threads scale + rollout_level -> model.df_scale / model.df_rollout_level,
  - a rollout with rollout_level>0 DIFFERS from the clean rollout (the noised past changed it),
  - the level embedding receives gradient in the noised-past training rollout,
  - the guard raises when rollout_level>0 but scale==0.
Run: docker compose exec -T app uv run --no-sync python -m quickdraw.smoke.df_rollout
"""
import torch
torch.set_num_threads(1)
from hydra import compose, initialize_config_dir

from ..training.setup import build_model


def _fail(m):
    print("FAIL:", m); raise SystemExit(1)


def _build(scale, roll):
    with initialize_config_dir(config_dir="/app/conf", version_base=None):
        cfg = compose(config_name="config", overrides=[
            "model=vl128_blockstack_flow",
            f"variations.noise_injection.observations_encoded_pre_fusion.scale={scale}",
            f"variations.noise_injection.observations_encoded_pre_fusion.rollout_level={roll}"])
    return build_model(cfg)


def main():
    torch.manual_seed(0)
    m = _build(0.3, 0.0).eval()
    print(f"built: df_scale={m.df_scale} df_rollout_level={m.df_rollout_level}")
    if abs(m.df_scale - 0.3) > 1e-9 or m.df_rollout_level != 0.0:
        _fail("config did not thread scale/rollout_level onto the model")

    P, H, B = 8, 12, 1
    ctx = {"proprio": torch.randn(B, P, 17),
           "cam_scene": torch.randn(B, P, 96, 128, 3),
           "cam_wrist": torch.randn(B, P, 96, 128, 3)}
    acts = torch.randn(B, P - 1 + H, 5)

    torch.manual_seed(1)
    with torch.no_grad():
        o0 = m.imagine_eval(ctx, acts, H, heads=["proprio"])["proprio"]
    m.df_rollout_level = 0.1                                   # flip the noised-past rollout ON
    torch.manual_seed(1)
    with torch.no_grad():
        o1 = m.imagine_eval(ctx, acts, H, heads=["proprio"])["proprio"]
    if torch.equal(o0, o1):
        _fail("df_rollout_level>0 did NOT change the rollout (noised past had no effect)")
    print(f"rollout clean vs ε=0.1 differ: max|Δ|={float((o0 - o1).abs().max()):.4g}   OK")

    m.train()
    tf = {"proprio": torch.randn(B, H, 17),
          "cam_scene": torch.randn(B, H, 96, 128, 3),
          "cam_wrist": torch.randn(B, H, 96, 128, 3)}
    preds = m.rollout_train(ctx, acts, tf, p_tf=0.0, detach_every=32)
    preds.sum().backward()
    g = sum(float(p.grad.abs().sum()) for p in m.df_level_emb.parameters() if p.grad is not None)
    if not (g > 0):
        _fail("df_level_emb received NO gradient in the noised-past rollout")
    print(f"df_level_emb gradient in noised-past rollout: {g:.4g}   OK")

    try:
        _build(0.0, 0.1)
        _fail("guard did NOT raise for rollout_level>0 with scale=0")
    except ValueError:
        print("guard raises: rollout_level>0 needs scale>0   OK")

    print("\nALL DF-ROLLOUT SMOKE PASSED")


if __name__ == "__main__":
    main()
