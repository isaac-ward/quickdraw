import sys, torch
sys.path.insert(0,"/app/src")
from hydra import compose, initialize_config_dir
from quickdraw.data.dataset import set_subsample, set_action_aggregate
from quickdraw.training.setup import build_model
with initialize_config_dir(config_dir="/app/conf", version_base=None):
    cfg = compose("config", overrides=["model=vl128_starling","data=starling2","environments=recorded",
        "data.subsample=4","data.action_aggregate=concat",
        "model.modalities.0.decode_kind=mse",
        "+model.modalities.0.derivative_weight=1.0","+model.modalities.1.derivative_weight=1.0"])
set_subsample(4); set_action_aggregate("concat")
m = build_model(cfg)
def n(mod): return sum(p.numel() for p in mod.parameters())
print("TOP-LEVEL:", ", ".join(f"{k}={n(v)/1e6:.3f}M" for k,v in m.named_children()))
for path in ("modalities.proprio.enc","modalities.image.ae","act_enc","backbone","flow",
             "modalities.proprio.decode_head","modalities.image.decode_head"):
    try:
        mod = m
        for p in path.split("."):
            mod = getattr(mod, p) if not p.isdigit() else mod[int(p)]
    except Exception as e:
        print(f"\n### {path}: {e}"); continue
    print(f"\n### {path}   [{n(mod)/1e6:.4f} M]")
    for name, sub in mod.named_modules():
        if name == "" or len(list(sub.children())): continue
        print(f"    {name:<46} {type(sub).__name__:<14} {n(sub):>9,}  {getattr(sub,'weight',None).shape if hasattr(sub,'weight') and sub.weight is not None else ''}")
