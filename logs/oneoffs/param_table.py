from hydra import compose, initialize_config_dir
from quickdraw.training.setup import build_model
def build(over):
    with initialize_config_dir(config_dir="/app/conf", version_base=None):
        cfg=compose(config_name="config", overrides=[
          "model=vl128_blockstack_2cam","data=block_stack","environments=recorded",
          "environments.obs_dim=17","environments.action_dim=5","data.subsample=10","data.F=64"]+over)
    return build_model(cfg)
def split(m):
    tot=sum(p.numel() for p in m.parameters())
    dec=sum(p.numel() for n,p in m.named_parameters() if "decode" in n or "modalities.cam" in n)
    return tot, dec
cfgs=[
  ("FULL (d128 h8 dep4 db64)", []),
  ("d64 h4 dep2 db64",        ["model.d=64","model.heads=4","model.depth=2"]),
  ("d64 h4 dep2 db32",        ["model.d=64","model.heads=4","model.depth=2","model.modalities.1.decode_base=32","model.modalities.2.decode_base=32"]),
  ("d64 h4 dep2 db16",        ["model.d=64","model.heads=4","model.depth=2","model.modalities.1.decode_base=16","model.modalities.2.decode_base=16"]),
  ("d128 h8 dep4 db16",       ["model.modalities.1.decode_base=16","model.modalities.2.decode_base=16"]),
  ("d64 h4 dep2 db16 tok16",  ["model.d=64","model.heads=4","model.depth=2","model.modalities.1.decode_base=16","model.modalities.2.decode_base=16","model.modalities.1.num_tokens=16","model.modalities.2.num_tokens=16"]),
  ("d96 h6 dep3 db32",        ["model.d=96","model.heads=6","model.depth=3","model.modalities.1.decode_base=32","model.modalities.2.decode_base=32"]),
]
print(f"{'config':28} {'total':>8} {'decoder':>8} {'backbone':>9}")
for name,ov in cfgs:
    try:
        m=build(ov); tot,dec=split(m)
        print(f"{name:28} {tot/1e6:7.2f}M {dec/1e6:7.2f}M {(tot-dec)/1e6:8.2f}M")
    except Exception as e:
        print(f"{name:28} BUILD FAIL: {type(e).__name__}: {str(e)[:60]}")
