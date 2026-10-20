from hydra import compose, initialize_config_dir
from quickdraw.data.dataset import DataConfig, load_split_episodes_mm
from quickdraw.training.setup import image_head_cams, image_head_sizes, resolve_data_root, normalizer
with initialize_config_dir(config_dir="/app/conf", version_base=None):
    cfg=compose(config_name="config", overrides=["model=vl128_blockstack_2cam_cube","data=block_stack",
      "environments=recorded","environments.obs_dim=49","environments.action_dim=5","data.subsample=10","data.F=64",
      "+data.obs_fields=[observation.state,observation.objects.cube]"])
dc=DataConfig.from_cfg(cfg); print("obs_fields:", dc.obs_fields)
root=resolve_data_root(cfg); camsd=image_head_cams(cfg)
eps=load_split_episodes_mm(root,"val",dcfg=dc,img_size=image_head_sizes(cfg),cam=camsd,repo_id="block-stack")
o,a,fr=eps[0]; print("loaded val ep0: obs shape", o.shape, "(expect (...,49))  action", a.shape)
import numpy as np
print("obs last 32 (cube) sample @ffirst frame:", np.round(o[0,17:22],3), "...")
n=normalizer(cfg); print("normalizer o_mean len:", n.o_mean.shape[0], "(expect 49)")
print("LOAD TEST OK" if o.shape[-1]==49 and n.o_mean.shape[0]==49 else "MISMATCH")
