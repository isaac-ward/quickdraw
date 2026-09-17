import os, json, numpy as np, imageio.v2 as imageio
from omegaconf import OmegaConf
from quickdraw.training.setup import image_head_cams, image_head_sizes
from quickdraw.data.dataset import load_split_episodes_mm, set_subsample
RUN=os.environ["RUN"]; IDX=[int(x) for x in os.environ["IDX"].split(",")]; OUT=os.environ["OUT"]
cfg=OmegaConf.create(json.load(open(f"{RUN}/logs/config.json")))
set_subsample(int(cfg.data.subsample))
cams=image_head_cams(cfg) or cfg.data.get("cam","fpv"); sizes=image_head_sizes(cfg) or 128
eps=load_split_episodes_mm(cfg.data.root,"val",img_size=sizes,cam=cams,repo_id=cfg.data.repo_id)
eps=sorted(eps, key=lambda e:-len(e[0]))
rows=[]
for i in IDX:
    im=eps[i][2]["image"]; T=len(im); jd=np.linspace(0,T-1,12).astype(int)
    rows.append(np.concatenate([im[j] for j in jd],axis=1))
imageio.imwrite(OUT, np.concatenate(rows,axis=0).astype(np.uint8))
print(f"rows (top->bottom) idx={IDX} lens={[len(eps[i][0]) for i in IDX]} -> {OUT}", flush=True)
