import os, json, numpy as np, torch, imageio.v2 as imageio
from omegaconf import OmegaConf
from quickdraw.training.setup import image_head_cams, image_head_sizes
from quickdraw.data.dataset import load_split_episodes_mm, set_subsample
RUN=os.environ["RUN"]; OUT=os.environ["OUT"]
cfg=OmegaConf.create(json.load(open(f"{RUN}/logs/config.json")))
set_subsample(int(cfg.data.subsample))
cams=image_head_cams(cfg) or cfg.data.get("cam","fpv"); sizes=image_head_sizes(cfg) or 128
H=[n for n,_ in [("image",0)]][0]  # fpv image head
eps=load_split_episodes_mm(cfg.data.root,"val",img_size=sizes,cam=cams,repo_id=cfg.data.repo_id)
eps=sorted(eps, key=lambda e:-len(e[0]))
rows=[]
for i,(o,a,fr) in enumerate(eps):
    im=fr["image"]; T=len(im); jd=np.linspace(0,T-1,6).astype(int)
    row=np.concatenate([im[j] for j in jd],axis=1)
    rows.append(row); 
    if i>=27: break
grid=np.concatenate(rows,axis=0)
imageio.imwrite(OUT, grid.astype(np.uint8))
print(f"{len(rows)} eps (sorted longest); len range {len(eps[0][0])}..{len(eps[min(27,len(eps)-1)][0])} -> {OUT}", flush=True)
