"""Discover the cube color palette: collect saturated non-table pixels across many frames/episodes/cameras,
histogram hue, report the dominant cube hues (so we know if there's purple/yellow/etc)."""
from __future__ import annotations
import os, hydra, numpy as np
from omegaconf import OmegaConf
def rgb2hue(im):
    im=im/255.0; r,g,b=im[...,0],im[...,1],im[...,2]; mx=im.max(-1); mn=im.min(-1); df=mx-mn+1e-6
    s=df/np.clip(mx,1e-6,1); h=np.zeros_like(mx); rm=(mx==r); gm=(mx==g)&~rm; bm=(mx==b)&~rm&~gm
    h[rm]=((g-b)/df)[rm]%6; h[gm]=((b-r)/df)[gm]+2; h[bm]=((r-g)/df)[bm]+4; return (h*60)%360,s,mx
@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg):
    from quickdraw.data.dataset import DataConfig, load_split_episodes_mm
    from quickdraw.training.setup import image_head_cams, image_head_sizes, resolve_data_root
    ck=os.path.expanduser(str(cfg.get("eval_ckpt","") or "")); run=os.path.dirname(os.path.dirname(ck))
    rcfg=OmegaConf.load(os.path.join(run,"checkpoints","config.resolved.yaml")); OmegaConf.set_struct(rcfg,False)
    cams,sizes=image_head_cams(rcfg),image_head_sizes(rcfg)
    eps=load_split_episodes_mm(resolve_data_root(rcfg),"eval_purple_stack",dcfg=DataConfig.from_cfg(rcfg),
                               img_size=sizes,cam=cams,repo_id=str(rcfg.data.get("repo_id","torus")))
    hues=[]; rgbs=[]
    for o,a,fr in eps[:int(cfg.get('n_ep',20))]:
        for cam in cams:
            f=fr[cam]
            for t in range(0,len(f),20):
                im=f[t].astype(float); h,s,v=rgb2hue(im); table=(h>=18)&(h<=55)
                m=(s>0.45)&(v>0.2)&(~table)
                hues.append(h[m]); rgbs.append(im[m])
    hues=np.concatenate(hues); rgbs=np.concatenate(rgbs)
    print(f"[pal] {len(hues)} cube-candidate pixels")
    hist,edges=np.histogram(hues,bins=36,range=(0,360))
    for i,c in enumerate(hist):
        if c> len(hues)*0.01: print(f"  hue {edges[i]:3.0f}-{edges[i+1]:3.0f}: {100*c/len(hues):4.1f}%  meanRGB={rgbs[(hues>=edges[i])&(hues<edges[i+1])].mean(0).astype(int)}")
    print("[pal] DONE")
if __name__=="__main__": main()
