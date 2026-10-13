"""ONE-OFF feasibility: can we cleanly segment cubes by color from block-stack frames? (de-risks the
pose-from-data idea). Read-only. HSV saturation+value mask -> connected components -> per-blob hue -> overlay.
  python logs/oneoffs/cube_seg_probe.py '+eval_ckpt="<any bs ckpt>"' +eval_out=logs/_cubeseg +n_frames=6
"""
from __future__ import annotations
import os, hydra, numpy as np
from omegaconf import OmegaConf

@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg):
    import imageio.v2 as imageio
    from scipy import ndimage
    import colorsys
    from quickdraw.data.dataset import DataConfig, load_split_episodes_mm
    from quickdraw.training.setup import image_head_cams, image_head_sizes, resolve_data_root
    ck=os.path.expanduser(str(cfg.get("eval_ckpt","") or "")); run=os.path.dirname(os.path.dirname(ck))
    rcfg=OmegaConf.load(os.path.join(run,"checkpoints","config.resolved.yaml")); OmegaConf.set_struct(rcfg,False)
    cams,sizes=image_head_cams(rcfg),image_head_sizes(rcfg)
    out=str(cfg.get("eval_out","logs/_cubeseg")); os.makedirs(out,exist_ok=True)
    nf=int(cfg.get("n_frames",6))
    eps=load_split_episodes_mm(resolve_data_root(rcfg),"val",dcfg=DataConfig.from_cfg(rcfg),
                               img_size=sizes,cam=cams,repo_id=str(rcfg.data.get("repo_id","torus")))
    head=list(cams)[0]; fr=eps[0][2][head]     # (T,H,W,3) uint8
    T=len(fr); idxs=np.linspace(20,T-20,nf).astype(int)
    def rgb2hue(im):                 # -> (H, in degrees 0..360), sat, val
        im=im/255.0; r,g,b=im[...,0],im[...,1],im[...,2]
        mx=im.max(-1); mn=im.min(-1); df=mx-mn+1e-6; v=mx; s=df/np.clip(mx,1e-6,1)
        h=np.zeros_like(mx)
        rm=(mx==r); gm=(mx==g)&~rm; bm=(mx==b)&~rm&~gm
        h[rm]=((g-b)/df)[rm]%6; h[gm]=((b-r)/df)[gm]+2; h[bm]=((r-g)/df)[bm]+4
        return (h*60)%360, s, v
    rows=[]
    for t in idxs:
        im=fr[t].astype(np.uint8); h,s,v=rgb2hue(im)
        table=(h>=18)&(h<=55)         # EXCLUDE the brown/orange table hue band
        mask=(s>0.45)&(v>0.2)&(~table)   # cube pixels: saturated, not-dark, not table-hue (R/G/B cubes)
        lab,n=ndimage.label(mask)
        overlay=im.copy(); found=[]
        for k in range(1,n+1):
            ys,xs=np.where(lab==k)
            if len(ys)<15: continue    # min blob size
            cy,cx=int(ys.mean()),int(xs.mean())
            col=im[ys,xs].mean(0).astype(int)
            found.append((cx,cy,tuple(col),len(ys)))
            overlay[max(cy-1,0):cy+2, max(cx-1,0):cx+2]=[255,255,0]   # yellow centroid dot
            # box
            y0,y1,x0,x1=ys.min(),ys.max(),xs.min(),xs.max()
            overlay[y0:y1+1,x0]=[255,255,0]; overlay[y0:y1+1,x1]=[255,255,0]
            overlay[y0,x0:x1+1]=[255,255,0]; overlay[y1,x0:x1+1]=[255,255,0]
        print(f"[seg] frame {t}: {len(found)} blobs  "+", ".join(f"@({cx},{cy})rgb{col}n{n_}" for cx,cy,col,n_ in found))
        rows.append(np.concatenate([im, overlay],axis=1))
    grid=np.concatenate(rows,axis=0)
    imageio.imwrite(os.path.join(out,f"cubeseg_{head}.png"),grid.astype(np.uint8))
    print(f"[seg] wrote cubeseg_{head}.png (left=raw, right=detected boxes) -> {out}")
    print("[seg] DONE")

if __name__=="__main__":
    main()
