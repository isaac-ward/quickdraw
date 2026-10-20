"""ONE-OFF: per-trajectory VIDEO of cube extraction — bbox + X + position label per cube, color-tracked,
with a VISIBLE / OCCLUDED(hold-last) status bar. De-risks the pose-from-data pipeline visually.
  python logs/oneoffs/cube_seg_video.py '+eval_ckpt="<bs ckpt>"' +eval_out=logs/_cubevid +n_ep=2 +stride=2
"""
from __future__ import annotations
import os, hydra, numpy as np
from omegaconf import OmegaConf

REF = {"R": np.array([175,85,70]), "G": np.array([40,130,95]), "B": np.array([90,150,190]), "P": np.array([84,44,139])}
COL = {"R": (255,60,60), "G": (60,220,90), "B": (70,150,255), "P": (200,80,230)}
TYP = 120  # typical full-cube blob size (px)
MINVIS = 0.15  # drop detections below this visibility (kills the spurious purple pink-cursor)

def rgb2hue(im):
    im=im/255.0; r,g,b=im[...,0],im[...,1],im[...,2]
    mx=im.max(-1); mn=im.min(-1); df=mx-mn+1e-6; v=mx; s=df/np.clip(mx,1e-6,1)
    h=np.zeros_like(mx); rm=(mx==r); gm=(mx==g)&~rm; bm=(mx==b)&~rm&~gm
    h[rm]=((g-b)/df)[rm]%6; h[gm]=((b-r)/df)[gm]+2; h[bm]=((r-g)/df)[bm]+4
    return (h*60)%360, s, v

@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg):
    import imageio.v2 as imageio
    from scipy import ndimage
    from PIL import Image, ImageDraw
    from quickdraw.data.dataset import DataConfig, load_split_episodes_mm
    from quickdraw.training.setup import image_head_cams, image_head_sizes, resolve_data_root
    ck=os.path.expanduser(str(cfg.get("eval_ckpt","") or "")); run=os.path.dirname(os.path.dirname(ck))
    rcfg=OmegaConf.load(os.path.join(run,"checkpoints","config.resolved.yaml")); OmegaConf.set_struct(rcfg,False)
    cams,sizes=image_head_cams(rcfg),image_head_sizes(rcfg)
    out=str(cfg.get("eval_out","logs/_cubevid")); os.makedirs(out,exist_ok=True)
    n_ep=int(cfg.get("n_ep",2)); stride=int(cfg.get("stride",2)); UP=4
    split=str(cfg.get("split","val"))
    eps=load_split_episodes_mm(resolve_data_root(rcfg),split,dcfg=DataConfig.from_cfg(rcfg),
                               img_size=sizes,cam=cams,repo_id=str(rcfg.data.get("repo_id","torus")))
    head=list(cams)[0]
    def detect(im):                       # per-PIXEL color -> per-COLOR components (touching cubes don't merge)
        h,s,v=rgb2hue(im); table=(h>=18)&(h<=55)
        cand=(s>0.45)&(v>0.2)&(~table)
        # assign each candidate pixel to the nearest palette anchor (within a distance gate)
        imf=im.astype(float); found={}
        dists=np.stack([((imf-REF[c])**2).sum(-1) for c in REF],axis=-1)   # (H,W,ncolors)
        nearest=np.argmin(dists,axis=-1); mind=np.min(dists,axis=-1)
        gate=(mind<70**2)                                                   # reject ambiguous/noise pixels
        for ci,c in enumerate(REF):
            m=cand&gate&(nearest==ci)
            if m.sum()==0: continue
            lab,n=ndimage.label(m)
            if n==0: continue
            sizes=ndimage.sum(np.ones_like(lab),lab,range(1,n+1))
            k=int(np.argmax(sizes))+1                                       # largest component of THIS color
            ys,xs=np.where(lab==k)
            if len(ys) < MINVIS*TYP: continue
            found[c]=(float(xs.mean()),float(ys.mean()),len(ys),int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max()))
        return found
    for e in range(min(n_ep,len(eps))):
        fr=eps[e][2][head]; T=len(fr); frames=[]
        last={c:None for c in REF}          # hold-last per color
        for t in range(0,T,stride):
            im=fr[t].astype(np.uint8); det=detect(im)
            big=Image.fromarray(im).resize((im.shape[1]*UP, im.shape[0]*UP), Image.NEAREST)
            dr=ImageDraw.Draw(big); H=im.shape[0]*UP
            status=[]
            for c in ("R","G","B","P"):
                if c in det:
                    cx,cy,a,x0,y0,x1,y1=det[c]; last[c]=(cx,cy); vis=True; vf=min(1.0,a/TYP)
                elif last[c] is not None:
                    cx,cy=last[c]; x0=y0=x1=y1=None; vis=False; vf=0.0
                else:
                    continue          # this color never seen (e.g. no purple cube) -> skip entirely
                X,Y=cx*UP,cy*UP; col=COL[c]
                if vis and x0 is not None:
                    dr.rectangle([x0*UP,y0*UP,x1*UP,y1*UP], outline=col, width=2)
                    dr.line([X-6,Y,X+6,Y],fill=col,width=2); dr.line([X,Y-6,X,Y+6],fill=col,width=2)  # solid X
                    dr.text((X+7,Y-6), f"{c}({cx:.0f},{cy:.0f})", fill=col)
                    status.append(f"{c}:V{vf:.1f}({cx:.0f},{cy:.0f})")
                else:                        # OCCLUDED -> dim hold-last X, no box
                    dr.line([X-5,Y,X+5,Y],fill=col,width=1); dr.line([X,Y-5,X,Y+5],fill=col,width=1)
                    dr.text((X+6,Y-6), f"{c}?", fill=col)
                    status.append(f"{c}:OCC")
            bar=Image.new("RGB",(big.width,18),(20,20,20)); ImageDraw.Draw(bar).text((3,3)," | ".join(status),fill=(255,255,255))
            frames.append(np.concatenate([np.asarray(bar), np.asarray(big)],axis=0))
        imageio.mimsave(os.path.join(out,f"cubetrack_{split}_ep{e}_{head}.mp4"), frames, fps=12)
        print(f"[vid] ep{e}: {len(frames)} frames -> cubetrack_{split}_ep{e}_{head}.mp4")
    print(f"[vid] DONE -> {out}")

if __name__=="__main__":
    main()
