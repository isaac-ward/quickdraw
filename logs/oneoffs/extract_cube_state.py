"""ONE-OFF: extract cube state over the ENTIRE block-stack dataset (all splits, both cameras, every frame) via
split-level hue-peak discovery. Canonical cube order R,G,B,P -> a FIXED-WIDTH per-frame vector so the obs layout
is consistent across every episode (absent colors = zeros, vis=0). Saves per-episode npz + optional review video.
Does NOT push to HF (that is a separate, reviewed, non-destructive step).
  python logs/oneoffs/extract_cube_state.py '+eval_ckpt="<bs ckpt>"' +out=logs/_cube_extract +render=true
Per cube, per camera: (cx/W, cy/H, size/TYP, soft_vis). 4 cubes x 2 cams x 4 = 32 dims appended to proprio.
"""
from __future__ import annotations
import os, json, hydra, numpy as np
from omegaconf import OmegaConf
CANON=[("R",5,(255,60,60)),("G",150,(60,220,90)),("B",205,(70,150,255)),("P",270,(200,80,230))]
ORDER=["R","G","B","P"]; TYP=120.0; MINVIS=0.12
def hsv(im):
    im=im/255.0; r,g,b=im[...,0],im[...,1],im[...,2]; mx=im.max(-1); mn=im.min(-1); df=mx-mn+1e-6
    s=df/np.clip(mx,1e-6,1); h=np.zeros_like(mx); rm=(mx==r); gm=(mx==g)&~rm; bm=(mx==b)&~rm&~gm
    h[rm]=((g-b)/df)[rm]%6; h[gm]=((b-r)/df)[gm]+2; h[bm]=((r-g)/df)[bm]+4; return (h*60)%360,s,mx
def candidate(im): h,s,v=hsv(im); table=(h>=18)&(h<=55); return (s>0.45)&(v>0.2)&(~table),h
def hdiff(a,b): d=abs(a-b)%360; return np.minimum(d,360-d)
def discover(hues):
    if len(hues)<200: return {}
    bins=72; hist,_=np.histogram(hues,bins=bins,range=(0,360)); hist=hist.astype(float)
    k=np.array([.25,.5,1,.5,.25]); hist=np.convolve(np.r_[hist[-2:],hist,hist[:2]],k,'same')[2:-2]; tot=hist.sum()
    pk=[]
    for i in range(bins):
        if hist[i]>=hist[(i-1)%bins] and hist[i]>=hist[(i+1)%bins] and hist[i]>0.03*tot: pk.append((i*5+2.5,hist[i]))
    pk.sort(key=lambda x:-x[1]); kept=[]
    for hue,c in pk:
        if all(hdiff(hue,k)>25 for k,_ in kept): kept.append((hue,c))
    named={}
    for hue,_ in kept:
        nm,ch,col=min(CANON,key=lambda c:hdiff(hue,c[1]))
        if nm not in named or hdiff(hue,ch)<hdiff(named[nm][0],ch): named[nm]=(hue,col)
    return named
@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg):
    from scipy import ndimage
    from quickdraw.data.dataset import DataConfig, load_split_episodes_mm
    from quickdraw.training.setup import image_head_cams, image_head_sizes, resolve_data_root
    ck=os.path.expanduser(str(cfg.get("eval_ckpt","") or "")); run=os.path.dirname(os.path.dirname(ck))
    rcfg=OmegaConf.load(os.path.join(run,"checkpoints","config.resolved.yaml")); OmegaConf.set_struct(rcfg,False)
    camsd=image_head_cams(rcfg); cams=list(camsd); sizes=image_head_sizes(rcfg); root=resolve_data_root(rcfg)
    repo=str(rcfg.data.get("repo_id","torus")); out=str(cfg.get("out","logs/_cube_extract")); os.makedirs(out,exist_ok=True)
    render=bool(cfg.get("render",True))
    splits=list(cfg.get("splits",[])) or ["train","val","eval_purple_play","eval_purple_stack"]
    coverage={}
    for split in splits:
        try: eps=load_split_episodes_mm(root,split,dcfg=DataConfig.from_cfg(rcfg),img_size=sizes,cam=camsd,repo_id=repo)
        except Exception as e: print(f"[ext] {split}: SKIP ({type(e).__name__}: {e})"); continue
        # split-level palette from cam_scene
        head=cams[0]; pool=[]
        for _,_,fr in eps[:min(len(eps),12)]:
            f=fr[head]
            for t in range(0,len(f),max(1,len(f)//40)): c,h=candidate(f[t].astype(float)); pool.append(h[c])
        named=discover(np.concatenate(pool) if pool else np.array([]))
        present=[nm for nm in ORDER if nm in named]
        print(f"[ext] {split}: {len(eps)} eps, palette {present}  ("+", ".join(f'{n}@{named[n][0]:.0f}' for n in present)+")")
        os.makedirs(os.path.join(out,split),exist_ok=True); cov={nm:[] for nm in ORDER}
        for e,(o,a,fr) in enumerate(eps):
            T=len(o); vec=np.zeros((T, len(ORDER)*len(cams)*4), np.float32)   # per-frame fixed-width cube state
            last={(nm,ci):None for nm in ORDER for ci in range(len(cams))}
            for ci,cam in enumerate(cams):
                f=fr[cam]; H,W=f.shape[1],f.shape[2]
                for t in range(T):
                    im=f[t].astype(float); c,h=candidate(im)
                    for j,nm in enumerate(ORDER):
                        base=(j*len(cams)+ci)*4
                        if nm not in named: continue
                        hue=named[nm][0]; m=c&(hdiff(h,hue)<22); a_=int(m.sum())
                        if a_>=MINVIS*TYP:
                            lab,n=ndimage.label(m); sz=ndimage.sum(np.ones_like(lab),lab,range(1,n+1)); kk=int(np.argmax(sz))+1
                            ys,xs=np.where(lab==kk); a2=len(ys); cx,cy=xs.mean()/W,ys.mean()/H; vis=min(1.0,a2/TYP)
                            last[(nm,ci)]=(cx,cy); vec[t,base:base+4]=[cx,cy,a2/TYP,vis]
                        elif last[(nm,ci)] is not None:
                            cx,cy=last[(nm,ci)]; vec[t,base:base+4]=[cx,cy,0.0,0.0]   # hold-last pos, vis=0
                        if ci==0: cov[nm].append(vec[t,base+3])
            np.savez_compressed(os.path.join(out,split,f"ep{e:03d}.npz"), cube_state=vec, order=np.array(ORDER),
                                cams=np.array(cams), present=np.array(present))
        coverage[split]={nm:(float(np.mean([v for v in cov[nm]])) if cov[nm] else 0.0) for nm in ORDER}
        print(f"[ext]   mean cam0 visibility: "+", ".join(f"{nm}={coverage[split][nm]:.2f}" for nm in present))
    json.dump({"order":ORDER,"cams":cams,"per_cube_per_cam_feats":["cx/W","cy/H","size/TYP","soft_vis"],
               "dims":len(ORDER)*len(cams)*4,"coverage":coverage}, open(os.path.join(out,"manifest.json"),"w"),indent=2)
    print(f"[ext] appended-dims = {len(ORDER)*len(cams)*4}; manifest.json + per-episode npz -> {out}")
    print("[ext] DONE")
if __name__=="__main__": main()
