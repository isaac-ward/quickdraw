"""ONE-OFF (redesign): per-episode hue-peak color DISCOVERY -> track only the colors actually present.
No fixed palette: pool cube-candidate hues over the episode, find hue peaks (each = a real cube color),
assign per-frame pixels only to discovered peaks. Fixes cross-contamination (no green in purple tasks,
no purple in RGB). Per-color connected-components, soft-visibility, hold-last. Renders a tracking video.
  python logs/oneoffs/cube_track.py '+eval_ckpt="<bs ckpt>"' +split=val +eval_out=logs/_track +n_ep=4 +stride=2
"""
from __future__ import annotations
import os, hydra, numpy as np
from omegaconf import OmegaConf
# canonical hue -> identity name + display color (for labels only; discovery is data-driven)
CANON=[("R",5,(255,60,60)),("G",150,(60,220,90)),("B",205,(70,150,255)),("P",270,(200,80,230)),("Y",55,(240,220,60))]
TYP=120; MINVIS=0.12
def hsv(im):
    im=im/255.0; r,g,b=im[...,0],im[...,1],im[...,2]; mx=im.max(-1); mn=im.min(-1); df=mx-mn+1e-6
    s=df/np.clip(mx,1e-6,1); h=np.zeros_like(mx); rm=(mx==r); gm=(mx==g)&~rm; bm=(mx==b)&~rm&~gm
    h[rm]=((g-b)/df)[rm]%6; h[gm]=((b-r)/df)[gm]+2; h[bm]=((r-g)/df)[bm]+4; return (h*60)%360,s,mx
def candidate(im):
    h,s,v=hsv(im); table=(h>=18)&(h<=55); return (s>0.45)&(v>0.2)&(~table), h
def hdiff(a,b): d=abs(a-b)%360; return np.minimum(d,360-d)
def discover_peaks(hues):
    if len(hues)<200: return []
    bins=72; hist,_=np.histogram(hues,bins=bins,range=(0,360)); hist=hist.astype(float)
    k=np.array([.25,.5,1,.5,.25]); hist=np.convolve(np.r_[hist[-2:],hist,hist[:2]],k,'same')[2:-2]  # circular smooth
    tot=hist.sum(); peaks=[]
    for i in range(bins):
        c=hist[i]
        if c>=hist[(i-1)%bins] and c>=hist[(i+1)%bins] and c>0.03*tot:
            peaks.append((i*5+2.5, c))
    peaks.sort(key=lambda x:-x[1]); kept=[]
    for hue,c in peaks:                              # merge peaks within 25 deg (keep the stronger)
        if all(hdiff(hue,k)>25 for k,_ in kept): kept.append((hue,c))
    return [h for h,_ in kept]
def name_of(hue): return min(CANON,key=lambda c:hdiff(hue,c[1]))
@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg):
    import imageio.v2 as imageio
    from scipy import ndimage
    from PIL import Image, ImageDraw
    from quickdraw.data.dataset import DataConfig, load_split_episodes_mm
    from quickdraw.training.setup import image_head_cams, image_head_sizes, resolve_data_root
    ck=os.path.expanduser(str(cfg.get("eval_ckpt","") or "")); run=os.path.dirname(os.path.dirname(ck))
    rcfg=OmegaConf.load(os.path.join(run,"checkpoints","config.resolved.yaml")); OmegaConf.set_struct(rcfg,False)
    cams,sizes=image_head_cams(rcfg),image_head_sizes(rcfg); out=str(cfg.get("eval_out","logs/_track")); os.makedirs(out,exist_ok=True)
    split=str(cfg.get("split","val")); UP=4; stride=int(cfg.get("stride",2))
    eps=load_split_episodes_mm(resolve_data_root(rcfg),split,dcfg=DataConfig.from_cfg(rcfg),
                               img_size=sizes,cam=cams,repo_id=str(rcfg.data.get("repo_id","torus")))
    head=list(cams)[0]; NE=min(int(cfg.get("n_ep",4)),len(eps))
    scope=str(cfg.get("discover","split"))     # split (pool across episodes, robust) | episode
    def discover_over(ep_range):
        pool=[]
        for e in ep_range:
            fr=eps[e][2][head]; T=len(fr)
            for t in range(0,T,max(1,T//40)):
                cand,h=candidate(fr[t].astype(float)); pool.append(h[cand])
        peaks=discover_peaks(np.concatenate(pool) if pool else np.array([]))
        named={}
        for hue in peaks:
            nm,ch,col=name_of(hue)
            if nm not in named or hdiff(hue,ch)<hdiff(named[nm][0],ch): named[nm]=(hue,col)
        return named
    if scope=="split":
        named_split=discover_over(range(min(len(eps),max(NE,12))))   # pool across up to 12 eps for a robust palette
        order_split=[nm for nm in ["R","G","B","P","Y"] if nm in named_split]
        print(f"[track] {split}: SPLIT-LEVEL palette -> "+", ".join(f"{nm}@hue{named_split[nm][0]:.0f}" for nm in order_split))
    for e in range(NE):
        fr=eps[e][2][head]; T=len(fr)
        if scope=="split":
            named,order=named_split,order_split
        else:
            named=discover_over([e]); order=[nm for nm in ["R","G","B","P","Y"] if nm in named]
            print(f"[track] {split} ep{e}: discovered {len(order)} cubes -> "+", ".join(f"{nm}@hue{named[nm][0]:.0f}" for nm in order))
        # PASS 2: per-frame track
        last={nm:None for nm in order}; frames=[]
        for t in range(0,T,stride):
            im=fr[t].astype(np.uint8); cand,h=candidate(im.astype(float))
            big=Image.fromarray(im).resize((im.shape[1]*UP,im.shape[0]*UP),Image.NEAREST); dr=ImageDraw.Draw(big); st=[]
            for nm in order:
                hue,col=named[nm]; m=cand&(hdiff(h,hue)<22)
                if m.sum()>0:
                    lab,n=ndimage.label(m); sz=ndimage.sum(np.ones_like(lab),lab,range(1,n+1)); k=int(np.argmax(sz))+1
                    ys,xs=np.where(lab==k); a=len(ys)
                else: a=0
                if a>=MINVIS*TYP:
                    cx,cy=float(xs.mean()),float(ys.mean()); last[nm]=(cx,cy); vf=min(1,a/TYP)
                    X,Y=cx*UP,cy*UP
                    dr.rectangle([xs.min()*UP,ys.min()*UP,xs.max()*UP,ys.max()*UP],outline=col,width=2)
                    dr.line([X-6,Y,X+6,Y],fill=col,width=2); dr.line([X,Y-6,X,Y+6],fill=col,width=2)
                    dr.text((X+7,Y-6),f"{nm} V{vf:.1f}",fill=col); st.append(f"{nm}:V{vf:.1f}({cx:.0f},{cy:.0f})")
                elif last[nm] is not None:
                    X,Y=last[nm][0]*UP,last[nm][1]*UP
                    dr.line([X-4,Y,X+4,Y],fill=col,width=1); dr.line([X,Y-4,X,Y+4],fill=col,width=1)
                    st.append(f"{nm}:OCC")
                else: st.append(f"{nm}:--")
            bar=Image.new("RGB",(big.width,18),(15,15,15)); ImageDraw.Draw(bar).text((3,3)," | ".join(st),fill=(255,255,255))
            frames.append(np.concatenate([np.asarray(bar),np.asarray(big)],0))
        imageio.mimsave(os.path.join(out,f"track_{split}_ep{e}_{head}.mp4"),frames,fps=12)
        print(f"[track]   -> track_{split}_ep{e}_{head}.mp4 ({len(frames)} frames)")
    print("[track] DONE")
if __name__=="__main__": main()
