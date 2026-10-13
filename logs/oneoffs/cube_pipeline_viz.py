"""ONE-OFF: visualize EVERY stage of cube extraction, so the pipeline (and its failure pixels) are legible.
Per frame, a labeled row: RAW | CANDIDATE MASK | PER-COLOR ASSIGNMENT | FINAL (boxes+X+soft-vis).
  python logs/oneoffs/cube_pipeline_viz.py '+eval_ckpt="<bs ckpt>"' +eval_out=logs/_cubepipe +n_frames=5
"""
from __future__ import annotations
import os, hydra, numpy as np
from omegaconf import OmegaConf
REF={"R":np.array([175,85,70]),"G":np.array([40,130,95]),"B":np.array([90,150,190]),"P":np.array([120,70,150])}
COL={"R":(255,60,60),"G":(60,220,90),"B":(70,150,255),"P":(200,80,230)}
TYP=120; GATE=70**2; MINVIS=0.15
def rgb2hue(im):
    im=im/255.0; r,g,b=im[...,0],im[...,1],im[...,2]; mx=im.max(-1); mn=im.min(-1); df=mx-mn+1e-6
    s=df/np.clip(mx,1e-6,1); h=np.zeros_like(mx); rm=(mx==r); gm=(mx==g)&~rm; bm=(mx==b)&~rm&~gm
    h[rm]=((g-b)/df)[rm]%6; h[gm]=((b-r)/df)[gm]+2; h[bm]=((r-g)/df)[bm]+4; return (h*60)%360,s,mx
@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg):
    import imageio.v2 as imageio
    from scipy import ndimage
    from PIL import Image, ImageDraw
    from quickdraw.data.dataset import DataConfig, load_split_episodes_mm
    from quickdraw.training.setup import image_head_cams, image_head_sizes, resolve_data_root
    ck=os.path.expanduser(str(cfg.get("eval_ckpt","") or "")); run=os.path.dirname(os.path.dirname(ck))
    rcfg=OmegaConf.load(os.path.join(run,"checkpoints","config.resolved.yaml")); OmegaConf.set_struct(rcfg,False)
    cams,sizes=image_head_cams(rcfg),image_head_sizes(rcfg); out=str(cfg.get("eval_out","logs/_cubepipe")); os.makedirs(out,exist_ok=True)
    eps=load_split_episodes_mm(resolve_data_root(rcfg),"val",dcfg=DataConfig.from_cfg(rcfg),
                               img_size=sizes,cam=cams,repo_id=str(rcfg.data.get("repo_id","torus")))
    head=list(cams)[0]; fr=eps[0][2][head]; T=len(fr); UP=4
    idxs=np.linspace(30,T-30,int(cfg.get("n_frames",5))).astype(int)
    def stages(im):
        h,s,v=rgb2hue(im); table=(h>=18)&(h<=55); cand=(s>0.45)&(v>0.2)&(~table)
        imf=im.astype(float); dists=np.stack([((imf-REF[c])**2).sum(-1) for c in REF],-1)
        nearest=np.argmin(dists,-1); mind=np.min(dists,-1); gate=mind<GATE
        # assignment paint
        assign=np.full_like(im,40)
        dets={}
        for ci,c in enumerate(REF):
            m=cand&gate&(nearest==ci); assign[m]=COL[c]
            if m.sum()==0: continue
            lab,n=ndimage.label(m)
            if n==0: continue
            sz=ndimage.sum(np.ones_like(lab),lab,range(1,n+1)); k=int(np.argmax(sz))+1
            ys,xs=np.where(lab==k); a=len(ys)
            if a/TYP<MINVIS: continue
            dets[c]=(float(xs.mean()),float(ys.mean()),a,int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max()))
        candimg=np.where(cand[...,None],255,0).astype(np.uint8).repeat(3,-1)
        return candimg, assign.astype(np.uint8), dets
    def up(a):
        return np.asarray(Image.fromarray(a).resize((a.shape[1]*UP,a.shape[0]*UP),Image.NEAREST))
    def cap(a,txt):
        from PIL import Image, ImageDraw
        bar=Image.new("RGB",(a.shape[1],16),(0,0,0)); ImageDraw.Draw(bar).text((3,3),txt,fill=(255,255,255))
        return np.concatenate([np.asarray(bar),a],0)
    rows=[]
    for t in idxs:
        im=fr[t].astype(np.uint8); candimg,assign,dets=stages(im)
        final=Image.fromarray(up(im)); dr=ImageDraw.Draw(final)
        st=[]
        for c in ("R","G","B","P"):
            if c not in dets: 
                if c!="P": st.append(f"{c}:OCC")
                continue
            cx,cy,a,x0,y0,x1,y1=dets[c]; X,Y=cx*UP,cy*UP; col=COL[c]; vf=min(1,a/TYP)
            dr.rectangle([x0*UP,y0*UP,x1*UP,y1*UP],outline=col,width=2)
            dr.line([X-6,Y,X+6,Y],fill=col,width=2); dr.line([X,Y-6,X,Y+6],fill=col,width=2)
            dr.text((X+7,Y-6),f"{c} V{vf:.1f}",fill=col); st.append(f"{c}:V{vf:.1f}")
        panels=[cap(up(im),f"1 RAW  t={t}"), cap(up(candimg),"2 CANDIDATE MASK"),
                cap(up(assign),"3 PER-COLOR ASSIGN"), cap(np.asarray(final)," ".join(st))]
        rows.append(np.concatenate(panels,axis=1))
    grid=np.concatenate(rows,axis=0)
    imageio.imwrite(os.path.join(out,f"pipeline_{head}.png"),grid)
    print(f"[pipe] wrote pipeline_{head}.png ({len(idxs)} frames x 4 stages) -> {out}")
    print("[pipe] DONE")
if __name__=="__main__": main()
