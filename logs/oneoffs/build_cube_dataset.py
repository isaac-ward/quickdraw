"""Stage the cube-augmented block-stack dataset locally (NO upload). Per split: split-level hue-peak palette
discovery, per-RAW-frame cube state (subsample=1, aligned to parquet rows), rewrite parquet
(observation_vector -> observation.state  +  add observation.objects.cube [32]), keeping every other column.
Compute observation.objects.cube TRAIN stats. Update meta/info.json + normalization_stats.json. Verify.
  python logs/oneoffs/build_cube_dataset.py '+eval_ckpt="<bs ckpt>"' +stage=logs/_hf_stage
"""
from __future__ import annotations
import os, json, shutil, hydra, numpy as np
from omegaconf import OmegaConf
CANON=[("R",5),("G",150),("B",205),("P",270)]; ORDER=["R","G","B","P"]; TYP=120.0; MINVIS=0.12
def hsv(im):
    im=im/255.0; r,g,b=im[...,0],im[...,1],im[...,2]; mx=im.max(-1); mn=im.min(-1); df=mx-mn+1e-6
    s=df/np.clip(mx,1e-6,1); h=np.zeros_like(mx); rm=(mx==r); gm=(mx==g)&~rm; bm=(mx==b)&~rm&~gm
    h[rm]=((g-b)/df)[rm]%6; h[gm]=((b-r)/df)[gm]+2; h[bm]=((r-g)/df)[bm]+4; return (h*60)%360,s,mx
def cand(im): h,s,v=hsv(im); table=(h>=18)&(h<=55); return (s>0.45)&(v>0.2)&(~table),h
def hd(a,b): d=abs(a-b)%360; return np.minimum(d,360-d)
def discover(hues):
    if len(hues)<200: return {}
    hist,_=np.histogram(hues,72,(0,360)); hist=hist.astype(float)
    k=np.array([.25,.5,1,.5,.25]); hist=np.convolve(np.r_[hist[-2:],hist,hist[:2]],k,'same')[2:-2]; tot=hist.sum(); pk=[]
    for i in range(72):
        if hist[i]>=hist[(i-1)%72] and hist[i]>=hist[(i+1)%72] and hist[i]>0.03*tot: pk.append((i*5+2.5,hist[i]))
    pk.sort(key=lambda x:-x[1]); kept=[]
    for hue,c in pk:
        if all(hd(hue,k)>25 for k,_ in kept): kept.append((hue,c))
    named={}
    for hue,_ in kept:
        nm,ch=min(CANON,key=lambda c:hd(hue,c[1]))
        if nm not in named or hd(hue,ch)<hd(named[nm],ch): named[nm]=hue
    return named
def cube_state_for_episode(frames_per_cam, cams, named):   # -> (T, 32) per raw frame
    import scipy.ndimage as ndi
    T=len(frames_per_cam[cams[0]]); vec=np.zeros((T,len(ORDER)*len(cams)*4),np.float32)
    for ci,cam in enumerate(cams):
        f=frames_per_cam[cam]; H,W=f.shape[1],f.shape[2]; last={nm:None for nm in ORDER}
        for t in range(T):
            c,h=cand(f[t].astype(float))
            for j,nm in enumerate(ORDER):
                base=(j*len(cams)+ci)*4
                if nm not in named: continue
                m=c&(hd(h,named[nm])<22)
                if int(m.sum())>=MINVIS*TYP:
                    lab,n=ndi.label(m); sz=ndi.sum(np.ones_like(lab),lab,range(1,n+1)); kk=int(np.argmax(sz))+1
                    ys,xs=np.where(lab==kk); a=len(ys); cx,cy=xs.mean()/W,ys.mean()/H
                    last[nm]=(cx,cy); vec[t,base:base+4]=[cx,cy,a/TYP,min(1.0,a/TYP)]
                elif last[nm] is not None:
                    cx,cy=last[nm]; vec[t,base:base+4]=[cx,cy,0.0,0.0]
    return vec
@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg):
    import pyarrow as pa, pyarrow.parquet as pq
    from quickdraw.data.dataset import DataConfig, load_split_episodes_mm
    from quickdraw.training.setup import image_head_cams, image_head_sizes, resolve_data_root
    ck=os.path.expanduser(str(cfg.get("eval_ckpt","") or "")); run=os.path.dirname(os.path.dirname(ck))
    rcfg=OmegaConf.load(os.path.join(run,"checkpoints","config.resolved.yaml")); OmegaConf.set_struct(rcfg,False)
    camsd=image_head_cams(rcfg); cams=list(camsd); sizes=image_head_sizes(rcfg)
    root=resolve_data_root(rcfg); repo=str(rcfg.data.get("repo_id","torus"))
    stage=str(cfg.get("stage","logs/_hf_stage")); shutil.rmtree(stage,ignore_errors=True); os.makedirs(stage)
    dc1=DataConfig(subsample=1)          # RAW frames, aligned to parquet rows
    splits=["train","val","eval_purple_play","eval_purple_stack"]; train_cube=[]
    for split in splits:
        eps=load_split_episodes_mm(root,split,dcfg=dc1,img_size=sizes,cam=camsd,repo_id=repo)
        # palette discovery over the split (cam_scene)
        head=cams[0]; pool=[]
        for _,_,fr in eps[:min(len(eps),12)]:
            f=fr[head]
            for t in range(0,len(f),max(1,len(f)//40)): c,h=cand(f[t].astype(float)); pool.append(h[c])
        named=discover(np.concatenate(pool) if pool else np.array([]))
        present=[nm for nm in ORDER if nm in named]
        cubes=[cube_state_for_episode(fr,cams,named) for _,_,fr in eps]      # per episode (T,32), in np.unique(ep) order
        allcube=np.concatenate(cubes,0)
        if split=="train": train_cube.append(allcube)
        # rewrite parquet
        pfile=os.path.join(root,split,"data","chunk-000","file-000.parquet"); t=pq.read_table(pfile)
        assert t.num_rows==len(allcube), f"{split}: parquet {t.num_rows} rows != cube {len(allcube)} (alignment!)"
        cols={n:t.column(n) for n in t.column_names}
        cols["observation.state"]=cols.pop("observation_vector")            # RENAME
        newt=pa.table(cols); newt=newt.append_column("observation.objects.cube",
              pa.array([row.tolist() for row in allcube], type=pa.list_(pa.float32())))
        od=os.path.join(stage,split,"data","chunk-000"); os.makedirs(od,exist_ok=True)
        pq.write_table(newt, os.path.join(od,"file-000.parquet"))
        print(f"[build] {split}: {t.num_rows} rows, palette {present}, cols {newt.column_names}", flush=True)
    # train stats for observation.objects.cube
    tc=np.concatenate(train_cube,0); cmean=tc.mean(0).tolist(); cstd=(tc.std(0)+1e-6).tolist()
    json.dump({"observation.objects.cube_mean":cmean,"observation.objects.cube_std":cstd,
               "train_frames":int(len(tc))}, open(os.path.join(stage,"_cube_stats.json"),"w"))
    print(f"[build] train cube stats over {len(tc)} frames; staged -> {stage}")
    print("[build] DONE (no upload)")
if __name__=="__main__": main()
