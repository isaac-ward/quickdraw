"""ONE-OFF: how well does the cube arm PREDICT the cube channel of the proprio (dims 17:49) in open-loop rollout?
Isolates 'cube well-predicted but not wired to pixels' vs 'cube badly predicted'. Reports per-horizon position
error (cx,cy) and visibility error, cube-state vs the persistence baseline (hold cube at its ctx-last value).
  python logs/oneoffs/cube_pred_bench.py '+eval_ckpt="<cube ckpt>"' +horizon=256 +n_ep=4
"""
from __future__ import annotations
import os, hydra, numpy as np, torch
from omegaconf import OmegaConf
@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg):
    from quickdraw.data.dataset import DataConfig, load_split_episodes_mm
    from quickdraw.training.setup import (build_model, image_head_cams, image_head_sizes, load_checkpoint,
                                          normalizer, resolve_data_root)
    ck=os.path.expanduser(str(cfg.get("eval_ckpt","") or "")); run=os.path.dirname(os.path.dirname(ck))
    rcfg=OmegaConf.load(os.path.join(run,"checkpoints","config.resolved.yaml")); OmegaConf.set_struct(rcfg,False)
    dev="cuda" if torch.cuda.is_available() else "cpu"
    m=build_model(rcfg).to(dev).eval(); load_checkpoint(m,ck); m=getattr(m,"_orig_mod",m)
    norm=normalizer(rcfg); cams=image_head_cams(rcfg); sizes=image_head_sizes(rcfg)
    P=int(rcfg.data.P); H=int(cfg.get("horizon",256)); n_ep=int(cfg.get("n_ep",4))
    dc=DataConfig.from_cfg(rcfg); imgh=list(cams)
    eps=load_split_episodes_mm(resolve_data_root(rcfg),"val",dcfg=dc,img_size=sizes,cam=cams,repo_id=str(rcfg.data.get("repo_id","torus")))
    n_ep=min(n_ep,len(eps)); H=min(H, min(len(o) for o,_,_ in eps[:n_ep])-P-1)
    ctx={"proprio":torch.stack([norm.norm_obs(torch.from_numpy(o[:P])) for o,_,_ in eps[:n_ep]]).float().to(dev)}
    for h in imgh: ctx[h]=torch.stack([torch.from_numpy(fr[h][:P]).float().div(255.) for _,_,fr in eps[:n_ep]]).to(dev)
    acts=torch.stack([norm.norm_act(torch.from_numpy(a[np.clip(np.arange(0,P+H-1),0,len(a)-1)])) for _,a,_ in eps[:n_ep]]).float().to(dev)
    truep=torch.stack([torch.from_numpy(o[P:P+H]) for o,_,_ in eps[:n_ep]]).float().to(dev)   # (n,H,49) ABSOLUTE (raw obs)
    with torch.no_grad():
        res=m.imagine_eval(ctx,acts,H,heads=["proprio"]+imgh,norm=norm)
        pred=norm.denorm_obs(res["proprio"])          # (n,H,49) absolute
    HS=[h for h in [1,8,16,32,64,128,256] if h<=H]
    # cube dims 17:49; position dims are indices 0,1 of each 4-block -> within cube: [0,1,4,5,...]; vis = idx 3 of each block
    cube_pred=pred[...,17:].cpu().numpy(); cube_true=truep[...,17:].cpu().numpy()   # (n,H,32)
    posmask=np.array([ (f%4) in (0,1) for f in range(32)]); vismask=np.array([ (f%4)==3 for f in range(32)])
    # persistence baseline: hold cube at ctx-last true value
    ctxlast=torch.stack([torch.from_numpy(o[P-1,17:]) for o,_,_ in eps[:n_ep]]).float().numpy()  # (n,32)
    print(f"[bench] {os.path.basename(run)}  H={H} n_ep={n_ep}")
    print(f"  {'h':>5} {'pos_err(model)':>15} {'pos_err(persist)':>17} {'vis_err(model)':>15}")
    for h in HS:
        t=h-1
        pe=np.sqrt(((cube_pred[:,t,posmask]-cube_true[:,t,posmask])**2).mean())
        pp=np.sqrt(((ctxlast[:,posmask]-cube_true[:,t,posmask])**2).mean())
        ve=np.abs(cube_pred[:,t,vismask]-cube_true[:,t,vismask]).mean()
        print(f"  @+{h:<3} {pe:15.4f} {pp:17.4f} {ve:15.4f}")
    print("[bench] pos in normalized image coords [0,1]; lower=better; model should beat persistence to be useful")
    print("[bench] DONE")
if __name__=="__main__": main()
