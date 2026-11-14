"""Standard cross-camera CMC/mAP/mINP; same-camera same-ID and junk are excluded."""
import numpy as np
import torch


def evaluate_rank(query,gallery,q_pids,g_pids,q_cams,g_cams,max_rank=50):
    query=torch.nn.functional.normalize(query.float(),dim=1).cpu()
    gallery=torch.nn.functional.normalize(gallery.float(),dim=1).cpu()
    pids=np.asarray(g_pids);cams=np.asarray(g_cams)
    cmcs=[];aps=[];inps=[];invalid=0
    for start in range(0,len(query),128):
        distances=(1-query[start:start+128]@gallery.T).numpy()
        for offset,dist in enumerate(distances):
            i=start+offset
            keep=(pids!=-1)&~((pids==q_pids[i])&(cams==q_cams[i]))
            order=np.argsort(dist,kind='stable');order=order[keep[order]]
            hit=(pids[order]==q_pids[i]).astype(np.int64)
            positions=np.flatnonzero(hit)
            if not len(positions):invalid+=1;continue
            curve=np.minimum(1,np.cumsum(hit))[:max_rank]
            if len(curve)<max_rank:curve=np.pad(curve,(0,max_rank-len(curve)),constant_values=curve[-1])
            cmcs.append(curve)
            aps.append(np.mean(np.arange(1,len(positions)+1)/(positions+1)))
            inps.append(len(positions)/(positions[-1]+1))
    if not aps:raise ValueError('No valid cross-camera query matches')
    cmc=np.mean(cmcs,axis=0)
    return {'mAP':float(np.mean(aps)*100),'rank1':float(cmc[0]*100),'rank5':float(cmc[min(4,max_rank-1)]*100),'mINP':float(np.mean(inps)*100),'valid_queries':len(aps),'invalid_queries':invalid,'gallery_size':len(gallery)}
