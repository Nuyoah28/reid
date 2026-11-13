"""Test the feature-leakage hypothesis on held-out Market training identities."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from PIL import Image,ImageDraw
from reid.data import Images,training_split
from reid.model import OcclusionOSNet
from reid.occlusion import paired_occlusion,influence_target

ROOT=Path(__file__).resolve().parents[1]


def heatmap(value):
    a=value.detach().float().cpu().squeeze().clamp(0,1).numpy()
    rgb=np.stack([a,1-abs(2*a-1),1-a],-1)
    return Image.fromarray((rgb*255).astype(np.uint8)).resize((128,256))


def photo(value):
    x=value.detach().float().cpu()*torch.tensor([.229,.224,.225])[:,None,None]+torch.tensor([.485,.456,.406])[:,None,None]
    return Image.fromarray((x.clamp(0,1).permute(1,2,0).numpy()*255).astype(np.uint8)).resize((128,256))


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batches',type=int,default=8)
    args=p.parse_args()
    torch.set_num_threads(4)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    config=saved['config']
    if config['protocol']!='dev':raise ValueError('Use a development checkpoint; no test tuning')
    model=OcclusionOSNet(len(config['split']['train_ids']),config['variant']).cuda().eval()
    model.load_state_dict(saved['model'])
    _,query,gallery,_=training_split(ROOT/'data'/'Market-1501-v15.09.15')
    records=query+gallery
    # Identity-sorted gallery batches can contain just one identity, for which
    # another-identity donor is undefined. Shuffle independently of training RNG.
    loader=DataLoader(Images(records,config['height'],config['width']),batch_size=24,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(8180))
    generator=torch.Generator().manual_seed(8181)
    drift=[];target_values=[];prediction_error=[];canvas=None;sample_count=0
    for batch,(images,labels,_) in enumerate(loader):
        if batch>=args.batches:break
        images=images.cuda()
        a,b,visible=paired_occlusion(images,labels,generator=generator)
        with torch.autocast('cuda'):
            out=model(torch.cat([a,b]),details=True)
        ma,mb=out['maps'].chunk(2)
        target=influence_target(ma,mb,visible,config['tau'])
        geometric=F.adaptive_avg_pool2d(visible,ma.shape[-2:])
        difference=(1-F.cosine_similarity(ma.float(),mb.float(),dim=1)).clamp_min(0).unsqueeze(1)
        drift.extend(difference[geometric>.999].cpu().tolist())
        target_values.extend(target[geometric>.999].cpu().tolist())
        if out['weights'] is not None:
            predicted=out['reliability_logits'].chunk(2)[0].sigmoid()
            prediction_error.extend((predicted-target).abs().flatten().cpu().tolist())
        if canvas is None:
            columns=['View A','View B','Geometry','Target','Predicted' if out['weights'] is not None else 'Feature drift']
            canvas=Image.new('RGB',(128*5,280*4),'white')
            draw=ImageDraw.Draw(canvas)
            for row in range(4):
                maps=[photo(a[row]),photo(b[row]),heatmap(geometric[row]),heatmap(target[row]),heatmap(predicted[row] if out['weights'] is not None else difference[row])]
                for col,(label,picture) in enumerate(zip(columns,maps)):
                    draw.text((col*128+4,row*280+3),label,fill='black')
                    canvas.paste(picture,(col*128,row*280+22))
        sample_count+=len(images)
    values=np.asarray(drift)
    result={'protocol':'Development held-out training identities; fixed two-donor interventions; eval mode, no BN updates','samples':sample_count,'fully_visible_feature_cells':len(values),'outside_rectangle_cosine_drift_mean':float(values.mean()),'outside_rectangle_cosine_drift_median':float(np.median(values)),'fraction_visible_cells_drift_gt_0_05':float((values>.05).mean()),'visible_cell_target_mean':float(np.mean(target_values)),'reliability_target_mae_all_cells':float(np.mean(prediction_error)) if prediction_error else None,'interpretation':'Feature sensitivity under synthetic donor replacement, not a ground-truth human visibility annotation or proof of causal identifiability.'}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'diagnostic.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    canvas.save(args.output/'diagnostic.png')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
