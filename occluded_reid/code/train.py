"""Reproducible OSNet/occlusion experiments. Run from repository root."""
import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from reid.data import Images,PKSampler,scan_market,training_split
from reid.losses import batch_hard
from reid.metrics import evaluate_rank
from reid.model import OcclusionOSNet
from reid.occlusion import paired_occlusion,influence_target,surviving_evidence_loss

ROOT=Path(__file__).resolve().parents[1]

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def seed_worker(worker_id):
    seed=torch.initial_seed()%2**32;random.seed(seed);np.random.seed(seed)

@torch.inference_mode()
def extract(model,loader,device):
    model.eval();features=[];ids=[];cams=[]
    for images,pids,cameras in loader:
        with torch.autocast(device.type,enabled=device.type=='cuda'):
            embeddings=model(images.to(device,non_blocking=True))
        features.append(embeddings.float().cpu());ids.extend(pids.tolist());cams.extend(cameras.tolist())
    return torch.cat(features),ids,cams

def run_evaluation(model,query,gallery,args,device):
    # Evaluation loader creation must not advance the training augmentation RNG.
    with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
        return _run_evaluation(model,query,gallery,args,device)


def _run_evaluation(model,query,gallery,args,device):
    loader_args={'batch_size':args.eval_batch_size,'num_workers':args.workers,'pin_memory':device.type=='cuda'}
    gall=extract(model,DataLoader(Images(gallery,args.height,args.width),**loader_args),device)
    results={}
    for amount in args.eval_occlusion:
        data=Images(query,args.height,args.width,occlusion=amount,seed=1729)
        q=extract(model,DataLoader(data,**loader_args),device)
        results['clean' if amount==0 else f'synthetic_noise_{amount:g}']=evaluate_rank(q[0],gall[0],q[1],gall[1],q[2],gall[2])
    return results

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--variant',choices=['plain','aug','visibility','clcp','full'],default='full')
    p.add_argument('--data',type=Path,default=ROOT/'data'/'Market-1501-v15.09.15')
    p.add_argument('--pretrained',type=Path,default=ROOT/'weights'/'osnet_x1_0_imagenet.pth')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--protocol',choices=['dev','official'],default='dev')
    p.add_argument('--epochs',type=int,default=40);p.add_argument('--steps',type=int,default=None)
    p.add_argument('--batch-size',type=int,default=24);p.add_argument('--instances',type=int,default=4)
    p.add_argument('--height',type=int,default=256);p.add_argument('--width',type=int,default=128)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--eval-batch-size',type=int,default=96)
    p.add_argument('--lr',type=float,default=.0003);p.add_argument('--seed',type=int,default=1)
    p.add_argument('--head-lr-multiplier',type=float,default=5.)
    p.add_argument('--fixbase-epochs',type=int,default=2)
    p.add_argument('--tau',type=float,default=.15);p.add_argument('--calibration-weight',type=float,default=.5)
    p.add_argument('--evidence-weight',type=float,default=.5)
    p.add_argument('--eval-occlusion',type=float,nargs='+',default=[0.,.3,.5])
    p.add_argument('--eval-every',type=int,default=0)
    p.add_argument('--resume',type=Path);p.add_argument('--evaluate-only',action='store_true')
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    if any(x<0 or x>=1 for x in args.eval_occlusion):p.error('Occlusion must be in [0,1)')
    if args.protocol=='official' and args.eval_every:p.error('Official test is evaluated only at fixed final epoch')
    if args.epochs<1:p.error('epochs must be positive')
    if args.steps is not None and args.steps<1:p.error('steps must be positive')
    if not 0<=args.fixbase_epochs<args.epochs:p.error('fixbase-epochs must be in [0, epochs)')
    if args.lr<=0 or args.head_lr_multiplier<=0:p.error('Learning rates must be positive')
    if args.tau<=0:p.error('tau must be positive')
    if args.calibration_weight<0 or args.evidence_weight<0:p.error('Loss weights must be nonnegative')
    if args.evaluate_only and not args.resume:p.error('--evaluate-only requires --resume')
    if args.output.joinpath('config.json').exists() and not args.resume:
        p.error('Output already contains an experiment; use a new output or --resume')
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no silent CPU fallback')
    torch.set_num_threads(args.threads);seed_all(args.seed)
    torch.backends.cudnn.benchmark=True
    args.output.mkdir(parents=True,exist_ok=True)
    if args.protocol=='dev':train,query,gallery,split=training_split(args.data)
    else:
        train=scan_market(args.data/'bounding_box_train');query=scan_market(args.data/'query');gallery=scan_market(args.data/'bounding_box_test')
        split={'train_ids':sorted({r[1] for r in train}),'test_ids':sorted({r[1] for r in query})}
    if {r[1] for r in train}&{r[1] for r in query}:raise RuntimeError('Identity leakage')
    model=OcclusionOSNet(len({r[1] for r in train}),args.variant)
    init=model.load_imagenet(args.pretrained)
    model=model.to(device)
    # Reset RNG after architecture-specific initialization so data/augmentation match.
    seed_all(args.seed)
    backbone=[];heads=[]
    for name,param in model.named_parameters():
        (heads if name.startswith(('classifier.','reliability.')) else backbone).append(param)
    optimizer=torch.optim.Adam([{'params':backbone,'lr':args.lr,'multiplier':1.}, {'params':heads,'lr':args.lr*args.head_lr_multiplier,'multiplier':args.head_lr_multiplier}],lr=args.lr,weight_decay=5e-4,amsgrad=True)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    sampler=PKSampler(train,args.batch_size,args.instances,args.seed,args.steps)
    loader_gen=torch.Generator().manual_seed(args.seed)
    occlusion_gen=torch.Generator().manual_seed(args.seed+5000)
    loader=DataLoader(Images(train,args.height,args.width,train=True),batch_size=args.batch_size,sampler=sampler,num_workers=args.workers,pin_memory=device.type=='cuda',worker_init_fn=seed_worker,generator=loader_gen,persistent_workers=False)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update({'torch':torch.__version__,'torchvision':__import__('torchvision').__version__,'cuda':torch.version.cuda,'python':platform.python_version(),'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu','train_images':len(train),'query_images':len(query),'gallery_images':len(gallery),'trainable_parameters':sum(v.numel() for v in model.parameters()),'inference_parameters':sum(v.numel() for k,v in model.named_parameters() if not k.startswith('classifier.')),'initialization':init,'cudnn_benchmark':True,'determinism':'seeded; cuDNN may use nondeterministic kernels','split':split})
    source_files=sorted((ROOT/'code'/'reid').glob('*.py'))+[Path(__file__),ROOT/'code'/'deep-person-reid'/'torchreid'/'models'/'osnet.py']
    config['source_sha256']={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    config['pretrained_sha256']=hashlib.sha256(args.pretrained.read_bytes()).hexdigest()
    config['recipe_version']=2
    start_epoch=0;history=[]
    if args.resume:
        saved=torch.load(args.resume,map_location='cpu',weights_only=False)
        for key in ('variant','seed','height','width','protocol','epochs','steps','batch_size','instances','workers','lr','tau','calibration_weight','evidence_weight','fixbase_epochs','head_lr_multiplier','split','source_sha256','pretrained_sha256'):
            if saved['config'][key]!=config[key]:raise ValueError(f'Resume config changed: {key}')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
        start_epoch=saved['epoch'];history=saved['history']
        random.setstate(saved['rng']['python']);np.random.set_state(saved['rng']['numpy']);torch.set_rng_state(saved['rng']['torch'])
        if device.type=='cuda':torch.cuda.set_rng_state_all(saved['rng']['cuda'])
        loader_gen.set_state(saved['rng']['loader']);occlusion_gen.set_state(saved['rng']['occlusion'])
    (args.output/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(json.dumps({'event':'start','variant':args.variant,'protocol':args.protocol,'images':len(train),'steps_per_epoch':sampler.steps,'gpu':config['gpu'],'inference_parameters':config['inference_parameters']}),flush=True)
    for epoch in range(start_epoch,args.epochs if not args.evaluate_only else start_epoch):
        model.train();sampler.epoch=epoch
        if epoch<args.fixbase_epochs:
            # Official OSNet likewise warms up its new classifier before fine-tuning.
            model.eval()
            for name,param in model.named_parameters():
                param.requires_grad_(name.startswith(('classifier.','reliability.')))
            model.classifier.train()
            if model.reliability is not None:model.reliability.train()
        else:
            for param in model.parameters():param.requires_grad_(True)
        warm=min(5,max(1,args.epochs//5))
        factor=(epoch+1)/warm if epoch<warm else .5*(1+math.cos(math.pi*(epoch-warm)/max(1,args.epochs-warm)))
        for group in optimizer.param_groups:group['lr']=args.lr*factor*group['multiplier']
        sums={k:0. for k in ['loss','ce','triplet','calibration','evidence','target_mean']};t0=time.perf_counter()
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
        for step,(images,labels,_) in enumerate(loader):
            images=images.to(device,non_blocking=True);labels=labels.to(device,non_blocking=True)
            if args.variant!='plain':
                a,b,visible=paired_occlusion(images,labels,generator=occlusion_gen)
                input_images=torch.cat([images,a,b]);views=3
            else:input_images=images;views=1
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type,enabled=device.type=='cuda'):
                out=model(input_images,details=True)
                ce=F.cross_entropy(out['logits'],labels.repeat(views),label_smoothing=.1)
                triplet=torch.stack([batch_hard(F.normalize(f.float(),dim=1),labels) for f in out['features'].chunk(views)]).mean()
                calibration=ce.new_zeros(());evidence=ce.new_zeros(());mean_target=ce.new_zeros(())
                if model.reliability is not None:
                    clean,ma,mb=out['maps'].chunk(3)
                    target=influence_target(ma,mb,visible,args.tau,args.variant=='visibility')
                    rc,ra,rb=out['reliability_logits'].chunk(3)
                    calibration=.5*(F.binary_cross_entropy_with_logits(ra.float(),target)+F.binary_cross_entropy_with_logits(rb.float(),target))
                    # Clean means no synthetic occluder, not that every pixel is foreground.
                    # No all-ones clean calibration target is imposed on natural clutter.
                    if args.variant=='full':evidence=surviving_evidence_loss(clean,ma,mb,target)
                    mean_target=target.mean()
                loss=ce+.5*triplet+args.calibration_weight*calibration+args.evidence_weight*evidence
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
            scaler.step(optimizer);scaler.update()
            for key,value in [('loss',loss),('ce',ce),('triplet',triplet),('calibration',calibration),('evidence',evidence),('target_mean',mean_target)]:sums[key]+=float(value.detach())
            if (step+1)%25==0:print(json.dumps({'event':'step','epoch':epoch+1,'step':step+1,'loss':sums['loss']/(step+1),'elapsed_seconds':round(time.perf_counter()-t0,1)}),flush=True)
        record={'epoch':epoch+1,'lr':args.lr*factor,'backbone_frozen':epoch<args.fixbase_epochs,'train_seconds':time.perf_counter()-t0,'peak_cuda_mb':torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else 0,**{k:v/(step+1) for k,v in sums.items()}}
        if args.eval_every and (epoch+1)%args.eval_every==0:record['validation']=run_evaluation(model,query,gallery,args,device)
        history.append(record)
        (args.output/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        rng={'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],'loader':loader_gen.get_state(),'occlusion':occlusion_gen.get_state()}
        checkpoint={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch+1,'history':history,'config':config,'rng':rng}
        torch.save(checkpoint,args.output/'last.tmp');(args.output/'last.tmp').replace(args.output/'last.pt')
        print(json.dumps({'event':'epoch',**record}),flush=True)
    result=run_evaluation(model,query,gallery,args,device)
    report={'protocol':args.protocol,'variant':args.variant,'seed':args.seed,'trained_epochs':history[-1]['epoch'] if history else 0,'evaluation':result,'config':config,'history':history,'note':'Synthetic query noise is a stress test; it is NOT an Occluded-ReID/Occluded-Duke result.'}
    (args.output/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'event':'complete','results':result}),flush=True)

if __name__=='__main__':main()
