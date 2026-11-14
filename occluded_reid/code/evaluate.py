"""Evaluate a frozen checkpoint, including genuinely occluded external images."""
import argparse
import json
from pathlib import Path
import torch
from train import extract
from torch.utils.data import DataLoader
from reid.data import Images,scan_occluded
from reid.metrics import evaluate_rank
from reid.model import OcclusionOSNet

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--data',type=Path,default=ROOT/'data')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    torch.set_num_threads(4)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    config=saved['config']
    model=OcclusionOSNet(len(config['split']['train_ids']),config['variant'])
    model.load_state_dict(saved['model'],strict=True)
    device=torch.device('cuda')
    model.to(device).eval()
    query,gallery=scan_occluded(args.data)
    # Cross-dataset namespaces differ; numeric PID equality to Market is irrelevant.
    def embeddings(records):
        dataset=Images(records,config['height'],config['width'])
        return extract(model,DataLoader(dataset,batch_size=96,num_workers=args.workers,pin_memory=True),device)
    q,g=embeddings(query),embeddings(gallery)
    results=evaluate_rank(q[0],g[0],q[1],g[1],q[2],g[2])
    report={'dataset':'Occluded-REID','protocol':'Market-trained cross-dataset; 200 held-out IDs; all 1000 occluded queries vs all 1000 whole gallery; no target tuning; no reranking','checkpoint':str(args.checkpoint),'variant':config['variant'],'train_protocol':config['protocol'],'epochs':saved['epoch'],'seed':config['seed'],'evaluation':results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
