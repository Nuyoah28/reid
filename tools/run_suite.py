"""Run sequential CUDA experiments; copy small auditable records into results/."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--protocol',choices=['dev','official'],required=True)
    p.add_argument('--variants',nargs='+',default=['aug','visibility','clcp','full'])
    p.add_argument('--seeds',nargs='+',type=int,default=[1])
    p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--steps',type=int,default=80,help='PK batches per epoch; 0 uses a full epoch')
    p.add_argument('--tag',default='pilot')
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--external',action='store_true',help='Only after recipe is frozen, never for selection')
    args=p.parse_args()
    if args.steps<0:p.error('steps must be nonnegative')
    args.steps=args.steps or None
    for seed in args.seeds:
        for variant in args.variants:
            name=f'{args.tag}_{args.protocol}_{variant}_s{seed}'
            output=ROOT/'runs'/name
            archive=ROOT/'results'/name
            if archive.exists() and any(archive.iterdir()) and not (output/'config.json').exists():
                raise FileExistsError(f'Archive {archive} already exists without its local run. '
                                      'Use a fresh --tag to preserve the published records.')
            archive.mkdir(parents=True,exist_ok=True)
            command=[sys.executable,str(ROOT/'code'/'train.py'),'--variant',variant,'--protocol',args.protocol,'--epochs',str(args.epochs),'--seed',str(seed),'--output',str(output),'--workers',str(args.workers)]
            if args.steps is not None:command+=['--steps',str(args.steps)]
            if (output/'config.json').exists():
                old=json.loads((output/'config.json').read_text(encoding='utf-8'))
                for key in ['protocol','epochs','steps','seed','variant']:
                    wanted={'protocol':args.protocol,'epochs':args.epochs,'steps':args.steps,'seed':seed,'variant':variant}[key]
                    if old[key]!=wanted:raise ValueError(f'Existing run {name}: {key} differs; use a new tag')
            if (output/'last.pt').exists():command+=['--resume',str(output/'last.pt')]
            if not (output/'results.json').exists():
                with (archive/'train.log').open('a',encoding='utf-8') as log:
                    print(f'Running {name}',flush=True)
                    completed=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                if completed.returncode:raise RuntimeError(f'{name} failed; inspect {archive}/train.log')
            for filename in ['config.json','history.json','results.json']:
                shutil.copy2(output/filename,archive/filename)
            if args.external:
                subprocess.run([sys.executable,str(ROOT/'code'/'evaluate.py'),'--checkpoint',str(output/'last.pt'),'--output',str(archive/'occluded_reid.json'),'--workers',str(args.workers)],cwd=ROOT,check=True)
            report=json.loads((output/'results.json').read_text(encoding='utf-8'))
            print(json.dumps({'run':name,'evaluation':report['evaluation']}),flush=True)


if __name__=='__main__':main()
