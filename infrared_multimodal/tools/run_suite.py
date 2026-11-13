"""Sequential four-way controls; use new tags for changed budgets or sources."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['regdb', 'sysu'], required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--protocol', choices=['dev', 'official'], default='dev')
    p.add_argument('--tag', required=True)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--lr', type=float, default=.1)
    p.add_argument('--trials', nargs='+', type=int, default=[1])
    p.add_argument('--seeds', nargs='+', type=int, default=[1])
    p.add_argument('--variants', nargs='+', choices=['agw', 'center', 'unbounded', 'bcrt'],
                   default=['agw', 'center', 'unbounded', 'bcrt'])
    args = p.parse_args()
    if Path(args.tag).name != args.tag or args.tag in {'.', '..'}:
        raise ValueError('Tag must be a simple directory name component.')
    if args.dataset == 'sysu' and args.trials != [1]:
        raise ValueError('SYSU uses one official training split; gallery trials are handled in evaluate.py.')
    for trial in args.trials:
        for seed in args.seeds:
            for variant in args.variants:
                name = f'{args.tag}_{args.dataset}_{args.protocol}_{variant}_t{trial}_s{seed}'
                run = ROOT/'runs'/name
                archive = ROOT/'results'/name
                if (archive/'config.json').exists() and not (run/'config.json').exists():
                    raise ValueError('Archived result exists without local run; choose a new tag.')
                command = [sys.executable, str(ROOT/'code/train.py'), '--dataset', args.dataset,
                           '--data', args.data, '--protocol', args.protocol, '--variant', variant,
                           '--trial', str(trial), '--seed', str(seed), '--epochs', str(args.epochs),
                           '--steps', str(args.steps), '--lr', str(args.lr), '--output', str(run)]
                if (run/'last.pt').exists():
                    command += ['--resume', str(run/'last.pt')]
                # train.py checks immutable configuration/source even for completed resumes.
                print(' '.join(command), flush=True)
                subprocess.run(command, cwd=ROOT, check=True)
                archive.mkdir(parents=True, exist_ok=True)
                for filename in ['config.json', 'history.json']:
                    (archive/filename).write_bytes((run/filename).read_bytes())
                subprocess.run([sys.executable, str(ROOT/'code/evaluate.py'), '--checkpoint', str(run/'last.pt'),
                                '--data', args.data, '--output', str(archive/'evaluation.json')], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
