"""Summarize completed runs without silently pooling failed recipes/protocols."""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARIANT_ORDER = {name: i for i, name in enumerate(('plain','aug','visibility','clcp','full'))}
CONDITION_ORDER = {name: i for i, name in enumerate(
    ('clean','synthetic_noise_0.3','synthetic_noise_0.5','real_occluded_reid'))}
VARIANT_COLORS = dict(zip(('plain','aug','visibility','clcp','full'),
                          ('tab:blue','tab:orange','tab:green','tab:purple','tab:red')))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tags', nargs='+', required=True, help='Explicit completed suite prefixes')
    p.add_argument('--output', type=Path, default=ROOT/'results'/'summary')
    args = p.parse_args()
    rows = []
    for tag in args.tags:
        for path in sorted((ROOT/'results').glob(f'{tag}_*/results.json')):
            report = json.loads(path.read_text(encoding='utf-8'))
            config = report['config']
            if config.get('recipe_version') != 2:
                raise ValueError(f'{path}: not the calibrated recipe; do not mix failed runs')
            conditions = dict(report['evaluation'])
            external = path.parent/'occluded_reid.json'
            if external.exists():
                conditions['real_occluded_reid'] = json.loads(external.read_text(encoding='utf-8'))['evaluation']
            for condition, metrics in conditions.items():
                rows.append({'run': path.parent.name, 'tag': tag, 'protocol': report['protocol'],
                             'variant': report['variant'], 'seed': report['seed'],
                             'epochs': report['trained_epochs'], 'steps': config['steps'],
                             'condition': condition, **metrics})
    if not rows:
        raise ValueError('No completed runs matched')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'metrics.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in ('tag','protocol','variant','epochs','steps','condition'))
        groups[key].append(row)
    aggregate = []
    table = ['| Tag / protocol | Variant | Epochs × steps | Condition | Seeds | mAP (%) | Rank-1 (%) |',
             '|---|---|---|---|---|---|---|']
    for key, values in sorted(groups.items(), key=lambda item: (
            item[0][0], item[0][1], VARIANT_ORDER.get(item[0][2], 99),
            CONDITION_ORDER.get(item[0][5], 99), item[0][5])):
        tag, protocol, variant, epochs, steps, condition = key
        item = dict(zip(('tag','protocol','variant','epochs','steps','condition'),key))
        item['seeds'] = [x['seed'] for x in values]
        for metric in ('mAP','rank1','rank5','mINP'):
            scores = [x[metric] for x in values]
            item[metric+'_mean'] = statistics.mean(scores)
            item[metric+'_sample_std'] = statistics.stdev(scores) if len(scores)>1 else None
        aggregate.append(item)
        def display(metric):
            value = f'{item[metric+"_mean"]:.2f}'
            deviation = item[metric+'_sample_std']
            return value + (f' ± {deviation:.2f}' if deviation is not None else ' (single seed)')
        table.append(f'| {tag} / {protocol} | {variant} | {epochs} × {steps or "full"} | {condition} | {item["seeds"]} | {display("mAP")} | {display("rank1")} |')
    (args.output/'metrics.json').write_text(json.dumps({'rows':rows,'aggregate':aggregate},indent=2),encoding='utf-8')
    (args.output/'table.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
    # One panel per protocol/tag, so unrelated budgets cannot look comparable.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels = sorted({(r['tag'],r['protocol']) for r in aggregate})
    fig, axes = plt.subplots(1,len(panels),figsize=(7*len(panels),5.2),squeeze=False)
    for ax, panel in zip(axes[0],panels):
        selected = [r for r in aggregate if (r['tag'],r['protocol'])==panel]
        conditions = sorted({r['condition'] for r in selected},
                            key=lambda name: (CONDITION_ORDER.get(name, 99), name))
        variants = sorted({r['variant'] for r in selected},key=lambda name:VARIANT_ORDER.get(name,99))
        width = .8/len(variants)
        for j, variant in enumerate(variants):
            lookup = {r['condition']:r for r in selected if r['variant']==variant}
            x = [i-.4+width*(j+.5) for i in range(len(conditions))]
            y = [lookup[c]['mAP_mean'] if c in lookup else float('nan') for c in conditions]
            err = [lookup[c]['mAP_sample_std'] or 0 if c in lookup else 0 for c in conditions]
            # A single seed has unknown variance, not a measured zero variance.
            bars={'yerr':err,'capsize':3} if any(lookup[c]['mAP_sample_std'] is not None for c in conditions if c in lookup) else {}
            container=ax.bar(x,y,width,label=variant,color=VARIANT_COLORS.get(variant),**bars)
            ax.bar_label(container,fmt='%.1f',fontsize=7,padding=2)
        ax.set_xticks(range(len(conditions)),[c.replace('synthetic_noise_','noise ').replace('real_occluded_reid','real occluded') for c in conditions],rotation=15)
        budgets = sorted({(r['epochs'],r['steps']) for r in selected},key=str)
        budget_text = '; '.join(f'{epochs} epochs x {steps or "full"} PK batches' for epochs,steps in budgets)
        seeds = sorted({seed for r in selected for seed in r['seeds']})
        ax.set_ylabel('mAP (%)');ax.set_title(' / '.join(panel)+'\n'+budget_text+f'; seeds {seeds}');ax.set_ylim(0,100)
        ax.legend();ax.grid(axis='y',alpha=.2)
    fig.text(.5,.012,
             'Short budget; single seeds do not establish significance. Plain uses 1 training view; others use 3.\n'
             'Noise is synthetic; real occluded is cross-dataset Occluded-REID.',
             ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.09,1,1));fig.savefig(args.output/'comparison.png',dpi=180);plt.close(fig)
    print('\n'.join(table))


if __name__=='__main__':main()
