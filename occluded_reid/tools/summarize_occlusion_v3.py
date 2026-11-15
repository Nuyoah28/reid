"""Create signed paired comparisons after v3 archive verification."""
import csv
import json
from pathlib import Path
import statistics

from verify_occlusion_v3 import ROOT, read, verify


def main():
    verification = verify()
    protocol = read(ROOT/'research/occlusion_v3_protocol.json')
    variants, seeds = protocol['variants_in_order'], protocol['seeds']
    destination = ROOT/'results/occlusion_v3_summary'
    rows, values = [], {}
    for variant in variants:
        for seed in seeds:
            folder = ROOT/'results'/f'occlusion_v3_{variant}_s{seed}'
            result, external = read(folder/'results.json'), read(folder/'occluded_reid.json')
            dev = result['evaluation']
            item = {'variant': variant, 'seed': seed,
                    'clean_map': dev['clean']['mAP'],
                    'noise30_map': dev['synthetic_noise_0.3']['mAP'],
                    'noise50_map': dev['synthetic_noise_0.5']['mAP'],
                    'development_map': (dev['synthetic_noise_0.3']['mAP']+dev['synthetic_noise_0.5']['mAP'])/2,
                    'external_map': external['evaluation']['mAP'],
                    'external_rank1': external['evaluation']['rank1'],
                    'external_rank5': external['evaluation']['rank5'],
                    'external_minp': external['evaluation']['mINP'],
                    'optimizer_updates': sum(h['optimizer_updates'] for h in result['history']),
                    'amp_skips': sum(h['amp_skipped_updates'] for h in result['history']),
                    'train_seconds': sum(h['train_seconds'] for h in result['history']),
                    'peak_cuda_mb': max(h['peak_cuda_mb'] for h in result['history']),
                    'inference_parameters': result['config']['inference_parameters']}
            rows.append(item)
            values[variant, seed] = item
    comparisons = []
    for candidate, base in (('transformer', 'baseline'), ('semantic_detail', 'baseline'),
                            ('semantic_detail', 'transformer')):
        for metric in ('development_map', 'external_map', 'external_rank1'):
            delta = [values[candidate, s][metric]-values[base, s][metric] for s in seeds]
            comparisons.append({'comparison': f'{candidate} - {base}', 'metric': metric,
                                'seed_deltas': delta, 'mean_delta': statistics.mean(delta),
                                'sample_std': statistics.stdev(delta)})
    summary = {'status': 'complete', 'verification': verification['status'],
               'units': 'Scores are percentages; deltas are percentage points; SD is across two seeds, not CI.',
               'runs': rows, 'paired_comparisons': comparisons,
               'limitations': ['8x80 short-budget feasibility screen.',
                               'Occluded-REID was viewed in v1/v2; exploratory transfer only.',
                               'No significance, convergence, or SOTA claim.']}
    (destination/'metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with (destination/'metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with (destination/'paired_deltas.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, list(comparisons[0]))
        writer.writeheader()
        for row in comparisons:
            writer.writerow({**row, 'seed_deltas': ','.join(f'{x:+.4f}' for x in row['seed_deltas'])})
    lines = ['# 第三阶段结构消融结果', '',
             '百分制，两种子均值 ± 样本标准差；误差不是置信区间。合成开发指标为 30%/50% 横向遮挡 mAP 的均值。', '',
             '| 模型 | seed | clean mAP | noise30 | noise50 | 开发遮挡均值 | 真实 mAP | 真实 R1 | 参数量 | 更新 / 跳步 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['variant']} | {r['seed']} | {r['clean_map']:.2f} | {r['noise30_map']:.2f} | {r['noise50_map']:.2f} | {r['development_map']:.2f} | {r['external_map']:.2f} | {r['external_rank1']:.2f} | {r['inference_parameters']:,} | {r['optimizer_updates']} / {r['amp_skips']} |")
    lines += ['', '## 两种子均值 ± 样本标准差', '',
              '| 模型 | 开发遮挡 mAP | 真实 mAP | 真实 Rank-1 | 训练秒数 | 峰值 CUDA MiB |',
              '|---|---:|---:|---:|---:|---:|']
    for variant in variants:
        rs = [values[variant, s] for s in seeds]
        cells = []
        for metric in ('development_map', 'external_map', 'external_rank1', 'train_seconds', 'peak_cuda_mb'):
            cells.append(f"{statistics.mean(r[metric] for r in rs):.2f} ± {statistics.stdev(r[metric] for r in rs):.2f}")
        lines.append('| '+variant+' | '+' | '.join(cells)+' |')
    lines += ['', '## 配对结构差值', '',
              '| 比较 | 指标 | s1 | s2 | 平均差 |', '|---|---|---:|---:|---:|']
    for row in comparisons:
        a, b = row['seed_deltas']
        lines.append(f"| {row['comparison']} | {row['metric']} | {a:+.2f} | {b:+.2f} | {row['mean_delta']:+.2f} |")
    lines += ['', '![第三阶段结构比较](comparison.png)', '',
              '真实集已在前两阶段查看，本表是探索性迁移。保留所有种子与负结果；逐轮记录和 checkpoint/source/data-record 哈希见各运行 JSON、freeze manifest 与 verification.json。', '']
    (destination/'table.md').write_text('\n'.join(lines), encoding='utf-8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for axis, metric, title in zip(axes, ('development_map', 'external_map'),
                                       ('Synthetic development', 'Occluded-REID exploratory transfer')):
            means = [statistics.mean(values[v, s][metric] for s in seeds) for v in variants]
            sds = [statistics.stdev(values[v, s][metric] for s in seeds) for v in variants]
            axis.bar(variants, means, yerr=sds, capsize=4,
                     color=['#718096', '#2774ae', '#3a8e78'])
            axis.set_ylabel('mAP (%)'); axis.set_title(title); axis.grid(axis='y', alpha=.2)
            axis.spines[['top', 'right']].set_visible(False)
        fig.tight_layout()
        fig.savefig(destination/'comparison.png', dpi=170)
        plt.close(fig)
    except ImportError:
        lines.append('Matplotlib was unavailable; numeric tables remain complete.')
    print(json.dumps({'status': 'complete', 'runs': len(rows), 'output': str(destination)}))


if __name__ == '__main__':
    main()
