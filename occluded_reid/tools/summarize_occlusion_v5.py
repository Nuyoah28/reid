"""Create paired-seed tables for the registered v5 experiment."""
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
VARIANTS = ('plain', 'occlusion_aug', 'pairwise_consistency', 'factorial_interaction')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def mean_sd(values):
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.


def main():
    from verify_occlusion_v5 import verify
    audit = verify()
    if audit['status'] != 'passed':
        raise RuntimeError('v5 verification did not pass; refusing to summarize results')
    protocol = read(ROOT/'research/occlusion_v5_protocol.json')
    summary_dir = ROOT/'results/occlusion_v5_summary'
    run_tag = protocol['run_tag']
    per_run = {}
    for variant in VARIANTS:
        for seed in protocol['seeds']:
            folder = ROOT/'results'/f'{run_tag}_{variant}_s{seed}'
            dev, external = read(folder/'results.json'), read(folder/'occluded_reid.json')
            per_run[(variant, seed)] = {'development': dev['evaluation'],
                                       'external': external['evaluation'],
                                       'seconds': sum(x['train_seconds'] for x in dev['history']),
                                       'updates': sum(x['optimizer_updates'] for x in dev['history']),
                                       'peak_cuda_mb': max(x['peak_cuda_mb'] for x in dev['history'])}
    rows = []
    table = ['# v5 配对遮挡交叉效应实验结果', '',
             '分数为百分制，均值 ± 跨种子样本标准差。真实 Occluded-REID 已在前序阶段查看，只作冻结模型的探索性迁移结果。', '',
             '| 变体 | 开发遮挡 mAP | 开发 clean mAP | 开发遮挡 R1 | Occluded-REID mAP | R1 | 训练分钟/seed | 参数量 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for variant in VARIANTS:
        values = [per_run[(variant, seed)] for seed in protocol['seeds']]
        metrics = [row['development'] for row in values]
        primary = [(m['synthetic_donor_0.3']['mAP'] + m['synthetic_donor_0.5']['mAP'])/2 for m in metrics]
        clean = [m['clean']['mAP'] for m in metrics]
        rank1 = [(m['synthetic_donor_0.3']['rank1'] + m['synthetic_donor_0.5']['rank1'])/2 for m in metrics]
        external = [per_run[(variant, seed)]['external'] for seed in protocol['seeds']]
        ext_map, ext_r1 = [x['mAP'] for x in external], [x['rank1'] for x in external]
        minutes = [row['seconds']/60 for row in values]
        config = read(ROOT/'results'/f'{run_tag}_{variant}_s{protocol["seeds"][0]}'/'config.json')
        map_mean, map_sd = mean_sd(primary)
        clean_mean, clean_sd = mean_sd(clean)
        rank_mean, rank_sd = mean_sd(rank1)
        ext_map_mean, ext_map_sd = mean_sd(ext_map)
        ext_rank_mean, ext_rank_sd = mean_sd(ext_r1)
        minute_mean, _ = mean_sd(minutes)
        table.append(f'| {variant} | {map_mean:.2f} ± {map_sd:.2f} | {clean_mean:.2f} ± {clean_sd:.2f} | '
                     f'{rank_mean:.2f} ± {rank_sd:.2f} | {ext_map_mean:.2f} ± {ext_map_sd:.2f} | '
                     f'{ext_rank_mean:.2f} ± {ext_rank_sd:.2f} | {minute_mean:.1f} | {config["inference_parameters"]:,} |')
        rows.append({'variant': variant, 'seed_values': {'development_occlusion_map': primary,
            'development_clean_map': clean, 'development_occlusion_rank1': rank1,
            'exploratory_external_map': ext_map, 'exploratory_external_rank1': ext_r1},
            'means_and_sample_sd': {'development_occlusion_map': [map_mean, map_sd],
                'development_clean_map': [clean_mean, clean_sd],
                'development_occlusion_rank1': [rank_mean, rank_sd],
                'exploratory_external_map': [ext_map_mean, ext_map_sd],
                'exploratory_external_rank1': [ext_rank_mean, ext_rank_sd]},
            'optimizer_updates': [row['updates'] for row in values],
            'training_seconds': [round(row['seconds'], 2) for row in values]})
    table.extend(['', '## 配对差值', '',
                  '| 比较 | 开发遮挡 mAP 差值（逐 seed） | 均值 ± 样本标准差 | 外测 mAP 差值（逐 seed） | 均值 ± 样本标准差 |',
                  '|---|---|---:|---|---:|'])
    comparisons = [('occlusion_aug - plain', 'occlusion_aug', 'plain'),
                   ('pairwise_consistency - occlusion_aug', 'pairwise_consistency', 'occlusion_aug'),
                   ('factorial_interaction - pairwise_consistency', 'factorial_interaction', 'pairwise_consistency'),
                   ('factorial_interaction - occlusion_aug', 'factorial_interaction', 'occlusion_aug')]
    paired_rows = []
    for label, left, right in comparisons:
        dev_delta, external_delta = [], []
        for seed in protocol['seeds']:
            lrow, rrow = per_run[(left, seed)], per_run[(right, seed)]
            ldev, rdev = lrow['development'], rrow['development']
            dev_delta.append((ldev['synthetic_donor_0.3']['mAP'] + ldev['synthetic_donor_0.5']['mAP'] -
                              rdev['synthetic_donor_0.3']['mAP'] - rdev['synthetic_donor_0.5']['mAP'])/2)
            external_delta.append(lrow['external']['mAP'] - rrow['external']['mAP'])
        dm, ds = mean_sd(dev_delta)
        em, es = mean_sd(external_delta)
        table.append(f'| {label} | {", ".join(f"{x:+.2f}" for x in dev_delta)} | {dm:+.2f} ± {ds:.2f} | '
                     f'{", ".join(f"{x:+.2f}" for x in external_delta)} | {em:+.2f} ± {es:.2f} |')
        paired_rows.append({'comparison': label, 'development_deltas': dev_delta,
                            'development_mean_sd': [dm, ds], 'external_deltas': external_delta,
                            'external_mean_sd': [em, es]})
    table.extend(['', '所有模块判断只依据开发集及配对消融；Occluded-REID 数字不用于模型选择。5 个种子用于机制筛选，不构成显著性或 SOTA 证据。'])
    (summary_dir/'table.md').write_text('\n'.join(table)+'\n', encoding='utf-8')
    (summary_dir/'metrics.json').write_text(json.dumps({'variants': rows, 'paired_comparisons': paired_rows},
        ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote {summary_dir / "table.md"}')


if __name__ == '__main__':
    main()
