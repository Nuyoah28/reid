"""Summarize paired MDTA versus frozen v3 baselines after audit."""
import json
from pathlib import Path
import statistics

from verify_occlusion_v4 import ROOT, read, verify


def mean_sd(values):
    return statistics.mean(values), statistics.stdev(values)


def main():
    audit = verify()
    protocol = read(ROOT/'research/occlusion_v4_protocol.json')
    output = ROOT/'results/occlusion_v4_summary'
    rows = []
    paired = []
    for ref in protocol['baseline_references']:
        seed = int(ref['run'].rsplit('s', 1)[1])
        folder = ROOT/'results'/f'occlusion_v4_mdta_s{seed}'
        result, external = read(folder/'results.json'), read(folder/'occluded_reid.json')
        development = result['evaluation']
        development_map = (development['synthetic_noise_0.3']['mAP']+
                            development['synthetic_noise_0.5']['mAP'])/2
        baseline = read(ROOT/'results'/ref['run']/'occluded_reid.json')['evaluation']
        baseline_dev = read(ROOT/'results'/ref['run']/'results.json')['evaluation']
        baseline_dev_map = (baseline_dev['synthetic_noise_0.3']['mAP']+
                            baseline_dev['synthetic_noise_0.5']['mAP'])/2
        row = {'seed': seed, 'clean_map': development['clean']['mAP'],
               'noise30_map': development['synthetic_noise_0.3']['mAP'],
               'noise50_map': development['synthetic_noise_0.5']['mAP'],
               'development_map': development_map,
               'external_map': external['evaluation']['mAP'],
               'external_rank1': external['evaluation']['rank1'],
               'external_rank5': external['evaluation']['rank5'],
               'external_minp': external['evaluation']['mINP'],
               'inference_parameters': result['config']['inference_parameters'],
               'optimizer_updates': sum(x['optimizer_updates'] for x in result['history']),
               'amp_skips': sum(x['amp_skipped_updates'] for x in result['history']),
               'peak_cuda_mb': max(x['peak_cuda_mb'] for x in result['history']),
               'development_delta_vs_baseline': development_map-baseline_dev_map,
               'external_map_delta_vs_baseline': external['evaluation']['mAP']-baseline['mAP'],
               'external_rank1_delta_vs_baseline': external['evaluation']['rank1']-baseline['rank1']}
        rows.append(row)
        paired.append({'seed': seed, 'baseline_run': ref['run'],
                       'baseline_checkpoint_sha256': ref['checkpoint_sha256'],
                       'mdta_checkpoint_sha256': external['checkpoint_sha256'],
                       'target_data_record_sha256': external['target_data_record_sha256'],
                       'development_delta': row['development_delta_vs_baseline'],
                       'external_map_delta': row['external_map_delta_vs_baseline'],
                       'external_rank1_delta': row['external_rank1_delta_vs_baseline']})
    summary = {'status': 'complete', 'verification': audit['status'],
               'units': 'Metrics are percent; differences are percentage points; SD is across two seeds, not CI.',
               'runs': rows, 'paired_comparisons': paired,
               'target_data_record_sha256': audit['target_data_record_sha256'],
               'limitations': ['8x80 short-budget feasibility screen.',
                               'Occluded-REID was viewed in v1-v3; exploratory transfer only.',
                               'The TIP 2024 second-order attention paper is not available locally; exact novelty overlap is unresolved.',
                               'No significance, convergence, or SOTA claim.']}
    (output/'metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    metrics = [('development_map', 'Synthetic occlusion development mAP'),
               ('external_map', 'Occluded-REID exploratory mAP'),
               ('external_rank1', 'Occluded-REID exploratory Rank-1')]
    lines = ['# 第四阶段 MDTA 结构筛选', '',
             '百分制；配对差值以相同 seed 的 v3 baseline 为参照。两 seed 的样本标准差不是置信区间。', '',
             '| seed | 开发遮挡 mAP | 真实 mAP | 真实 Rank-1 | 开发差值 | 真实 mAP 差值 | 参数量 | 更新 / 跳步 |',
             '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['seed']} | {row['development_map']:.2f} | {row['external_map']:.2f} | {row['external_rank1']:.2f} | {row['development_delta_vs_baseline']:+.2f} | {row['external_map_delta_vs_baseline']:+.2f} | {row['inference_parameters']:,} | {row['optimizer_updates']} / {row['amp_skips']} |")
    lines += ['', '## 两 seed 均值 ± 样本标准差', '',
              '| 指标 | MDTA | 配对差值 |', '|---|---:|---:|']
    for metric, label in metrics:
        values = [row[metric] for row in rows]
        delta_key = {'development_map': 'development_delta_vs_baseline',
                     'external_map': 'external_map_delta_vs_baseline',
                     'external_rank1': 'external_rank1_delta_vs_baseline'}[metric]
        delta = [row[delta_key] for row in rows]
        m, sd = mean_sd(values); dm, dsd = mean_sd(delta)
        lines.append(f'| {label} | {m:.2f} ± {sd:.2f} | {dm:+.2f} ± {dsd:.2f} |')
    lines += ['', '真实目标数据清单在训练前注册，checkpoint 在全部开发训练结束后统一冻结。Occluded-REID 已在前序阶段暴露，因此仅作探索性迁移。',
              '', '完整逐项结果与 hash 见 `metrics.json`、`external_freeze_manifest.json` 和 `verification.json`。']
    (output/'table.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'status': 'complete', 'runs': len(rows), 'output': str(output)}))


if __name__ == '__main__':
    main()
