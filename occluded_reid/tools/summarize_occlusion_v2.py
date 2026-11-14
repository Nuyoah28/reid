"""Generate complete signed two-seed tables/figures after frozen external evaluation."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
from verify_occlusion_v2 import verify, read, ROOT

METRICS = ('clean_map', 'noise30_map', 'noise50_map', 'development_map', 'external_map', 'external_rank1')
PAIRS = [('paste3', 'repeat3', '遮挡粘贴（相同三视图预算）'),
         ('stability', 'paste3', '旧可靠性头与稳定性监督组合'),
         ('peer', 'stability', '替换为跨图身份目标'),
         ('peer_stability', 'peer', '联合旧外观稳定性'),
         ('peer_evidence', 'peer_stability', '添加旧剩余证据一致性')]


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(destination, variants, seeds, by_run, aggregates, comparisons):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none'})
    colors = ['#718096', '#2764ad', '#4d9787', '#d48c29', '#a05ba4', '#bf5b58']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for axis, metric, title in zip(axes, ('development_map', 'external_map'),
                                   ('Synthetic development: mean of 30% / 50% noise', 'Real Occluded-REID: exploratory transfer')):
        means = [aggregates[v][metric]['mean'] for v in variants]
        sds = [aggregates[v][metric]['sample_std'] for v in variants]
        x = np.arange(len(variants))
        axis.bar(x, means, yerr=sds, color=colors, alpha=.8, capsize=4)
        for i, seed in enumerate(seeds):
            axis.scatter(x + (i - .5)*.13, [by_run[v, seed][metric] for v in variants],
                         marker=('o', 'x')[i], color='#172335', s=25, zorder=3, label=f'seed {seed}')
        axis.set_xticks(x, [v.replace('_', '\n') for v in variants], fontsize=9)
        axis.set_ylabel('mAP (%)'); axis.set_title(title, fontsize=11)
        axis.set_ylim(0, max(m+s for m, s in zip(means, sds))*1.2)
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='y', alpha=.2); axis.set_axisbelow(True)
    axes[0].legend(loc='upper left', fontsize=9)
    fig.suptitle('Occlusion v2 | all 6 variants x 2 seeds | OSNet / 651 train IDs / 640 PK batches', fontsize=13)
    fig.text(.5, .015, 'Bars: mean; error bars: sample SD (n=2), not confidence intervals. Fixed final epoch; no reranking.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, .94))
    for ext in ('png', 'svg'):
        fig.savefig(destination/f'comparison.{ext}', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    labels = [a+' − '+b for a, b, _ in PAIRS]
    for axis, metric, title in zip(axes, ('development_map', 'external_map'), ('Synthetic development delta', 'Real transfer delta')):
        means = [comparisons[a+'-'+b][metric]['mean'] for a, b, _ in PAIRS]
        sd = [comparisons[a+'-'+b][metric]['sample_std'] for a, b, _ in PAIRS]
        y = list(range(len(PAIRS)))
        axis.barh(y, means, xerr=sd, color=['#257a63' if v >= 0 else '#bb534f' for v in means], alpha=.78, capsize=4)
        for i, seed in enumerate(seeds):
            values = [comparisons[a+'-'+b][metric]['per_seed'][str(seed)] for a, b, _ in PAIRS]
            axis.scatter(values, [v+(i-.5)*.12 for v in y], marker=('o', 'x')[i], s=25, color='#172335', zorder=3)
        axis.axvline(0, color='#25364b', linewidth=1)
        axis.set_yticks(y, labels, fontsize=9); axis.invert_yaxis()
        axis.set_title(title); axis.set_xlabel('Paired mAP difference (percentage points)')
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='x', alpha=.2); axis.set_axisbelow(True)
    fig.suptitle('Every registered module comparison, including negative effects', fontsize=13)
    fig.tight_layout(rect=(0, .02, 1, .94))
    for ext in ('png', 'svg'):
        fig.savefig(destination/f'paired_deltas.{ext}', dpi=170)
    plt.close(fig)
    # Matplotlib emits trailing spaces inside multiline SVG path attributes.
    # Remove only line-end whitespace; XML still separates coordinates by LF.
    for name in ('comparison.svg', 'paired_deltas.svg'):
        path = destination/name
        source = path.read_text(encoding='utf-8')
        path.write_text('\n'.join(line.rstrip() for line in source.splitlines())+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', default='peer_v2')
    args = parser.parse_args()
    verification = verify(args.tag)
    protocol = read(ROOT/'research/occlusion_v2_protocol.json')
    variants, seeds = protocol['variants_in_order'], protocol['seeds']
    destination = ROOT/'results'/f'{args.tag}_summary'
    rows, by_run = [], {}
    for variant in variants:
        for seed in seeds:
            folder = ROOT/'results'/f'{args.tag}_{variant}_s{seed}'
            result, external = read(folder/'results.json'), read(folder/'occluded_reid.json')
            e, history = result['evaluation'], result['history']
            row = dict(variant=variant, seed=seed, clean_map=e['clean']['mAP'], clean_rank1=e['clean']['rank1'],
                       noise30_map=e['synthetic_noise_0.3']['mAP'], noise30_rank1=e['synthetic_noise_0.3']['rank1'],
                       noise50_map=e['synthetic_noise_0.5']['mAP'], noise50_rank1=e['synthetic_noise_0.5']['rank1'],
                       development_map=(e['synthetic_noise_0.3']['mAP']+e['synthetic_noise_0.5']['mAP'])/2,
                       external_map=external['evaluation']['mAP'], external_rank1=external['evaluation']['rank1'],
                       external_rank5=external['evaluation']['rank5'], external_minp=external['evaluation']['mINP'],
                       pk_batches=sum(h['pk_batches'] for h in history),
                       optimizer_updates=sum(h['optimizer_updates'] for h in history),
                       amp_skipped_updates=sum(h['amp_skipped_updates'] for h in history),
                       train_seconds=sum(h['train_seconds'] for h in history),
                       peak_cuda_mb=max(h['peak_cuda_mb'] for h in history),
                       inference_parameters=result['config']['inference_parameters'])
            row.update({'last_'+k: history[-1][k] for k in
                        ('loss', 'ce', 'triplet', 'calibration', 'evidence', 'target_mean', 'identity_mean', 'margin_mean', 'calibration_mae')})
            rows.append(row); by_run[variant, seed] = row
    aggregates = {v: {m: {'mean': statistics.mean(by_run[v, s][m] for s in seeds),
                          'sample_std': statistics.stdev(by_run[v, s][m] for s in seeds)} for m in METRICS} for v in variants}
    comparisons, paired_rows = {}, []
    for added, base, description in PAIRS:
        key = added+'-'+base
        comparisons[key] = {'description': description}
        for metric in METRICS:
            values = {str(s): by_run[added, s][metric] - by_run[base, s][metric] for s in seeds}
            comparisons[key][metric] = {'per_seed': values, 'mean': statistics.mean(values.values()),
                                       'sample_std': statistics.stdev(values.values())}
            paired_rows.append(dict(comparison=key, description=description, metric=metric,
                                    **{'seed_'+s: value for s, value in values.items()},
                                    mean=statistics.mean(values.values()), sample_std=statistics.stdev(values.values())))
    summary = {'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'tag': args.tag,
               'units': 'Metrics percent; paired deltas in percentage points; SD is sample SD across two seeds, not CI.',
               'runs': rows, 'aggregates': aggregates, 'paired_comparisons': comparisons,
               'verification': verification['status'],
               'warning': '651-ID/640-PK-batch feasibility screen. No significance, convergence, SOTA or novelty claim. External domain was seen in v1.'}
    (destination/'metrics.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (destination/'verification.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding='utf-8')
    write_csv(destination/'metrics.csv', rows)
    write_csv(destination/'paired_deltas.csv', paired_rows)
    aggregate_rows = [dict(variant=v, metric=m, **aggregates[v][m]) for v in variants for m in METRICS]
    write_csv(destination/'aggregates.csv', aggregate_rows)
    lines = ['# 第二阶段完整结果（自动从冻结档案生成）', '',
             '数值单位为百分制；差值为百分点。开发遮挡均值 = (30% + 50% 合成噪声 mAP)/2。', '',
             '## 每个种子的原始结果', '',
             '| 变体 | seed | clean mAP | noise30 | noise50 | 开发遮挡均值 | 真实 mAP | 真实 R1 | 实际更新 / 批次 | AMP跳步 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append('| {variant} | {seed} | {clean_map:.2f} | {noise30_map:.2f} | {noise50_map:.2f} | {development_map:.2f} | {external_map:.2f} | {external_rank1:.2f} | {optimizer_updates}/{pk_batches} | {amp_skipped_updates} |'.format(**r))
    lines += ['', '## 两种子均值 ± 样本标准差', '', '误差是两个种子的样本标准差，不是置信区间。', '',
              '| 变体 | clean mAP | noise30 | noise50 | 开发遮挡均值 | 真实 mAP | 真实 R1 |', '|---|---:|---:|---:|---:|---:|---:|']
    for variant in variants:
        lines.append('| '+variant+' | '+' | '.join(f"{aggregates[variant][m]['mean']:.2f} ± {aggregates[variant][m]['sample_std']:.2f}" for m in METRICS)+' |')
    lines += ['', '## 预注册的配对模块差值', '',
              '| 比较（左减右） | 开发 s1 | 开发 s2 | 开发平均 | 真实 s1 | 真实 s2 | 真实平均 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for a, b, _ in PAIRS:
        record = comparisons[a+'-'+b]
        values = []
        for m in ('development_map', 'external_map'):
            values += [record[m]['per_seed'][str(s)] for s in seeds]+[record[m]['mean']]
        lines.append('| '+a+' − '+b+' | '+' | '.join(f'{v:+.2f}' for v in values)+' |')
    lines += ['', '## 末轮训练诊断（不是独立可见性真值评估）', '',
              '| 变体 / seed | target均值 | 身份支持均值 | margin均值 | 校准MAE | 校准loss | 一致性loss | 训练秒数 | 峰值MiB |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append('| {variant} / {seed} | {last_target_mean:.4f} | {last_identity_mean:.4f} | {last_margin_mean:.4f} | {last_calibration_mae:.4f} | {last_calibration:.4f} | {last_evidence:.4f} | {train_seconds:.1f} | {peak_cuda_mb:.1f} |'.format(**r))
    lines += ['', '无对应模块时诊断值为 0 占位，不表示校准完美。不同 target 的 MAE 不代表相同语义任务，不能单凭较小 MAE 判断检索更好。', '',
              '![全部结果](comparison.png)', '', '![模块正负效果](paired_deltas.png)', '',
              '所有组均为 651-ID / 8×80 批次训练，与 v1 的 751-ID / 20×100 正式组不构成直接模块对照。',
              '全部开发完成后才冻结并外测；旧阶段已观察过此真实数据集，因此只能称探索性跨域确认。',
              '预算、checkpoint/source/data-record 哈希、原始日志和顺序校验见 verification.json 与 external_freeze_manifest.json。', '']
    (destination/'table.md').write_text('\n'.join(lines), encoding='utf-8')
    plot(destination, variants, seeds, by_run, aggregates, comparisons)
    print(json.dumps({'status': 'complete', 'runs': len(rows), 'output': str(destination)}, ensure_ascii=True))


if __name__ == '__main__':
    main()
