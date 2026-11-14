"""Label retained v1 narrative without editing old experimental data."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]/'research'
for name in ['研究方案.md', '创新来源清单.md', '实验记录.md', '交付清单.md', '论文框架.md', '下一步研究决策.md']:
    path = root/name
    text = path.read_text(encoding='utf-8')
    if '**阶段说明（2026-11-14 更新）' in text:
        continue
    first, rest = text.split('\n', 1)
    note = ('\n\n> **阶段说明（2026-11-14 更新）：本文正文保留为 v1 历史记录。** '
            '当前主线仍为遮挡 ReID；本阶段新增跨源图身份支持监督，并执行六变体、两种子的 v2 对照。'
            '当前方法见 [第二阶段方法与来源](第二阶段方法与来源.md)，实测正负结果见 '
            '[第二阶段实验报告](第二阶段实验报告.md)，新网络图见 '
            '[第二阶段网络图](architecture/第二阶段网络图.html)。'
            '正文中的“本轮”“尚未追加”“下一次”指 v1 当时状态，不代表 v2 状态。旧实验数字与原始档案保留。\n')
    path.write_text(first+note+rest, encoding='utf-8')
path = root/'architecture/网络结构图谱.html'
text = path.read_text(encoding='utf-8')
text = text.replace('<h1>OSNet 基线与遮挡 ReID 候选模型结构图谱</h1>',
                    '<h1>v1 历史图谱：OSNet 基线与遮挡 ReID 候选模型</h1>\n'
                    '<div class="note"><strong>这是第一阶段 v1 的历史结构与结果。</strong>'
                    '本阶段 v2 的身份原型、局部 margin 和双干预目标见 '
                    '<a href="第二阶段网络图.html">第二阶段完整网络图</a>。'
                    '骨干及可靠性头结构沿用，v1 的 full 不等同于 v2 的 peer_evidence。</div>')
text = text.replace('对应仓库当前代码：', '对应仓库保留的 v1 代码：')
path.write_text(text, encoding='utf-8')
print('Marked six documents and the v1 atlas as historical.')
