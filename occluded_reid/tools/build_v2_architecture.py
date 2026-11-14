"""Build a local interactive method atlas, reusing verified OSNet block diagrams."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/'research/architecture'
legacy = (DEST/'网络结构图谱.html').read_text(encoding='utf-8')
style = re.search(r'<style>(.*?)</style>', legacy, re.S).group(1)
sections = []
for key in ('baseline', 'block', 'head'):
    section = re.search(r'<section id="'+key+r'">.*?</section>', legacy, re.S).group(0)
    section = section.replace('512→751', '512→651')
    section = section.replace('① 原始 baseline', '3.1 原始 baseline').replace('③ OSBlock', '3.2 OSBlock').replace('④ 可靠性头', '3.3 可靠性头')
    section = section.replace('该图是遮挡外观变化下的稳定性预测，并非人体分割图。',
                              '本阶段预测的是所选变体的软目标（G·A、G·I 或 G·I·A），并非人体分割图。')
    sections.append(section)


def box(x, y, w, h, title, lines, category='box', ident=None):
    identity = f' id="{ident}"' if ident else ''
    result = f'<g{identity}><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" class="{category}"/>'
    result += f'<text x="{x+w/2}" y="{y+28}" text-anchor="middle" class="t">{title}</text>'
    for i, line in enumerate(lines):
        result += f'<text x="{x+w/2}" y="{y+53+i*23}" text-anchor="middle" class="s">{line}</text>'
    return result+'</g>'


def arrow(path, dashed=False):
    return f'<path d="{path}" class="arrow'+(' dash' if dashed else '')+'"/>'


def svg(content, height, name):
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1300 {height}" role="img" aria-label="{name}"><defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10z" fill="#526985"/></marker></defs>{content}</svg>'


inference = ''.join([
    box(25, 55, 140, 100, '单张 RGB', ['3×256×128', 'Resize / Normalize'], 'std'),
    box(200, 55, 195, 100, 'OSNet x1.0', ['6 个 OSBlock', 'ICCV 2019 · 原样引用']),
    box(430, 55, 160, 100, '空间特征 F', ['512×16×8', '共 128 个位置']),
    box(805, 55, 210, 100, '加权空间池化', ['Σ(W·F) / ΣW', '无头变体使用 GAP'], 'new', 'pool'),
    box(1050, 55, 225, 100, 'FC → L2 normalize', ['512→512 / BN / ReLU', '独立的 512 维检索向量']),
    box(430, 215, 335, 123, '沿用 v1 可靠性头', ['F.detach → 1×1 Conv 512→32', 'GN4 / SiLU → DW3×3 / SiLU', '1×1 Conv 32→1 → sigmoid'], 'new', 'reliability'),
    box(805, 215, 210, 123, '权重下限保护', ['W = 0.05 + 0.95·r', '输出 1×16×8', '只需当前这张图'], 'new', 'floor'),
    arrow('M165 105H200'), arrow('M395 105H430'), arrow('M590 105H805'), arrow('M1015 105H1050'),
    arrow('M510 155V215', True), arrow('M765 274H805'), arrow('M910 215V155'),
    '<text x="27" y="245" class="s">推理不使用：</text><text x="27" y="272" class="s">身份标签 / 其他源图 / donor</text><text x="27" y="299" class="s">原型 / mask 真值 / CLIP / 姿态</text>',
    '<text x="27" y="381" class="s">训练另接 512→651 分类器；CE 与 Triplet 可沿加权池化回传。detach 仅截断头输入回到骨干的支路。</text>'
])
inference = svg(inference, 410, '第二阶段单图推理拓扑')

training = ''.join([
    box(25, 25, 220, 100, 'PK 训练批次', ['6 IDs × 4 draws', '源图索引 u / 身份 y'], 'std'),
    box(280, 25, 270, 100, '固定矩形，替换两个 donor', ['clean x / 遮挡 xa / 遮挡 xb', 'V：未被矩形覆盖的区域'], 'new', 'input_views'),
    box(585, 25, 265, 100, '共享 OSNet · 一次前向', ['72×3×256×128', 'F0 / Fa / Fb：各 24×512×16×8']),
    box(885, 25, 390, 100, '三视图检索与身份训练', ['同一 FC / 分类器；CE + 0.5 Triplet', 'Triplet 按视图分别计算后平均'], 'std'),
    arrow('M245 75H280'), arrow('M550 75H585'), arrow('M850 75H885'),
    box(25, 210, 340, 148, 'clean → 临时身份原型', ['GAP(F0) → 同身份其他源图 p', '排除 anchor 的全部源图副本', '重复源图按出现次数倒数加权', '其他身份批内原型 n；无 memory bank'], 'candidate', 'peer_prototypes'),
    box(405, 210, 405, 148, '两个视图的局部身份 margin', ['m_v(s) = cos(Fv(s), p)', '− max[其他身份] cos(Fv(s), n)', 'I = sigmoid(min(m_a,m_b) / 0.1)', '要求两种 donor 下都保留身份支持'], 'candidate', 'peer_margin'),
    box(850, 210, 425, 148, '几何项 G 与旧稳定性 A', ['G = AvgPool(V) → 1×16×8', 'A = exp(−[1−cos(Fa,Fb)] / 0.15)', 'stability: G·A；peer: G·I', '联合目标: G·I·A'], 'new', 'stability_target'),
    arrow('M700 125V165H195V210', True), arrow('M720 125V180H608V210', True),
    arrow('M740 125V165H1062V210', True), arrow('M365 284H405'),
    box(405, 440, 405, 122, '停止梯度的空间目标 T', ['B×1×16×8；这是软监督', '不是人体分割或身份概率真值', '整个原型 / margin / T 分支 no_grad'], 'candidate', 'detached_target'),
    box(25, 440, 340, 122, '可靠性校准 BCE', ['平均 BCE(ra,T) 和 BCE(rb,T)', '仅对遮挡视图显式校准', '损失权重 0.5；ra/rb 来自 v1 头'], 'new', 'calibration'),
    box(850, 440, 425, 122, '剩余证据一致性（可选）', ['T 重池化 clean 教师与 Fa / Fb', 'clean 教师 detach；cosine 对齐', '原样复用 v1 loss，权重 0.5'], 'new', 'evidence'),
    arrow('M608 358V440'), arrow('M1062 358V400H675V440'),
    arrow('M405 500H365'), arrow('M810 500H850'),
    '<text x="26" y="610" class="s">紫色：本阶段自行实现的监督候选；橙色：v1 沿用；蓝色：OSNet；绿色：常规训练。全部差值均需消融检验。</text>',
    '<text x="26" y="638" class="s">RGANet（TIFS 2024）已有判别性 / 不变性联合评估；DPM（MM 2022）已有原型掩码。此图不构成首次创新声明。</text>'
])
training = svg(training, 666, '第二阶段训练目标生成与梯度边界')

html = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>遮挡 ReID 第二阶段 · 跨图身份证据校准</title><style>'''+style+'''
svg .candidate{fill:#eee6ff;stroke:#7953ab;stroke-width:1.5}.choice{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0}.choice button{border:1px solid #afbdd0;border-radius:6px;padding:8px 12px;cursor:pointer;background:white;font:14px Consolas,monospace}.choice button.active{color:white;background:#355a89;border-color:#355a89}.dim{opacity:.17}#selection{padding:12px 16px;background:#f0f4fa;border-radius:8px;min-height:96px}details summary{cursor:pointer;font-size:20px;font-weight:bold;padding:10px}details{margin:16px 0}.badge{display:inline-block;padding:3px 12px;background:#eee6ff;color:#64428d;border-radius:20px;font-size:13px;font-weight:bold}.stage{font-size:13px;letter-spacing:.12em;color:#526986}footer{color:#4d5f76;font-size:14px}</style></head><body><main>
<header><div class="stage">OCCLUDED PERSON RE-IDENTIFICATION / V2 / 2026-11-14</div><h1>跨图身份证据校准：完整网络与训练分支</h1><p>研究主线：<strong>遮挡下保留可辨认的身份证据</strong>。骨干与可靠性头沿用；本阶段改动的是空间监督目标。参数量仅为设备背景，不作为轻量化贡献。</p><div class="legend"><span class="tag baseline">OSNet · ICCV 2019 · 引用</span><span class="tag project">v1 模块 · 本阶段沿用</span><span class="badge">v2 候选监督 · 自行实现</span><span class="tag standard">通用算子与训练</span></div><div class="note">这是可检验的研究候选，不能据结构图宣称已有效或已证新颖。<a href="../第二阶段实验报告.md">实测正负结果</a> · <a href="../第二阶段方法与来源.md">完整公式与来源</a> · <a href="../原型方法查重记录.md">DPM / RGANet 查重</a></div><nav><a href="#compare">六组对照</a><a href="#infer">单图推理</a><a href="#train">训练目标</a><a href="#baseline">原始 baseline</a><a href="#block">每个 OSBlock</a><a href="#head">可靠性卷积头</a><a href="#sources">来源与边界</a></nav></header>
<section id="compare"><h2>六组使用相同数据划分和三视图批次预算</h2><p>选择变体，查看实际启用的训练目标。图中的淡色块为该变体未使用的部分。</p><div class="choice" role="group" aria-label="选择变体">'''+''.join('<button type="button" data-variant="'+v+'">'+v+'</button>' for v in ('repeat3','paste3','stability','peer','peer_stability','peer_evidence'))+'''</div><div id="selection" aria-live="polite"></div></section>
<section id="infer"><h2>1. 推理：一张图独立得到一个检索向量</h2><p>四个有可靠性头的变体使用完全相同的推理拓扑；repeat3 / paste3 改用 GAP。标签与跨图原型都不进入推理。</p><div class="diagram">'''+inference+'''</div></section>
<section id="train"><h2>2. 训练：用同身份其他源图判断局部证据</h2><p>batch 为 24 张源图；拼接三视图后是 72 张。空间目标的原型、相似度、minimum 和 sigmoid 全部停止梯度；可靠性头和检索网络通过各自损失训练。</p><div class="diagram">'''+training+'''</div><p><strong>repeat3 的特别约定：</strong>同一个已作常规增强的 clean tensor 复制三份，不是三个独立随机增强；它也消耗相同 donor 随机数序列。合成矩形未覆盖区域不等于人体前景。</p></section>
<section><h2>3. 所有卷积与采样块：沿用并展开的骨干</h2><p>下面的 baseline / OSBlock / 可靠性头来自旧版代码追踪图，结构未修改；分类器已按本阶段更新为 <strong>651 类</strong>。可靠性头的监督语义以本页为准。旧版 full 指 v1，不能与 peer_evidence 混称。</p></section>
'''+''.join(sections)+'''
<section id="sources"><h2>4. 来源与贡献边界</h2><table><thead><tr><th>部分</th><th>论文 / 时间</th><th>本项目的关系</th></tr></thead><tbody>
<tr><td>OSNet 全部 stem、OSBlock、ChannelGate、FC</td><td>Zhou et al., ICCV 2019</td><td>官方代码固定版本；原样引用。</td></tr>
<tr><td>遮挡粘贴</td><td>SPT, AAAI 2024 等已有范式</td><td>项目简单矩形双 donor；未引入其 saliency/OIoU/mask rolling。</td></tr>
<tr><td>外观稳定性、可靠性头、一致性</td><td>本项目 v1，2026-11-14 之前已实现</td><td>本阶段沿用；通用 Conv/GN/SiLU，不能再称新模块。</td></tr>
<tr><td>判别性与不变性联合评估</td><td>RGANet, TIFS 2024</td><td>已有概念；未复制其 CLIP 区域与人体部位 memory bank。</td></tr>
<tr><td>原型辅助可见模式选择</td><td>DPM, ACM MM 2022</td><td>已有动机；本项目未移植其 HMG/HEM，必须比较。</td></tr>
<tr><td>排除自身源图的身份 margin + 双干预最小支持 target</td><td>本项目 v2 候选，2026-11-14</td><td>自行定义和实现的监督构造；学术首次与有效性未获确认。</td></tr>
</tbody></table><p>RGANet DOI 中有 2023，但正式论文年为 2024。TGPG（IEEE SPL 2024）目前仅核对元数据/摘要，未取得全文；查重仍有边界。</p><p><a href="../第二阶段方法与来源.md">查看公式、梯度、风险及逐项来源</a> · <a href="网络结构图谱.html">v1 历史图谱</a> · <a href="v2_inference.svg">导出推理 SVG</a> · <a href="v2_training.svg">导出训练 SVG</a></p></section>
<footer>本地交互，无外部请求。对应 code/occlusion_v2/method.py、code/reid/model.py 与固定 OSNet 官方源码。结构图不能替代受控实验。</footer></main>
<script>
const descriptions={repeat3:'同一个增强后的 clean tensor 复制三份；GAP；没有可靠性校准或一致性。作用：匹配批次图像数与 BN 预算。',paste3:'clean + 两个矩形 donor 视图；GAP；没有可靠性校准。与 repeat3 比较遮挡粘贴的作用。',stability:'相同三视图；v1 可靠性头；目标 G × A。与 paste3 比较的是头部、池化与监督整体。',peer:'相同三视图与可靠性头；目标 G × I。用跨图身份 margin 替换外观稳定性；直接比较 peer − stability。',peer_stability:'目标 G × I × A；在 peer 上加入旧外观稳定性衰减。几何项只乘一次。',peer_evidence:'目标 G × I × A；再加入 v1 剩余证据一致性 loss。该 loss 是直接复用，不是本阶段第二个新模块。'};
function labels(id, values){document.querySelectorAll('#'+id+' text').forEach((t,i)=>{if(i<values.length)t.textContent=values[i]});}
function select(v){
  const head=!['repeat3','paste3'].includes(v),peer=v.startsWith('peer');
  document.querySelectorAll('[data-variant]').forEach(b=>{b.classList.toggle('active',b.dataset.variant===v);b.setAttribute('aria-pressed',b.dataset.variant===v)});
  document.getElementById('selection').textContent=v+'：'+descriptions[v];
  for(const id of ['reliability','floor','calibration','detached_target'])document.getElementById(id).classList.toggle('dim',!head);
  for(const id of ['peer_prototypes','peer_margin'])document.getElementById(id).classList.toggle('dim',!peer);
  document.getElementById('stability_target').classList.toggle('dim',!head);
  document.getElementById('evidence').classList.toggle('dim',v!=='peer_evidence');
  labels('pool',head?['加权空间池化','Σ(W·F) / ΣW','单图预测的空间权重']:['全局平均池化 GAP','mean(F, spatial)','不使用可靠性头']);
  document.querySelector('#pool rect').setAttribute('class',head?'new':'box');
  labels('input_views',v==='repeat3'?['同一增强 tensor 复制三份','输入 x / x / x；非独立增强','仍消耗同一 donor 随机数']:['固定矩形，替换两个 donor','clean x / 遮挡 xa / 遮挡 xb','V：未被矩形覆盖的区域']);
  const target=v==='stability'?'G·A':v==='peer'?'G·I':'G·I·A';
  labels('detached_target',['停止梯度的目标 T = '+target,'B×1×16×8；这是软监督','不是人体分割或身份概率真值','目标生成分支全部 no_grad']);
  document.querySelector('#detached_target rect').setAttribute('class',peer?'candidate':'new');
  labels('stability_target',v==='peer'?['几何项 G（此变体不使用 A）','G = AvgPool(V) → 1×16×8','外观稳定性不进入 peer 的 target','当前目标: G·I','各视图均按同一几何项加权']:['几何项 G 与旧稳定性 A','G = AvgPool(V) → 1×16×8','A = exp(−[1−cos(Fa,Fb)] / 0.15)','当前目标: '+target,'几何项只乘一次']);
  document.documentElement.dataset.variant=v;
}
document.querySelectorAll('[data-variant]').forEach(b=>b.addEventListener('click',()=>select(b.dataset.variant)));select(new URLSearchParams(location.search).get('variant') in descriptions ? new URLSearchParams(location.search).get('variant') : 'peer_evidence');
</script></body></html>'''
(DEST/'第二阶段网络图.html').write_text(html, encoding='utf-8')
svg_style = 'text{font-family:Microsoft YaHei,Arial,sans-serif;fill:#18243a}.t{font-size:17px;font-weight:700}.s{font-size:13px}.box{fill:#e7f0ff;stroke:#3774bb;stroke-width:1.5}.new{fill:#fff1d4;stroke:#bf8319;stroke-width:1.5}.std{fill:#e5f5ed;stroke:#5c9a75;stroke-width:1.5}.candidate{fill:#eee6ff;stroke:#7953ab;stroke-width:1.5}.arrow{stroke:#526985;stroke-width:2;fill:none;marker-end:url(#arr)}.dash{stroke-dasharray:6 4}'
for name, content in [('v2_inference', inference), ('v2_training', training)]:
    content = content.replace('<defs>', '<style>'+svg_style+'</style><rect width="1300" height="100%" fill="white"/><defs>', 1)
    (DEST/f'{name}.svg').write_text(content, encoding='utf-8')
print(DEST/'第二阶段网络图.html')
