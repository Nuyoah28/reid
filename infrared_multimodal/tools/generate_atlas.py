"""Generate code-native SVG diagrams and a self-contained explorable network atlas."""
from pathlib import Path
from html import escape
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'research/architecture'
BLUE, ORANGE, TEAL = '#215da8', '#bf5b19', '#167d79'


class Diagram:
    def __init__(self, title, subtitle, height=850):
        self.height = height
        self.parts = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="{height}" viewBox="0 0 1500 {height}" role="img" aria-label="{escape(title)}">
        <defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10" fill="#60758a"/></marker></defs>
        <style>text{{font-family:'Microsoft YaHei','Noto Sans CJK SC',sans-serif;fill:#18344b}}.title{{font-size:28px;font-weight:700}}.sub{{font-size:16px;fill:#52697b}}.label{{font-size:18px;font-weight:700}}.line{{font-size:15px}}.src{{font-size:13px;fill:#52697b}}.wire{{fill:none;stroke:#60758a;stroke-width:2;marker-end:url(#arr)}}</style>
        <rect width="1500" height="{height}" fill="#f7fafc"/>
        <text x="42" y="50" class="title">{escape(title)}</text><text x="42" y="80" class="sub">{escape(subtitle)}</text>''']

    def text(self, x, y, value, cls='line'):
        self.parts.append(f'<text x="{x}" y="{y}" class="{cls}">{escape(value)}</text>')

    def box(self, x, y, w, h, title, lines=(), source='', color=BLUE):
        self.parts.append(f'<g data-box="{x},{y},{w},{h}">')
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="white" stroke="{color}" stroke-width="1.7"/><rect x="{x}" y="{y+12}" width="5" height="{h-24}" rx="2" fill="{color}"/>')
        self.text(x+16,y+29,title,'label')
        for i,line in enumerate(lines):
            self.text(x+16,y+54+i*24,line)
        if source:
            self.text(x+16,y+h-13,source,'src')
        self.parts.append('</g>')

    def arrow(self, points, dashed=False):
        pts = ' '.join(f'{x},{y}' for x,y in points)
        dash = ' stroke-dasharray="7 5"' if dashed else ''
        self.parts.append(f'<polyline points="{pts}" class="wire"{dash}/>')

    def finish(self, name):
        result = '\n'.join(self.parts)+f'<text x="42" y="{self.height-20}" class="src">实现依据：code/model.py · code/objectives.py · 实测尺寸见 layer_trace.json ｜ 2026-11-13</text></svg>'
        (OUT/name).write_text(result,encoding='utf-8')
        return result


def backbone(proposed=False):
    title = '02 / AGW + BCRT：仅在训练阶段增加受约束特征增强' if proposed else '01 / Baseline：AGW 可见光—红外双流网络'
    d = Diagram(title,'蓝色：复用的图像编码路径　橙色：本项目新增训练计算　虚线：训练损失分支；形状省略 batch 维',1080 if proposed else 830)
    d.box(42,120,235,125,'可见光 VIS', ['3 × 288 × 144','RGB 图像；独立 stem'], '训练 8 张 / 推理按模态输入')
    d.box(42,280,235,125,'红外 IR', ['3 × 288 × 144','读取后转换为 3 通道'], '训练 8 张；非像素配对')
    for y,mod in [(120,'VIS'),(280,'IR')]:
        d.box(325,y,390,125,f'{mod} stem：模态独立参数', ['Conv 7×7, 3→64, s2, p3 → BN → ReLU','MaxPool 3×3, s2, p1 → 64×72×36'], 'ResNet [S1]；双流结构 [S7]；AGW [S5]')
        d.arrow([(277,y+63),(325,y+63)])
    d.box(785,202,660,120,'按 batch 拼接 → 后续所有层共享参数', ['训练：8 VIS + 8 IR；按相同身份采样 P=4, K=2/模态','没有 VIS/IR 图像通道拼接；推理只编码当前输入模态'], '双流共享骨干是 AGW 基线的一部分 [S5,S7]')
    d.arrow([(715,182),(750,182),(750,245),(785,245)])
    d.arrow([(715,342),(750,342),(750,280),(785,280)])
    specs = [('Stage 1 · 3 blocks','64 → 256 × 72 × 36','B1 投影捷径；B2–B3 恒等捷径','无 Non-local'),
             ('Stage 2 · 4 blocks','256 → 512 × 36 × 18','B1 的 Conv3×3 stride=2','B3、B4 后各一个 Non-local'),
             ('Stage 3 · 6 blocks','512 → 1024 × 18 × 9','B1 的 Conv3×3 stride=2','B4、B5、B6 后各一个 Non-local'),
             ('Stage 4 · 3 blocks','1024 → 2048 × 18 × 9','B1 stride=1；保留较高分辨率','无 Non-local')]
    for i,(name,shape,conv,nl) in enumerate(specs):
        x=42+i*358
        d.box(x,455,328,143,name,[shape,conv,nl],'ResNet-50 [S1]；Non-local [S2,S5]')
        if i<3: d.arrow([(x+328,523),(x+358,523)])
    d.arrow([(1115,322),(1115,425),(205,425),(205,455)])
    d.box(42,652,390,126,'GeM 全局池化 → f ∈ R²⁰⁴⁸', ['max(x,0)³ → 空间均值 → 立方根','固定 p=3；FP32 计算避免 AMP 溢出'], 'GeM [S3]；AGW 固定幂实现 [S5]')
    d.arrow([(1280,598),(1280,624),(237,624),(237,652)])
    d.box(500,652,390,126,'BNNeck → h ∈ R²⁰⁴⁸', ['BatchNorm1d(2048)，bias 冻结','推理：L2(h) → 2048 维检索向量'], 'BNNeck [S4]；AGW [S5]')
    d.arrow([(432,711),(500,711)])
    d.box(958,652,487,126,'训练：ID 分类 + 加权正则 Triplet', ['Linear 2048→N_id，无 bias → CE','WRT 使用池化特征 f；推理为余弦相似度'], 'CE：通用目标；WRT / AGW [S5]')
    d.arrow([(890,690),(958,690)],True)
    d.arrow([(432,752),(466,752),(466,797),(1200,797),(1200,778)],True)
    if proposed:
        d.box(42,857,390,140,'新增：跨身份残差供体', ['z=L2(f)；每身份 VIS / IR 均值','δ_j = cᵀ_j − cⱽ_j；供体 j ≠ y','δ 和原型停止梯度'], '本项目组合设计 BCRT [S6]，训练专用',ORANGE)
        d.box(500,857,390,140,'新增：解析计算可行步长', ['针对当前 batch 中每个异身份原型','保留至少 ρ=0.5 的初始欧式间隔','t=min(α=0.5, 所有有效边界上限)'], '本项目推导与实现 [S6]，非泛化证明',ORANGE)
        d.box(958,857,487,140,'新增：虚拟特征的原型监督', ['v=L2(z+t·d) → 原型余弦 logits','温度 T=0.1 → CE_virtual','总损失 CE + WRT + 0.2 CE_virtual'], '前 5 epoch 辅助权重为 0；无新增推理参数',ORANGE)
        d.arrow([(235,778),(235,857)],True)
        d.arrow([(432,921),(500,921)],True)
        d.arrow([(890,921),(958,921)],True)
        d.text(42,1033,'BCRT 是待验证的研究候选；“残差生成/原型监督”已有前人研究，创新主张限定为跨身份供体与身份间隔约束的结合。','sub')
    return d.finish('bcrt.svg' if proposed else 'baseline.svg')


def bottleneck():
    d=Diagram('03 / ResNet Bottleneck：每个卷积与捷径','16 个块均遵循此结构；下方网页逐块列出输入通道、stride、输出尺寸与 Non-local 插入位置。',690)
    d.box(42,155,215,115,'输入 X',['C_in × H × W'],'ResNet-50 [S1]')
    d.box(310,155,265,115,'Conv 1×1 → BN → ReLU',['C_in → C_mid','stride=1，padding=0'],'降维；bias=False')
    d.box(625,155,265,115,'Conv 3×3 → BN → ReLU',['C_mid → C_mid','stride=s，padding=1'],'提取空间特征；bias=False')
    d.box(940,155,265,115,'Conv 1×1 → BN',['C_mid → 4 C_mid','stride=1，padding=0'],'升维；这里不先做 ReLU')
    d.box(1260,155,198,115,'Add → ReLU',['与捷径逐元素相加','输出 C_out'],'C_out=4 C_mid')
    for a,b in [(257,310),(575,625),(890,940),(1205,1260)]: d.arrow([(a,210),(b,210)])
    d.box(460,340,650,123,'捷径分支', ['每个 stage 的 B1：Conv 1×1, C_in→C_out, stride=s → BN','其余块：Identity；输出与主分支形状相同'], '投影也使用 bias=False；即使 s=1，通道改变仍需投影')
    d.arrow([(150,270),(150,400),(460,400)])
    d.arrow([(1110,400),(1355,400),(1355,270)])
    d.box(42,510,1416,116,'本项目实际 stride / 通道配置', ['Stage1: C_mid=64, 3 blocks, s=1；Stage2: C_mid=128, 4 blocks, 首块 s=2；Stage3: C_mid=256, 6 blocks, 首块 s=2',
        'Stage4: C_mid=512, 3 blocks, 首块 s=1。没有上采样、解码器或转置卷积；最大空间下采样倍数为 16。'],'详细模块名对应 stages.0–3；stages.s.b 内含 conv1/bn1/conv2/bn2/conv3/bn3/relu/downsample')
    return d.finish('bottleneck.svg')


def nonlocal_detail():
    d=Diagram('04 / Non-local：保留官方 AGW 的窄通道实现','来源：Wang et al., Non-local Neural Networks, CVPR 2018；实际公式与通道数按 AGW commit e817a267。',850)
    d.box(42,170,215,120,'输入 X',['B × C × H × W','N = H·W'],'C=512 或 1024 [S2,S5]')
    for y,name,note in [(120,'θ','B × N × 1'),(285,'φ','B × 1 × N'),(450,'g','B × N × 1')]:
        d.box(335,y,290,120,f'{name}: Conv 1×1, C→1',['stride=1, bias=True','展平 / 转置 → '+note],'AGW 代码中 inter_channels=1')
        d.arrow([(257,225),(290,225),(290,y+60),(335,y+60)])
    d.box(720,193,350,130,'A = (θ(X) @ φ(X)) / N',['形状 B × N × N','点积后除以位置数 N；没有 softmax'],'S2: N=648；S3: N=162')
    d.arrow([(625,180),(670,180),(670,230),(720,230)])
    d.arrow([(625,345),(670,345),(670,285),(720,285)])
    d.box(1140,380,310,140,'A @ g(X) → reshape',['B × N × 1','重排成 B × 1 × H × W'],'聚合全图上下文')
    d.arrow([(1070,260),(1290,260),(1290,380)])
    d.arrow([(625,510),(1010,510),(1010,450),(1140,450)])
    d.box(720,630,370,125,'输出投影：Conv 1×1, 1→C',['bias=True → BN2d(C)','BN 的 γ、β 均以 0 初始化'],'因此初始时残差支路为 0')
    d.box(310,630,340,125,'Add：X + projected context',['输出 B × C × H × W','没有额外 ReLU 或空间采样'],'位置：S2B3/B4；S3B4/B5/B6')
    d.arrow([(1290,520),(1290,690),(1090,690)])
    d.arrow([(720,690),(650,690)])
    d.arrow([(150,290),(150,690),(310,690)])
    d.text(42,802,'注意：此处的 1 通道不是通行的 C/2 配置；为了可核对地复用 AGW，本项目保留上游行为并提供数值等价测试。','sub')
    return d.finish('nonlocal.svg')


def transport():
    d=Diagram('05 / BCRT：供体残差 → 身份间隔 → 安全步长 → 训练监督','本项目候选机制（2026）；固定 prototype 的欧式几何有解析约束，后续归一化余弦 CE 和泛化没有该保证。',1070)
    d.box(42,125,425,160,'① 采样与特征归一化',['4 个身份 × 每模态 2 张 = 16 张','共享图像编码只执行一次','z_i=L2(f_i)；无额外图像前向'],'输入来自 baseline 的 GeM 输出',TEAL)
    d.box(538,125,425,160,'② 估计模态中心与原型',['cⱽ_j=mean(z of VIS identity j)','cᵀ_j=mean(z of IR identity j)','p_j=stopgrad((cⱽ_j+cᵀ_j)/2)'],'batch 局部原型；不跨 batch 存储',ORANGE)
    d.box(1030,125,420,160,'③ 选择所有其他身份供体',['δ_j=stopgrad(cᵀ_j−cⱽ_j), j≠y_i','VIS 锚点：d=+δ_j；IR 锚点：d=−δ_j','16 × (4−1) = 48 个候选虚拟对'],'排除同一身份以减少标签专属残差捷径',ORANGE)
    d.arrow([(467,205),(538,205)]); d.arrow([(963,205),(1030,205)])
    d.box(42,365,675,178,'④ 对每个异身份 k 计算欧式间隔',['g_k(0) = ||z−p_k||² − ||z−p_y||²','a_k = 2 d · (p_y−p_k)','g_k(t) = g_k(0) + t a_k','任一初始间隔 ≤ 1e−6：该锚点不参与辅助损失'],'原型 / 方向 / 步长停止梯度；当前 z 保留梯度',ORANGE)
    d.box(785,365,665,178,'⑤ 解析求可行上限，不增加可学习参数',['要求 g_k(t) ≥ ρ g_k(0)，ρ=0.5','仅 a_k < −1e−8 时产生上界：','u_k = (1−ρ) g_k(0) / (−a_k)','t = clip(min_k u_k, 0, α=0.5)；无上界时 t=α'],'约束对象是未归一化的 z+t·d 与固定原型',ORANGE)
    d.arrow([(1240,285),(1240,325),(375,325),(375,365)])
    d.arrow([(717,450),(785,450)])
    d.box(42,620,675,153,'⑥ 虚拟特征监督',['v=L2(z+t·d)','logits=v · L2(p)ᵀ / T，T=0.1','L_virtual=mean(CE(logits,y))，只计 eligible 对'],'反向传播到当前锚点及编码网络；供体与原型为常量目标',ORANGE)
    d.box(785,620,665,153,'⑦ 总损失与推理',['L = L_CE + L_WRT + λ L_virtual','λ=0（前 5 epoch），之后 λ=0.2','推理丢弃整个 BCRT 计算；仍为 L2(BNNeck(f))'],'改进只发生在训练目标中；额外推理参数数目=0',TEAL)
    d.arrow([(1110,543),(1110,580),(375,580),(375,620)])
    d.arrow([(717,694),(785,694)])
    d.box(42,850,1408,147,'四组消融各自回答什么问题',['AGW：原基线 CE+WRT。Center：加普通跨模态中心 L2，检验简单对齐是否已足够。',
        'Unbounded：相同供体 / eligible 掩码 / 虚拟 CE，但 t 恒为 α。BCRT：仅将 t 改为解析受约束步长。',
        '报告 capped_fraction、eligible_fraction、间隔违反数及 Rank-1 / mAP；若约束几乎不触发或精度不升，必须如实报告。'],'特征生成与原型学习不是本项目首创：相关工作 PDM / ICASSP 2025；这里未复制其模块或代码')
    return d.finish('transport.svg')


SOURCES = [
('S1','ResNet-50 / Bottleneck','He et al. · Deep Residual Learning for Image Recognition · CVPR 2016','https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html','复用：16 个瓶颈块与投影捷径；AGW 将最后 stage 首块 stride 改为 1。'),
('S2','Non-local','Wang et al. · Non-local Neural Networks · CVPR 2018','https://openaccess.thecvf.com/content_cvpr_2018/html/Wang_Non-Local_Neural_Networks_CVPR_2018_paper.html','复用：长程关系聚合。以 AGW 实际实现为准，C→1、除 N、无 softmax。'),
('S3','GeM','Radenović et al. · Fine-Tuning CNN Image Retrieval with No Human Annotation · TPAMI online 2018 / 卷期 2019','https://doi.org/10.1109/TPAMI.2018.2846566','复用池化思想；AGW 固定 p=3；FP32 是本项目数值稳定性适配，不算研究创新。'),
('S4','BNNeck','Luo et al. · Bag of Tricks and a Strong Baseline for Deep Person Re-identification · CVPRW 2019','https://openaccess.thecvf.com/content_CVPRW_2019/html/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.html','复用：区分分类/度量使用的特征；2048 维 BN 归一化。'),
('S5','AGW baseline / WRT','Ye et al. · Deep Learning for Person Re-identification: A Survey and Outlook · arXiv 2020 / TPAMI online 2021 / 卷期 2022','https://github.com/mangye16/Cross-Modal-Re-ID-baseline','完整基线选择，固定 commit e817a2672c1e14e8802617acb980a79520ef9545，保留 MIT。'),
('S6','BCRT 研究候选','本项目 · 2026-11-13 · 尚无已发表论文','../创新来源与推导.md','自行推导并实现供体排除、方向符号、所有原型间隔上限及相同 eligible 对照；不能称已证实新颖或有效。'),
('S7','可见光 / 红外双流设计','Ye et al. · Bi-directional Center-Constrained Top-Ranking for Visible Thermal Person Re-Identification · TIFS 2019','https://doi.org/10.1109/TIFS.2018.2885254','AGW 官方 README 引用的双流结构来源；本项目通过 AGW 继承，并非新增贡献。'),
('S8','相关工作：PDM','Prototype-Driven Multi-Feature Generation for Visible-Infrared Person Re-identification · ICASSP 2025（预印本 2024）','https://arxiv.org/abs/2409.05642','特征生成、原型对齐已有先例；用于界定创新边界，未复制 PDM 模块。')]


def blocks():
    sections=[]
    current=64
    h,w=72,36
    for s,(count,mid) in enumerate(zip([3,4,6,3],[64,128,256,512]),1):
        cards=[]
        for b in range(1,count+1):
            stride=2 if s in [2,3] and b==1 else 1
            oh,ow=h//stride,w//stride
            nl=(s==2 and b in [3,4]) or (s==3 and b in [4,5,6])
            shortcut=f'Conv1×1 {current}→{4*mid}, s{stride} → BN · bias=False' if b==1 else 'Identity（恒等映射）'
            wide_shortcut_paths='<path d="M43 100 V146 H245" class="ba"/><path d="M550 146 H708 V105" class="ba"/>'
            flow=f'''<svg class="blockflow" viewBox="0 0 760 188" role="img" aria-label="S{s}B{b} 逐卷积结构">
              <defs><marker id="arr{s}_{b}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10" fill="#60758a"/></marker></defs>
              <style>.bf{{font:13px 'Microsoft YaHei',sans-serif;fill:#18344b}}.bt{{font:700 13px 'Microsoft YaHei',sans-serif;fill:#18344b}}.ba{{fill:none;stroke:#60758a;stroke-width:1.7;marker-end:url(#arr{s}_{b})}}.bb{{fill:#fff;stroke:#215da8;stroke-width:1.3}}.bs{{fill:#fff7ef;stroke:#bf5b19;stroke-width:1.3}}</style>
              <rect x="5" y="48" width="77" height="52" rx="6" class="bb"/><text x="43" y="68" text-anchor="middle" class="bt">输入</text><text x="43" y="87" text-anchor="middle" class="bf">{current}×{h}×{w}</text>
              <rect x="111" y="38" width="145" height="70" rx="6" class="bb"/><text x="183" y="59" text-anchor="middle" class="bt">Conv1 + BN + ReLU</text><text x="183" y="79" text-anchor="middle" class="bf">1×1 / s=1 / p=0</text><text x="183" y="98" text-anchor="middle" class="bf">{current}→{mid} 通道</text>
              <rect x="287" y="38" width="145" height="70" rx="6" class="bb"/><text x="359" y="59" text-anchor="middle" class="bt">Conv2 + BN + ReLU</text><text x="359" y="79" text-anchor="middle" class="bf">3×3 / s={stride} / p=1</text><text x="359" y="98" text-anchor="middle" class="bf">{mid}→{mid} 通道</text>
              <rect x="463" y="38" width="145" height="70" rx="6" class="bb"/><text x="535" y="59" text-anchor="middle" class="bt">Conv3 + BN</text><text x="535" y="79" text-anchor="middle" class="bf">1×1 / s=1 / p=0</text><text x="535" y="98" text-anchor="middle" class="bf">{mid}→{4*mid} 通道</text>
              <rect x="661" y="42" width="94" height="62" rx="6" class="bb"/><text x="708" y="64" text-anchor="middle" class="bt">Add</text><text x="708" y="83" text-anchor="middle" class="bf">ReLU</text>
              <path d="M82 74 H111 M256 74 H287 M432 74 H463 M608 74 H661" class="ba"/>
              {wide_shortcut_paths}<rect x="245" y="126" width="305" height="40" rx="5" class="bs"/><text x="397" y="150" text-anchor="middle" class="bf">捷径：{escape(shortcut)}</text>
              <text x="12" y="179" class="bf">输出 {4*mid}×{oh}×{ow}{' · Non-local 紧随此块' if nl else ''}</text></svg>'''
            mobile_shortcut = f'投影 1×1, s{stride}, p=0, {current}→{4*mid} → BN' if b==1 else 'Identity（恒等映射，无参数）'
            mobile_shortcut_paths = ('<path d="M100 29 H14 V389 H25" class="ma"/><path d="M335 389 H344 V315 H242" class="ma"/>'
                                     if b==1 else '<path d="M100 29 H18 V315 H118" class="ma"/>')
            mobile_flow=f'''<svg class="blockflow-mobile" viewBox="0 0 360 440" role="img" aria-label="S{s}B{b} 窄屏逐卷积结构">
              <defs><marker id="marr{s}_{b}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10" fill="#60758a"/></marker></defs>
              <style>.mf{{font:12px 'Microsoft YaHei',sans-serif;fill:#18344b}}.mt{{font:700 12px 'Microsoft YaHei',sans-serif;fill:#18344b}}.ma{{fill:none;stroke:#60758a;stroke-width:1.7;marker-end:url(#marr{s}_{b})}}.mb{{fill:#fff;stroke:#215da8;stroke-width:1.3}}.ms{{fill:#fff7ef;stroke:#bf5b19;stroke-width:1.3}}</style>
              <rect x="100" y="5" width="160" height="48" rx="6" class="mb"/><text x="180" y="25" text-anchor="middle" class="mt">输入 {current}×{h}×{w}</text><text x="180" y="43" text-anchor="middle" class="mf">进入主支路与捷径</text>
              <rect x="42" y="77" width="276" height="54" rx="6" class="mb"/><text x="180" y="97" text-anchor="middle" class="mt">Conv1 1×1 / stride=1 / padding=0</text><text x="180" y="118" text-anchor="middle" class="mf">{current}→{mid} · BN · ReLU · bias=False</text>
              <rect x="42" y="146" width="276" height="54" rx="6" class="mb"/><text x="180" y="166" text-anchor="middle" class="mt">Conv2 3×3 / stride={stride} / padding=1</text><text x="180" y="187" text-anchor="middle" class="mf">{mid}→{mid} · BN · ReLU · bias=False</text>
              <rect x="42" y="215" width="276" height="54" rx="6" class="mb"/><text x="180" y="235" text-anchor="middle" class="mt">Conv3 1×1 / stride=1 / padding=0</text><text x="180" y="256" text-anchor="middle" class="mf">{mid}→{4*mid} · BN · bias=False</text>
              <rect x="118" y="292" width="124" height="47" rx="6" class="mb"/><text x="180" y="312" text-anchor="middle" class="mt">残差相加</text><text x="180" y="330" text-anchor="middle" class="mf">Add · ReLU</text>
              <path d="M180 53 V77 M180 131 V146 M180 200 V215 M180 269 V292" class="ma"/>
              {mobile_shortcut_paths}
              <rect x="25" y="365" width="310" height="48" rx="5" class="ms"/><text x="180" y="385" text-anchor="middle" class="mt">捷径：{escape(mobile_shortcut)}</text><text x="180" y="403" text-anchor="middle" class="mf">输出 {4*mid}×{oh}×{ow}</text>
              </svg>'''
            cards.append(f'''<article class="block"><h4>S{s}B{b} <code>stages.{s-1}.{b-1}</code></h4>
            <p class="shape">输入 {current}×{h}×{w} → 输出 {4*mid}×{oh}×{ow}</p>{flow}{mobile_flow}<p class="small">来源 [S1]{' + [S2,S5]' if nl else ''}。卷积均 bias=False；BN eps=1e−5, momentum=0.1。</p></article>''')
            current,h,w=4*mid,oh,ow
        sections.append(f'<h3>Stage {s} · {count} 个块</h3><div class="blocks">'+''.join(cards)+'</div>')
    return ''.join(sections)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    figures = [('baseline',backbone()),('bcrt',backbone(True)),('bottleneck',bottleneck()),('nonlocal',nonlocal_detail()),('transport',transport())]
    figure_html=''.join(f'<section id="{name}" class="figure"><div class="figurebar"><a href="{name}.svg" download>下载 SVG</a><button onclick="zoom(this,1.2)">放大</button><button onclick="zoom(this,0.833333)">缩小</button><button onclick="resetZoom(this)">适应宽度</button></div><div class="viewport">{svg}</div></section>' for name,svg in figures)
    sources=''.join(f'<tr id="{ident}"><td>{ident}<br><b>{escape(name)}</b></td><td><a href="{escape(url)}">{escape(citation)}</a></td><td>{escape(use)}</td></tr>' for ident,name,citation,url,use in SOURCES)
    trace=json.loads((OUT/'layer_trace.json').read_text(encoding='utf-8'))
    trace_rows=''.join(f'<tr><td>{i+1}</td><td><code>{escape(r["module"])}</code></td><td>{escape(r["type"])}</td><td>{escape(r["spec"])}</td><td>{escape(str(r["output"]))}</td></tr>' for i,r in enumerate(trace))
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AGW 与 BCRT · 红外多模态网络图谱</title>
    <style>*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f0f5f8;color:#18344b;font:16px/1.7 'Microsoft YaHei',sans-serif}header,main{max-width:1564px;margin:auto;padding:30px 32px}header{padding-top:45px}h1{font-size:34px;line-height:1.4;margin:0 0 12px}h2{font-size:26px;margin:0 0 12px}h3{margin:28px 0 14px}p{margin:8px 0 15px}.kicker{color:#215da8;font-size:13px;letter-spacing:2px;font-weight:bold}nav{position:sticky;top:0;z-index:10;background:#fff;border-block:1px solid #d6e2eb;padding:13px 25px;display:flex;gap:23px;flex-wrap:wrap}a{color:#215da8;text-decoration:none}a:hover{text-decoration:underline}section{scroll-margin-top:90px;margin-bottom:30px}.panel{background:white;border:1px solid #d9e4ec;border-radius:14px;padding:26px}.figure{border:1px solid #cddce7;border-radius:14px;overflow:hidden;background:#fff}.figurebar{display:flex;gap:16px;padding:12px 20px;align-items:center;font-size:14px}.viewport{overflow:auto}.viewport svg{display:block;width:100%;height:auto;max-width:none}.figurebar button{border:1px solid #ccdce7;background:#f7fafc;padding:7px 13px;border-radius:6px;cursor:pointer;color:#18344b}.blocks{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.block{border:1px solid #d9e4ec;border-left:4px solid #215da8;border-radius:8px;padding:18px 20px;background:#fafcfe;overflow-x:auto}.blockflow{width:100%;min-width:700px;height:auto;display:block;margin:4px 0}.block h4{margin:0 0 8px;font-size:19px}.block code{font-size:13px;margin-left:12px}.shape{color:#167d79;font-weight:600}.small{font-size:13px;color:#567083}li{margin:4px 0}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;vertical-align:top;border-bottom:1px solid #dde6ed;padding:12px}th{background:#eef4f8}code{font:13px/1.6 Consolas,monospace;overflow-wrap:anywhere}.tablewrap{overflow:auto}.callout{border-left:4px solid #bf5b19;background:#fff6ed;padding:16px 20px;border-radius:7px}.two{display:grid;grid-template-columns:1fr 1fr;gap:22px}.metric{font-size:30px;font-weight:bold}input[type=range]{width:100%}.toy{background:#f5f9fc;border-radius:10px;padding:18px}summary{cursor:pointer;font-weight:700}#traceSearch{padding:10px;border:1px solid #bccfdd;border-radius:6px;width:100%;margin:15px 0}.pill{display:inline-block;padding:4px 10px;background:#e6f3f1;border-radius:20px;font-size:13px;color:#167d79}footer{padding:30px;color:#52697b;text-align:center}@media(max-width:850px){header,main{padding:20px 14px}.blocks,.two{grid-template-columns:1fr}h1{font-size:27px}.panel{padding:18px}.viewport svg{min-width:1100px}nav{gap:12px;font-size:14px}}
    </style><header><p class="kicker">VISIBLE–INFRARED PERSON RE-IDENTIFICATION / 2026</p><h1>AGW 与 BCRT：从整网到每个卷积块</h1><p>具体方向：监督式可见光—红外行人重识别；以不增加推理分支的身份边界约束特征增强为研究候选。</p><span class="pill">RTX 4060 · mmi-unet</span> <span class="pill">两模态 · 单个 2048 维检索向量</span></header>
    <nav><a href="#baseline">Baseline</a><a href="#bcrt">改进模型</a><a href="#bottleneck">卷积与捷径</a><a href="#nonlocal">Non-local</a><a href="#transport">BCRT 细节</a><a href="#every-block">16 块逐层表</a><a href="#sources">出处</a><a href="#trace">运行层记录</a></nav><main>
    <section class="panel"><h2>如何阅读</h2><p>两张整网图共用完全相同的图像编码器。蓝色框来自既有论文与 AGW 实现；橙色框为本项目训练阶段的候选机制。下采样仅出现在 stem、Stage2 首块和 Stage3 首块；本网络没有上采样。各图可放大、横向滚动或单独下载 SVG。</p><div class="callout">“自行设计”描述代码和推导来源，不等于已证明学术首创。特征生成、跨模态中心和原型监督均有已有工作。BCRT 的可检验主张是：从其他身份借用残差时，通过解析步长减少当前 batch 中固定身份原型的间隔破坏。实测结论见 <a href="../实验记录.md">实验记录</a>。</div></section>'''+figure_html+'''
    <section id="every-block" class="panel"><h2>06 / 16 个 Bottleneck 完整展开</h2><p>张量格式为 C×H×W。每个 stage 首块都有投影捷径；S1B1 和 S4B1 虽然 stride=1，仍因通道变化使用投影。Non-local 紧接该块最终 ReLU 之后。</p>'''+blocks()+'''</section>
    <section class="panel"><h2>07 / 身份间隔约束的可交互例子</h2><p>一维演示：当前特征 x=0.2，本身份原型 p₀=0，异身份原型 p₁=1，方向 d=2。此例仅说明几何机制，不能作为检索精度结果。</p><div class="two"><div class="toy"><label>候选步长 α：<b id="alphaText"></b><input id="alpha" type="range" min="0" max="0.6" step="0.005" value="0.5"></label><label>间隔保留比例 ρ：<b id="rhoText"></b><input id="rho" type="range" min="0" max="1" step="0.01" value="0.5"></label><p id="toyFormula"></p></div><div class="toy"><p>未约束 / 受约束移动后的欧式间隔</p><div class="metric" id="toyResult"></div><p id="toyExplanation"></p></div></div><p class="small">在真实训练中，约束同时面对 batch 内所有异身份原型；用最小上限得到可行步长。后续特征与原型归一化会改变决策几何，因此不宣称此间隔保证自动延伸到余弦分类器。</p></section>
    <section id="sources" class="panel"><h2>08 / 模块出处、年份与复用性质</h2><div class="tablewrap"><table><thead><tr><th>编号 / 模块</th><th>出处与时间</th><th>本项目中的作用与归属</th></tr></thead><tbody>'''+sources+'''</tbody></table></div><p>普通 center 消融只是均值中心的 L2 对齐，不是对某篇 Hetero-center Triplet 论文的完整复现。BatchNorm、ReLU、SGD、交叉熵、L2 归一化是常用运算，不计入创新贡献。</p></section>
    <section id="trace" class="panel"><h2>09 / 真实前向运行的全部叶子层记录</h2><p>输入为各一张 VIS/IR，即拼接后 B=2；下表直接来自 layer_trace.json。重复的 ReLU 是同一模块在瓶颈内被调用多次。函数运算 cat、残差 add、GeM 的 pow/mean、矩阵乘法与 normalize 已在上方图中补全。</p><details><summary>展开 '''+str(len(trace))+''' 次叶子模块调用（可搜索）</summary><input id="traceSearch" placeholder="筛选，例如 stages.2.3 或 Conv2d" oninput="filterTrace()"><div class="tablewrap"><table id="traceTable"><thead><tr><th>#</th><th>模块路径</th><th>类型</th><th>卷积 / BN / Pool 参数</th><th>输出形状</th></tr></thead><tbody>'''+trace_rows+'''</tbody></table></div></details></section>
    </main><footer>本地可复查图谱 · 图由 tools/generate_atlas.py 生成 · 不依赖外部脚本或网络服务</footer><style>.blockflow-mobile{display:none}@media(min-width:851px){.blockflow{min-width:0}}@media(max-width:850px){.blockflow{display:none}.blockflow-mobile{display:block;width:100%;height:auto;min-width:0}}</style><script>
    function zoom(btn,f){const v=btn.closest('.figure').querySelector('svg');const n=(+(v.dataset.zoom||1))*f;v.dataset.zoom=n;v.style.width=(100*n)+'%'}function resetZoom(btn){const v=btn.closest('.figure').querySelector('svg');v.dataset.zoom=1;v.style.width='100%'}
    function toy(){const a=+document.querySelector('#alpha').value,r=+document.querySelector('#rho').value,t=Math.min(a,(1-r)*0.6/4);document.querySelector('#alphaText').textContent=a.toFixed(3);document.querySelector('#rhoText').textContent=r.toFixed(2);document.querySelector('#toyFormula').textContent='g(t)=0.6−4t；t=min(α, (1−ρ)×0.6/4)='+t.toFixed(4);document.querySelector('#toyResult').textContent=(.6-4*a).toFixed(3)+' / '+(.6-4*t).toFixed(3);document.querySelector('#toyExplanation').textContent='目标至少保留 '+(.6*r).toFixed(3)+' 间隔。'+(a>t?'当前约束触发，缩短移动步长。':'当前方向与候选步长无需截断。')};document.querySelector('#alpha').addEventListener('input',toy);document.querySelector('#rho').addEventListener('input',toy);toy();function filterTrace(){const q=document.querySelector('#traceSearch').value.toLowerCase();document.querySelectorAll('#traceTable tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q))}
    </script></html>'''
    (OUT/'network.html').write_text(html,encoding='utf-8')
    print(f'Wrote network.html and {len(figures)} SVG diagrams; {len(trace)} layer calls.')


if __name__=='__main__':
    main()
