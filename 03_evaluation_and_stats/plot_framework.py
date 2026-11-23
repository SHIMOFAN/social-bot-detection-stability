"""Framework diagram (K=4 + MH5 + SubBag-ProbVar) as a pure-stdlib SVG, faithful to run_robust_k4.py / run_robust_exp.py."""
from pathlib import Path
import shutil
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIG_DIR = HERE / "figures"
PKG_DIR = ROOT / "experiment_code_package" / "04_results_and_docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PKG_DIR.mkdir(parents=True, exist_ok=True)

FONT = "Microsoft YaHei, PingFang SC, SimHei, Arial, sans-serif"
NAVY, CRIMSON, STEEL, GRAY, AMBER, TEAL = "#1B3B6F", "#C0392B", "#4A7FB5", "#7F8C8D", "#E67E22", "#2E8B8B"
DARK, MUTED, LIGHT, BORDER = "#1E293B", "#475569", "#F8FAFC", "#CBD5E1"

W, H = 1360, 665
o = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
    "<defs>",
]
for name, col in (("navy", NAVY), ("crimson", CRIMSON), ("gray", GRAY)):
    o.append(f'<marker id="a-{name}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 1 L 10 5 L 0 9 z" fill="{col}"/></marker>')
o += ["</defs>", f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>']


def rect(x, y, w, h, fill, stroke=None, rx=6, dash=None, sw=1.2):
    s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"'
    if stroke:
        s += f' stroke="{stroke}" stroke-width="{sw}"'
    if dash:
        s += f' stroke-dasharray="{dash}"'
    o.append(s + "/>")


def text(x, y, s, size=12, weight="400", fill=DARK, anchor="middle"):
    o.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" fill="{fill}">{escape(s)}</text>')


def arrow(pts, col=NAVY, dash=None, sw=1.8, both=False):
    mk = {NAVY: "navy", CRIMSON: "crimson", GRAY: "gray"}[col]
    d = "M " + " L ".join(f"{x},{y}" for x, y in pts)
    s = f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{sw}" marker-end="url(#a-{mk})"'
    if both:
        s += f' marker-start="url(#a-{mk})"'
    if dash:
        s += f' stroke-dasharray="{dash}"'
    o.append(s + "/>")


# ---------- Title ----------
text(W / 2, 32, "方法框架：K=4 局部解冻 + 多头集成 (MH5) + SubBag-ProbVar 子包概率方差训练", 20, "700", NAVY)

# ---------- Training zone ----------
rect(15, 52, W - 30, 448, "#FBFCFE", BORDER, 10)
text(35, 76, "训练阶段：每个账号的 ≤16 条推文只过编码器 1 次，再复用抽取 8 个子包", 13.5, "700", NAVY, "start")

# Col A: inputs
rect(30, 95, 140, 195, LIGHT, BORDER)
text(100, 114, "推文池 (≤16 条)", 12.5, "700")
for i in range(16):
    r, c = divmod(i, 4)
    rect(40 + c * 32, 126 + r * 26, 28, 20, "#DCE7F5", STEEL, 3, sw=1)
text(100, 244, "t₁ … t₁₆", 11.5, "400", MUTED)
text(100, 266, "固定缓存，不额外采集", 11, "400", MUTED)
rect(30, 425, 140, 50, LIGHT, BORDER)
text(100, 446, "账号画像 uᵢ", 12.5, "700")
text(100, 464, "392 维特征", 11.5, "400", MUTED)

# Col B: encoder panel
rect(205, 95, 240, 300, "#FFFFFF", NAVY, 8, sw=1.6)
text(325, 114, "推文分支 · MiniLM-L12", 12.5, "700", NAVY)
rect(220, 125, 210, 52, "#E5E7EB", GRAY, 6)
text(325, 146, "第 1–8 层 · 冻结", 12.5, "700", MUTED)
text(325, 164, "输入隐藏态 fp16 预缓存", 11, "400", MUTED)
rect(220, 192, 210, 75, "#FBE9E7", CRIMSON, 6, sw=1.8)
text(325, 213, "第 9–12 层 · 解冻 (K=4)", 13, "700", CRIMSON)
text(325, 232, "可训练 · encoder_lr = 2e-5", 11.5, "400", CRIMSON)
text(325, 250, "eval 模式（无 Dropout）", 11.5, "400", CRIMSON)
rect(220, 280, 210, 44, "#EEF3FA", STEEL, 6)
text(325, 299, "均值池化 + L2 归一化", 12, "700", NAVY)
text(325, 315, "eⱼ：384 维单位向量", 11, "400", MUTED)
rect(220, 337, 210, 44, "#EEF3FA", STEEL, 6)
text(325, 356, "文本投影 f_text", 12, "700", NAVY)
text(325, 372, "384 → 64 维", 11, "400", MUTED)
arrow([(325, 177), (325, 191)], NAVY, sw=1.5)
arrow([(325, 267), (325, 279)], NAVY, sw=1.5)
arrow([(325, 324), (325, 336)], NAVY, sw=1.5)
arrow([(170, 160), (188, 160), (188, 150), (219, 150)], NAVY)

# profile MLP
rect(220, 425, 210, 50, "#EEF3FA", STEEL, 6)
text(325, 446, "画像分支 f_prof", 12, "700", NAVY)
text(325, 464, "392 → 64 维", 11, "400", MUTED)
arrow([(170, 450), (219, 450)], NAVY)

# Col C: sub-bag sampler
rect(480, 95, 180, 300, "#FFFFFF", AMBER, 8, sw=1.6)
text(570, 114, "子包采样器", 12.5, "700", AMBER)
text(570, 132, "无放回抽 S=8 个子包", 11.5, "700", DARK)
text(570, 148, "每个子包 8 条推文", 11, "400", MUTED)
for i in range(8):
    r, c = divmod(i, 2)
    rect(494 + c * 80, 162 + r * 30, 72, 24, "#FDEBD3", AMBER, 4, sw=1)
    text(530 + c * 80, 179 + r * 30, f"子包 V{i + 1}", 11.5, "600", "#9A5A12")
rect(494, 292, 152, 88, "#FFF8EE", AMBER, 6, sw=1)
text(570, 312, "子包内池化", 12, "700", DARK)
text(570, 330, "均值池化 / SetAttn", 11.5, "400", MUTED)
text(570, 348, "x̄(Vₛ)：64 维", 11.5, "400", MUTED)
text(570, 366, "编码零额外开销", 11, "700", CRIMSON)
arrow([(445, 359), (462, 359), (462, 215), (480, 215)], NAVY)
arrow([(570, 278), (570, 291)], NAVY, sw=1.5)

# Col D: concat + heads
rect(710, 95, 170, 56, "#EEF3FA", STEEL, 6)
text(795, 117, "拼接 [x̄ ; f_prof(u)]", 12, "700", NAVY)
text(795, 135, "128 维", 11, "400", MUTED)
rect(710, 168, 170, 175, "#FFFFFF", TEAL, 8, sw=1.6)
text(795, 187, "MH5：5 个分类头", 12.5, "700", TEAL)
for i in range(5):
    rect(724, 197 + i * 28, 142, 22, "#E3F3F3", TEAL, 4, sw=1)
    text(795, 212 + i * 28, f"头 h{i + 1}（独立初始化）", 11.5, "600", "#1E6B6B")
rect(710, 358, 170, 44, "#E3F3F3", TEAL, 6)
text(795, 377, "σ 后取平均", 12, "700", "#1E6B6B")
text(795, 393, "p_ens(Vₛ)", 11, "400", MUTED)
arrow([(660, 336), (672, 336), (672, 108), (710, 108)], NAVY)
arrow([(430, 450), (689, 450), (689, 138), (710, 138)], NAVY)
arrow([(795, 151), (795, 167)], NAVY, sw=1.5)
arrow([(795, 343), (795, 357)], NAVY, sw=1.5)

# Col E: loss
rect(920, 95, 395, 300, "#FFFFFF", CRIMSON, 8, sw=1.6)
text(1117, 114, "训练目标（类比 R-Drop 的一致性约束，但扰动来自推文子包）", 12, "700", CRIMSON)
for i in range(8):
    rect(938 + i * 46, 128, 40, 24, "#FDECEA", CRIMSON, 4, sw=1)
    text(958 + i * 46, 145, f"p{i + 1}", 11.5, "700", CRIMSON)
arrow([(938, 168), (1296, 168)], CRIMSON, sw=2, both=True)
text(1117, 187, "最小化 8 个子包概率的方差  4·Var(p_ens)", 12, "700", CRIMSON)
rect(938, 202, 359, 108, "#FFF5F4", CRIMSON, 6, sw=1)
text(1117, 224, "L = L_BCE + λ_var · 4·Var(p_ens) + λ_ac · L_anti-collapse", 12.5, "700", DARK)
text(1117, 246, "L_BCE：5 头 × 8 子包逐一计算，pos_weight 平衡类别", 11.5, "400", MUTED)
text(1117, 266, "λ_var = 1.5（边界敏感，p≈0.5 处梯度最大）", 11.5, "400", MUTED)
text(1117, 286, "λ_ac = 0.1（防止推文增量塌缩） · CVaR 权重 = 0", 11.5, "400", MUTED)
rect(938, 324, 359, 56, "#F1F5F9", BORDER, 6, sw=1)
text(1117, 346, "只更新：第 9–12 层 + f_text / f_prof + 5 个分类头", 12, "700", NAVY)
text(1117, 366, "第 1–8 层与缓存隐藏态保持不变", 11.5, "400", MUTED)
arrow([(880, 380), (900, 380), (900, 215), (920, 215)], NAVY)

# gradient return path
arrow([(1117, 395), (1117, 412), (325, 412), (325, 397)], CRIMSON, dash="6,4", sw=1.5)
text(790, 430, "反向传播", 11.5, "700", CRIMSON)

# ---------- Inference zone ----------
rect(15, 512, W - 30, 108, "#F4F9F6", TEAL, 10, dash="7,4", sw=1.4)
rect(30, 540, 150, 40, TEAL, None, 20)
text(105, 565, "推理（部署）", 13.5, "700", "#FFFFFF")
steps = [
    ("账号：固定 8 条推文", "单视图，无社交图"),
    ("MiniLM（K=4 微调权重）", "每条推文编码 1 次"),
    ("均值池化 → f_text ⊕ f_prof", "128 维账号表示"),
    ("MH5 → p_ens", "5 头概率平均"),
    ("p_ens ≥ τ → Bot / Human", "阈值 τ 在 cal 集校准"),
]
for i, (a, b) in enumerate(steps):
    x = 205 + i * 224
    rect(x, 534, 192, 52, "#FFFFFF", TEAL, 6, sw=1.4)
    text(x + 96, 556, a, 12, "700", "#1E6B6B")
    text(x + 96, 574, b, 11, "400", MUTED)
    if i < len(steps) - 1:
        arrow([(x + 193, 560), (x + 223, 560)], GRAY, sw=1.6)
text(W / 2, 608, "评测：同一账号抽 30 个 8 推文视图 × 5 个训练种子，统计 Churn（预测翻转率）与稳定误差；官方 test 集不使用", 11.5, "600", MUTED)

# ---------- Legend ----------
lx = 330
rect(lx, 634, 700, 24, "#F1F5F9", BORDER, 12, sw=1)
rect(lx + 20, 640, 14, 12, "#E5E7EB", GRAY, 2)
text(lx + 40, 650, "冻结层", 11.5, "600", MUTED, "start")
rect(lx + 120, 640, 14, 12, "#FBE9E7", CRIMSON, 2)
text(lx + 140, 650, "解冻可训练层 (K=4)", 11.5, "600", CRIMSON, "start")
rect(lx + 300, 640, 14, 12, "#FDEBD3", AMBER, 2)
text(lx + 320, 650, "子包采样", 11.5, "600", "#9A5A12", "start")
rect(lx + 420, 640, 14, 12, "#E3F3F3", TEAL, 2)
text(lx + 440, 650, "多头集成", 11.5, "600", "#1E6B6B", "start")
o.append(f'<line x1="{lx + 540}" y1="646" x2="{lx + 566}" y2="646" stroke="{CRIMSON}" stroke-width="1.5" stroke-dasharray="6,4"/>')
text(lx + 574, 650, "梯度回传", 11.5, "600", CRIMSON, "start")

o.append("</svg>")
svg = "\n".join(o)
p = FIG_DIR / "fig4_method_framework.svg"
p.write_text(svg, encoding="utf8")
shutil.copy2(p, PKG_DIR / p.name)
print(f"Wrote {p} ({len(svg)} bytes)")
