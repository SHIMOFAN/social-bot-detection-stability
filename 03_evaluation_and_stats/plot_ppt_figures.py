"""Generate 3 publication-grade vector SVG figures for PPT directly from verified CSV/NPZ outputs."""
import csv
import json
from pathlib import Path
import shutil
import sys
from xml.sax.saxutils import escape

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for p in (str(ROOT / "scripts"), str(ROOT / "work" / "antigravity_three_experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

import common_eval as ce
from experiment_common import prediction_data, probability

FIG_DIR = HERE / "figures"
PKG_FIG_DIR = ROOT / "experiment_code_package" / "04_results_and_docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PKG_FIG_DIR.mkdir(parents=True, exist_ok=True)

FONT = "Microsoft YaHei, PingFang SC, SimHei, Arial, sans-serif"
NAVY = "#1B3B6F"
CRIMSON = "#C0392B"
STEEL = "#4A7FB5"
TEAL = "#2E8B8B"
AMBER = "#E67E22"
GRAY = "#7F8C8D"
LIGHT_BG = "#F8FAFC"
GRID = "#E2E8F0"
TEXT_DARK = "#1E293B"
TEXT_MUTED = "#475569"


def load_summary():
    rows = {}
    with (HERE / "robust_summary.csv").open("r", encoding="utf8") as f:
        for r in csv.DictReader(f):
            rows[(r["method"], r["calibration_mode"])] = {
                k: (float(v) if k not in ("method", "calibration_mode") else v)
                for k, v in r.items()
            }
    return rows


def star_points(cx, cy, r_out=10, r_in=4.5):
    pts = []
    for i in range(10):
        ang = np.pi / 2 + i * np.pi / 5
        r = r_out if i % 2 == 0 else r_in
        pts.append(f"{cx + r * np.cos(ang):.2f},{cy - r * np.sin(ang):.2f}")
    return " ".join(pts)


def svg_header(w, h):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        "<defs>",
        '  <marker id="arrow-crimson" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
        f'    <path d="M 0 1 L 10 5 L 0 9 z" fill="{CRIMSON}"/>',
        "  </marker>",
        '  <marker id="arrow-navy" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
        f'    <path d="M 0 1 L 10 5 L 0 9 z" fill="{NAVY}"/>',
        "  </marker>",
        "</defs>",
        f'<rect width="{w}" height="{h}" fill="#FFFFFF" rx="8"/>',
    ]


def build_fig1_pareto(rows):
    w, h = 1280, 560
    s = svg_header(w, h)
    # Main title
    s.append(f'<text x="{w/2}" y="34" text-anchor="middle" font-family="{FONT}" font-size="20" font-weight="700" fill="{NAVY}">Pareto 前沿外推：K=4 局部解冻 + SubBag-ProbVar 同时打破 AP 上限与采样方差瓶颈</text>')

    panels = [
        {
            "title": "(a) 跨视图采样方差 (SpVar) vs. AP",
            "xlabel": "跨视图采样方差 SpVar（越左越稳定）",
            "x0": 95, "x1": 605, "y0": 80, "y1": 445,
            "xmin": 0.11, "xmax": 0.34, "xticks": [0.12, 0.16, 0.20, 0.24, 0.28, 0.32],
            "xfmt": lambda v: f"{v:.2f}",
            "mode": "cal_matched_human_fp", "xkey": "sampling_var", "xscale": 1.0,
            "points": [
                ("EXP-0", "EXP-0 (K=0)", "k0", -8, 16, "end"),
                ("EXP-4", "EXP-4", "k1", 8, 5, "start"),
                ("EXP-8-MH5-KL", "EXP-8-MH5-KL", "k1", 8, 12, "start"),
                ("EXP-8-MH5-KL05", "MH5-KL05", "k1", -8, 13, "end"),
                ("MH5-SubBag-Clean", "SubBag-Clean", "k1", -10, -10, "end"),
                ("MH5-SetAttn-Clean", "SetAttn-Clean", "k1", 10, -8, "start"),
                ("MH5-SubBag-ProbVar", "SubBag-ProbVar", "k1", 10, -7, "start"),
                ("MH5-ResGate-SubBag", "ResGate-SubBag", "k1", 10, 14, "start"),
                ("K4-MH5-KL01", "K4-MH5-KL01", "k4", 12, 4, "start"),
                ("K4-MH5-SetAttn", "K4-MH5-SetAttn", "k4", -12, 16, "end"),
                ("K4-MH5-SubBag", "K4-MH5-SubBag (0.9299)", "k4_best", 13, 4, "start"),
            ],
        },
        {
            "title": "(b) 等稳态误报 (HumFP=18.55%) 下 Churn vs. AP",
            "xlabel": "配对预测翻转率 Churn (%)（越左越稳定）",
            "x0": 725, "x1": 1235, "y0": 80, "y1": 445,
            "xmin": 3.45, "xmax": 4.50, "xticks": [3.5, 3.7, 3.9, 4.1, 4.3, 4.5],
            "xfmt": lambda v: f"{v:.1f}%",
            "mode": "dev_iso_human_stable_fp", "xkey": "churn", "xscale": 100.0,
            "points": [
                ("EXP-0", "EXP-0 (K=0)", "k0", -8, 16, "end"),
                ("EXP-4", "EXP-4", "k1", -10, 5, "end"),
                ("EXP-8-MH5-KL", "EXP-8-MH5-KL", "k1", 10, 12, "start"),
                ("EXP-8-MH5-KL05", "MH5-KL05", "k1", -10, 4, "end"),
                ("MH5-SetAttn-Clean", "SetAttn-Clean", "k1", 10, -8, "start"),
                ("MH5-SubBag-ProbVar", "SubBag-ProbVar", "k1", -10, 14, "end"),
                ("K4-MH5-KL01", "K4-MH5-KL01 (-12.1%)", "k4", 12, 4, "start"),
                ("K4-MH5-SetAttn", "K4-MH5-SetAttn (-15.1%)", "k4", -12, 16, "end"),
                ("K4-MH5-SubBag", "K4-MH5-SubBag (-16.1%)", "k4_best", 13, 4, "start"),
            ],
        },
    ]

    ymin, ymax = 0.9175, 0.9315
    yticks = [0.918, 0.920, 0.922, 0.924, 0.926, 0.928, 0.930]

    for p in panels:
        x0, x1, y0, y1 = p["x0"], p["x1"], p["y0"], p["y1"]
        def mx(v): return x0 + (v - p["xmin"]) / (p["xmax"] - p["xmin"]) * (x1 - x0)
        def my(v): return y1 - (v - ymin) / (ymax - ymin) * (y1 - y0)

        # Panel card
        s.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="{LIGHT_BG}" stroke="#CBD5E1" stroke-width="1"/>')
        s.append(f'<text x="{(x0+x1)/2}" y="{y0-14}" text-anchor="middle" font-family="{FONT}" font-size="15" font-weight="700" fill="{TEXT_DARK}">{escape(p["title"])}</text>')

        # Gridlines & ticks
        for yv in yticks:
            yy = my(yv)
            s.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}" stroke="{GRID}" stroke-width="1"/>')
            s.append(f'<text x="{x0-8}" y="{yy+4:.1f}" text-anchor="end" font-family="{FONT}" font-size="11.5" fill="{TEXT_MUTED}">{yv:.3f}</text>')
        for xv in p["xticks"]:
            xx = mx(xv)
            s.append(f'<line x1="{xx:.1f}" y1="{y0}" x2="{xx:.1f}" y2="{y1}" stroke="{GRID}" stroke-width="1"/>')
            s.append(f'<text x="{xx:.1f}" y="{y1+18}" text-anchor="middle" font-family="{FONT}" font-size="11.5" fill="{TEXT_MUTED}">{p["xfmt"](xv)}</text>')

        # K<=1 ceiling dashed line at AP = 0.9244
        y_ceil = my(0.9244)
        s.append(f'<line x1="{x0}" y1="{y_ceil:.1f}" x2="{x1}" y2="{y_ceil:.1f}" stroke="{STEEL}" stroke-width="1.5" stroke-dasharray="6,4" opacity="0.8"/>')
        s.append(f'<text x="{x0+8}" y="{y_ceil-6:.1f}" text-anchor="start" font-family="{FONT}" font-size="11" font-weight="600" fill="{STEEL}">K ≤ 1 表征上限 (AP ≈ 0.924)</text>')

        # Shaded K=4 region highlight
        s.append(f'<rect x="{x0+2}" y="{y0+2}" width="{(x1-x0)*0.56:.1f}" height="{y_ceil-y0-4:.1f}" fill="{CRIMSON}" opacity="0.045"/>')

        # Arrow from EXP-0 to K4-MH5-SubBag
        r0 = rows[("EXP-0", p["mode"])]
        rb = rows[("K4-MH5-SubBag", p["mode"])]
        ax0, ay0 = mx(r0[p["xkey"]] * p["xscale"]), my(r0["ap"])
        ax1, ay1 = mx(rb[p["xkey"]] * p["xscale"]), my(rb["ap"])
        # shorten slightly
        dx, dy = ax1 - ax0, ay1 - ay0
        dist = np.hypot(dx, dy)
        s.append(
            f'<line x1="{ax0 + dx/dist*12:.1f}" y1="{ay0 + dy/dist*12:.1f}" '
            f'x2="{ax1 - dx/dist*16:.1f}" y2="{ay1 - dy/dist*16:.1f}" '
            f'stroke="{CRIMSON}" stroke-width="1.6" stroke-dasharray="5,3" opacity="0.65" marker-end="url(#arrow-crimson)"/>'
        )

        # Plot points
        for mname, label, grp, ldx, ldy, anchor in p["points"]:
            r = rows[(mname, p["mode"])]
            cx, cy = mx(r[p["xkey"]] * p["xscale"]), my(r["ap"])
            if grp == "k0":
                s.append(f'<rect x="{cx-6:.1f}" y="{cy-6:.1f}" width="12" height="12" fill="{GRAY}" stroke="#FFFFFF" stroke-width="1.5"/>')
                lcol, fw = TEXT_MUTED, "600"
            elif grp == "k1":
                s.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5.5" fill="{STEEL}" stroke="#FFFFFF" stroke-width="1.4"/>')
                lcol, fw = TEXT_DARK, "400"
            elif grp == "k4":
                s.append(f'<polygon points="{cx:.1f},{cy-7.5:.1f} {cx+7.5:.1f},{cy:.1f} {cx:.1f},{cy+7.5:.1f} {cx-7.5:.1f},{cy:.1f}" fill="{NAVY}" stroke="#FFFFFF" stroke-width="1.5"/>')
                lcol, fw = NAVY, "700"
            else:
                s.append(f'<polygon points="{star_points(cx, cy, 10.5, 4.8)}" fill="{CRIMSON}" stroke="#FFFFFF" stroke-width="1.3"/>')
                lcol, fw = CRIMSON, "700"
            s.append(f'<text x="{cx+ldx:.1f}" y="{cy+ldy:.1f}" text-anchor="{anchor}" font-family="{FONT}" font-size="11.5" font-weight="{fw}" fill="{lcol}">{escape(label)}</text>')

        # Axes labels
        s.append(f'<text x="{(x0+x1)/2}" y="{y1+42}" text-anchor="middle" font-family="{FONT}" font-size="12.5" font-weight="600" fill="{TEXT_DARK}">{escape(p["xlabel"])}</text>')
        s.append(f'<text x="{x0-58}" y="{(y0+y1)/2}" text-anchor="middle" font-family="{FONT}" font-size="12.5" font-weight="600" fill="{TEXT_DARK}" transform="rotate(-90 {x0-58} {(y0+y1)/2})">Average Precision (AP)</text>')

    # Legend bar
    ly = 528
    s.append(f'<rect x="260" y="{ly-18}" width="760" height="30" rx="15" fill="#F1F5F9" stroke="#CBD5E1" stroke-width="1"/>')
    s.append(f'<rect x="290" y="{ly-9}" width="11" height="11" fill="{GRAY}"/>')
    s.append(f'<text x="308" y="{ly+1}" font-family="{FONT}" font-size="12" fill="{TEXT_DARK}">K=0 冻结基线 (EXP-0)</text>')
    s.append(f'<circle cx="485" cy="{ly-3}" r="5.5" fill="{STEEL}"/>')
    s.append(f'<text x="498" y="{ly+1}" font-family="{FONT}" font-size="12" fill="{TEXT_DARK}">K=1 浅层微调与消融</text>')
    s.append(f'<polygon points="660,{ly-10} 667,{ly-3} 660,{ly+4} 653,{ly-3}" fill="{NAVY}"/>')
    s.append(f'<text x="674" y="{ly+1}" font-family="{FONT}" font-size="12" font-weight="600" fill="{NAVY}">K=4 深层解冻对照</text>')
    s.append(f'<polygon points="{star_points(825, ly-3, 8.5, 3.8)}" fill="{CRIMSON}"/>')
    s.append(f'<text x="840" y="{ly+1}" font-family="{FONT}" font-size="12" font-weight="700" fill="{CRIMSON}">K4-MH5-SubBag（最优前沿）</text>')

    s.append("</svg>")
    return "\n".join(s)


def compute_conservation_data():
    analysis = json.loads((HERE / "robust_analysis.json").read_text(encoding="utf8"))
    models = [
        ("EXP-0", ce.FORMAL_FULL_ROOT / "EXP-0"),
        ("EXP-8-MH5-KL", ce.EXP_ROOT / "exp2_ensemble" / "full" / "EXP-8-MH5-KL"),
        ("K4-MH5-KL01", HERE / "runs" / "full" / "K4-MH5-KL01"),
        ("K4-MH5-SubBag", HERE / "runs" / "full" / "K4-MH5-SubBag"),
    ]
    out = {}
    for mode in ("cal_matched_human_fp", "dev_iso_human_stable_fp"):
        out[mode] = []
        for name, mdir in models:
            pr = prediction_data(mdir / "predictions.npz")
            prob = probability(pr["logits"])
            y = pr["labels"]
            cuts = np.array(analysis["methods"][name]["modes"][mode]["thresholds"])[None, :, None]
            hpos = (prob >= cuts)[y == 0]
            r1 = float(hpos.mean()) * 100.0
            r30 = float(hpos.all(-1).mean()) * 100.0
            flip_contrib = r1 - r30
            out[mode].append((name, r30, flip_contrib, r1))
    return out


def build_fig2_conservation(data):
    w, h = 1280, 540
    s = svg_header(w, h)
    s.append(f'<text x="{w/2}" y="34" text-anchor="middle" font-family="{FONT}" font-size="20" font-weight="700" fill="{NAVY}">多视图误报守恒律实证拆解：R̄₁(H) = R_stable(H) + E[(Sᵢ/M)·Fᵢ | Y=0]</text>')

    panels = [
        {
            "mode": "cal_matched_human_fp",
            "title": "(a) cal_matched_human_fp：锁定单视图误报 (~29.5%)",
            "subtitle": "翻转贡献缩小 (-2.66 pp) → 30v 稳定误报必然被动抬升 (+3.12 pp)",
            "x0": 95, "x1": 605, "y0": 95, "y1": 430,
            "ref_y": 29.12, "ref_label": "EXP-0 单视图误报基准 (29.12%)",
            "badge_col": CRIMSON,
        },
        {
            "mode": "dev_iso_human_stable_fp",
            "title": "(b) dev_iso_human_stable_fp：等 30v 稳定误报 (18.55%)",
            "subtitle": "固定稳态误报后，K4-MH5-SubBag 单视图误报净降 2.39 pp",
            "x0": 725, "x1": 1235, "y0": 95, "y1": 430,
            "ref_y": 18.55, "ref_label": "锁定 30v 稳定误报 (18.55%)",
            "badge_col": TEAL,
        },
    ]

    ymax = 38.0
    yticks = [0, 5, 10, 15, 20, 25, 30, 35]

    for p in panels:
        x0, x1, y0, y1 = p["x0"], p["x1"], p["y0"], p["y1"]
        def my(v): return y1 - (v / ymax) * (y1 - y0)

        s.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="{LIGHT_BG}" stroke="#CBD5E1" stroke-width="1"/>')
        s.append(f'<text x="{(x0+x1)/2}" y="{y0-28}" text-anchor="middle" font-family="{FONT}" font-size="15" font-weight="700" fill="{TEXT_DARK}">{escape(p["title"])}</text>')
        s.append(f'<text x="{(x0+x1)/2}" y="{y0-9}" text-anchor="middle" font-family="{FONT}" font-size="12" font-weight="600" fill="{p["badge_col"]}">{escape(p["subtitle"])}</text>')

        for yv in yticks:
            yy = my(yv)
            s.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}" stroke="{GRID}" stroke-width="1"/>')
            s.append(f'<text x="{x0-8}" y="{yy+4:.1f}" text-anchor="end" font-family="{FONT}" font-size="11.5" fill="{TEXT_MUTED}">{yv}%</text>')

        items = data[p["mode"]]
        slot_w = (x1 - x0) / len(items)
        bar_w = 64
        for idx, (name, r30, flip_c, r1) in enumerate(items):
            cx = x0 + (idx + 0.5) * slot_w
            bx = cx - bar_w / 2
            y_r30 = my(r30)
            y_r1 = my(r1)
            # Bottom bar: 30v stable FP
            s.append(f'<rect x="{bx:.1f}" y="{y_r30:.1f}" width="{bar_w}" height="{y1-y_r30:.1f}" fill="{NAVY}" rx="2"/>')
            # Top bar: flipping contribution
            s.append(f'<rect x="{bx:.1f}" y="{y_r1:.1f}" width="{bar_w}" height="{y_r30-y_r1:.1f}" fill="{AMBER}" rx="2"/>')
            # Segment value labels inside bars
            s.append(f'<text x="{cx:.1f}" y="{(y1+y_r30)/2+4:.1f}" text-anchor="middle" font-family="{FONT}" font-size="11.5" font-weight="700" fill="#FFFFFF">{r30:.2f}%</text>')
            s.append(f'<text x="{cx:.1f}" y="{(y_r30+y_r1)/2+4:.1f}" text-anchor="middle" font-family="{FONT}" font-size="11" font-weight="700" fill="#FFFFFF">{flip_c:.2f}%</text>')
            # Total on top
            tcol = CRIMSON if (p["mode"] == "dev_iso_human_stable_fp" and name == "K4-MH5-SubBag") else TEXT_DARK
            s.append(f'<text x="{cx:.1f}" y="{y_r1-7:.1f}" text-anchor="middle" font-family="{FONT}" font-size="12" font-weight="700" fill="{tcol}">{r1:.2f}%</text>')
            # X label
            s.append(f'<text x="{cx:.1f}" y="{y1+22}" text-anchor="middle" font-family="{FONT}" font-size="11.5" font-weight="600" fill="{TEXT_DARK}">{escape(name)}</text>')

        # Reference dashed line + clean top-right pill label (zero bar collision)
        ry = my(p["ref_y"])
        s.append(f'<line x1="{x0}" y1="{ry:.1f}" x2="{x1}" y2="{ry:.1f}" stroke="{CRIMSON}" stroke-width="1.6" stroke-dasharray="6,4"/>')
        s.append(f'<rect x="{x1-215}" y="{y0+10}" width="205" height="22" rx="4" fill="#FFFFFF" stroke="{CRIMSON}" stroke-width="1"/>')
        s.append(f'<line x1="{x1-207}" y1="{y0+21}" x2="{x1-189}" y2="{y0+21}" stroke="{CRIMSON}" stroke-width="1.8" stroke-dasharray="4,2"/>')
        s.append(f'<text x="{x1-184}" y="{y0+25}" text-anchor="start" font-family="{FONT}" font-size="10.8" font-weight="600" fill="{CRIMSON}">{escape(p["ref_label"])}</text>')

        s.append(f'<text x="{x0-54}" y="{(y0+y1)/2}" text-anchor="middle" font-family="{FONT}" font-size="12.5" font-weight="600" fill="{TEXT_DARK}" transform="rotate(-90 {x0-54} {(y0+y1)/2})">Human 误报率 (%)</text>')

    # Legend
    ly = 505
    s.append(f'<rect x="255" y="{ly-18}" width="770" height="30" rx="15" fill="#F1F5F9" stroke="#CBD5E1" stroke-width="1"/>')
    s.append(f'<rect x="285" y="{ly-9}" width="16" height="12" rx="2" fill="{NAVY}"/>')
    s.append(f'<text x="308" y="{ly+1}" font-family="{FONT}" font-size="12" font-weight="600" fill="{TEXT_DARK}">30v 稳定误报 R_stable(H)</text>')
    s.append(f'<rect x="505" y="{ly-9}" width="16" height="12" rx="2" fill="{AMBER}"/>')
    s.append(f'<text x="528" y="{ly+1}" font-family="{FONT}" font-size="12" font-weight="600" fill="{TEXT_DARK}">翻转区间误报贡献 E[(Sᵢ/M)·Fᵢ | Y=0]</text>')
    s.append(f'<text x="805" y="{ly+1}" font-family="{FONT}" font-size="12" fill="{TEXT_MUTED}">（柱顶为单视图总误报）</text>')

    s.append("</svg>")
    return "\n".join(s)


def build_fig3_variance(rows):
    w, h = 1280, 540
    s = svg_header(w, h)
    s.append(f'<text x="{w/2}" y="34" text-anchor="middle" font-family="{FONT}" font-size="20" font-weight="700" fill="{NAVY}">交叉随机效应方差分解：训练种子方差 (TrVar) vs. 推文采样方差 (SpVar)</text>')

    methods = [
        ("EXP-0", "EXP-0\n(K=0)"),
        ("EXP-4", "EXP-4\n(K=1)"),
        ("EXP-8-MH5-KL", "EXP-8\nMH5-KL"),
        ("MH5-SubBag-ProbVar", "K1-SubBag\nProbVar"),
        ("MH5-ResGate-SubBag", "K1-ResGate\nSubBag"),
        ("K4-MH5-KL01", "K4-MH5\nKL01"),
        ("K4-MH5-SetAttn", "K4-MH5\nSetAttn"),
        ("K4-MH5-SubBag", "K4-MH5\nSubBag"),
    ]

    x0, x1, y0, y1 = 95, 1235, 105, 425
    ymax = 0.54
    yticks = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
    def my(v): return y1 - (v / ymax) * (y1 - y0)

    s.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="{LIGHT_BG}" stroke="#CBD5E1" stroke-width="1"/>')
    for yv in yticks:
        yy = my(yv)
        s.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}" stroke="{GRID}" stroke-width="1"/>')
        s.append(f'<text x="{x0-10}" y="{yy+4:.1f}" text-anchor="end" font-family="{FONT}" font-size="11.5" fill="{TEXT_MUTED}">{yv:.2f}</text>')

    # Group separator between K<=1 and K=4
    slot_w = (x1 - x0) / len(methods)
    sep_x = x0 + 5 * slot_w
    s.append(f'<line x1="{sep_x:.1f}" y1="{y0}" x2="{sep_x:.1f}" y2="{y1}" stroke="#94A3B8" stroke-width="1.4" stroke-dasharray="5,4"/>')
    s.append(f'<text x="{(x0+sep_x)/2:.1f}" y="{y0-10}" text-anchor="middle" font-family="{FONT}" font-size="13" font-weight="700" fill="{STEEL}">浅层模型区间 (K ≤ 1)</text>')
    s.append(f'<text x="{(sep_x+x1)/2:.1f}" y="{y0-10}" text-anchor="middle" font-family="{FONT}" font-size="13" font-weight="700" fill="{CRIMSON}">深层局部解冻区间 (K = 4)</text>')

    bar_w = 34
    gap = 6
    for idx, (mname, disp) in enumerate(methods):
        r = rows[(mname, "cal_matched_human_fp")]
        trv, spv = r["training_var"], r["sampling_var"]
        cx = x0 + (idx + 0.5) * slot_w
        bx1 = cx - bar_w - gap / 2
        bx2 = cx + gap / 2
        y_tr = my(trv)
        y_sp = my(spv)
        sp_col = CRIMSON if mname.startswith("K4-") else NAVY

        s.append(f'<rect x="{bx1:.1f}" y="{y_tr:.1f}" width="{bar_w}" height="{y1-y_tr:.1f}" fill="#7FA6CC" rx="2"/>')
        s.append(f'<rect x="{bx2:.1f}" y="{y_sp:.1f}" width="{bar_w}" height="{y1-y_sp:.1f}" fill="{sp_col}" rx="2"/>')

        s.append(f'<text x="{bx1+bar_w/2:.1f}" y="{y_tr-6:.1f}" text-anchor="middle" font-family="{FONT}" font-size="11" font-weight="600" fill="{TEXT_MUTED}">{trv:.3f}</text>')
        s.append(f'<text x="{bx2+bar_w/2:.1f}" y="{y_sp-6:.1f}" text-anchor="middle" font-family="{FONT}" font-size="11" font-weight="700" fill="{sp_col}">{spv:.3f}</text>')

        lines = disp.split("\n")
        for li, line in enumerate(lines):
            fw = "700" if mname.startswith("K4-") else "600"
            col = NAVY if mname.startswith("K4-") else TEXT_DARK
            s.append(f'<text x="{cx:.1f}" y="{y1+20+li*15}" text-anchor="middle" font-family="{FONT}" font-size="11.5" font-weight="{fw}" fill="{col}">{escape(line)}</text>')

    # Callout badges at top inside chart
    # 1. EXP-0 dominance
    s.append(f'<rect x="115" y="118" width="245" height="28" rx="6" fill="#FFFFFF" stroke="{NAVY}" stroke-width="1.2"/>')
    s.append(f'<text x="237" y="136" text-anchor="middle" font-family="{FONT}" font-size="11.5" font-weight="700" fill="{NAVY}">EXP-0：SpVar 为 TrVar 的 1.90 倍</text>')

    # 2. K4-SetAttn TrVar drop
    s.append(f'<rect x="835" y="118" width="380" height="28" rx="6" fill="#FFFFFF" stroke="{CRIMSON}" stroke-width="1.2"/>')
    s.append(f'<text x="1025" y="136" text-anchor="middle" font-family="{FONT}" font-size="11.5" font-weight="700" fill="{CRIMSON}">K4-SubBag: SpVar -42.3% ｜ SetAttn(lr=1.2e-5): TrVar -41.6%</text>')

    # Y axis label
    s.append(f'<text x="36" y="{(y0+y1)/2}" text-anchor="middle" font-family="{FONT}" font-size="12.5" font-weight="600" fill="{TEXT_DARK}" transform="rotate(-90 36 {(y0+y1)/2})">Logit 方差分量估计值</text>')

    # Legend
    ly = 505
    s.append(f'<rect x="275" y="{ly-18}" width="730" height="30" rx="15" fill="#F1F5F9" stroke="#CBD5E1" stroke-width="1"/>')
    s.append(f'<rect x="305" y="{ly-9}" width="16" height="12" rx="2" fill="#7FA6CC"/>')
    s.append(f'<text x="328" y="{ly+1}" font-family="{FONT}" font-size="12" font-weight="600" fill="{TEXT_DARK}">TrVar（训练随机种子方差）</text>')
    s.append(f'<rect x="520" y="{ly-9}" width="16" height="12" rx="2" fill="{NAVY}"/>')
    s.append(f'<text x="543" y="{ly+1}" font-family="{FONT}" font-size="12" font-weight="600" fill="{TEXT_DARK}">SpVar（K≤1 推文采样方差）</text>')
    s.append(f'<rect x="745" y="{ly-9}" width="16" height="12" rx="2" fill="{CRIMSON}"/>')
    s.append(f'<text x="768" y="{ly+1}" font-family="{FONT}" font-size="12" font-weight="700" fill="{CRIMSON}">SpVar（K=4 推文采样方差）</text>')

    s.append("</svg>")
    return "\n".join(s)


def main():
    rows = load_summary()
    cons_data = compute_conservation_data()
    figs = {
        "fig1_pareto_frontier.svg": build_fig1_pareto(rows),
        "fig2_error_conservation.svg": build_fig2_conservation(cons_data),
        "fig3_variance_decomposition.svg": build_fig3_variance(rows),
    }
    for fname, content in figs.items():
        p1 = FIG_DIR / fname
        p2 = PKG_FIG_DIR / fname
        p1.write_text(content, encoding="utf8")
        shutil.copy2(p1, p2)
        print(f"Wrote {p1} ({len(content)} bytes) -> copied to {p2}")


if __name__ == "__main__":
    main()
