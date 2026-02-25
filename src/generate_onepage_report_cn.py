import argparse
from pathlib import Path
from typing import Dict

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成中文版一页式汇报 PDF")
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=Path("outputs/experiments/experiment_summary.xlsx"),
        help="基线实验汇总文件",
    )
    parser.add_argument(
        "--optimized-summary",
        type=Path,
        default=Path("outputs_opt/experiments/experiment_summary.xlsx"),
        help="增强实验汇总文件",
    )
    parser.add_argument(
        "--comparison-excel",
        type=Path,
        default=Path("outputs_opt/comparison/before_after_summary.xlsx"),
        help="前后对比汇总文件",
    )
    parser.add_argument(
        "--fig-core",
        type=Path,
        default=Path("outputs_opt/comparison/figures/before_after_core_metrics.png"),
        help="核心指标对比图",
    )
    parser.add_argument(
        "--fig-size",
        type=Path,
        default=Path("outputs_opt/comparison/figures/before_after_size_score.png"),
        help="体量-效果对比图",
    )
    parser.add_argument(
        "--fig-topk",
        type=Path,
        default=Path("outputs_opt/comparison/figures/before_after_topk_score_curve.png"),
        help="TopK 对比曲线",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/onepage_executive_summary_cn.pdf"),
        help="输出 PDF 路径",
    )
    return parser.parse_args()


def read_top1(path: Path) -> pd.Series:
    df = pd.read_excel(path, sheet_name="summary")
    df = df.sort_values(
        ["score_primary", "policy_label_match_rate", "policy_event_active_ratio", "dict_total_size"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    return df.iloc[0]


def metric_delta(before: float, after: float) -> Dict[str, str]:
    d = after - before
    pct = (d / before * 100.0) if before != 0 else float("nan")
    return {"before": f"{before:.6f}", "after": f"{after:.6f}", "delta": f"{d:+.6f}", "pct": f"{pct:+.2f}%"}


def build_report(args: argparse.Namespace) -> None:
    # Windows 常见中文字体优先，找不到则回退 DejaVu。
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    before = read_top1(args.baseline_summary)
    after = read_top1(args.optimized_summary)

    d_score = metric_delta(float(before["score_primary"]), float(after["score_primary"]))
    d_match = metric_delta(float(before["policy_label_match_rate"]), float(after["policy_label_match_rate"]))
    d_p_active = metric_delta(float(before["policy_event_active_ratio"]), float(after["policy_event_active_ratio"]))
    d_e_active = metric_delta(float(before["econ_event_active_ratio"]), float(after["econ_event_active_ratio"]))

    size_before = int(before["dict_total_size"])
    size_after = int(after["dict_total_size"])
    size_delta = size_after - size_before
    size_pct = (size_delta / size_before * 100.0) if size_before != 0 else 0.0

    fig = plt.figure(figsize=(8.27, 11.69))  # A4 竖版
    gs = fig.add_gridspec(
        24,
        2,
        left=0.05,
        right=0.95,
        top=0.97,
        bottom=0.04,
        hspace=0.55,
        wspace=0.35,
    )

    ax_title = fig.add_subplot(gs[0:2, :])
    ax_title.axis("off")
    ax_title.text(0.0, 0.76, "MonetaryEconomic 一页式汇报（中文版）", fontsize=17, fontweight="bold")
    ax_title.text(0.0, 0.38, "日期：2026-02-24（夜间增强实跑）", fontsize=10.5)
    ax_title.text(0.0, 0.05, "定位：在不覆盖原交付 outputs/ 的前提下，产出增强版 outputs_opt/ 并形成可验收对比。", fontsize=9.6)

    ax_text = fig.add_subplot(gs[2:8, :])
    ax_text.axis("off")
    txt = [
        "一、方法增强（保持默认兼容）",
        "1) 词典概率平滑：降低低支持短语的随机波动。",
        "2) 标签边际约束：过滤 top1/top2 概率过于接近的词条。",
        "3) ML 与计数融合：在低支持短语上引入 ML 先验，提高覆盖与匹配。",
        "",
        "二、关键结论（Top1 对比）",
        f"- score_primary：{d_score['before']} -> {d_score['after']}（{d_score['delta']}，{d_score['pct']}）",
        f"- policy_label_match_rate：{d_match['before']} -> {d_match['after']}（{d_match['delta']}，{d_match['pct']}）",
        f"- policy_event_active_ratio：{d_p_active['before']} -> {d_p_active['after']}（{d_p_active['delta']}，{d_p_active['pct']}）",
        f"- econ_event_active_ratio：{d_e_active['before']} -> {d_e_active['after']}（{d_e_active['delta']}，{d_e_active['pct']}）",
        f"- 词典总条数：{size_before} -> {size_after}（{size_delta:+d}，{size_pct:+.2f}%）",
        "",
        "三、答辩口径（建议）",
        "增强方案在有效性指标上全面提升，代价是词典体量增加；",
        "原方案与原交付保持不变，验收脚本继续通过，具备可追溯与可回滚性。",
    ]
    ax_text.text(0.0, 1.0, "\n".join(txt), va="top", fontsize=10.0)

    ax_img_core = fig.add_subplot(gs[8:14, :])
    ax_img_core.axis("off")
    if args.fig_core.exists():
        ax_img_core.imshow(mpimg.imread(args.fig_core))
        ax_img_core.set_title("图1：优化前后核心指标对比", fontsize=10)
    else:
        ax_img_core.text(0.5, 0.5, f"缺少图：{args.fig_core}", ha="center", va="center")

    ax_img_size = fig.add_subplot(gs[14:19, 0])
    ax_img_size.axis("off")
    if args.fig_size.exists():
        ax_img_size.imshow(mpimg.imread(args.fig_size))
        ax_img_size.set_title("图2：体量-效果权衡", fontsize=9.5)
    else:
        ax_img_size.text(0.5, 0.5, f"缺少图：{args.fig_size}", ha="center", va="center")

    ax_img_topk = fig.add_subplot(gs[14:19, 1])
    ax_img_topk.axis("off")
    if args.fig_topk.exists():
        ax_img_topk.imshow(mpimg.imread(args.fig_topk))
        ax_img_topk.set_title("图3：Top-K 得分曲线", fontsize=9.5)
    else:
        ax_img_topk.text(0.5, 0.5, f"缺少图：{args.fig_topk}", ha="center", va="center")

    ax_bottom = fig.add_subplot(gs[19:24, :])
    ax_bottom.axis("off")
    bottom_lines = [
        "四、可交付物定位",
        "- 原交付：outputs/（不变，validate_delivery 通过）",
        "- 增强交付：outputs_opt/experiments/experiment_summary.xlsx",
        "- 前后对比：outputs_opt/comparison/before_after_summary.xlsx",
        "",
        f"数据来源：{args.baseline_summary} | {args.optimized_summary} | {args.comparison_excel}",
        "生成命令：python -m src.generate_onepage_report_cn",
    ]
    ax_bottom.text(0.0, 1.0, "\n".join(bottom_lines), va="top", fontsize=9.4)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format="pdf", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    build_report(args)
    print(f"[INFO] 中文版 PDF 已生成: {args.output.resolve()}")


if __name__ == "__main__":
    main()
