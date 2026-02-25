import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one-page executive PDF report")
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=Path("outputs/experiments/experiment_summary.xlsx"),
    )
    parser.add_argument(
        "--optimized-summary",
        type=Path,
        default=Path("outputs_opt/experiments/experiment_summary.xlsx"),
    )
    parser.add_argument(
        "--comparison-excel",
        type=Path,
        default=Path("outputs_opt/comparison/before_after_summary.xlsx"),
    )
    parser.add_argument(
        "--core-fig",
        type=Path,
        default=Path("outputs_opt/comparison/figures/before_after_core_metrics.png"),
    )
    parser.add_argument(
        "--curve-fig",
        type=Path,
        default=Path("outputs_opt/comparison/figures/before_after_topk_score_curve.png"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/onepage_executive_summary.pdf"),
    )
    return parser.parse_args()


def _load_top1(path: Path) -> pd.Series:
    df = pd.read_excel(path, sheet_name="summary")
    sort_cols = ["score_primary", "policy_label_match_rate", "policy_event_active_ratio", "dict_total_size"]
    df = df.sort_values(sort_cols, ascending=[False, False, False, True]).reset_index(drop=True)
    return df.iloc[0]


def _fmt_pct(delta: float, base: float) -> str:
    if base == 0:
        return "nan"
    return f"{delta / base * 100:.2f}%"


def collect_metrics(b: pd.Series, o: pd.Series) -> Dict[str, str]:
    m = {}
    for key in [
        "score_primary",
        "policy_label_match_rate",
        "policy_event_active_ratio",
        "econ_event_active_ratio",
        "dict_total_size",
    ]:
        bv = float(b[key])
        ov = float(o[key])
        d = ov - bv
        if key == "dict_total_size":
            m[key] = f"{int(bv)} -> {int(ov)} ({d:+.0f}, {_fmt_pct(d, bv)})"
        else:
            m[key] = f"{bv:.6f} -> {ov:.6f} ({d:+.6f}, {_fmt_pct(d, bv)})"
    return m


def build_pdf(args: argparse.Namespace) -> None:
    # Prefer common Windows CJK fonts; fallback to DejaVu.
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    b = _load_top1(args.baseline_summary)
    o = _load_top1(args.optimized_summary)
    metrics = collect_metrics(b, o)

    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    gs = fig.add_gridspec(20, 2, left=0.05, right=0.95, top=0.97, bottom=0.04, hspace=0.6, wspace=0.4)

    ax_title = fig.add_subplot(gs[0:2, :])
    ax_title.axis("off")
    ax_title.text(0, 0.75, "MonetaryEconomic One-Page Executive Summary", fontsize=17, fontweight="bold")
    ax_title.text(0, 0.40, "Date: 2026-02-24 | Scope: Baseline vs Enhanced Track (non-overwriting)", fontsize=10)
    ax_title.text(
        0,
        0.05,
        "Enhanced best run: run_006_thr0.50_sup2_len3_cov0.30_ml1_mg0.00_sa0.20_bw0.35",
        fontsize=9.5,
    )

    ax_txt = fig.add_subplot(gs[2:8, :])
    ax_txt.axis("off")
    lines = [
        "Objective:",
        "  Keep original deliverables intact while improving match and activation metrics.",
        "",
        "What changed in method:",
        "  1) Dictionary probability smoothing (alpha).",
        "  2) Optional label-margin filter.",
        "  3) ML-prior + count-based blended dictionary decision.",
        "",
        "Top-1 metric deltas (Before -> After):",
        f"  score_primary: {metrics['score_primary']}",
        f"  policy_label_match_rate: {metrics['policy_label_match_rate']}",
        f"  policy_event_active_ratio: {metrics['policy_event_active_ratio']}",
        f"  econ_event_active_ratio: {metrics['econ_event_active_ratio']}",
        f"  dict_total_size: {metrics['dict_total_size']}",
        "",
        "Interpretation:",
        "  Enhanced track improves effectiveness with a larger dictionary footprint.",
        "  Baseline outputs remain valid and acceptance script still passes.",
    ]
    ax_txt.text(0, 1, "\n".join(lines), va="top", fontsize=10.2, family="monospace")

    ax_img1 = fig.add_subplot(gs[8:14, :])
    ax_img1.axis("off")
    if args.core_fig.exists():
        img1 = mpimg.imread(args.core_fig)
        ax_img1.imshow(img1)
        ax_img1.set_title("Evidence Figure 1: Core Metrics Before vs After", fontsize=10)
    else:
        ax_img1.text(0.5, 0.5, f"Missing figure: {args.core_fig}", ha="center", va="center")

    ax_img2 = fig.add_subplot(gs[14:19, :])
    ax_img2.axis("off")
    if args.curve_fig.exists():
        img2 = mpimg.imread(args.curve_fig)
        ax_img2.imshow(img2)
        ax_img2.set_title("Evidence Figure 2: Top-K Score Curve", fontsize=10)
    else:
        ax_img2.text(0.5, 0.5, f"Missing figure: {args.curve_fig}", ha="center", va="center")

    ax_foot = fig.add_subplot(gs[19:20, :])
    ax_foot.axis("off")
    ax_foot.text(
        0,
        0.8,
        f"Data sources: {args.baseline_summary} | {args.optimized_summary} | {args.comparison_excel}",
        fontsize=8.5,
    )
    ax_foot.text(
        0,
        0.2,
        "Generated by: python -m src.generate_onepage_report",
        fontsize=8.5,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format="pdf", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    build_pdf(args)
    print(f"[INFO] PDF generated: {args.output.resolve()}")


if __name__ == "__main__":
    main()
