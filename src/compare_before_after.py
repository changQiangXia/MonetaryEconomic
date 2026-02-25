import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing matplotlib. Install with: pip install matplotlib") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline and optimized experiment outputs")
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=Path("outputs/experiments/experiment_summary.xlsx"),
        help="Baseline experiment summary Excel",
    )
    parser.add_argument(
        "--optimized-summary",
        type=Path,
        default=Path("outputs_opt/experiments/experiment_summary.xlsx"),
        help="Optimized experiment summary Excel",
    )
    parser.add_argument(
        "--baseline-exp-dir",
        type=Path,
        default=Path("outputs/experiments"),
        help="Baseline experiments directory (contains run_* folders)",
    )
    parser.add_argument(
        "--optimized-exp-dir",
        type=Path,
        default=Path("outputs_opt/experiments"),
        help="Optimized experiments directory (contains run_* folders)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_opt/comparison"))
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def read_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")
    df = pd.read_excel(path, sheet_name="summary")
    if df.empty:
        raise ValueError(f"Summary sheet is empty: {path}")
    sort_cols = ["score_primary", "policy_label_match_rate", "policy_event_active_ratio", "dict_total_size"]
    return df.sort_values(sort_cols, ascending=[False, False, False, True]).reset_index(drop=True)


def get_top_row(df: pd.DataFrame) -> pd.Series:
    return df.iloc[0]


def build_metric_comparison(baseline_top: pd.Series, optimized_top: pd.Series) -> pd.DataFrame:
    metrics: List[Dict[str, object]] = []
    rows = [
        ("score_primary", "Primary Score", "higher_better"),
        ("policy_label_match_rate", "Policy Match Rate", "higher_better"),
        ("policy_event_active_ratio", "Policy Active Ratio", "higher_better"),
        ("econ_event_active_ratio", "Economic Active Ratio", "higher_better"),
        ("dict_total_size", "Dictionary Total Size", "lower_better"),
    ]
    for key, display_name, preference in rows:
        before = float(baseline_top[key])
        after = float(optimized_top[key])
        delta_abs = after - before
        delta_pct = (delta_abs / before * 100.0) if before != 0 else np.nan
        if preference == "higher_better":
            improved = delta_abs > 0
            gain = delta_abs
        else:
            improved = delta_abs < 0
            gain = -delta_abs
        metrics.append(
            {
                "metric": key,
                "metric_display": display_name,
                "preference": preference,
                "baseline": before,
                "optimized": after,
                "delta_abs": delta_abs,
                "delta_pct": delta_pct,
                "improved": int(improved),
                "gain_for_preference": gain,
            }
        )
    return pd.DataFrame(metrics)


def save_core_metric_bar(comp_df: pd.DataFrame, out_path: Path) -> None:
    plot_df = comp_df[comp_df["metric"] != "dict_total_size"].copy()
    if plot_df.empty:
        return

    x = np.arange(len(plot_df))
    width = 0.36
    plt.figure(figsize=(10.5, 6))
    plt.bar(x - width / 2, plot_df["baseline"], width=width, label="Before")
    plt.bar(x + width / 2, plot_df["optimized"], width=width, label="After")
    plt.xticks(x, plot_df["metric_display"], rotation=15)
    plt.ylim(0, 1.05)
    plt.ylabel("Score / Ratio")
    plt.title("Before vs After: Core Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_size_score_dual(comp_df: pd.DataFrame, out_path: Path) -> None:
    score_row = comp_df[comp_df["metric"] == "score_primary"].iloc[0]
    size_row = comp_df[comp_df["metric"] == "dict_total_size"].iloc[0]
    labels = ["Before", "After"]
    score_values = [float(score_row["baseline"]), float(score_row["optimized"])]
    size_values = [float(size_row["baseline"]), float(size_row["optimized"])]

    x = np.arange(2)
    fig, ax1 = plt.subplots(figsize=(9.5, 5.6))
    bar = ax1.bar(x, size_values, color=["#4C78A8", "#F58518"], alpha=0.75, width=0.55)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Dictionary Size")
    ax1.set_title("Before vs After: Size and Primary Score")

    ax2 = ax1.twinx()
    ax2.plot(x, score_values, color="#54A24B", marker="o", linewidth=2.2)
    ax2.set_ylabel("Primary Score")
    ax2.set_ylim(0, 1.0)

    for rect in bar:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width() / 2, h + 8, f"{int(h)}", ha="center", va="bottom", fontsize=9)
    for xi, sv in zip(x, score_values):
        ax2.text(xi, sv + 0.015, f"{sv:.4f}", ha="center", va="bottom", fontsize=9, color="#2F6B2F")

    fig.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_topk_curve(baseline_df: pd.DataFrame, optimized_df: pd.DataFrame, top_k: int, out_path: Path) -> None:
    k = max(1, top_k)
    b = baseline_df["score_primary"].head(k).to_numpy()
    o = optimized_df["score_primary"].head(k).to_numpy()
    n = max(len(b), len(o))
    if n == 0:
        return
    x = np.arange(1, n + 1)
    b = np.pad(b, (0, max(0, n - len(b))), constant_values=np.nan)
    o = np.pad(o, (0, max(0, n - len(o))), constant_values=np.nan)

    plt.figure(figsize=(10.5, 5.6))
    plt.plot(x, b, marker="o", linewidth=1.8, label="Before")
    plt.plot(x, o, marker="o", linewidth=1.8, label="After")
    plt.xlabel("Rank in Top-K")
    plt.ylabel("Primary Score")
    plt.title(f"Top-{n} Score Curve: Before vs After")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _load_best_run_indices(exp_dir: Path, run_name: str) -> pd.DataFrame:
    path = exp_dir / run_name / "event_indices.xlsx"
    if not path.exists():
        return pd.DataFrame(columns=["publish_time", "Imp", "Ieo"])
    df = pd.read_excel(path)
    if df.empty:
        return pd.DataFrame(columns=["publish_time", "Imp", "Ieo"])
    df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")
    df = df.dropna(subset=["publish_time"]).sort_values("publish_time")
    return df[["publish_time", "Imp", "Ieo"]]


def save_timeseries_overlay(
    baseline_exp_dir: Path,
    optimized_exp_dir: Path,
    baseline_top: pd.Series,
    optimized_top: pd.Series,
    out_path: Path,
) -> None:
    before_df = _load_best_run_indices(baseline_exp_dir, str(baseline_top["run_name"]))
    after_df = _load_best_run_indices(optimized_exp_dir, str(optimized_top["run_name"]))
    if before_df.empty or after_df.empty:
        return

    before_month = (
        before_df.set_index("publish_time")[["Imp", "Ieo"]]
        .resample("ME")
        .mean()
        .rename(columns={"Imp": "Imp_before", "Ieo": "Ieo_before"})
    )
    after_month = (
        after_df.set_index("publish_time")[["Imp", "Ieo"]]
        .resample("ME")
        .mean()
        .rename(columns={"Imp": "Imp_after", "Ieo": "Ieo_after"})
    )
    merged = before_month.join(after_month, how="inner")
    if merged.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(merged.index, merged["Imp_before"], label="Imp Before", linewidth=1.7)
    axes[0].plot(merged.index, merged["Imp_after"], label="Imp After", linewidth=1.7)
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Imp")
    axes[0].set_title("Monthly Imp: Before vs After (Best Runs)")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    axes[1].plot(merged.index, merged["Ieo_before"], label="Ieo Before", linewidth=1.7)
    axes[1].plot(merged.index, merged["Ieo_after"], label="Ieo After", linewidth=1.7)
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Ieo")
    axes[1].set_title("Monthly Ieo: Before vs After (Best Runs)")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    axes[1].set_xlabel("Month")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_excel_report(
    out_path: Path,
    baseline_df: pd.DataFrame,
    optimized_df: pd.DataFrame,
    baseline_top: pd.Series,
    optimized_top: pd.Series,
    comp_df: pd.DataFrame,
    top_k: int,
) -> None:
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        baseline_top.to_frame().T.to_excel(writer, sheet_name="before_top1", index=False)
        optimized_top.to_frame().T.to_excel(writer, sheet_name="after_top1", index=False)
        comp_df.to_excel(writer, sheet_name="metric_comparison", index=False)
        baseline_df.head(top_k).to_excel(writer, sheet_name="before_topk", index=False)
        optimized_df.head(top_k).to_excel(writer, sheet_name="after_topk", index=False)
        baseline_df.to_excel(writer, sheet_name="before_all", index=False)
        optimized_df.to_excel(writer, sheet_name="after_all", index=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    baseline_df = read_summary(args.baseline_summary)
    optimized_df = read_summary(args.optimized_summary)
    baseline_top = get_top_row(baseline_df)
    optimized_top = get_top_row(optimized_df)
    comp_df = build_metric_comparison(baseline_top, optimized_top)

    save_excel_report(
        out_path=args.output_dir / "before_after_summary.xlsx",
        baseline_df=baseline_df,
        optimized_df=optimized_df,
        baseline_top=baseline_top,
        optimized_top=optimized_top,
        comp_df=comp_df,
        top_k=max(1, args.top_k),
    )

    save_core_metric_bar(comp_df, fig_dir / "before_after_core_metrics.png")
    save_size_score_dual(comp_df, fig_dir / "before_after_size_score.png")
    save_topk_curve(baseline_df, optimized_df, args.top_k, fig_dir / "before_after_topk_score_curve.png")
    save_timeseries_overlay(
        baseline_exp_dir=args.baseline_exp_dir,
        optimized_exp_dir=args.optimized_exp_dir,
        baseline_top=baseline_top,
        optimized_top=optimized_top,
        out_path=fig_dir / "before_after_imp_ieo_monthly_overlay.png",
    )

    print("[INFO] Comparison outputs:")
    print(f"  - Excel: {(args.output_dir / 'before_after_summary.xlsx').resolve()}")
    print("  - PNG:")
    for p in sorted(fig_dir.glob("*.png")):
        print(f"    * {p.resolve()}")
    print("[INFO] Top before:")
    print(baseline_top.to_string())
    print("[INFO] Top after:")
    print(optimized_top.to_string())


if __name__ == "__main__":
    main()
