import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing matplotlib. Install with: pip install matplotlib") from exc

from src.config import (
    DEFAULT_HEADER_ROW,
    DEFAULT_INPUT_FILE,
    DEFAULT_SHEET_INDEX,
    DICTIONARY_PROB_THRESHOLD,
    ECON_TONES,
    MAX_PHRASE_CHARS,
    MAX_PHRASE_TOKENS,
    MIN_DICT_PHRASE_CHARS,
    MIN_PHRASE_CHARS,
    MIN_PHRASE_SUPPORT,
    MIN_PHRASE_TOKENS,
    POLICY_TONES,
    THEME_ECONOMIC,
    THEME_MONETARY,
)
from src.run_pipeline import (
    build_dictionary,
    build_event_indices,
    build_event_phrase_df,
    build_length_distribution_tables,
    build_phrase_dataframe,
    build_phrase_stats,
    load_events,
)


@dataclass(frozen=True)
class ExperimentConfig:
    dict_threshold: float
    min_support: int
    min_dict_chars: int
    max_event_coverage: float
    dict_min_label_margin: float
    dict_prob_smoothing_alpha: float
    dict_ml_blend_weight: float
    use_ml_prior: bool


def parse_list_floats(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_list_ints(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid experiments for dictionary/index parameters")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--sheet-index", type=int, default=DEFAULT_SHEET_INDEX)
    parser.add_argument("--header-row", type=int, default=DEFAULT_HEADER_ROW)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments"))

    parser.add_argument("--min-phrase-chars", type=int, default=MIN_PHRASE_CHARS)
    parser.add_argument("--max-phrase-chars", type=int, default=MAX_PHRASE_CHARS)
    parser.add_argument("--min-phrase-tokens", type=int, default=MIN_PHRASE_TOKENS)
    parser.add_argument("--max-phrase-tokens", type=int, default=MAX_PHRASE_TOKENS)

    parser.add_argument("--thresholds", type=str, default=f"{DICTIONARY_PROB_THRESHOLD},0.55,0.60")
    parser.add_argument("--supports", type=str, default=f"{MIN_PHRASE_SUPPORT},3,5")
    parser.add_argument("--min-dict-chars-list", type=str, default=f"{MIN_DICT_PHRASE_CHARS},4")
    parser.add_argument("--max-event-coverages", type=str, default="0.30,0.35")
    parser.add_argument("--min-label-margins", type=str, default="0.00")
    parser.add_argument("--prob-smoothing-alphas", type=str, default="0.00")
    parser.add_argument("--ml-blend-weights", type=str, default="0.00")
    parser.add_argument("--use-ml-prior", action="store_true")
    parser.add_argument("--ml-tendency-min-conf", type=float, default=0.55)
    parser.add_argument("--ml-blend-support-pivot", type=int, default=5)

    parser.add_argument("--save-details", action="store_true", help="Save dictionary/index files for each run")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k experiments in ranking sheet")
    return parser.parse_args()


def iter_experiment_configs(args: argparse.Namespace) -> Iterable[ExperimentConfig]:
    thresholds = parse_list_floats(args.thresholds)
    supports = parse_list_ints(args.supports)
    min_dict_chars_list = parse_list_ints(args.min_dict_chars_list)
    max_event_coverages = parse_list_floats(args.max_event_coverages)
    min_label_margins = parse_list_floats(args.min_label_margins)
    prob_smoothing_alphas = parse_list_floats(args.prob_smoothing_alphas)
    ml_blend_weights = parse_list_floats(args.ml_blend_weights)

    seen = set()
    for threshold, support, min_chars, coverage, min_margin, smoothing_alpha, blend_weight in product(
        thresholds,
        supports,
        min_dict_chars_list,
        max_event_coverages,
        min_label_margins,
        prob_smoothing_alphas,
        ml_blend_weights,
    ):
        key = (
            float(threshold),
            int(support),
            int(min_chars),
            float(coverage),
            float(min_margin),
            float(smoothing_alpha),
            float(blend_weight),
            bool(args.use_ml_prior),
        )
        if key in seen:
            continue
        seen.add(key)
        yield ExperimentConfig(
            dict_threshold=threshold,
            min_support=support,
            min_dict_chars=min_chars,
            max_event_coverage=coverage,
            dict_min_label_margin=min_margin,
            dict_prob_smoothing_alpha=smoothing_alpha,
            dict_ml_blend_weight=blend_weight,
            use_ml_prior=bool(args.use_ml_prior),
        )


def format_run_name(run_id: int, cfg: ExperimentConfig) -> str:
    base = (
        f"run_{run_id:03d}_thr{cfg.dict_threshold:.2f}_sup{cfg.min_support}"
        f"_len{cfg.min_dict_chars}_cov{cfg.max_event_coverage:.2f}"
    )
    if (
        cfg.use_ml_prior
        or cfg.dict_min_label_margin > 0
        or cfg.dict_prob_smoothing_alpha > 0
        or cfg.dict_ml_blend_weight > 0
    ):
        base += (
            f"_ml{1 if cfg.use_ml_prior else 0}"
            f"_mg{cfg.dict_min_label_margin:.2f}"
            f"_sa{cfg.dict_prob_smoothing_alpha:.2f}"
            f"_bw{cfg.dict_ml_blend_weight:.2f}"
        )
    return base


def argmax_label(row: pd.Series, labels: List[str]) -> str:
    cols = [f"P_{label}" for label in labels]
    if row[cols].sum() <= 0:
        return ""
    return max(labels, key=lambda label: row[f"P_{label}"])


def summarize_experiment(
    run_id: int,
    run_name: str,
    cfg: ExperimentConfig,
    events: pd.DataFrame,
    policy_dict_df: pd.DataFrame,
    econ_dict_df: pd.DataFrame,
    event_indices_df: pd.DataFrame,
) -> dict:
    policy_prob_cols = [f"P_{x}" for x in POLICY_TONES]
    econ_prob_cols = [f"P_{x}" for x in ECON_TONES]

    policy_active = (event_indices_df[policy_prob_cols].sum(axis=1) > 0).mean()
    econ_active = (event_indices_df[econ_prob_cols].sum(axis=1) > 0).mean()

    predicted_policy = event_indices_df.apply(argmax_label, axis=1, labels=POLICY_TONES)
    policy_match_rate = (predicted_policy == event_indices_df["policy_tone"]).mean()

    return {
        "run_id": run_id,
        "run_name": run_name,
        "dict_threshold": cfg.dict_threshold,
        "min_support": cfg.min_support,
        "min_dict_chars": cfg.min_dict_chars,
        "max_event_coverage": cfg.max_event_coverage,
        "dict_min_label_margin": cfg.dict_min_label_margin,
        "dict_prob_smoothing_alpha": cfg.dict_prob_smoothing_alpha,
        "dict_ml_blend_weight": cfg.dict_ml_blend_weight,
        "use_ml_prior": int(cfg.use_ml_prior),
        "policy_dict_size": len(policy_dict_df),
        "econ_dict_size": len(econ_dict_df),
        "dict_total_size": len(policy_dict_df) + len(econ_dict_df),
        "policy_event_active_ratio": float(policy_active),
        "econ_event_active_ratio": float(econ_active),
        "policy_label_match_rate": float(policy_match_rate),
        "Imp_mean": float(event_indices_df["Imp"].mean()),
        "Imp_std": float(event_indices_df["Imp"].std(ddof=1)),
        "Ieo_mean": float(event_indices_df["Ieo"].mean()),
        "Ieo_std": float(event_indices_df["Ieo"].std(ddof=1)),
        "events_count": len(events),
    }


def add_ranking_scores(summary_df: pd.DataFrame) -> pd.DataFrame:
    out = summary_df.copy()
    size_min = out["dict_total_size"].min()
    size_max = out["dict_total_size"].max()
    denom = (size_max - size_min) if size_max != size_min else 1.0
    out["size_norm"] = (out["dict_total_size"] - size_min) / denom

    out["score_primary"] = (
        0.50 * out["policy_label_match_rate"]
        + 0.25 * out["policy_event_active_ratio"]
        + 0.25 * out["econ_event_active_ratio"]
        - 0.05 * out["size_norm"]
    )
    out["score_coverage_first"] = (
        0.35 * out["policy_label_match_rate"]
        + 0.325 * out["policy_event_active_ratio"]
        + 0.325 * out["econ_event_active_ratio"]
        - 0.03 * out["size_norm"]
    )
    return out


def build_pivot_table(
    summary_df: pd.DataFrame,
    value_col: str,
) -> pd.DataFrame:
    pivot = pd.pivot_table(
        summary_df,
        index="dict_threshold",
        columns="min_support",
        values=value_col,
        aggfunc="mean",
    )
    return pivot.sort_index().sort_index(axis=1)


def _save_topk_bar(top_k_df: pd.DataFrame, fig_path: Path) -> None:
    n = min(len(top_k_df), 10)
    if n == 0:
        return
    plot_df = top_k_df.head(n).copy()
    labels = [f"#{int(i)}" for i in plot_df["run_id"]]
    x = np.arange(n)
    width = 0.25

    plt.figure(figsize=(11, 6))
    plt.bar(x - width, plot_df["policy_label_match_rate"], width=width, label="Policy Match")
    plt.bar(x, plot_df["policy_event_active_ratio"], width=width, label="Policy Active")
    plt.bar(x + width, plot_df["econ_event_active_ratio"], width=width, label="Economic Active")
    plt.xticks(x, labels)
    plt.ylim(0, 1.05)
    plt.xlabel("Run ID Rank")
    plt.ylabel("Ratio")
    plt.title("Top-K Config Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()


def _save_tradeoff_scatter(summary_df: pd.DataFrame, fig_path: Path) -> None:
    plt.figure(figsize=(10.5, 6))
    scatter = plt.scatter(
        summary_df["dict_total_size"],
        summary_df["policy_label_match_rate"],
        c=summary_df["policy_event_active_ratio"],
        cmap="viridis",
        s=70,
        alpha=0.9,
        edgecolors="black",
        linewidths=0.3,
    )
    top = summary_df.head(5)
    for row in top.itertuples(index=False):
        plt.annotate(f"#{int(row.run_id)}", (row.dict_total_size, row.policy_label_match_rate), fontsize=9)
    cbar = plt.colorbar(scatter)
    cbar.set_label("Policy Active Ratio")
    plt.xlabel("Dictionary Total Size")
    plt.ylabel("Policy Label Match Rate")
    plt.title("Tradeoff: Dictionary Size vs Policy Match")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()


def _draw_heatmap(ax, pivot_df: pd.DataFrame, title: str) -> None:
    if pivot_df.empty:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No Data", ha="center", va="center")
        ax.axis("off")
        return

    mat = pivot_df.values
    im = ax.imshow(mat, aspect="auto", cmap="YlGnBu")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(pivot_df.columns)))
    ax.set_yticks(np.arange(len(pivot_df.index)))
    ax.set_xticklabels([str(x) for x in pivot_df.columns])
    ax.set_yticklabels([f"{x:.2f}" for x in pivot_df.index])
    ax.set_xlabel("min_support")
    ax.set_ylabel("dict_threshold")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=8, color="black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _save_metric_heatmaps(summary_df: pd.DataFrame, fig_path: Path) -> None:
    pivot_match = build_pivot_table(summary_df, "policy_label_match_rate")
    pivot_policy_active = build_pivot_table(summary_df, "policy_event_active_ratio")
    pivot_econ_active = build_pivot_table(summary_df, "econ_event_active_ratio")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    _draw_heatmap(axes[0], pivot_match, "Policy Match")
    _draw_heatmap(axes[1], pivot_policy_active, "Policy Active")
    _draw_heatmap(axes[2], pivot_econ_active, "Economic Active")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()


def _save_indices_timeseries(event_indices_df: pd.DataFrame, fig_path: Path) -> None:
    if event_indices_df.empty:
        return
    plot_df = event_indices_df.copy()
    plot_df["publish_time"] = pd.to_datetime(plot_df["publish_time"], errors="coerce")
    plot_df = plot_df.sort_values("publish_time")

    plt.figure(figsize=(12, 5.5))
    plt.plot(plot_df["publish_time"], plot_df["Imp"], label="Imp", linewidth=1.8)
    plt.plot(plot_df["publish_time"], plot_df["Ieo"], label="Ieo", linewidth=1.8)
    plt.axhline(0, color="gray", linewidth=1, linestyle="--", alpha=0.7)
    plt.ylim(-1.05, 1.05)
    plt.title("Best Run: Imp / Ieo Time Series")
    plt.xlabel("Publish Time")
    plt.ylabel("Index Value")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()


def save_png_reports(
    fig_dir: Path,
    summary_df: pd.DataFrame,
    top_k_df: pd.DataFrame,
    best_event_indices_df: pd.DataFrame,
) -> Dict[str, Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "topk_bar": fig_dir / "topk_metric_comparison.png",
        "tradeoff_scatter": fig_dir / "size_vs_match_tradeoff.png",
        "metric_heatmaps": fig_dir / "metric_heatmaps.png",
        "best_timeseries": fig_dir / "best_run_imp_ieo_timeseries.png",
    }
    _save_topk_bar(top_k_df, outputs["topk_bar"])
    _save_tradeoff_scatter(summary_df, outputs["tradeoff_scatter"])
    _save_metric_heatmaps(summary_df, outputs["metric_heatmaps"])
    _save_indices_timeseries(best_event_indices_df, outputs["best_timeseries"])
    return outputs


def save_excel_report(
    output_path: Path,
    summary_df: pd.DataFrame,
    top_k_df: pd.DataFrame,
    length_summary_df: pd.DataFrame,
    length_bins_df: pd.DataFrame,
    best_event_indices_df: pd.DataFrame,
) -> None:
    pivot_match = build_pivot_table(summary_df, "policy_label_match_rate")
    pivot_policy_active = build_pivot_table(summary_df, "policy_event_active_ratio")
    pivot_econ_active = build_pivot_table(summary_df, "econ_event_active_ratio")
    pivot_size = build_pivot_table(summary_df, "dict_total_size")

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        top_k_df.to_excel(writer, sheet_name="top_k", index=False)
        length_summary_df.to_excel(writer, sheet_name="base_len_summary", index=False)
        length_bins_df.to_excel(writer, sheet_name="base_len_bins", index=False)
        pivot_match.to_excel(writer, sheet_name="pivot_policy_match")
        pivot_policy_active.to_excel(writer, sheet_name="pivot_policy_active")
        pivot_econ_active.to_excel(writer, sheet_name="pivot_econ_active")
        pivot_size.to_excel(writer, sheet_name="pivot_dict_size")
        best_event_indices_df.to_excel(writer, sheet_name="best_run_event_indices", index=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not 0 <= args.ml_tendency_min_conf <= 1:
        raise ValueError("--ml-tendency-min-conf must be within [0, 1]")
    if args.ml_blend_support_pivot < 1:
        raise ValueError("--ml-blend-support-pivot must be >= 1")

    events = load_events(args.input, args.sheet_index, args.header_row)
    stats = build_phrase_stats(
        events=events,
        min_phrase_chars=args.min_phrase_chars,
        max_phrase_chars=args.max_phrase_chars,
        min_phrase_tokens=args.min_phrase_tokens,
        max_phrase_tokens=args.max_phrase_tokens,
        ml_tendency_min_conf=args.ml_tendency_min_conf,
    )
    phrase_df = build_phrase_dataframe(
        phrase_total_counter=stats["phrase_total_counter"],
        theme_cache=stats["theme_cache"],
        phrase_event_support=stats["phrase_event_support"],
        total_events=len(events),
    )
    length_summary_df, length_bins_df = build_length_distribution_tables(phrase_df)
    event_phrase_df = build_event_phrase_df(stats["event_phrase_counter"])

    summary_rows = []
    run_indices_map: Dict[int, pd.DataFrame] = {}
    configs = list(iter_experiment_configs(args))

    for run_id, cfg in enumerate(configs, start=1):
        run_name = format_run_name(run_id, cfg)

        policy_dict_df = build_dictionary(
            phrase_label_counter=stats["policy_label_counter"],
            labels=POLICY_TONES,
            theme_name=THEME_MONETARY,
            threshold=cfg.dict_threshold,
            min_support=cfg.min_support,
            phrase_event_support=stats["phrase_event_support"],
            total_events=len(events),
            min_dict_chars=cfg.min_dict_chars,
            max_event_coverage=cfg.max_event_coverage,
            tendency_prior=stats.get("policy_tendency_prior") if cfg.use_ml_prior else None,
            tendency_ml_min_conf=args.ml_tendency_min_conf,
            prob_smoothing_alpha=cfg.dict_prob_smoothing_alpha,
            min_label_margin=cfg.dict_min_label_margin,
            ml_blend_weight=cfg.dict_ml_blend_weight,
            ml_blend_support_pivot=args.ml_blend_support_pivot,
        )
        econ_dict_df = build_dictionary(
            phrase_label_counter=stats["econ_label_counter"],
            labels=ECON_TONES,
            theme_name=THEME_ECONOMIC,
            threshold=cfg.dict_threshold,
            min_support=cfg.min_support,
            phrase_event_support=stats["phrase_event_support"],
            total_events=len(events),
            min_dict_chars=cfg.min_dict_chars,
            max_event_coverage=cfg.max_event_coverage,
            tendency_prior=stats.get("econ_tendency_prior") if cfg.use_ml_prior else None,
            tendency_ml_min_conf=args.ml_tendency_min_conf,
            prob_smoothing_alpha=cfg.dict_prob_smoothing_alpha,
            min_label_margin=cfg.dict_min_label_margin,
            ml_blend_weight=cfg.dict_ml_blend_weight,
            ml_blend_support_pivot=args.ml_blend_support_pivot,
        )
        event_indices_df = build_event_indices(events, event_phrase_df, policy_dict_df, econ_dict_df)
        run_indices_map[run_id] = event_indices_df

        summary_rows.append(
            summarize_experiment(
                run_id=run_id,
                run_name=run_name,
                cfg=cfg,
                events=events,
                policy_dict_df=policy_dict_df,
                econ_dict_df=econ_dict_df,
                event_indices_df=event_indices_df,
            )
        )

        if args.save_details:
            run_dir = args.output_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            with pd.ExcelWriter(run_dir / "realtime_dictionary.xlsx", engine="openpyxl") as writer:
                pd.concat([policy_dict_df, econ_dict_df], ignore_index=True).to_excel(
                    writer, sheet_name="dictionary_all", index=False
                )
                policy_dict_df.to_excel(writer, sheet_name="monetary_theme", index=False)
                econ_dict_df.to_excel(writer, sheet_name="economic_theme", index=False)
            event_indices_df.to_excel(run_dir / "event_indices.xlsx", index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = add_ranking_scores(summary_df)
    summary_df = summary_df.sort_values(
        ["score_primary", "policy_label_match_rate", "policy_event_active_ratio", "dict_total_size"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    top_k_df = summary_df.head(max(args.top_k, 1))

    best_run_id = int(summary_df.iloc[0]["run_id"])
    best_event_indices_df = run_indices_map[best_run_id]

    summary_xlsx = args.output_dir / "experiment_summary.xlsx"
    save_excel_report(
        output_path=summary_xlsx,
        summary_df=summary_df,
        top_k_df=top_k_df,
        length_summary_df=length_summary_df,
        length_bins_df=length_bins_df,
        best_event_indices_df=best_event_indices_df,
    )
    summary_df.to_csv(args.output_dir / "experiment_summary.csv", index=False, encoding="utf-8-sig")

    fig_outputs = save_png_reports(
        fig_dir=args.output_dir / "figures",
        summary_df=summary_df,
        top_k_df=top_k_df,
        best_event_indices_df=best_event_indices_df,
    )

    print(f"[INFO] Grid runs: {len(summary_df)}")
    print(f"[INFO] Output Excel: {summary_xlsx.resolve()}")
    print("[INFO] Output PNG files:")
    for p in fig_outputs.values():
        print(f"  - {p.resolve()}")
    print("[INFO] Top configuration:")
    print(top_k_df.head(1).to_string(index=False))


if __name__ == "__main__":
    main()
