from pathlib import Path
import sys

import pandas as pd


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    raise SystemExit(1)


def pass_msg(msg: str) -> None:
    print(f"[PASS] {msg}")


def main() -> None:
    base = Path("outputs")
    exp = base / "experiments"
    figs = exp / "figures"

    required_files = [
        base / "events_clean.xlsx",
        base / "phrase_distribution.xlsx",
        base / "realtime_dictionary.xlsx",
        base / "event_indices.xlsx",
        base / "sentence_labels.xlsx",
        base / "ml_labeling_report.xlsx",
        exp / "experiment_summary.xlsx",
        exp / "experiment_summary.csv",
        figs / "topk_metric_comparison.png",
        figs / "size_vs_match_tradeoff.png",
        figs / "metric_heatmaps.png",
        figs / "best_run_imp_ieo_timeseries.png",
    ]

    missing = [str(p) for p in required_files if not p.exists()]
    if missing:
        fail(f"缺少输出文件: {missing}")
    pass_msg("输出文件完整")

    events = pd.read_excel(base / "events_clean.xlsx")
    if len(events) < 200:
        fail(f"events_clean 行数异常: {len(events)}")
    pass_msg(f"events_clean 行数={len(events)}")

    sentence_xl = pd.ExcelFile(base / "sentence_labels.xlsx")
    if "sentence_labels" not in sentence_xl.sheet_names:
        fail("sentence_labels.xlsx 缺少 sentence_labels sheet")
    sentence_df = pd.read_excel(base / "sentence_labels.xlsx", sheet_name="sentence_labels")
    required_sentence_cols = {
        "clause_id",
        "event_id",
        "clause_text",
        "clause_theme_final",
        "sentence_policy_tone_label",
        "sentence_econ_tone_label",
    }
    if not required_sentence_cols.issubset(set(sentence_df.columns)):
        fail("sentence_labels 缺少关键列")
    if len(sentence_df) == 0:
        fail("sentence_labels 为空")
    pass_msg(f"短句标签行数={len(sentence_df)}")

    ml_xl = pd.ExcelFile(base / "ml_labeling_report.xlsx")
    for sheet in ["phrase_theme_ml", "phrase_tendency_ml", "model_metrics"]:
        if sheet not in ml_xl.sheet_names:
            fail(f"ml_labeling_report 缺少 {sheet}")
    metrics = pd.read_excel(base / "ml_labeling_report.xlsx", sheet_name="model_metrics")
    if metrics.empty:
        fail("model_metrics 为空")
    if "available" not in metrics.columns:
        fail("model_metrics 缺少 available 列")
    if int(metrics["available"].sum()) < 2:
        fail("可用 ML 模型数量过少")
    pass_msg(f"可用 ML 模型数={int(metrics['available'].sum())}/{len(metrics)}")

    dictionary = pd.read_excel(base / "realtime_dictionary.xlsx", sheet_name="dictionary_all")
    required_dict_cols = {"phrase", "theme", "tendency", "tendency_prob"}
    if not required_dict_cols.issubset(set(dictionary.columns)):
        fail("realtime_dictionary 缺少关键列")
    if dictionary["tendency_prob"].min() <= 0.5:
        fail("dictionary 含有 tendency_prob <= 0.5 的词条")
    if dictionary["theme"].nunique() < 2:
        fail("dictionary 主题类别不足")
    pass_msg(f"词典条数={len(dictionary)}，主题数={dictionary['theme'].nunique()}")

    indices = pd.read_excel(base / "event_indices.xlsx")
    required_idx_cols = {
        "event_id",
        "Imp",
        "Ieo",
        "P_宽松",
        "P_稳健",
        "P_从紧",
        "P_正面",
        "P_中性",
        "P_负面",
    }
    if not required_idx_cols.issubset(set(indices.columns)):
        fail("event_indices 缺少关键列")
    if len(indices) != len(events):
        fail("event_indices 与 events_clean 行数不一致")
    pass_msg(f"事件指数行数={len(indices)}")

    exp_summary = pd.read_excel(exp / "experiment_summary.xlsx", sheet_name="summary")
    if len(exp_summary) < 10:
        fail("实验配置数量过少")
    pass_msg(f"实验配置数={len(exp_summary)}")

    print("[OK] 所有验收项通过")


if __name__ == "__main__":
    main()
