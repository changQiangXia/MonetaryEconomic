import argparse
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import (
    BOUNDARY_STOPWORDS,
    DEFAULT_HEADER_ROW,
    DEFAULT_INPUT_FILE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SHEET_INDEX,
    DICTIONARY_EXACT_BLACKLIST,
    DICTIONARY_PROB_THRESHOLD,
    ECONOMIC_GATE_KEYWORDS,
    ECON_NEGATIVE_KEYWORDS,
    ECON_POSITIVE_KEYWORDS,
    ECON_TONES,
    ECONOMIC_KEYWORDS,
    LENGTH_BINS,
    MAX_ASCII_RATIO,
    MAX_DICT_EVENT_COVERAGE,
    MAX_DIGIT_RATIO,
    MAX_PHRASE_CHARS,
    MAX_PHRASE_TOKENS,
    MIN_CJK_CHAR_COUNT,
    MIN_DICT_PHRASE_CHARS,
    MIN_PHRASE_CHARS,
    MIN_PHRASE_SUPPORT,
    MIN_PHRASE_TOKENS,
    MONETARY_GATE_KEYWORDS,
    MONETARY_KEYWORDS,
    PHRASE_STOPWORDS,
    POLICY_TONES,
    THEME_CONTEXT_WEIGHT,
    THEME_ECONOMIC,
    THEME_KEYWORD_WEIGHT,
    THEME_MONETARY,
    THEME_SCORE_GAP_TO_ASSIGN,
    THEME_UNKNOWN,
)

try:
    import jieba  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit("缺少依赖 jieba，请先执行: pip install jieba") from exc

try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.metrics import accuracy_score  # type: ignore
    from sklearn.model_selection import train_test_split  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit("缺少依赖 scikit-learn，请先执行: pip install scikit-learn") from exc


RANDOM_SEED = 42

CLAUSE_SPLIT_PATTERN = re.compile(r"[\u3002\uff01\uff1f!?；;\uff0c,\n\r]+")
WHITESPACE_PATTERN = re.compile(r"\s+")
VALID_CHAR_PATTERN = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")
CJK_CHAR_PATTERN = re.compile(r"[\u4e00-\u9fff]")
ASCII_ALPHA_PATTERN = re.compile(r"[A-Za-z]")
DIGIT_PATTERN = re.compile(r"\d")
DATE_LIKE_PATTERN = re.compile(r"^\d+([./-]\d+)*[%年月日点]?$")

THEME_MIXED = "混合"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MonetaryEconomic pipeline runner")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_FILE, help="Input Excel path")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--sheet-index", type=int, default=DEFAULT_SHEET_INDEX, help="Excel sheet index")
    parser.add_argument("--header-row", type=int, default=DEFAULT_HEADER_ROW, help="Header row index")

    parser.add_argument("--min-phrase-chars", type=int, default=MIN_PHRASE_CHARS)
    parser.add_argument("--max-phrase-chars", type=int, default=MAX_PHRASE_CHARS)
    parser.add_argument("--min-phrase-tokens", type=int, default=MIN_PHRASE_TOKENS)
    parser.add_argument("--max-phrase-tokens", type=int, default=MAX_PHRASE_TOKENS)

    parser.add_argument("--dict-threshold", type=float, default=DICTIONARY_PROB_THRESHOLD)
    parser.add_argument("--min-support", type=int, default=MIN_PHRASE_SUPPORT)
    parser.add_argument("--min-dict-chars", type=int, default=MIN_DICT_PHRASE_CHARS)
    parser.add_argument("--max-dict-event-coverage", type=float, default=MAX_DICT_EVENT_COVERAGE)
    parser.add_argument("--dict-prob-smoothing-alpha", type=float, default=0.0)
    parser.add_argument("--dict-min-label-margin", type=float, default=0.0)
    parser.add_argument("--dict-ml-blend-weight", type=float, default=0.0)
    parser.add_argument("--dict-ml-blend-support-pivot", type=int, default=5)

    parser.add_argument("--ml-theme-high-conf", type=float, default=0.60)
    parser.add_argument("--ml-theme-low-conf", type=float, default=0.50)
    parser.add_argument("--ml-tendency-min-conf", type=float, default=0.55)
    parser.add_argument("--ml-seed-min-support", type=int, default=3)
    parser.add_argument("--ml-seed-min-prob", type=float, default=0.60)
    return parser.parse_args()


def parse_excel_date(value) -> Tuple[pd.Timestamp, bool]:
    if pd.isna(value):
        return pd.NaT, False
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value), False
    if isinstance(value, (int, float)):
        converted = pd.Timestamp("1899-12-30") + pd.to_timedelta(float(value), unit="D")
        return converted, True
    converted = pd.to_datetime(value, errors="coerce")
    return converted, False


def normalize_policy_tone(raw_tone: str) -> str:
    text = str(raw_tone)
    if "宽松" in text:
        return "宽松"
    if "紧缩" in text:
        return "从紧"
    if "稳健" in text:
        return "稳健"
    return "稳健"


def normalize_text(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", str(text)).strip()


def split_to_clauses(text: str) -> List[str]:
    parts = [p.strip(" ：:、\t") for p in CLAUSE_SPLIT_PATTERN.split(text)]
    return [p for p in parts if p]


def tokenize(text: str) -> List[str]:
    tokens = []
    for token in jieba.cut(text):
        token = token.strip()
        if not token:
            continue
        if not VALID_CHAR_PATTERN.search(token):
            continue
        tokens.append(token)
    return tokens


def score_keywords(text: str, keywords: List[str]) -> int:
    return sum(len(keyword) for keyword in keywords if keyword in text)


def infer_clause_theme_hint(clause_text: str) -> Tuple[int, int]:
    monetary_hint = score_keywords(clause_text, MONETARY_KEYWORDS)
    economic_hint = score_keywords(clause_text, ECONOMIC_KEYWORDS)
    return monetary_hint, economic_hint


def infer_clause_theme_seed(monetary_hint: int, economic_hint: int) -> str:
    if monetary_hint > economic_hint and monetary_hint > 0:
        return THEME_MONETARY
    if economic_hint > monetary_hint and economic_hint > 0:
        return THEME_ECONOMIC
    if monetary_hint > 0 and economic_hint > 0:
        return THEME_MIXED
    return THEME_UNKNOWN


def classify_econ_tone(clause_text: str) -> str:
    positive = score_keywords(clause_text, ECON_POSITIVE_KEYWORDS)
    negative = score_keywords(clause_text, ECON_NEGATIVE_KEYWORDS)
    if positive > negative and positive > 0:
        return "正面"
    if negative > positive and negative > 0:
        return "负面"
    return "中性"


def is_noise_phrase(
    phrase: str,
    phrase_tokens: List[str],
    token_count: int,
    min_tokens: int,
    max_tokens: int,
) -> bool:
    if not phrase:
        return True
    if token_count < min_tokens or token_count > max_tokens:
        return True
    if phrase in PHRASE_STOPWORDS:
        return True
    if DATE_LIKE_PATTERN.fullmatch(phrase):
        return True

    first_token = phrase_tokens[0]
    last_token = phrase_tokens[-1]
    if first_token in BOUNDARY_STOPWORDS or last_token in BOUNDARY_STOPWORDS:
        return True
    if all(token in BOUNDARY_STOPWORDS for token in phrase_tokens):
        return True

    cjk_count = len(CJK_CHAR_PATTERN.findall(phrase))
    if cjk_count < MIN_CJK_CHAR_COUNT:
        return True

    char_count = len(phrase)
    if char_count <= 0:
        return True
    digit_ratio = len(DIGIT_PATTERN.findall(phrase)) / char_count
    ascii_ratio = len(ASCII_ALPHA_PATTERN.findall(phrase)) / char_count
    if digit_ratio > MAX_DIGIT_RATIO:
        return True
    if ascii_ratio > MAX_ASCII_RATIO:
        return True
    return False


def generate_phrases(
    tokens: List[str],
    min_chars: int,
    max_chars: int,
    min_tokens: int,
    max_tokens: int,
) -> Iterable[str]:
    token_count = len(tokens)
    for i in range(token_count):
        chunk_tokens: List[str] = []
        chunk_text = ""
        for j in range(i, token_count):
            token = tokens[j]
            chunk_tokens.append(token)
            chunk_text += token

            char_len = len(chunk_text)
            if char_len > max_chars:
                break
            if char_len < min_chars:
                continue
            if not VALID_CHAR_PATTERN.search(chunk_text):
                continue
            if is_noise_phrase(chunk_text, chunk_tokens, j - i + 1, min_tokens, max_tokens):
                continue
            yield chunk_text


def classify_theme(phrase: str, context_counts: Mapping[str, float]) -> str:
    monetary_kw = score_keywords(phrase, MONETARY_KEYWORDS)
    economic_kw = score_keywords(phrase, ECONOMIC_KEYWORDS)
    monetary_ctx = float(context_counts.get(THEME_MONETARY, 0.0))
    economic_ctx = float(context_counts.get(THEME_ECONOMIC, 0.0))

    monetary_score = monetary_kw * THEME_KEYWORD_WEIGHT + monetary_ctx * THEME_CONTEXT_WEIGHT
    economic_score = economic_kw * THEME_KEYWORD_WEIGHT + economic_ctx * THEME_CONTEXT_WEIGHT

    if monetary_score <= 0 and economic_score <= 0:
        return THEME_UNKNOWN
    if abs(monetary_score - economic_score) < THEME_SCORE_GAP_TO_ASSIGN:
        if monetary_kw == 0 and economic_kw == 0:
            return THEME_UNKNOWN
    if monetary_score > economic_score:
        return THEME_MONETARY
    if economic_score > monetary_score:
        return THEME_ECONOMIC
    return THEME_UNKNOWN


def load_events(input_path: Path, sheet_index: int, header_row: int) -> pd.DataFrame:
    raw = pd.read_excel(input_path, sheet_name=sheet_index, header=header_row)
    if raw.shape[1] < 5:
        raise ValueError("输入文件字段数不足，至少需要前5列。")

    events = raw.iloc[:, :5].copy()
    events.columns = ["event_id", "source", "publish_time_raw", "tone_raw", "text_raw"]

    events["event_id"] = pd.to_numeric(events["event_id"], errors="coerce")
    events = events[events["event_id"].notna()].copy()
    events["event_id"] = events["event_id"].astype(int)

    parsed_dates = events["publish_time_raw"].apply(parse_excel_date)
    events["publish_time"] = parsed_dates.apply(lambda x: x[0])
    events["date_fixed_from_serial"] = parsed_dates.apply(lambda x: x[1])

    events["policy_tone"] = events["tone_raw"].apply(normalize_policy_tone)
    events["text"] = events["text_raw"].fillna("").map(normalize_text)
    events["source"] = events["source"].fillna("").map(normalize_text)
    events["is_duplicate_text"] = events["text"].duplicated(keep=False)
    events["source_missing"] = events["source"].eq("")

    events = events.sort_values(["publish_time", "event_id"]).reset_index(drop=True)
    return events


def fit_char_ngram_classifier(
    texts: List[str],
    labels: List[str],
    model_name: str,
    min_samples_per_class: int = 3,
) -> Dict[str, object]:
    counts = Counter(labels)
    classes = sorted(counts.keys())

    result: Dict[str, object] = {
        "model_name": model_name,
        "available": False,
        "reason": "",
        "train_samples": len(labels),
        "class_count": len(classes),
        "class_distribution": dict(counts),
        "eval_accuracy": np.nan,
        "vectorizer": None,
        "model": None,
    }

    if len(classes) < 2:
        result["reason"] = "classes_lt_2"
        return result
    if min(counts.values()) < min_samples_per_class:
        result["reason"] = "class_samples_too_small"
        return result

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1)
    X = vectorizer.fit_transform(texts)
    y = np.array(labels)

    eval_accuracy = np.nan
    if len(labels) >= 30 and min(counts.values()) >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=RANDOM_SEED,
            stratify=y,
        )
        eval_model = LogisticRegression(max_iter=2000, class_weight="balanced")
        eval_model.fit(X_train, y_train)
        y_pred = eval_model.predict(X_test)
        eval_accuracy = float(accuracy_score(y_test, y_pred))

    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X, y)

    result.update(
        {
            "available": True,
            "reason": "ok",
            "eval_accuracy": eval_accuracy,
            "vectorizer": vectorizer,
            "model": model,
        }
    )
    return result


def predict_char_ngram_classifier(bundle: Dict[str, object], texts: List[str]) -> Tuple[List[str], List[float]]:
    if not bundle.get("available"):
        return ([""] * len(texts), [float("nan")] * len(texts))

    vectorizer = bundle["vectorizer"]
    model = bundle["model"]
    X = vectorizer.transform(texts)
    proba = model.predict_proba(X)
    pred_idx = np.argmax(proba, axis=1)
    classes = model.classes_
    pred_labels = [str(classes[i]) for i in pred_idx]
    pred_probs = [float(proba[i, pred_idx[i]]) for i in range(len(pred_idx))]
    return pred_labels, pred_probs


def seed_phrase_theme_label(phrase: str, context_counts: Mapping[str, float]) -> str:
    monetary_kw = score_keywords(phrase, MONETARY_KEYWORDS)
    economic_kw = score_keywords(phrase, ECONOMIC_KEYWORDS)
    if monetary_kw > economic_kw and monetary_kw > 0:
        return THEME_MONETARY
    if economic_kw > monetary_kw and economic_kw > 0:
        return THEME_ECONOMIC

    monetary_ctx = float(context_counts.get(THEME_MONETARY, 0.0))
    economic_ctx = float(context_counts.get(THEME_ECONOMIC, 0.0))
    if monetary_ctx > economic_ctx and monetary_ctx > 0:
        return THEME_MONETARY
    if economic_ctx > monetary_ctx and economic_ctx > 0:
        return THEME_ECONOMIC
    return THEME_UNKNOWN


def finalize_theme_label(
    heuristic_label: str,
    seed_label: str,
    ml_label: str,
    ml_prob: float,
    high_conf: float,
    low_conf: float,
) -> Tuple[str, str]:
    if ml_label and not np.isnan(ml_prob) and ml_prob >= high_conf:
        return ml_label, "ml_high_conf"
    if heuristic_label != THEME_UNKNOWN:
        return heuristic_label, "heuristic"
    if seed_label in (THEME_MONETARY, THEME_ECONOMIC):
        return seed_label, "seed"
    if ml_label and not np.isnan(ml_prob) and ml_prob >= low_conf:
        return ml_label, "ml_low_conf"
    return THEME_UNKNOWN, "unknown"


def finalize_clause_theme_label(
    seed_label: str,
    ml_label: str,
    ml_prob: float,
    high_conf: float,
    low_conf: float,
) -> Tuple[str, str]:
    if ml_label and not np.isnan(ml_prob) and ml_prob >= high_conf:
        return ml_label, "ml_high_conf"
    if seed_label in (THEME_MONETARY, THEME_ECONOMIC):
        return seed_label, "seed"
    if seed_label == THEME_MIXED:
        return THEME_MIXED, "seed_mixed"
    if ml_label and not np.isnan(ml_prob) and ml_prob >= low_conf:
        return ml_label, "ml_low_conf"
    if ml_label:
        return ml_label, "ml_fallback"
    return THEME_UNKNOWN, "unknown"


def build_tendency_seed_df(
    phrase_label_counter: Dict[str, Counter],
    labels: List[str],
    min_seed_support: int,
    min_seed_prob: float,
    theme_name: str,
) -> pd.DataFrame:
    rows = []
    for phrase, counts in phrase_label_counter.items():
        total = int(sum(counts.values()))
        if total < min_seed_support:
            continue
        probs = {label: counts.get(label, 0) / total for label in labels}
        top_label = max(labels, key=lambda x: probs[x])
        top_prob = probs[top_label]
        if top_prob < min_seed_prob:
            continue
        rows.append(
            {
                "phrase": phrase,
                "theme": theme_name,
                "seed_tendency_label": top_label,
                "seed_tendency_prob": top_prob,
                "seed_support_count": total,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["phrase", "theme", "seed_tendency_label", "seed_tendency_prob", "seed_support_count"]
        )
    return pd.DataFrame(rows)


def model_metric_row(bundle: Dict[str, object]) -> Dict[str, object]:
    return {
        "model_name": bundle.get("model_name"),
        "available": bool(bundle.get("available")),
        "reason": bundle.get("reason", ""),
        "train_samples": int(bundle.get("train_samples", 0)),
        "class_count": int(bundle.get("class_count", 0)),
        "class_distribution": str(bundle.get("class_distribution", {})),
        "eval_accuracy": bundle.get("eval_accuracy", np.nan),
    }


def build_sentence_label_summary(sentence_df: pd.DataFrame) -> pd.DataFrame:
    if sentence_df.empty:
        return pd.DataFrame(columns=["metric", "value"])

    rows = []
    rows.append({"metric": "sentence_count", "value": len(sentence_df)})
    rows.append(
        {
            "metric": "theme_monetary_ratio",
            "value": float((sentence_df["clause_theme_final"] == THEME_MONETARY).mean()),
        }
    )
    rows.append(
        {
            "metric": "theme_economic_ratio",
            "value": float((sentence_df["clause_theme_final"] == THEME_ECONOMIC).mean()),
        }
    )
    rows.append(
        {
            "metric": "theme_mixed_ratio",
            "value": float((sentence_df["clause_theme_final"] == THEME_MIXED).mean()),
        }
    )
    rows.append(
        {
            "metric": "theme_unknown_ratio",
            "value": float((sentence_df["clause_theme_final"] == THEME_UNKNOWN).mean()),
        }
    )

    for label in POLICY_TONES:
        rows.append(
            {
                "metric": f"policy_sentence_ratio_{label}",
                "value": float((sentence_df["sentence_policy_tone_label"] == label).mean()),
            }
        )
    for label in ECON_TONES:
        rows.append(
            {
                "metric": f"econ_sentence_ratio_{label}",
                "value": float((sentence_df["sentence_econ_tone_label"] == label).mean()),
            }
        )
    return pd.DataFrame(rows)


def build_phrase_stats(
    events: pd.DataFrame,
    min_phrase_chars: int,
    max_phrase_chars: int,
    min_phrase_tokens: int,
    max_phrase_tokens: int,
    ml_theme_high_conf: float = 0.60,
    ml_theme_low_conf: float = 0.50,
    ml_tendency_min_conf: float = 0.55,
    ml_seed_min_support: int = 3,
    ml_seed_min_prob: float = 0.60,
) -> Dict[str, object]:
    phrase_total_counter: Counter = Counter()
    event_phrase_counter: Dict[int, Counter] = defaultdict(Counter)
    phrase_event_support: Counter = Counter()
    phrase_policy_counter: Dict[str, Counter] = defaultdict(Counter)
    phrase_econ_counter: Dict[str, Counter] = defaultdict(Counter)
    phrase_theme_context: Dict[str, Counter] = defaultdict(Counter)

    clause_rows: List[Dict[str, object]] = []
    phrase_occurrence_rows: List[Dict[str, object]] = []
    clause_count = 0
    clause_id = 0

    for row in events.itertuples(index=False):
        clauses = split_to_clauses(row.text)
        if not clauses:
            continue

        event_counter = event_phrase_counter[row.event_id]
        for clause_idx, clause in enumerate(clauses, start=1):
            clause_count += 1
            clause_id += 1
            clause_econ_tone = classify_econ_tone(clause)
            clause_monetary_hint, clause_economic_hint = infer_clause_theme_hint(clause)
            clause_theme_seed = infer_clause_theme_seed(clause_monetary_hint, clause_economic_hint)
            tokens = tokenize(clause)

            clause_rows.append(
                {
                    "clause_id": clause_id,
                    "event_id": int(row.event_id),
                    "clause_idx_in_event": int(clause_idx),
                    "publish_time": row.publish_time,
                    "source": row.source,
                    "tone_raw": row.tone_raw,
                    "policy_tone": row.policy_tone,
                    "clause_text": clause,
                    "clause_char_len": len(clause),
                    "clause_token_count": len(tokens),
                    "clause_monetary_hint": int(clause_monetary_hint),
                    "clause_economic_hint": int(clause_economic_hint),
                    "clause_theme_seed": clause_theme_seed,
                    "clause_econ_tone_rule": clause_econ_tone,
                }
            )

            if not tokens:
                continue

            for phrase in generate_phrases(
                tokens=tokens,
                min_chars=min_phrase_chars,
                max_chars=max_phrase_chars,
                min_tokens=min_phrase_tokens,
                max_tokens=max_phrase_tokens,
            ):
                if event_counter[phrase] == 0:
                    phrase_event_support[phrase] += 1

                phrase_total_counter[phrase] += 1
                event_counter[phrase] += 1
                phrase_policy_counter[phrase][row.policy_tone] += 1
                phrase_econ_counter[phrase][clause_econ_tone] += 1

                if clause_monetary_hint > clause_economic_hint and clause_monetary_hint > 0:
                    phrase_theme_context[phrase][THEME_MONETARY] += 1
                elif clause_economic_hint > clause_monetary_hint and clause_economic_hint > 0:
                    phrase_theme_context[phrase][THEME_ECONOMIC] += 1
                elif clause_monetary_hint > 0 and clause_economic_hint > 0:
                    phrase_theme_context[phrase][THEME_MONETARY] += 0.5
                    phrase_theme_context[phrase][THEME_ECONOMIC] += 0.5

                phrase_occurrence_rows.append(
                    {
                        "clause_id": clause_id,
                        "event_id": int(row.event_id),
                        "phrase": phrase,
                        "policy_tone": row.policy_tone,
                        "econ_tone": clause_econ_tone,
                        "clause_theme_seed": clause_theme_seed,
                    }
                )

    clause_df = pd.DataFrame(clause_rows)
    phrase_occurrence_df = pd.DataFrame(phrase_occurrence_rows)
    # Phrase-level theme seed and heuristic
    phrase_theme_rows = []
    for phrase in phrase_total_counter:
        context = phrase_theme_context.get(phrase, Counter())
        seed_label = seed_phrase_theme_label(phrase, context)
        heuristic_label = classify_theme(phrase, context)
        phrase_theme_rows.append(
            {
                "phrase": phrase,
                "theme_seed_label": seed_label,
                "theme_heuristic": heuristic_label,
                "context_monetary": float(context.get(THEME_MONETARY, 0.0)),
                "context_economic": float(context.get(THEME_ECONOMIC, 0.0)),
            }
        )
    phrase_theme_ml_df = pd.DataFrame(phrase_theme_rows)
    if phrase_theme_ml_df.empty:
        phrase_theme_ml_df = pd.DataFrame(
            columns=[
                "phrase",
                "theme_seed_label",
                "theme_heuristic",
                "context_monetary",
                "context_economic",
                "theme_ml_pred",
                "theme_ml_prob",
                "theme_final",
                "theme_final_source",
            ]
        )

    phrase_theme_train = phrase_theme_ml_df[
        phrase_theme_ml_df["theme_seed_label"].isin([THEME_MONETARY, THEME_ECONOMIC])
    ]
    phrase_theme_bundle = fit_char_ngram_classifier(
        texts=phrase_theme_train["phrase"].astype(str).tolist(),
        labels=phrase_theme_train["theme_seed_label"].astype(str).tolist(),
        model_name="phrase_theme_classifier",
    )

    if not phrase_theme_ml_df.empty:
        theme_pred, theme_prob = predict_char_ngram_classifier(
            phrase_theme_bundle, phrase_theme_ml_df["phrase"].astype(str).tolist()
        )
        phrase_theme_ml_df["theme_ml_pred"] = theme_pred
        phrase_theme_ml_df["theme_ml_prob"] = theme_prob
        final_themes = []
        final_sources = []
        for row in phrase_theme_ml_df.itertuples(index=False):
            final_theme, final_source = finalize_theme_label(
                heuristic_label=row.theme_heuristic,
                seed_label=row.theme_seed_label,
                ml_label=row.theme_ml_pred,
                ml_prob=row.theme_ml_prob if not pd.isna(row.theme_ml_prob) else float("nan"),
                high_conf=ml_theme_high_conf,
                low_conf=ml_theme_low_conf,
            )
            final_themes.append(final_theme)
            final_sources.append(final_source)
        phrase_theme_ml_df["theme_final"] = final_themes
        phrase_theme_ml_df["theme_final_source"] = final_sources

    theme_cache: Dict[str, str] = {
        row.phrase: row.theme_final for row in phrase_theme_ml_df.itertuples(index=False)
    }

    # Clause-level theme ML (for sentence labels)
    if clause_df.empty:
        clause_df = pd.DataFrame(
            columns=[
                "clause_id",
                "event_id",
                "clause_idx_in_event",
                "publish_time",
                "source",
                "tone_raw",
                "policy_tone",
                "clause_text",
                "clause_char_len",
                "clause_token_count",
                "clause_monetary_hint",
                "clause_economic_hint",
                "clause_theme_seed",
                "clause_econ_tone_rule",
                "clause_theme_ml_pred",
                "clause_theme_ml_prob",
                "clause_theme_final",
                "clause_theme_final_source",
                "sentence_policy_tone_label",
                "sentence_econ_tone_label",
            ]
        )
        clause_theme_bundle = {
            "model_name": "clause_theme_classifier",
            "available": False,
            "reason": "empty_clause_df",
            "train_samples": 0,
            "class_count": 0,
            "class_distribution": {},
            "eval_accuracy": np.nan,
            "vectorizer": None,
            "model": None,
        }
    else:
        clause_theme_train = clause_df[clause_df["clause_theme_seed"].isin([THEME_MONETARY, THEME_ECONOMIC])]
        clause_theme_bundle = fit_char_ngram_classifier(
            texts=clause_theme_train["clause_text"].astype(str).tolist(),
            labels=clause_theme_train["clause_theme_seed"].astype(str).tolist(),
            model_name="clause_theme_classifier",
        )
        clause_pred, clause_prob = predict_char_ngram_classifier(
            clause_theme_bundle, clause_df["clause_text"].astype(str).tolist()
        )
        clause_df["clause_theme_ml_pred"] = clause_pred
        clause_df["clause_theme_ml_prob"] = clause_prob

        final_clause_themes = []
        final_clause_sources = []
        sentence_policy_labels = []
        sentence_econ_labels = []
        for row in clause_df.itertuples(index=False):
            final_theme, final_source = finalize_clause_theme_label(
                seed_label=row.clause_theme_seed,
                ml_label=row.clause_theme_ml_pred,
                ml_prob=row.clause_theme_ml_prob if not pd.isna(row.clause_theme_ml_prob) else float("nan"),
                high_conf=ml_theme_high_conf,
                low_conf=ml_theme_low_conf,
            )
            final_clause_themes.append(final_theme)
            final_clause_sources.append(final_source)

            sentence_policy_labels.append(row.policy_tone if final_theme == THEME_MONETARY else "")
            sentence_econ_labels.append(row.clause_econ_tone_rule if final_theme == THEME_ECONOMIC else "")

        clause_df["clause_theme_final"] = final_clause_themes
        clause_df["clause_theme_final_source"] = final_clause_sources
        clause_df["sentence_policy_tone_label"] = sentence_policy_labels
        clause_df["sentence_econ_tone_label"] = sentence_econ_labels

    # Build final theme counters using final phrase themes
    policy_label_counter: Dict[str, Counter] = defaultdict(Counter)
    econ_label_counter: Dict[str, Counter] = defaultdict(Counter)
    unknown_theme_count = 0

    for phrase in phrase_total_counter:
        final_theme = theme_cache.get(phrase, THEME_UNKNOWN)
        if final_theme == THEME_MONETARY:
            policy_label_counter[phrase].update(phrase_policy_counter[phrase])
        elif final_theme == THEME_ECONOMIC:
            econ_label_counter[phrase].update(phrase_econ_counter[phrase])
        else:
            unknown_theme_count += 1

    # ML tendency labels (phrase text classifier over pseudo-seeds)
    policy_tendency_seed_df = build_tendency_seed_df(
        phrase_label_counter=policy_label_counter,
        labels=POLICY_TONES,
        min_seed_support=ml_seed_min_support,
        min_seed_prob=ml_seed_min_prob,
        theme_name=THEME_MONETARY,
    )
    econ_tendency_seed_df = build_tendency_seed_df(
        phrase_label_counter=econ_label_counter,
        labels=ECON_TONES,
        min_seed_support=ml_seed_min_support,
        min_seed_prob=ml_seed_min_prob,
        theme_name=THEME_ECONOMIC,
    )

    policy_tendency_bundle = fit_char_ngram_classifier(
        texts=policy_tendency_seed_df["phrase"].astype(str).tolist(),
        labels=policy_tendency_seed_df["seed_tendency_label"].astype(str).tolist(),
        model_name="phrase_policy_tendency_classifier",
    )
    econ_tendency_bundle = fit_char_ngram_classifier(
        texts=econ_tendency_seed_df["phrase"].astype(str).tolist(),
        labels=econ_tendency_seed_df["seed_tendency_label"].astype(str).tolist(),
        model_name="phrase_econ_tendency_classifier",
    )

    policy_tendency_prior: Dict[str, Tuple[str, float]] = {}
    econ_tendency_prior: Dict[str, Tuple[str, float]] = {}
    policy_tendency_ml_rows = []
    econ_tendency_ml_rows = []

    policy_phrases = list(policy_label_counter.keys())
    if policy_phrases:
        pred, prob = predict_char_ngram_classifier(policy_tendency_bundle, policy_phrases)
        for phrase, p_label, p_prob in zip(policy_phrases, pred, prob):
            if p_label and (not np.isnan(p_prob)) and p_prob >= ml_tendency_min_conf:
                policy_tendency_prior[phrase] = (p_label, p_prob)
            policy_tendency_ml_rows.append(
                {
                    "phrase": phrase,
                    "theme": THEME_MONETARY,
                    "tendency_ml_pred": p_label,
                    "tendency_ml_prob": p_prob,
                }
            )

    econ_phrases = list(econ_label_counter.keys())
    if econ_phrases:
        pred, prob = predict_char_ngram_classifier(econ_tendency_bundle, econ_phrases)
        for phrase, p_label, p_prob in zip(econ_phrases, pred, prob):
            if p_label and (not np.isnan(p_prob)) and p_prob >= ml_tendency_min_conf:
                econ_tendency_prior[phrase] = (p_label, p_prob)
            econ_tendency_ml_rows.append(
                {
                    "phrase": phrase,
                    "theme": THEME_ECONOMIC,
                    "tendency_ml_pred": p_label,
                    "tendency_ml_prob": p_prob,
                }
            )

    tendency_ml_df = pd.concat(
        [pd.DataFrame(policy_tendency_ml_rows), pd.DataFrame(econ_tendency_ml_rows)],
        ignore_index=True,
    )
    if tendency_ml_df.empty:
        tendency_ml_df = pd.DataFrame(columns=["phrase", "theme", "tendency_ml_pred", "tendency_ml_prob"])

    model_metrics_df = pd.DataFrame(
        [
            model_metric_row(phrase_theme_bundle),
            model_metric_row(clause_theme_bundle),
            model_metric_row(policy_tendency_bundle),
            model_metric_row(econ_tendency_bundle),
        ]
    )

    sentence_label_summary_df = build_sentence_label_summary(clause_df)

    return {
        "phrase_total_counter": phrase_total_counter,
        "event_phrase_counter": event_phrase_counter,
        "phrase_event_support": phrase_event_support,
        "policy_label_counter": policy_label_counter,
        "econ_label_counter": econ_label_counter,
        "theme_cache": theme_cache,
        "clause_count": clause_count,
        "unknown_theme_count": unknown_theme_count,
        "clause_df": clause_df,
        "phrase_occurrence_df": phrase_occurrence_df,
        "phrase_theme_ml_df": phrase_theme_ml_df,
        "tendency_ml_df": tendency_ml_df,
        "model_metrics_df": model_metrics_df,
        "policy_tendency_prior": policy_tendency_prior,
        "econ_tendency_prior": econ_tendency_prior,
        "sentence_label_summary_df": sentence_label_summary_df,
    }


def build_phrase_dataframe(
    phrase_total_counter: Counter,
    theme_cache: Dict[str, str],
    phrase_event_support: Counter,
    total_events: int,
) -> pd.DataFrame:
    rows = []
    for phrase, count in phrase_total_counter.items():
        event_support = int(phrase_event_support.get(phrase, 0))
        rows.append(
            {
                "phrase": phrase,
                "count": int(count),
                "event_support": event_support,
                "event_coverage": event_support / total_events if total_events else 0.0,
                "char_len": len(phrase),
                "theme": theme_cache.get(phrase, THEME_UNKNOWN),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["phrase", "count", "event_support", "event_coverage", "char_len", "theme"])
    df = pd.DataFrame(rows).sort_values(["count", "char_len"], ascending=[False, True]).reset_index(drop=True)
    return df


def _safe_mode(values: pd.Series) -> float:
    mode_series = values.mode()
    if mode_series.empty:
        return np.nan
    return float(mode_series.iloc[0])


def build_length_distribution_tables(df_phrase: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df_phrase.empty:
        summary = pd.DataFrame(
            [
                {"metric": "mean", "value": np.nan},
                {"metric": "std", "value": np.nan},
                {"metric": "median", "value": np.nan},
                {"metric": "mode", "value": np.nan},
                {"metric": "unique_phrase_count", "value": 0},
                {"metric": "occurrence_count", "value": 0},
            ]
        )
        bins = pd.DataFrame(columns=["length_bin", "unique_count", "unique_share", "occurrence_count", "occ_share"])
        return summary, bins

    length = df_phrase["char_len"]
    unique_total = len(df_phrase)
    occ_total = int(df_phrase["count"].sum())

    summary = pd.DataFrame(
        [
            {"metric": "mean", "value": float(length.mean())},
            {"metric": "std", "value": float(length.std(ddof=1)) if unique_total > 1 else 0.0},
            {"metric": "median", "value": float(length.median())},
            {"metric": "mode", "value": _safe_mode(length)},
            {"metric": "unique_phrase_count", "value": int(unique_total)},
            {"metric": "occurrence_count", "value": int(occ_total)},
        ]
    )

    bin_rows = []
    for left, right in LENGTH_BINS:
        mask = (df_phrase["char_len"] >= left) & (df_phrase["char_len"] <= right)
        unique_count = int(mask.sum())
        occurrence_count = int(df_phrase.loc[mask, "count"].sum())
        bin_rows.append(
            {
                "length_bin": f"[{left}, {right}]",
                "unique_count": unique_count,
                "unique_share": unique_count / unique_total if unique_total else 0.0,
                "occurrence_count": occurrence_count,
                "occ_share": occurrence_count / occ_total if occ_total else 0.0,
            }
        )
    bins = pd.DataFrame(bin_rows)
    return summary, bins

def build_dictionary(
    phrase_label_counter: Dict[str, Counter],
    labels: List[str],
    theme_name: str,
    threshold: float,
    min_support: int,
    phrase_event_support: Counter,
    total_events: int,
    min_dict_chars: int,
    max_event_coverage: float,
    tendency_prior: Optional[Mapping[str, Tuple[str, float]]] = None,
    tendency_ml_min_conf: float = 0.55,
    prob_smoothing_alpha: float = 0.0,
    min_label_margin: float = 0.0,
    ml_blend_weight: float = 0.0,
    ml_blend_support_pivot: int = 5,
) -> pd.DataFrame:
    def to_ml_probabilities(ml_label: str, ml_prob: float) -> Dict[str, float]:
        if ml_label not in labels or pd.isna(ml_prob):
            return {}
        if len(labels) == 1:
            return {labels[0]: 1.0}
        p = float(np.clip(float(ml_prob), 0.0, 1.0))
        other = (1.0 - p) / (len(labels) - 1)
        return {label: (p if label == ml_label else other) for label in labels}

    rows = []
    for phrase, counts in phrase_label_counter.items():
        if phrase in DICTIONARY_EXACT_BLACKLIST:
            continue
        if not passes_dictionary_gate(phrase=phrase, theme_name=theme_name):
            continue

        total = int(sum(counts.values()))
        if total < min_support:
            continue
        if len(phrase) < min_dict_chars:
            continue

        event_support = int(phrase_event_support.get(phrase, 0))
        event_coverage = event_support / total_events if total_events else 0.0
        if event_coverage > max_event_coverage:
            continue

        raw_probabilities = {label: counts.get(label, 0) / total for label in labels}
        if prob_smoothing_alpha > 0:
            smooth_denom = total + prob_smoothing_alpha * len(labels)
            probabilities = {
                label: (counts.get(label, 0) + prob_smoothing_alpha) / smooth_denom for label in labels
            }
        else:
            probabilities = raw_probabilities

        decision_scores = dict(probabilities)
        tendency_count_based = max(labels, key=lambda label: probabilities[label])
        tendency = tendency_count_based
        tendency_source = "count_majority"
        ml_tendency_pred = ""
        ml_tendency_prob = np.nan
        effective_ml_blend_weight = 0.0

        if tendency_prior and phrase in tendency_prior:
            ml_tendency_pred, ml_tendency_prob = tendency_prior[phrase]
            ml_probabilities = to_ml_probabilities(ml_tendency_pred, ml_tendency_prob)
            if ml_probabilities and ml_blend_weight > 0:
                support_decay = min(1.0, ml_blend_support_pivot / max(total, 1))
                effective_ml_blend_weight = ml_blend_weight * support_decay
                decision_scores = {
                    label: (1.0 - effective_ml_blend_weight) * probabilities[label]
                    + effective_ml_blend_weight * ml_probabilities[label]
                    for label in labels
                }
                tendency = max(labels, key=lambda label: decision_scores[label])
                tendency_source = "ml_blend"
            elif ml_tendency_pred in labels and ml_tendency_prob >= tendency_ml_min_conf:
                tendency = ml_tendency_pred
                tendency_source = "ml_prior"

        tendency_prob = decision_scores[tendency] if tendency_source == "ml_blend" else probabilities[tendency]
        sorted_scores = sorted(decision_scores.values(), reverse=True)
        tendency_margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 1.0
        if tendency_margin < min_label_margin:
            continue
        if tendency_prob <= threshold:
            continue

        row = {
            "phrase": phrase,
            "theme": theme_name,
            "tendency": tendency,
            "tendency_prob": tendency_prob,
            "support_count": total,
            "event_support": event_support,
            "event_coverage": event_coverage,
            "char_len": len(phrase),
            "tendency_source": tendency_source,
            "tendency_count_based": tendency_count_based,
            "tendency_ml_pred": ml_tendency_pred,
            "tendency_ml_prob": ml_tendency_prob,
            "tendency_margin": tendency_margin,
            "tendency_prob_count": raw_probabilities.get(tendency, np.nan),
            "tendency_prob_decision": tendency_prob,
            "dict_prob_smoothing_alpha": prob_smoothing_alpha,
            "effective_ml_blend_weight": effective_ml_blend_weight,
        }
        for label in labels:
            row[f"count_{label}"] = int(counts.get(label, 0))
            row[f"prob_{label}"] = probabilities[label]
        rows.append(row)

    if not rows:
        base_cols = [
            "phrase",
            "theme",
            "tendency",
            "tendency_prob",
            "support_count",
            "event_support",
            "event_coverage",
            "char_len",
            "tendency_source",
            "tendency_count_based",
            "tendency_ml_pred",
            "tendency_ml_prob",
            "tendency_margin",
            "tendency_prob_count",
            "tendency_prob_decision",
            "dict_prob_smoothing_alpha",
            "effective_ml_blend_weight",
        ]
        base_cols.extend([f"count_{x}" for x in labels])
        base_cols.extend([f"prob_{x}" for x in labels])
        return pd.DataFrame(columns=base_cols)

    df = pd.DataFrame(rows).sort_values(
        ["support_count", "tendency_prob", "event_coverage", "char_len"],
        ascending=[False, False, True, True],
    )
    return df.reset_index(drop=True)


def passes_dictionary_gate(phrase: str, theme_name: str) -> bool:
    if theme_name == THEME_MONETARY:
        return score_keywords(phrase, MONETARY_GATE_KEYWORDS) > 0
    if theme_name == THEME_ECONOMIC:
        return score_keywords(phrase, ECONOMIC_GATE_KEYWORDS) > 0
    return True


def build_event_phrase_df(event_phrase_counter: Dict[int, Counter]) -> pd.DataFrame:
    rows = []
    for event_id, counter in event_phrase_counter.items():
        for phrase, count in counter.items():
            rows.append({"event_id": int(event_id), "phrase": phrase, "count": int(count)})
    if not rows:
        return pd.DataFrame(columns=["event_id", "phrase", "count"])
    return pd.DataFrame(rows)


def compute_event_probabilities(
    event_phrase_df: pd.DataFrame,
    dictionary_df: pd.DataFrame,
    labels: List[str],
) -> pd.DataFrame:
    output_columns = ["event_id"] + [f"P_{label}" for label in labels]
    if event_phrase_df.empty or dictionary_df.empty:
        return pd.DataFrame(columns=output_columns)

    merged = event_phrase_df.merge(
        dictionary_df[["phrase", "tendency", "tendency_prob"]],
        on="phrase",
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame(columns=output_columns)

    merged["weight"] = merged["count"] * merged["tendency_prob"]
    denominator = merged.groupby("event_id")["weight"].sum()
    numerator = merged.groupby(["event_id", "tendency"])["weight"].sum().unstack(fill_value=0)

    for label in labels:
        if label not in numerator.columns:
            numerator[label] = 0.0
    numerator = numerator[labels]
    prob = numerator.div(denominator, axis=0).fillna(0.0)
    prob.columns = [f"P_{col}" for col in prob.columns]
    return prob.reset_index()


def build_event_indices(
    events: pd.DataFrame,
    event_phrase_df: pd.DataFrame,
    policy_dict: pd.DataFrame,
    econ_dict: pd.DataFrame,
) -> pd.DataFrame:
    policy_prob = compute_event_probabilities(event_phrase_df, policy_dict, POLICY_TONES)
    econ_prob = compute_event_probabilities(event_phrase_df, econ_dict, ECON_TONES)

    base = events[
        ["event_id", "publish_time", "source", "tone_raw", "policy_tone", "text"]
    ].copy()
    result = base.merge(policy_prob, on="event_id", how="left").merge(econ_prob, on="event_id", how="left")

    for label in POLICY_TONES:
        col = f"P_{label}"
        if col not in result.columns:
            result[col] = 0.0
    for label in ECON_TONES:
        col = f"P_{label}"
        if col not in result.columns:
            result[col] = 0.0

    prob_columns = [f"P_{label}" for label in POLICY_TONES + ECON_TONES]
    result[prob_columns] = result[prob_columns].fillna(0.0)

    result["Imp"] = result["P_从紧"] - result["P_宽松"] + result["P_稳健"]
    result["Ieo"] = result["P_负面"] - result["P_正面"] + result["P_中性"]
    return result.sort_values(["publish_time", "event_id"]).reset_index(drop=True)


def export_results(
    output_dir: Path,
    events: pd.DataFrame,
    phrase_df: pd.DataFrame,
    length_summary_df: pd.DataFrame,
    length_bins_df: pd.DataFrame,
    policy_dict_df: pd.DataFrame,
    econ_dict_df: pd.DataFrame,
    event_indices_df: pd.DataFrame,
    sentence_labels_df: Optional[pd.DataFrame] = None,
    sentence_label_summary_df: Optional[pd.DataFrame] = None,
    phrase_theme_ml_df: Optional[pd.DataFrame] = None,
    tendency_ml_df: Optional[pd.DataFrame] = None,
    model_metrics_df: Optional[pd.DataFrame] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    events.to_excel(output_dir / "events_clean.xlsx", index=False)

    with pd.ExcelWriter(output_dir / "phrase_distribution.xlsx", engine="openpyxl") as writer:
        phrase_df.to_excel(writer, sheet_name="phrases_all", index=False)
        length_summary_df.to_excel(writer, sheet_name="length_summary", index=False)
        length_bins_df.to_excel(writer, sheet_name="length_bins", index=False)
        phrase_df.head(200).to_excel(writer, sheet_name="top200_by_count", index=False)

    dictionary_all = pd.concat([policy_dict_df, econ_dict_df], ignore_index=True)
    with pd.ExcelWriter(output_dir / "realtime_dictionary.xlsx", engine="openpyxl") as writer:
        dictionary_all.to_excel(writer, sheet_name="dictionary_all", index=False)
        policy_dict_df.to_excel(writer, sheet_name="monetary_theme", index=False)
        econ_dict_df.to_excel(writer, sheet_name="economic_theme", index=False)

    event_indices_df.to_excel(output_dir / "event_indices.xlsx", index=False)

    if sentence_labels_df is not None:
        with pd.ExcelWriter(output_dir / "sentence_labels.xlsx", engine="openpyxl") as writer:
            sentence_labels_df.to_excel(writer, sheet_name="sentence_labels", index=False)
            if sentence_label_summary_df is not None:
                sentence_label_summary_df.to_excel(writer, sheet_name="summary", index=False)

    if any(x is not None for x in [phrase_theme_ml_df, tendency_ml_df, model_metrics_df]):
        with pd.ExcelWriter(output_dir / "ml_labeling_report.xlsx", engine="openpyxl") as writer:
            if phrase_theme_ml_df is not None:
                phrase_theme_ml_df.to_excel(writer, sheet_name="phrase_theme_ml", index=False)
            if tendency_ml_df is not None:
                tendency_ml_df.to_excel(writer, sheet_name="phrase_tendency_ml", index=False)
            if model_metrics_df is not None:
                model_metrics_df.to_excel(writer, sheet_name="model_metrics", index=False)


def print_run_summary(
    events: pd.DataFrame,
    clause_count: int,
    phrase_df: pd.DataFrame,
    policy_dict_df: pd.DataFrame,
    econ_dict_df: pd.DataFrame,
    event_indices_df: pd.DataFrame,
    unknown_theme_count: int,
    sentence_labels_df: Optional[pd.DataFrame],
    model_metrics_df: Optional[pd.DataFrame],
    output_dir: Path,
) -> None:
    serial_fix_count = int(events["date_fixed_from_serial"].sum())
    duplicate_count = int(events["is_duplicate_text"].sum())
    source_missing_count = int(events["source_missing"].sum())
    policy_cov = (
        (event_indices_df[["P_宽松", "P_稳健", "P_从紧"]].sum(axis=1) > 0).mean()
        if not event_indices_df.empty
        else 0.0
    )
    econ_cov = (
        (event_indices_df[["P_正面", "P_中性", "P_负面"]].sum(axis=1) > 0).mean()
        if not event_indices_df.empty
        else 0.0
    )

    print(f"[INFO] 事件数量: {len(events)}")
    print(f"[INFO] 分句数量: {clause_count}")
    if sentence_labels_df is not None and not sentence_labels_df.empty:
        print(f"[INFO] 短句标签行数: {len(sentence_labels_df)}")
    print(f"[INFO] 候选短语数量(去重): {len(phrase_df)}")
    print(f"[INFO] 未分类短语数量: {unknown_theme_count}")
    print(f"[INFO] 日期序列值修正条数: {serial_fix_count}")
    print(f"[INFO] 重复文本条数: {duplicate_count}")
    print(f"[INFO] 来源缺失条数: {source_missing_count}")
    print(f"[INFO] 货币政策词典短语数: {len(policy_dict_df)}")
    print(f"[INFO] 经济形势词典短语数: {len(econ_dict_df)}")
    print(f"[INFO] 货币政策事件激活率: {policy_cov:.4f}")
    print(f"[INFO] 经济形势事件激活率: {econ_cov:.4f}")
    if model_metrics_df is not None and not model_metrics_df.empty:
        avail = int(model_metrics_df["available"].sum())
        print(f"[INFO] 机器学习模型可用数: {avail}/{len(model_metrics_df)}")
    print(f"[INFO] 输出目录: {output_dir.resolve()}")


def run_pipeline(args: argparse.Namespace) -> Dict[str, pd.DataFrame]:
    if args.min_phrase_chars < 1 or args.max_phrase_chars < args.min_phrase_chars:
        raise ValueError("短语长度参数非法，请检查 --min-phrase-chars 和 --max-phrase-chars。")
    if args.min_phrase_tokens < 1 or args.max_phrase_tokens < args.min_phrase_tokens:
        raise ValueError("短语词数参数非法，请检查 --min-phrase-tokens 和 --max-phrase-tokens。")
    if args.min_support < 1:
        raise ValueError("--min-support 必须 >= 1。")
    if not 0 <= args.dict_threshold < 1:
        raise ValueError("--dict-threshold 建议在 [0, 1) 区间。")
    if not 0 < args.max_dict_event_coverage <= 1:
        raise ValueError("--max-dict-event-coverage 必须在 (0, 1] 区间。")
    if args.dict_prob_smoothing_alpha < 0:
        raise ValueError("--dict-prob-smoothing-alpha 必须 >= 0。")
    if not 0 <= args.dict_min_label_margin < 1:
        raise ValueError("--dict-min-label-margin 建议在 [0, 1) 区间。")
    if not 0 <= args.dict_ml_blend_weight <= 1:
        raise ValueError("--dict-ml-blend-weight 必须在 [0, 1] 区间。")
    if args.dict_ml_blend_support_pivot < 1:
        raise ValueError("--dict-ml-blend-support-pivot 必须 >= 1。")
    if not 0 <= args.ml_theme_low_conf <= args.ml_theme_high_conf <= 1:
        raise ValueError("请检查 ML 主题置信度阈值。")
    if not 0 <= args.ml_tendency_min_conf <= 1:
        raise ValueError("请检查 --ml-tendency-min-conf 范围。")
    if args.ml_seed_min_support < 1:
        raise ValueError("--ml-seed-min-support 必须 >= 1。")
    if not 0 <= args.ml_seed_min_prob <= 1:
        raise ValueError("请检查 --ml-seed-min-prob 范围。")

    events = load_events(args.input, args.sheet_index, args.header_row)
    stats = build_phrase_stats(
        events=events,
        min_phrase_chars=args.min_phrase_chars,
        max_phrase_chars=args.max_phrase_chars,
        min_phrase_tokens=args.min_phrase_tokens,
        max_phrase_tokens=args.max_phrase_tokens,
        ml_theme_high_conf=args.ml_theme_high_conf,
        ml_theme_low_conf=args.ml_theme_low_conf,
        ml_tendency_min_conf=args.ml_tendency_min_conf,
        ml_seed_min_support=args.ml_seed_min_support,
        ml_seed_min_prob=args.ml_seed_min_prob,
    )

    phrase_df = build_phrase_dataframe(
        stats["phrase_total_counter"],
        stats["theme_cache"],
        stats["phrase_event_support"],
        total_events=len(events),
    )
    if not phrase_df.empty and "phrase_theme_ml_df" in stats:
        merge_cols = [
            "phrase",
            "theme_seed_label",
            "theme_heuristic",
            "theme_ml_pred",
            "theme_ml_prob",
            "theme_final_source",
        ]
        phrase_df = phrase_df.merge(stats["phrase_theme_ml_df"][merge_cols], on="phrase", how="left")

    length_summary_df, length_bins_df = build_length_distribution_tables(phrase_df)

    policy_dict_df = build_dictionary(
        phrase_label_counter=stats["policy_label_counter"],
        labels=POLICY_TONES,
        theme_name=THEME_MONETARY,
        threshold=args.dict_threshold,
        min_support=args.min_support,
        phrase_event_support=stats["phrase_event_support"],
        total_events=len(events),
        min_dict_chars=args.min_dict_chars,
        max_event_coverage=args.max_dict_event_coverage,
        tendency_prior=stats.get("policy_tendency_prior"),
        tendency_ml_min_conf=args.ml_tendency_min_conf,
        prob_smoothing_alpha=args.dict_prob_smoothing_alpha,
        min_label_margin=args.dict_min_label_margin,
        ml_blend_weight=args.dict_ml_blend_weight,
        ml_blend_support_pivot=args.dict_ml_blend_support_pivot,
    )
    econ_dict_df = build_dictionary(
        phrase_label_counter=stats["econ_label_counter"],
        labels=ECON_TONES,
        theme_name=THEME_ECONOMIC,
        threshold=args.dict_threshold,
        min_support=args.min_support,
        phrase_event_support=stats["phrase_event_support"],
        total_events=len(events),
        min_dict_chars=args.min_dict_chars,
        max_event_coverage=args.max_dict_event_coverage,
        tendency_prior=stats.get("econ_tendency_prior"),
        tendency_ml_min_conf=args.ml_tendency_min_conf,
        prob_smoothing_alpha=args.dict_prob_smoothing_alpha,
        min_label_margin=args.dict_min_label_margin,
        ml_blend_weight=args.dict_ml_blend_weight,
        ml_blend_support_pivot=args.dict_ml_blend_support_pivot,
    )

    event_phrase_df = build_event_phrase_df(stats["event_phrase_counter"])
    event_indices_df = build_event_indices(events, event_phrase_df, policy_dict_df, econ_dict_df)

    export_results(
        output_dir=args.output_dir,
        events=events,
        phrase_df=phrase_df,
        length_summary_df=length_summary_df,
        length_bins_df=length_bins_df,
        policy_dict_df=policy_dict_df,
        econ_dict_df=econ_dict_df,
        event_indices_df=event_indices_df,
        sentence_labels_df=stats.get("clause_df"),
        sentence_label_summary_df=stats.get("sentence_label_summary_df"),
        phrase_theme_ml_df=stats.get("phrase_theme_ml_df"),
        tendency_ml_df=stats.get("tendency_ml_df"),
        model_metrics_df=stats.get("model_metrics_df"),
    )
    print_run_summary(
        events=events,
        clause_count=stats["clause_count"],
        phrase_df=phrase_df,
        policy_dict_df=policy_dict_df,
        econ_dict_df=econ_dict_df,
        event_indices_df=event_indices_df,
        unknown_theme_count=stats["unknown_theme_count"],
        sentence_labels_df=stats.get("clause_df"),
        model_metrics_df=stats.get("model_metrics_df"),
        output_dir=args.output_dir,
    )
    return {
        "events": events,
        "phrase_df": phrase_df,
        "policy_dict_df": policy_dict_df,
        "econ_dict_df": econ_dict_df,
        "event_indices_df": event_indices_df,
        "sentence_labels_df": stats.get("clause_df", pd.DataFrame()),
        "model_metrics_df": stats.get("model_metrics_df", pd.DataFrame()),
        "phrase_theme_ml_df": stats.get("phrase_theme_ml_df", pd.DataFrame()),
        "tendency_ml_df": stats.get("tendency_ml_df", pd.DataFrame()),
    }


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
