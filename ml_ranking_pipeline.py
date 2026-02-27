#!/usr/bin/env python3
"""Gradient boosted consensus scoring pipeline for disease prioritisation.

Each disease is represented by rich time-series features (level, trend,
volatility, severity priors, and a normalised monthly profile). A composite
target is derived from five epidemiological pillars—burden, recent incidence,
risk, trend, and policy severity—whose weights follow an information-entropy
principle, keeping the method hyper-parameter free.

A LightGBM regressor fits the entropy-weighted target, capturing non-linear
interactions while remaining interpretable via TreeSHAP contributions. Outputs
under ``analysis_ml_ranking/outputs`` include ranked tables, SHAP artefacts,
and pillar weights suitable for publication.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "merged_disease_cases_by_month_use.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

SEVERITY_MAP: Dict[str, float] = {
    "鼠疫": 1.0,
    "霍乱": 1.0,
    "狂犬病": 0.95,
    "人感染高致病性禽流感": 0.95,
    "传染性非典型肺炎": 0.95,
    "新型冠状病毒肺炎": 0.95,
    "艾滋病": 0.95,
    "病毒性肝炎": 0.9,
    "甲型肝炎": 0.8,
    "梅毒": 0.9,
    "乙型肝炎": 0.9,
    "丙型肝炎": 0.9,
    "丁型肝炎": 0.9,
    "肺结核": 0.9,
    "流行性出血热": 0.9,
    "人感染禽流感": 0.9,
    "疟疾": 0.9,
    "戊型肝炎": 0.85,
    "未分型肝炎": 0.85,
    "脊髓灰质炎": 0.85,
    "白喉": 0.85,
    "新生儿破伤风": 0.85,
    "炭疽": 0.85,
    "流行性脑脊髓膜炎": 0.85,
    "登革热": 0.8,
    "麻疹": 0.8,
    "百日咳": 0.8,
    "伤寒和副伤寒": 0.8,
    "斑疹伤寒": 0.8,
    "细菌性和阿米巴性痢疾": 0.8,
    "流行性乙型脑炎": 0.8,
    "钩端螺旋体病": 0.75,
    "血吸虫病": 0.75,
    "布鲁氏菌病": 0.75,
    "淋病": 0.75,
    "猴痘": 0.85,
    "黑热病": 0.75,
    "丝虫病": 0.7,
    "包虫病": 0.7,
    "流行性感冒": 0.7,
    "手足口病": 0.6,
    "其他感染性腹泻病": 0.7,
    "急性出血性结膜炎": 0.65,
    "猩红热": 0.65,
    "流行性腮腺炎": 0.65,
    "风疹": 0.6,
    "麻风病": 0.6,
}
DEFAULT_SEVERITY = 0.6
LIGHTGBM_PARAMS: Dict[str, float | int | str] = {
    "objective": "regression",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "min_child_samples": 6,
    "random_state": 42,
    "verbosity": -1,
}
HYPERPARAM_CANDIDATES: List[Dict[str, float | int | str]] = [
    {
        "objective": "regression",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "min_child_samples": 6,
        "random_state": 42,
        "verbosity": -1,
    },
    {
        "objective": "regression",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.03,
        "n_estimators": 800,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "min_child_samples": 6,
        "random_state": 42,
        "verbosity": -1,
    },
    {
        "objective": "regression",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.10,
        "n_estimators": 300,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "min_child_samples": 6,
        "random_state": 42,
        "verbosity": -1,
    },
    {
        "objective": "regression",
        "boosting_type": "gbdt",
        "num_leaves": 15,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "min_child_samples": 6,
        "random_state": 42,
        "verbosity": -1,
    },
    {
        "objective": "regression",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "min_child_samples": 10,
        "random_state": 42,
        "verbosity": -1,
    },
    {
        "objective": "regression",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "min_child_samples": 6,
        "random_state": 42,
        "verbosity": -1,
    },
    {
        "objective": "regression",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "min_child_samples": 6,
        "random_state": 42,
        "verbosity": -1,
    },
    {
        "objective": "regression",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 700,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "min_child_samples": 3,
        "random_state": 42,
        "verbosity": -1,
    },
]


def ensure_output_dir(path: Path) -> None:
    """Ensure the output directory exists, creating parents when needed."""

    path.mkdir(parents=True, exist_ok=True)


def parse_month_columns(columns: List[str]) -> Tuple[List[str], np.ndarray]:
    """Convert YYYY_MM labels to YYYY-MM strings and centred numeric index."""

    months = []
    for col in columns:
        try:
            months.append(pd.to_datetime(col, format="%Y_%m"))
        except ValueError as exc:
            raise ValueError(f"Column '{col}' is not in YYYY_MM format") from exc
    idx = np.arange(len(months), dtype=float)
    idx -= idx.mean()
    return [m.strftime("%Y-%m") for m in months], idx


def summarise_series(values: np.ndarray, time_index: np.ndarray) -> Dict[str, float]:
    """Summarise one disease’s monthly counts into epidemiological features."""

    log_values = np.log1p(values)
    slope = 0.0
    if not np.allclose(log_values, log_values[0]):
        slope = float(np.polyfit(time_index, log_values, 1)[0])

    total_cases = float(values.sum())
    mean_val = float(values.mean())
    std_val = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    std_log = float(log_values.std(ddof=1)) if len(values) > 1 else 0.0

    max_z = 0.0
    if std_val > 0:
        max_z = float(np.max((values - mean_val) / std_val))
        max_z = max(0.0, max_z)

    last_val = float(values[-1])
    last_mean = float(values[-3:].mean()) if len(values) >= 3 else last_val
    prev_mean = float(values[-6:-3].mean()) if len(values) >= 6 else mean_val
    momentum_ratio = np.nan
    if prev_mean == 0 and last_mean > 0:
        momentum_ratio = np.inf
    elif prev_mean == 0:
        momentum_ratio = 0.0
    else:
        momentum_ratio = last_mean / prev_mean

    cv = std_val / mean_val if mean_val > 0 else 0.0

    recent_window = min(6, len(values))
    baseline_window = min(6, len(values))
    recent_avg = float(values[-recent_window:].mean())
    baseline_avg = float(values[:baseline_window].mean())
    recent_change = np.inf if baseline_avg == 0 and recent_avg > 0 else (
        0.0 if baseline_avg == 0 else (recent_avg - baseline_avg) / baseline_avg
    )

    return {
        "total_cases": total_cases,
        "trend_slope": slope,
        "risk_score": 0.6 * std_log + 0.4 * max_z,
        "max_zscore": max_z,
        "std_log1p": std_log,
        "mean_monthly": mean_val,
        "std_monthly": std_val,
        "cv_monthly": cv,
        "last_month": last_val,
        "last3_mean": last_mean,
        "momentum_ratio": momentum_ratio,
        "recent_avg": recent_avg,
        "baseline_avg": baseline_avg,
        "recent_change_pct": recent_change,
        "max_monthly": float(values.max()),
        "min_monthly": float(values.min()),
    }


def build_feature_table(df: pd.DataFrame, value_cols: List[str], time_index: np.ndarray) -> pd.DataFrame:
    """Construct a per-disease feature table combining stats and profiles."""

    records: List[Dict[str, float]] = []
    for _, row in df.iterrows():
        disease = row["disease"]
        values = row[value_cols].to_numpy(dtype=float)
        stats = summarise_series(values, time_index)

        norm_profile = values / (stats["total_cases"] + 1e-9)
        profile_feats = {f"month_{col}": val for col, val in zip(value_cols, norm_profile)}

        record = {"disease": disease}
        record.update(stats)
        record.update(profile_feats)
        record["severity_score"] = SEVERITY_MAP.get(disease, DEFAULT_SEVERITY)
        records.append(record)

    feature_df = pd.DataFrame(records)

    momentum = feature_df["momentum_ratio"].replace([np.inf, -np.inf], np.nan)
    momentum_fallback = np.nanmax(momentum.to_numpy()) if np.isnan(momentum.to_numpy()).sum() < len(momentum) else 0.0
    feature_df["momentum_ratio"] = momentum.fillna(momentum_fallback)

    recent_change = feature_df["recent_change_pct"].replace([np.inf, -np.inf], np.nan)
    recent_fallback = np.nanmax(recent_change.to_numpy()) if np.isnan(recent_change.to_numpy()).sum() < len(recent_change) else 0.0
    feature_df["recent_change_pct"] = recent_change.fillna(recent_fallback)

    return feature_df


def normalise_series(series: pd.Series) -> pd.Series:
    """Normalise a numeric series to [0,1], handling constant columns."""

    min_val = series.min()
    max_val = series.max()
    if np.isclose(max_val, min_val):
        return pd.Series(np.zeros_like(series), index=series.index)
    return (series - min_val) / (max_val - min_val)


def entropy_weights(matrix: pd.DataFrame) -> pd.Series:
    """Compute entropy-based weights for each column in ``matrix``."""

    Z = matrix.fillna(0.0).to_numpy(dtype=float)
    Z = np.clip(Z, 0.0, None)
    column_sums = Z.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.divide(Z, column_sums, where=column_sums != 0)
    P = np.where(P <= 0, 1e-12, P)
    k = 1.0 / np.log(Z.shape[0]) if Z.shape[0] > 1 else 1.0
    entropy = -k * np.sum(P * np.log(P), axis=0)
    diversity = 1.0 - entropy
    if np.allclose(diversity.sum(), 0):
        weights = np.ones_like(diversity) / len(diversity)
    else:
        weights = diversity / diversity.sum()
    return pd.Series(weights, index=matrix.columns)


def compute_teacher_target(features: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Build the entropy-weighted consensus target used for LightGBM training."""

    log_total = np.log1p(features["total_cases"])
    cases_score = normalise_series(pd.Series(log_total, index=features.index))

    trend_clip = features["trend_slope"].clip(
        lower=features["trend_slope"].quantile(0.05),
        upper=features["trend_slope"].quantile(0.95),
    )
    trend_score = normalise_series(trend_clip)

    risk_clip = features["risk_score"].clip(upper=features["risk_score"].quantile(0.90))
    risk_score = normalise_series(risk_clip)

    recent_score = normalise_series(pd.Series(np.log1p(features["recent_avg"]), index=features.index))
    severity_series = features["severity_score"].astype(float)
    severity_norm = normalise_series(severity_series)

    metrics_matrix = pd.DataFrame(
        {
            "cases": cases_score,
            "recent": recent_score,
            "risk": risk_score,
            "trend": trend_score,
            "severity": severity_norm,
            "cases_severity": cases_score * severity_norm,
        },
        index=features.index,
    )
    weights = entropy_weights(metrics_matrix)
    final_score = (metrics_matrix * weights).sum(axis=1)
    return final_score, weights, metrics_matrix


def topk_overlap_rate(y_true: np.ndarray, y_pred: np.ndarray, k: int = 15) -> float:
    """Compute set-overlap hit-rate between true and predicted top-k disease lists."""

    k_eff = min(k, len(y_true))
    top_true = set(np.argsort(-y_true)[:k_eff].tolist())
    top_pred = set(np.argsort(-y_pred)[:k_eff].tolist())
    overlap = len(top_true & top_pred)
    return float(overlap) / float(k_eff)


def train_regressor(
    features: pd.DataFrame,
    feature_cols: List[str],
    model_params: Dict[str, float | int | str],
) -> Tuple[lgb.LGBMRegressor, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Train LightGBM on entropy-weighted scores and return model artefacts."""

    X = features[feature_cols]
    final_score, weights, metrics_matrix = compute_teacher_target(features)

    regressor = lgb.LGBMRegressor(**model_params)

    regressor.fit(X, final_score)

    predictions = regressor.predict(X)
    features = features.copy()
    for col_name, values in metrics_matrix.items():
        features[f"metric_{col_name}"] = values
    features["final_target"] = final_score
    features["model_score"] = predictions
    features["final_rank"] = features["model_score"].rank(method="dense", ascending=False).astype(int)

    booster = regressor.booster_
    # TreeSHAP contribution for each feature; last column corresponds to bias.
    shap_values = booster.predict(X, pred_contrib=True)
    shap_columns = feature_cols + ["bias"]
    shap_df = pd.DataFrame(shap_values, columns=shap_columns)
    shap_df.insert(0, "disease", features["disease"])  # Align for clarity

    return regressor, features, shap_df, weights


def run_lodo_cv(
    features: pd.DataFrame,
    feature_cols: List[str],
    target: pd.Series,
    model_params: Dict[str, float | int | str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run leave-one-disease-out CV and report regression/ranking metrics."""

    X = features[feature_cols].to_numpy(dtype=float)
    y = target.to_numpy(dtype=float)
    diseases = features["disease"].astype(str).to_numpy()
    n_samples = len(features)

    oof_pred = np.zeros(n_samples, dtype=float)
    for idx in range(n_samples):
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[idx] = False
        model = lgb.LGBMRegressor(**model_params)
        model.fit(X[train_mask], y[train_mask])
        x_test = features.iloc[[idx]][feature_cols]
        oof_pred[idx] = float(model.predict(x_test)[0])

    err = oof_pred - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    spearman = float(pd.Series(y).corr(pd.Series(oof_pred), method="spearman"))
    kendall = float(pd.Series(y).corr(pd.Series(oof_pred), method="kendall"))
    overlap_15 = topk_overlap_rate(y, oof_pred, k=15)

    oof_df = pd.DataFrame(
        {
            "disease": diseases,
            "final_target": y,
            "lodo_pred": oof_pred,
            "abs_error": np.abs(err),
        }
    )
    oof_df["target_rank"] = oof_df["final_target"].rank(method="dense", ascending=False).astype(int)
    oof_df["pred_rank"] = oof_df["lodo_pred"].rank(method="dense", ascending=False).astype(int)

    metrics_df = pd.DataFrame(
        [
            {
                "cv_strategy": "LODO",
                "n_diseases": n_samples,
                "n_folds": n_samples,
                "mae": mae,
                "rmse": rmse,
                "spearman": spearman,
                "kendall_tau": kendall,
                "overlap_at_15": overlap_15,
            }
        ]
    )
    return oof_df, metrics_df


def run_hyperparam_search(
    features: pd.DataFrame,
    feature_cols: List[str],
    target: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Evaluate candidate hyperparameters under disease-level LODO and pick the best set."""

    candidate_rows: List[Dict[str, Any]] = []
    result_rows: List[Dict[str, Any]] = []
    for idx, params in enumerate(HYPERPARAM_CANDIDATES, start=1):
        candidate_row = {"candidate_id": idx}
        candidate_row.update(params)
        candidate_rows.append(candidate_row)

        _, metrics_df = run_lodo_cv(features, feature_cols, target, params)
        row = metrics_df.iloc[0].to_dict()
        row["candidate_id"] = idx
        row["selection_key"] = (
            float(row["overlap_at_15"]),
            float(row["spearman"]),
            -float(row["mae"]),
        )
        result_rows.append(row)

    candidates_df = pd.DataFrame(candidate_rows)
    search_df = pd.DataFrame(result_rows).drop(columns=["selection_key"])

    # Deterministic selection: max overlap@15, then max Spearman, then min MAE.
    best_row = max(result_rows, key=lambda x: x["selection_key"])
    best_id = int(best_row["candidate_id"])
    selected_params = dict(HYPERPARAM_CANDIDATES[best_id - 1])
    selected_params["selected_candidate_id"] = best_id

    search_df = search_df.sort_values(
        by=["overlap_at_15", "spearman", "mae"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    return candidates_df, search_df, selected_params


def aggregate_shap(shap_df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Summarise SHAP contributions for the main epidemiological pillars."""

    focus_cols = ["total_cases", "trend_slope", "risk_score", "recent_change_pct", "max_zscore", "severity_score"]
    valid_cols = [c for c in focus_cols if c in shap_df.columns]
    melted = shap_df.melt(id_vars="disease", value_vars=valid_cols, var_name="feature", value_name="shap_value")
    summary = (
        melted.groupby("feature")["shap_value"]
        .agg(["mean", "median", "std"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    summary.rename(columns={"mean": "mean_contribution", "median": "median_contribution", "std": "std_contribution"}, inplace=True)
    return summary


def save_outputs(
    output_dir: Path,
    ranked_df: pd.DataFrame,
    shap_df: pd.DataFrame,
    shap_summary: pd.DataFrame,
    lodo_oof_df: pd.DataFrame,
    lodo_metrics_df: pd.DataFrame,
    hyper_candidates_df: pd.DataFrame,
    hyper_search_df: pd.DataFrame,
    selected_hyperparams: Dict[str, Any],
) -> None:
    """Persist ranking tables and interpretability artefacts to ``output_dir``."""

    ensure_output_dir(output_dir)
    ranked_df.sort_values("final_rank").to_csv(output_dir / "ml_ranked_diseases.csv", index=False)
    shap_df.to_csv(output_dir / "ml_shap_contributions.csv", index=False)
    shap_summary.to_csv(output_dir / "ml_shap_summary.csv", index=False)
    lodo_oof_df.to_csv(output_dir / "ml_lodo_cv_oof_predictions.csv", index=False)
    lodo_metrics_df.to_csv(output_dir / "ml_lodo_cv_metrics.csv", index=False)
    hyper_candidates_df.to_csv(output_dir / "ml_hyperparam_candidates.csv", index=False)
    hyper_search_df.to_csv(output_dir / "ml_hyperparam_search_results.csv", index=False)
    with open(output_dir / "ml_selected_hyperparams.json", "w", encoding="utf-8") as fp:
        json.dump(selected_hyperparams, fp, indent=2, ensure_ascii=False)


def main() -> None:
    """Entry point: build features, fit the model, and report rankings."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    value_cols = [c for c in df.columns if c != "disease"]
    month_labels, time_index = parse_month_columns(value_cols)

    feature_df = build_feature_table(df, value_cols, time_index)

    feature_cols = [c for c in feature_df.columns if c != "disease"]
    teacher_target, _, _ = compute_teacher_target(feature_df)
    hyper_candidates_df, hyper_search_df, selected_hyperparams = run_hyperparam_search(
        feature_df, feature_cols, teacher_target
    )
    selected_model_params = {
        k: v for k, v in selected_hyperparams.items() if k != "selected_candidate_id"
    }

    model, ranked_df, shap_df, entropy_w = train_regressor(
        feature_df, feature_cols, selected_model_params
    )
    lodo_oof_df, lodo_metrics_df = run_lodo_cv(
        feature_df,
        feature_cols,
        ranked_df["final_target"],
        selected_model_params,
    )
    shap_summary = aggregate_shap(shap_df, ranked_df)

    save_outputs(
        OUTPUT_DIR,
        ranked_df,
        shap_df,
        shap_summary,
        lodo_oof_df,
        lodo_metrics_df,
        hyper_candidates_df,
        hyper_search_df,
        selected_hyperparams,
    )

    display_cols = [
        "disease",
        "final_rank",
        "model_score",
        "final_target",
        "total_cases",
        "trend_slope",
        "risk_score",
        "severity_score",
    ]
    print("\nMachine-Learning Derived Disease Ranking")
    print("======================================")
    print(
        ranked_df.sort_values("final_rank")[display_cols]
        .head(15)
        .to_string(index=False, formatters={
            "final_rank": lambda x: f"{int(x)}",
            "model_score": lambda x: f"{x:,.3f}",
            "final_target": lambda x: f"{x:,.3f}",
            "total_cases": lambda x: f"{x:,.0f}",
            "trend_slope": lambda x: f"{x:,.3f}",
            "risk_score": lambda x: f"{x:,.3f}",
            "severity_score": lambda x: f"{x:,.2f}",
        })
    )

    print("\nFeature Contribution Summary (TreeSHAP)")
    print("======================================")
    print(
        shap_summary.to_string(
            index=False,
            formatters={
                "mean_contribution": lambda x: f"{x:,.4f}",
                "median_contribution": lambda x: f"{x:,.4f}",
                "std_contribution": lambda x: f"{x:,.4f}",
            },
        )
    )

    print("\nEntropy-Derived Pillar Weights")
    print("==============================")
    for name, value in entropy_w.items():
        print(f"{name:>16}: {value:,.4f}")

    print("\nLODO Cross-Validation Metrics")
    print("=============================")
    print(
        lodo_metrics_df.to_string(
            index=False,
            formatters={
                "mae": lambda x: f"{x:,.4f}",
                "rmse": lambda x: f"{x:,.4f}",
                "spearman": lambda x: f"{x:,.4f}",
                "kendall_tau": lambda x: f"{x:,.4f}",
            },
        )
    )

    print("\nHyperparameter Search (Top 5 by overlap@15/Spearman/MAE)")
    print("=======================================================")
    print(
        hyper_search_df.head(5).to_string(
            index=False,
            formatters={
                "mae": lambda x: f"{x:,.4f}",
                "rmse": lambda x: f"{x:,.4f}",
                "spearman": lambda x: f"{x:,.4f}",
                "kendall_tau": lambda x: f"{x:,.4f}",
                "overlap_at_15": lambda x: f"{x:,.4f}",
            },
        )
    )
    print(
        f"\nSelected candidate: #{selected_hyperparams['selected_candidate_id']} "
        f"with params {selected_model_params}"
    )

    metadata = {
        "months": month_labels,
        "lightgbm_version": lgb.__version__,
        "model_params": model.get_params(),
        "tested_hyperparams_file": str((OUTPUT_DIR / "ml_hyperparam_candidates.csv").name),
        "hyperparam_search_file": str((OUTPUT_DIR / "ml_hyperparam_search_results.csv").name),
        "selected_hyperparams_file": str((OUTPUT_DIR / "ml_selected_hyperparams.json").name),
        "selected_hyperparams": selected_hyperparams,
        "entropy_weights": {k: float(v) for k, v in entropy_w.items()},
        "cv_metrics": lodo_metrics_df.iloc[0].to_dict(),
    }
    with open(OUTPUT_DIR / "ml_metadata.json", "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
