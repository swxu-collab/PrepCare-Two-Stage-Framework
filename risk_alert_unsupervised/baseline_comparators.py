#!/usr/bin/env python3
"""Stage-2 comparator baselines for alerting performance.

Methods compared on the same 24-month calibration + 12-month monitoring split:
1. Ours (STL residual score with quantile threshold)
2. Raw-threshold baseline (raw count quantile threshold)
3. Seasonal-naive baseline (x_t - x_{t-12} residual quantile threshold)

Outputs:
- outputs/baseline_ours_alerts.csv
- outputs/baseline_raw_threshold_alerts.csv
- outputs/baseline_seasonal_naive_alerts.csv
- outputs/baseline_comparison_summary.csv
- outputs/baseline_comparison_by_disease.csv
- outputs/baseline_tradeoff_by_rho.csv
- outputs/baseline_matched_burden_comparison.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "merged_disease_cases_by_month_use.csv"
RANKING_PATH = PROJECT_ROOT / "analysis_ml_ranking" / "outputs" / "ml_ranked_diseases.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

import sys

sys.path.append(str(PROJECT_ROOT / "analysis_ml_ranking"))
from en_label_utils import translate  # type: ignore  # noqa: E402


CALIBRATION_MONTHS = 24
PERIOD = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-2 alert comparator baselines")
    parser.add_argument("--topk", type=int, default=15, help="Top-K diseases from Stage-1 ranking.")
    parser.add_argument(
        "--rho",
        type=float,
        action="append",
        help="Repeatable target exceedance rate(s) for quantile threshold, e.g., --rho 0.01 --rho 0.05.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.05,
        help="Minimum threshold floor for score-like methods (ours and seasonal-naive).",
    )
    parser.add_argument(
        "--raw-min-threshold",
        type=float,
        default=0.0,
        help="Lower bound for raw-count threshold (kept at 0 by default).",
    )
    return parser.parse_args()


def load_top_diseases(top_k: int) -> List[str]:
    ranking_df = pd.read_csv(RANKING_PATH)
    return ranking_df.sort_values("final_rank").head(top_k)["disease"].tolist()


def load_series_map(disease_names: List[str]) -> Dict[str, pd.Series]:
    df = pd.read_csv(DATA_PATH)
    month_cols = [c for c in df.columns if c != "disease"]
    month_index = pd.to_datetime(month_cols, format="%Y_%m")
    mapping: Dict[str, pd.Series] = {}
    for _, row in df.iterrows():
        disease = row["disease"]
        if disease in disease_names:
            mapping[disease] = pd.Series(row[month_cols].to_numpy(dtype=float), index=month_index)
    return mapping


def evaluate_ours(
    series: pd.Series, rho: float, min_score: float
) -> Tuple[pd.DataFrame, float]:
    log_series = np.log1p(series)
    stl = STL(log_series, period=PERIOD, robust=True).fit()
    resid = stl.resid

    calib_end = series.index[CALIBRATION_MONTHS - 1]
    calib_mask = pd.Series(series.index <= calib_end, index=series.index)
    monitor_mask = ~calib_mask

    resid_calib = resid[calib_mask]
    median_calib = float(np.median(resid_calib))
    mad_calib = float(np.median(np.abs(resid_calib - median_calib)) + 1e-6)
    score = np.maximum(0.0, resid) / mad_calib

    threshold = float(max(np.quantile(score[calib_mask], 1 - rho), min_score))
    alert = (score >= threshold).astype(int)

    out = pd.DataFrame(
        {
            "method": "ours_stl_residual",
            "disease": series.name,
            "disease_en": translate(series.name),
            "month": series.index,
            "cases": series.values,
            "signal_value": resid.values,
            "score": score,
            "threshold": threshold,
            "alert": alert,
            "phase": np.where(monitor_mask, "monitor", "calibration"),
        }
    )
    return out, threshold


def evaluate_raw_threshold(
    series: pd.Series, rho: float, raw_min_threshold: float
) -> Tuple[pd.DataFrame, float]:
    calib_end = series.index[CALIBRATION_MONTHS - 1]
    calib_mask = pd.Series(series.index <= calib_end, index=series.index)
    monitor_mask = ~calib_mask

    threshold = float(max(np.quantile(series[calib_mask], 1 - rho), raw_min_threshold))
    alert = (series >= threshold).astype(int)

    out = pd.DataFrame(
        {
            "method": "baseline_raw_threshold",
            "disease": series.name,
            "disease_en": translate(series.name),
            "month": series.index,
            "cases": series.values,
            "signal_value": series.values,
            "score": series.values,
            "threshold": threshold,
            "alert": alert,
            "phase": np.where(monitor_mask, "monitor", "calibration"),
        }
    )
    return out, threshold


def evaluate_seasonal_naive(
    series: pd.Series, rho: float, min_score: float
) -> Tuple[pd.DataFrame, float]:
    # Seasonal naive prediction: x_hat_t = x_{t-12}
    lag = PERIOD
    x = series.to_numpy(dtype=float)
    resid = np.full_like(x, np.nan, dtype=float)
    resid[lag:] = x[lag:] - x[:-lag]

    # Score-like one-sided residual, aligned with anomaly-above-baseline logic.
    score = np.maximum(0.0, np.nan_to_num(resid, nan=0.0))

    month_index = series.index
    calib_end = month_index[CALIBRATION_MONTHS - 1]
    calib_mask = pd.Series(month_index <= calib_end, index=month_index)
    monitor_mask = ~calib_mask

    # Use calibration months with valid lag residuals (months 13..24).
    valid_calib = calib_mask & pd.Series(~np.isnan(resid), index=month_index)
    if valid_calib.sum() == 0:
        threshold = float(min_score)
    else:
        threshold = float(max(np.quantile(score[valid_calib], 1 - rho), min_score))
    alert = (score >= threshold).astype(int)

    out = pd.DataFrame(
        {
            "method": "baseline_seasonal_naive",
            "disease": series.name,
            "disease_en": translate(series.name),
            "month": month_index,
            "cases": x,
            "signal_value": resid,
            "score": score,
            "threshold": threshold,
            "alert": alert,
            "phase": np.where(monitor_mask, "monitor", "calibration"),
        }
    )
    return out, threshold


def build_residual_top3_reference(series_map: Dict[str, pd.Series]) -> Dict[str, set]:
    """Build disease-specific top-3 anomalous months from STL residual scores.

    Reference months are defined on the 12-month monitoring window using the
    one-sided robust score from the STL residual model (same score family as ours).
    """
    ref_map: Dict[str, set] = {}
    for disease, series in series_map.items():
        s = series.copy()
        s.name = disease
        log_series = np.log1p(s)
        stl = STL(log_series, period=PERIOD, robust=True).fit()
        resid = stl.resid

        calib_end = s.index[CALIBRATION_MONTHS - 1]
        calib_mask = pd.Series(s.index <= calib_end, index=s.index)
        monitor_mask = ~calib_mask

        resid_calib = resid[calib_mask]
        median_calib = float(np.median(resid_calib))
        mad_calib = float(np.median(np.abs(resid_calib - median_calib)) + 1e-6)
        score = np.maximum(0.0, resid) / mad_calib

        monitor_df = pd.DataFrame({"month": s.index, "score": score})
        monitor_df = monitor_df[monitor_mask.values].copy()
        top3 = monitor_df.nlargest(3, "score")
        ref_map[disease] = set(top3["month"].tolist())
    return ref_map


def summarize_method(
    method_df: pd.DataFrame, residual_top3_ref: Dict[str, set]
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    monitor_df = method_df[method_df["phase"] == "monitor"].copy()
    per_disease_rows: List[Dict[str, float | int | str]] = []
    total_alert_events = 0
    alerts_on_top3_months = 0
    alerts_on_residual_top3_months = 0

    for disease, g in monitor_df.groupby("disease"):
        g = g.sort_values("month")
        alert_months = set(g.loc[g["alert"] == 1, "month"].tolist())

        # High-incidence references inside the monitoring window.
        top1_cases = g["cases"].max()
        top1_months = set(g.loc[g["cases"] == top1_cases, "month"].tolist())

        top3 = g.nlargest(3, "cases")
        top3_months = set(top3["month"].tolist())

        top1_hit = int(len(alert_months & top1_months) > 0)
        top3_hit = int(len(alert_months & top3_months) > 0)
        total_alert_events += int(g["alert"].sum())
        if alert_months:
            alerts_on_top3_months += int(len(alert_months & top3_months))
            residual_top3_months = residual_top3_ref.get(disease, set())
            alerts_on_residual_top3_months += int(len(alert_months & residual_top3_months))

        per_disease_rows.append(
            {
                "method": g["method"].iloc[0],
                "disease": disease,
                "disease_en": g["disease_en"].iloc[0],
                "monitor_months": int(g.shape[0]),
                "alerts_monitor": int(g["alert"].sum()),
                "top1_hit": top1_hit,
                "top3_hit": top3_hit,
            }
        )

    disease_df = pd.DataFrame(per_disease_rows).sort_values(
        ["method", "alerts_monitor"], ascending=[True, False]
    )
    if disease_df.empty:
        return disease_df, {}

    method_name = str(disease_df["method"].iloc[0])
    n_dis = int(disease_df.shape[0])
    total_alerts = int(disease_df["alerts_monitor"].sum())
    summary = {
        "method": method_name,
        "n_diseases": n_dis,
        "total_alerts_monitor": total_alerts,
        "mean_alerts_per_disease": float(disease_df["alerts_monitor"].mean()),
        "diseases_with_any_alert": int((disease_df["alerts_monitor"] > 0).sum()),
        "top1_coverage_rate": float(disease_df["top1_hit"].mean()),
        "top3_coverage_rate": float(disease_df["top3_hit"].mean()),
        "top3_event_precision": float(alerts_on_top3_months / total_alert_events) if total_alert_events > 0 else 0.0,
        "residual_top3_event_precision": float(alerts_on_residual_top3_months / total_alert_events)
        if total_alert_events > 0
        else 0.0,
    }
    return disease_df, summary


def build_matched_burden_table(tradeoff_df: pd.DataFrame) -> pd.DataFrame:
    """Match baseline rows to each ours row by closest total alert burden."""
    rows: List[Dict[str, float | str | int]] = []
    ours_df = tradeoff_df[tradeoff_df["method"] == "ours_stl_residual"].copy()
    for _, ours in ours_df.iterrows():
        for baseline_method in ["baseline_raw_threshold", "baseline_seasonal_naive"]:
            cand = tradeoff_df[tradeoff_df["method"] == baseline_method].copy()
            if cand.empty:
                continue
            cand["burden_gap"] = (cand["total_alerts_monitor"] - ours["total_alerts_monitor"]).abs()
            cand = cand.sort_values(["burden_gap", "rho"]).reset_index(drop=True)
            best = cand.iloc[0]
            rows.append(
                {
                    "ours_rho": float(ours["rho"]),
                    "ours_total_alerts": int(ours["total_alerts_monitor"]),
                    "ours_top3_coverage": float(ours["top3_coverage_rate"]),
                    "baseline_method": baseline_method,
                    "baseline_rho": float(best["rho"]),
                    "baseline_total_alerts": int(best["total_alerts_monitor"]),
                    "baseline_top3_coverage": float(best["top3_coverage_rate"]),
                    "alert_gap_abs": int(abs(int(best["total_alerts_monitor"]) - int(ours["total_alerts_monitor"]))),
                    "coverage_delta_ours_minus_baseline": float(
                        ours["top3_coverage_rate"] - best["top3_coverage_rate"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rho_list = args.rho if args.rho else [0.05]
    rho_list = sorted(set(float(r) for r in rho_list))

    top_diseases = load_top_diseases(args.topk)
    series_map = load_series_map(top_diseases)
    residual_top3_ref = build_residual_top3_reference(series_map)

    ours_frames: List[pd.DataFrame] = []
    raw_frames: List[pd.DataFrame] = []
    naive_frames: List[pd.DataFrame] = []

    for disease in top_diseases:
        if disease not in series_map:
            continue
        s = series_map[disease].copy()
        s.name = disease
        for rho in rho_list:
            ours_df, _ = evaluate_ours(s, rho=rho, min_score=args.min_score)
            raw_df, _ = evaluate_raw_threshold(s, rho=rho, raw_min_threshold=args.raw_min_threshold)
            naive_df, _ = evaluate_seasonal_naive(s, rho=rho, min_score=args.min_score)
            ours_df["rho"] = rho
            raw_df["rho"] = rho
            naive_df["rho"] = rho

            ours_frames.append(ours_df)
            raw_frames.append(raw_df)
            naive_frames.append(naive_df)

    ours_all = pd.concat(ours_frames, ignore_index=True)
    raw_all = pd.concat(raw_frames, ignore_index=True)
    naive_all = pd.concat(naive_frames, ignore_index=True)

    ours_all.to_csv(OUTPUT_DIR / "baseline_ours_alerts.csv", index=False)
    raw_all.to_csv(OUTPUT_DIR / "baseline_raw_threshold_alerts.csv", index=False)
    naive_all.to_csv(OUTPUT_DIR / "baseline_seasonal_naive_alerts.csv", index=False)

    disease_parts: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, float]] = []
    for rho in rho_list:
        for df in [
            ours_all[ours_all["rho"] == rho].copy(),
            raw_all[raw_all["rho"] == rho].copy(),
            naive_all[naive_all["rho"] == rho].copy(),
        ]:
            by_dis, summary = summarize_method(df, residual_top3_ref)
            if by_dis.empty:
                continue
            by_dis["rho"] = rho
            disease_parts.append(by_dis)
            summary["rho"] = rho
            summary_rows.append(summary)

    by_disease_df = pd.concat(disease_parts, ignore_index=True)
    by_disease_df.to_csv(OUTPUT_DIR / "baseline_comparison_by_disease.csv", index=False)

    summary_df = pd.DataFrame(summary_rows).sort_values(["rho", "method"]).reset_index(drop=True)
    summary_df.to_csv(OUTPUT_DIR / "baseline_comparison_summary.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "baseline_tradeoff_by_rho.csv", index=False)

    matched_df = build_matched_burden_table(summary_df)
    matched_df.to_csv(OUTPUT_DIR / "baseline_matched_burden_comparison.csv", index=False)

    print("Saved:")
    print(f"  - {OUTPUT_DIR / 'baseline_ours_alerts.csv'}")
    print(f"  - {OUTPUT_DIR / 'baseline_raw_threshold_alerts.csv'}")
    print(f"  - {OUTPUT_DIR / 'baseline_seasonal_naive_alerts.csv'}")
    print(f"  - {OUTPUT_DIR / 'baseline_comparison_by_disease.csv'}")
    print(f"  - {OUTPUT_DIR / 'baseline_comparison_summary.csv'}")
    print(f"  - {OUTPUT_DIR / 'baseline_tradeoff_by_rho.csv'}")
    print(f"  - {OUTPUT_DIR / 'baseline_matched_burden_comparison.csv'}")
    print("\nSummary:")
    print(
        summary_df.to_string(
            index=False,
            formatters={
                "top1_coverage_rate": lambda x: f"{x:.3f}",
                "top3_coverage_rate": lambda x: f"{x:.3f}",
                "mean_alerts_per_disease": lambda x: f"{x:.3f}",
            },
        )
    )


if __name__ == "__main__":
    main()
