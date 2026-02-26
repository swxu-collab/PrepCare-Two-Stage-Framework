#!/usr/bin/env python3
"""Stage-2 window sensitivity: compare 24+12 vs 12+12 under same monitor length.

Outputs:
- outputs/stage2_window_sensitivity_comparison.csv
- outputs/stage2_window_sensitivity_comparison_pivot.csv
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
PERIOD = 12

import sys

sys.path.append(str(PROJECT_ROOT / "analysis_ml_ranking"))
from en_label_utils import translate  # type: ignore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-2 window sensitivity (24+12 vs 12+12)")
    parser.add_argument("--topk", type=int, default=15)
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--raw-min-threshold", type=float, default=0.0)
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
            mapping[disease] = pd.Series(row[month_cols].to_numpy(dtype=float), index=month_index, name=disease)
    return mapping


def window_slice(series: pd.Series, calib_months: int, monitor_months: int) -> pd.Series:
    total = calib_months + monitor_months
    if series.shape[0] < total:
        raise ValueError(f"Series '{series.name}' has only {series.shape[0]} months, need {total}.")
    return series.iloc[-total:]


def build_residual_top3_reference(series: pd.Series, calib_months: int) -> set:
    log_series = np.log1p(series)
    stl = STL(log_series, period=PERIOD, robust=True).fit()
    resid = stl.resid
    score = np.maximum(0.0, resid) / (np.median(np.abs(resid[:calib_months] - np.median(resid[:calib_months]))) + 1e-6)
    monitor_df = pd.DataFrame({"month": series.index[calib_months:], "score": score[calib_months:]})
    return set(monitor_df.nlargest(3, "score")["month"].tolist())


def eval_ours(series: pd.Series, calib_months: int, rho: float, min_score: float) -> pd.DataFrame:
    log_series = np.log1p(series)
    stl = STL(log_series, period=PERIOD, robust=True).fit()
    resid = stl.resid
    resid_calib = resid[:calib_months]
    mad_calib = float(np.median(np.abs(resid_calib - np.median(resid_calib))) + 1e-6)
    score = np.maximum(0.0, resid) / mad_calib
    threshold = float(max(np.quantile(score[:calib_months], 1 - rho), min_score))
    alert = (score >= threshold).astype(int)
    return pd.DataFrame(
        {
            "method": "ours_stl_residual",
            "disease": series.name,
            "disease_en": translate(series.name),
            "month": series.index,
            "cases": series.values,
            "alert": alert,
            "phase": ["calibration"] * calib_months + ["monitor"] * (len(series) - calib_months),
        }
    )


def eval_raw(series: pd.Series, calib_months: int, rho: float, raw_min_threshold: float) -> pd.DataFrame:
    threshold = float(max(np.quantile(series.values[:calib_months], 1 - rho), raw_min_threshold))
    alert = (series.values >= threshold).astype(int)
    return pd.DataFrame(
        {
            "method": "baseline_raw_threshold",
            "disease": series.name,
            "disease_en": translate(series.name),
            "month": series.index,
            "cases": series.values,
            "alert": alert,
            "phase": ["calibration"] * calib_months + ["monitor"] * (len(series) - calib_months),
        }
    )


def eval_naive(series: pd.Series, calib_months: int, rho: float, min_score: float) -> pd.DataFrame:
    x = series.to_numpy(dtype=float)
    resid = np.full_like(x, np.nan, dtype=float)
    resid[PERIOD:] = x[PERIOD:] - x[:-PERIOD]
    score = np.maximum(0.0, np.nan_to_num(resid, nan=0.0))
    valid_calib = (~np.isnan(resid[:calib_months]))
    if valid_calib.sum() == 0:
        threshold = float(min_score)
    else:
        threshold = float(max(np.quantile(score[:calib_months][valid_calib], 1 - rho), min_score))
    alert = (score >= threshold).astype(int)
    return pd.DataFrame(
        {
            "method": "baseline_seasonal_naive",
            "disease": series.name,
            "disease_en": translate(series.name),
            "month": series.index,
            "cases": series.values,
            "alert": alert,
            "phase": ["calibration"] * calib_months + ["monitor"] * (len(series) - calib_months),
        }
    )


def summarize(method_df: pd.DataFrame, ref_map: Dict[str, set]) -> Dict[str, float]:
    monitor_df = method_df[method_df["phase"] == "monitor"].copy()
    rows = []
    total_alert_events = 0
    alerts_on_raw_top3 = 0
    alerts_on_residual_top3 = 0
    for disease, g in monitor_df.groupby("disease"):
        alert_months = set(g.loc[g["alert"] == 1, "month"].tolist())
        top3_months = set(g.nlargest(3, "cases")["month"].tolist())
        total_alert_events += int(g["alert"].sum())
        alerts_on_raw_top3 += int(len(alert_months & top3_months))
        alerts_on_residual_top3 += int(len(alert_months & ref_map.get(disease, set())))
        rows.append(int(g["alert"].sum()))
    total_alerts = int(sum(rows))
    n_dis = len(rows)
    return {
        "total_alerts_monitor": total_alerts,
        "mean_alerts_per_disease": float(total_alerts / n_dis if n_dis else 0.0),
        "top3_event_precision": float(alerts_on_raw_top3 / total_alert_events) if total_alert_events else 0.0,
        "residual_top3_event_precision": float(alerts_on_residual_top3 / total_alert_events)
        if total_alert_events
        else 0.0,
    }


def run_setting(
    series_map: Dict[str, pd.Series],
    top_diseases: List[str],
    calib_months: int,
    monitor_months: int,
    rho: float,
    min_score: float,
    raw_min_threshold: float,
) -> pd.DataFrame:
    method_buffers: Dict[str, List[pd.DataFrame]] = {
        "ours_stl_residual": [],
        "baseline_raw_threshold": [],
        "baseline_seasonal_naive": [],
    }
    residual_ref_by_disease: Dict[str, set] = {}
    for disease in top_diseases:
        s = window_slice(series_map[disease], calib_months=calib_months, monitor_months=monitor_months)
        residual_ref_by_disease[disease] = build_residual_top3_reference(s, calib_months=calib_months)
        method_frames = [
            eval_ours(s, calib_months=calib_months, rho=rho, min_score=min_score),
            eval_raw(s, calib_months=calib_months, rho=rho, raw_min_threshold=raw_min_threshold),
            eval_naive(s, calib_months=calib_months, rho=rho, min_score=min_score),
        ]
        for mdf in method_frames:
            method_name = str(mdf["method"].iloc[0])
            method_buffers[method_name].append(mdf)

    out_rows = []
    for method, parts in method_buffers.items():
        if not parts:
            continue
        full_df = pd.concat(parts, ignore_index=True)
        sm = summarize(full_df, residual_ref_by_disease)
        out_rows.append(
            {
                "method": method,
                **sm,
            }
        )
    return pd.DataFrame(out_rows)


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    top_diseases = load_top_diseases(args.topk)
    series_map = load_series_map(top_diseases)

    settings = [
        ("w36_cal24_mon12", 24, 12),
        ("w24_cal12_mon12", 12, 12),
    ]

    all_rows = []
    for setting_name, calib_months, monitor_months in settings:
        df = run_setting(
            series_map=series_map,
            top_diseases=top_diseases,
            calib_months=calib_months,
            monitor_months=monitor_months,
            rho=args.rho,
            min_score=args.min_score,
            raw_min_threshold=args.raw_min_threshold,
        )
        df["setting"] = setting_name
        df["rho"] = args.rho
        df["min_score"] = args.min_score
        all_rows.append(df)

    out_df = pd.concat(all_rows, ignore_index=True)
    out_df = out_df[
        [
            "setting",
            "method",
            "total_alerts_monitor",
            "mean_alerts_per_disease",
            "top3_event_precision",
            "residual_top3_event_precision",
            "rho",
            "min_score",
        ]
    ].sort_values(["setting", "method"])

    pivot = out_df.pivot(
        index="method",
        columns="setting",
        values=[
            "total_alerts_monitor",
            "mean_alerts_per_disease",
            "top3_event_precision",
            "residual_top3_event_precision",
        ],
    )
    pivot.columns = [f"{a}__{b}" for a, b in pivot.columns]
    pivot = pivot.reset_index()

    out_csv = OUTPUT_DIR / "stage2_window_sensitivity_comparison.csv"
    out_pivot_csv = OUTPUT_DIR / "stage2_window_sensitivity_comparison_pivot.csv"
    out_df.to_csv(out_csv, index=False)
    pivot.to_csv(out_pivot_csv, index=False)

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_pivot_csv}")
    print("\nComparison table:")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
