#!/usr/bin/env python3
"""Export Stage-2 ours-only sensitivity to min_score and rho.

Outputs:
- outputs/stage2_ours_min_score_sensitivity.csv
- outputs/stage2_ours_min_score_alert_pivot.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from baseline_comparators import (
    load_top_diseases,
    load_series_map,
    evaluate_ours,
    summarize_method,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-2 ours sensitivity export")
    parser.add_argument("--topk", type=int, default=15, help="Top-K diseases from Stage-1 ranking.")
    parser.add_argument(
        "--rho",
        type=float,
        nargs="+",
        default=[0.01, 0.02, 0.05, 0.10, 0.20, 0.30],
        help="Rho grid to evaluate.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        nargs="+",
        default=[0.05, 0.01, 0.0],
        help="Min-score grid to evaluate.",
    )
    return parser.parse_args()


def run_ours_sensitivity(topk: int, rho_list: List[float], min_score_list: List[float]) -> pd.DataFrame:
    diseases = load_top_diseases(topk)
    series_map = load_series_map(diseases)

    rows = []
    for min_score in min_score_list:
        for rho in rho_list:
            frames = []
            for disease in diseases:
                if disease not in series_map:
                    continue
                s = series_map[disease].copy()
                s.name = disease
                df, _ = evaluate_ours(s, rho=rho, min_score=min_score)
                df["rho"] = rho
                frames.append(df)

            all_df = pd.concat(frames, ignore_index=True)
            _, summary = summarize_method(all_df)
            rows.append(
                {
                    "min_score": float(min_score),
                    "rho": float(rho),
                    "total_alerts_monitor": int(summary["total_alerts_monitor"]),
                    "mean_alerts_per_disease": float(summary["mean_alerts_per_disease"]),
                    "top1_coverage_rate": float(summary["top1_coverage_rate"]),
                    "top3_coverage_rate": float(summary["top3_coverage_rate"]),
                    "top3_event_precision": float(summary["top3_event_precision"]),
                }
            )

    out_df = pd.DataFrame(rows).sort_values(["min_score", "rho"]).reset_index(drop=True)
    return out_df


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rho_list = sorted(set(float(r) for r in args.rho))
    min_score_list = sorted(set(float(m) for m in args.min_score), reverse=True)

    sensitivity_df = run_ours_sensitivity(args.topk, rho_list, min_score_list)
    sensitivity_path = OUTPUT_DIR / "stage2_ours_min_score_sensitivity.csv"
    sensitivity_df.to_csv(sensitivity_path, index=False)

    pivot_df = (
        sensitivity_df.pivot(index="min_score", columns="rho", values="total_alerts_monitor")
        .sort_index(ascending=False)
        .reset_index()
    )
    pivot_path = OUTPUT_DIR / "stage2_ours_min_score_alert_pivot.csv"
    pivot_df.to_csv(pivot_path, index=False)

    print(f"Saved: {sensitivity_path}")
    print(f"Saved: {pivot_path}")
    print("\nOurs sensitivity summary:")
    print(sensitivity_df.to_string(index=False, formatters={
        "top1_coverage_rate": "{:.3f}".format,
        "top3_coverage_rate": "{:.3f}".format,
        "top3_event_precision": "{:.3f}".format,
        "mean_alerts_per_disease": "{:.3f}".format,
    }))


if __name__ == "__main__":
    main()

