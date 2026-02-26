#!/usr/bin/env python3
"""Stage-1 ranking baselines against the teacher target.

This script computes single-pillar ranking comparators (burden/recent/risk/
trend/severity) and evaluates them against the entropy-weighted teacher target
with rank-based metrics:
  - MAE
  - RMSE
  - Spearman correlation
  - Kendall tau
  - Top-15 overlap rate

Outputs are written to ``analysis_ml_ranking/outputs``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ml_ranking_pipeline import (  # local module in the same folder
    DATA_PATH,
    OUTPUT_DIR,
    build_feature_table,
    compute_teacher_target,
    parse_month_columns,
    topk_overlap_rate,
)


def evaluate_single_pillars() -> pd.DataFrame:
    """Compute Stage-1 single-pillar baselines vs teacher target."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    value_cols = [c for c in df.columns if c != "disease"]
    _, time_index = parse_month_columns(value_cols)
    feature_df = build_feature_table(df, value_cols, time_index)

    teacher_target, _, metrics_matrix = compute_teacher_target(feature_df)
    y_true = teacher_target.to_numpy(dtype=float)

    baseline_cols: Dict[str, str] = {
        "Burden-only": "cases",
        "Recent-only": "recent",
        "Risk-only": "risk",
        "Trend-only": "trend",
        "Severity-only": "severity",
    }

    rows: List[Dict[str, float | str]] = []
    for baseline_name, metric_col in baseline_cols.items():
        score = metrics_matrix[metric_col].to_numpy(dtype=float)
        err = y_true - score
        rows.append(
            {
                "baseline_name": baseline_name,
                "pillar": metric_col,
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err ** 2))),
                "spearman": float(pd.Series(y_true).corr(pd.Series(score), method="spearman")),
                "kendall_tau": float(pd.Series(y_true).corr(pd.Series(score), method="kendall")),
                "overlap_at_15": float(topk_overlap_rate(y_true, score, k=15)),
            }
        )

    return pd.DataFrame(rows).sort_values(
        by=["overlap_at_15", "spearman", "mae", "kendall_tau"],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "stage1_single_pillar_baselines.csv"
    result = evaluate_single_pillars()
    result.to_csv(output_path, index=False)
    print(result.to_string(index=False, formatters={
        "mae": lambda x: f"{x:.4f}",
        "rmse": lambda x: f"{x:.4f}",
        "spearman": lambda x: f"{x:.4f}",
        "kendall_tau": lambda x: f"{x:.4f}",
        "overlap_at_15": lambda x: f"{x:.4f}",
    }))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
