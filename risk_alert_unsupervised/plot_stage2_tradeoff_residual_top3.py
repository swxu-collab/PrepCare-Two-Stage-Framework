#!/usr/bin/env python3
"""Plot Stage-2 comparator trade-off: burden vs residual-top3 precision."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = Path(__file__).resolve().parent / "outputs" / "baseline_tradeoff_by_rho.csv"
OUT_DIR = PROJECT_ROOT / "frontier_paper" / "figs" / "code_directly_generated_figures"
OUT_STEM = "stage2_tradeoff_residual_top3_precision_vs_burden"


METHOD_NAME = {
    "ours_stl_residual": "Ours (STL residual)",
    "baseline_raw_threshold": "Raw threshold",
    "baseline_seasonal_naive": "Seasonal-naive residual",
}

METHOD_COLOR = {
    "ours_stl_residual": "#1f77b4",
    "baseline_raw_threshold": "#d62728",
    "baseline_seasonal_naive": "#2ca02c",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_CSV).copy()
    if "residual_top3_event_precision" not in df.columns:
        raise ValueError(
            "Missing residual_top3_event_precision in baseline_tradeoff_by_rho.csv. "
            "Please rerun baseline_comparators.py first."
        )
    df = df.sort_values(["method", "rho"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7.4, 5.4))

    for method, g in df.groupby("method"):
        label = METHOD_NAME.get(method, method)
        color = METHOD_COLOR.get(method, None)
        ax.plot(
            g["total_alerts_monitor"],
            g["residual_top3_event_precision"],
            marker="o",
            linewidth=1.8,
            markersize=5,
            label=label,
            color=color,
        )
        for _, row in g.iterrows():
            ax.annotate(
                f"ρ={row['rho']:.2f}",
                (row["total_alerts_monitor"], row["residual_top3_event_precision"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
                color=color if color else "black",
            )

    ours = df[(df["method"] == "ours_stl_residual") & (df["rho"].round(2) == 0.05)]
    if not ours.empty:
        row = ours.iloc[0]
        ax.scatter(
            [row["total_alerts_monitor"]],
            [row["residual_top3_event_precision"]],
            s=80,
            facecolors="none",
            edgecolors=METHOD_COLOR["ours_stl_residual"],
            linewidths=2.0,
            zorder=5,
        )
        ax.annotate(
            "Recommended point\n(min_score=0.00, ρ=0.05)",
            (row["total_alerts_monitor"], row["residual_top3_event_precision"]),
            textcoords="offset points",
            xytext=(10, -28),
            fontsize=9,
            color=METHOD_COLOR["ours_stl_residual"],
        )

    ax.set_xlabel("Total alerts in monitoring window")
    ax.set_ylabel("Residual-top3 event precision")
    ax.set_title("Stage-2 Alerting Trade-off: Burden vs Residual-top3 Precision")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()

    pdf_path = OUT_DIR / f"{OUT_STEM}.pdf"
    png_path = OUT_DIR / f"{OUT_STEM}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()

