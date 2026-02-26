#!/usr/bin/env python3
"""Generate publication-ready SVG figures for disease importance analysis.

Usage
-----
/home/robbie/miniconda3/envs/disease/bin/python analysis_ml_ranking/plot_figures.py

All figures are saved into ``analysis_ml_ranking/figures``.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import json

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
FIGURE_DIR = Path(__file__).resolve().parent / "figures"

RANKED_PATH = OUTPUT_DIR / "ml_ranked_diseases.csv"
SHAP_SUMMARY_PATH = OUTPUT_DIR / "ml_shap_summary.csv"
METADATA_PATH = OUTPUT_DIR / "ml_metadata.json"

plt.style.use("seaborn-v0_8")
try:
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if Path(font_path).exists():
        font_manager.fontManager.addfont(font_path)
        default_cjk = font_manager.FontProperties(fname=font_path).get_name()
    else:
        default_cjk = "Noto Sans CJK SC"
except Exception:
    default_cjk = "Noto Sans CJK SC"
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.sans-serif": [default_cjk, "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
})


def ensure_directories() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, list[str]]:
    ranked = pd.read_csv(RANKED_PATH)
    shap_summary = pd.read_csv(SHAP_SUMMARY_PATH)
    with METADATA_PATH.open("r", encoding="utf-8") as fp:
        metadata = json.load(fp)
    entropy_weights = pd.Series(metadata["entropy_weights"], dtype=float)
    months = metadata["months"]
    return ranked, shap_summary, entropy_weights, months


def plot_entropy_weights(weights: pd.Series) -> None:
    display = weights.rename(index={"cases_severity": "severity"})
    display.index = [label.replace("_", " ").title() for label in display.index]
    fig, ax = plt.subplots(figsize=(6, 4))
    display.sort_values().plot(kind="barh", ax=ax, color="#386cb0")
    ax.set_xlabel("Entropy Weight")
    ax.set_title("Pillar Weights (Entropy-Derived)")
    ax.bar_label(ax.containers[0], fmt="{:.3f}", padding=3)
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "entropy_weights.svg", transparent=True)
    plt.close(fig)


def plot_pillar_heatmap(ranked: pd.DataFrame, top_n: int = 15) -> None:
    cols = ["metric_cases", "metric_recent", "metric_risk", "metric_trend", "metric_severity", "metric_cases_severity"]
    data = ranked.sort_values("final_rank").head(top_n)
    matrix = data[cols].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="viridis")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.replace("metric_", "").replace("cases_severity", "Cases x Severity").title() for c in cols], rotation=30, ha="right")
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data["disease"].tolist())
    ax.set_title("Top Diseases: Normalised Pillar Profile")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Normalised Contribution")
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "pillar_heatmap.svg", transparent=True)
    plt.close(fig)


def plot_shap_contributions(shap_summary: pd.DataFrame) -> None:
    cols = shap_summary.copy()
    cols["abs_mean"] = cols["mean_contribution"].abs()
    cols = cols.sort_values("abs_mean", ascending=True)
    total = cols["abs_mean"].sum()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(cols["feature"], cols["abs_mean"], color="#7fc97f")
    ax.set_xlabel("Mean |SHAP| Contribution")
    ax.set_title("Feature Importance (TreeSHAP)")
    max_width = cols["abs_mean"].max() if not cols.empty else 0.0
    for bar, value in zip(bars, cols["abs_mean"]):
        pct = value / total if total else 0.0
        ax.text(
            bar.get_width() + (max_width * 0.05 if max_width else 0.02),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f} ({pct:.1%})",
            va="center",
            fontsize=9,
        )
    ax.set_xlim(0, max_width * 1.2 if max_width else 1)
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "shap_summary.svg", transparent=True)
    plt.close(fig)


def plot_risk_vs_burden(ranked: pd.DataFrame, top_n: int = 30, label_top: int = 12) -> None:
    data = ranked.nsmallest(top_n, "final_rank")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    scatter = ax.scatter(
        np.log1p(data["total_cases"]),
        data["risk_score"],
        s=200 * data["final_target"],
        c=data["severity_score"],
        cmap="plasma",
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
    )
    ylim = ax.get_ylim()
    min_gap = 0.075 * (ylim[1] - ylim[0])
    y_positions: list[float] = []
    labeled = data.nsmallest(label_top, "final_rank")
    for _, row in labeled.iterrows():
        x = np.log1p(row["total_cases"])
        y = row["risk_score"]
        text_y = y
        for attempt in range(12):
            if all(abs(text_y - existing) > min_gap for existing in y_positions):
                break
            direction = 1 if attempt % 2 == 0 else -1
            text_y += direction * min_gap * (0.6 + attempt * 0.1)
        ax.plot([x, x], [y, text_y], color="0.6", linewidth=0.6, alpha=0.8)
        ax.text(x, text_y, row["disease"], fontsize=8, ha="left", va="center")
        y_positions.append(text_y)
    ax.set_xlabel("log(1 + Total Cases)")
    ax.set_ylabel("Risk Score")
    ax.set_title("Risk vs Burden (bubble size = final score, colour = severity)")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Severity Weight")
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "risk_vs_burden.svg", transparent=True)
    plt.close(fig)


def plot_time_series(ranked: pd.DataFrame, months: Sequence[str], top_n: int = 5, step: int = 4) -> None:
    data = ranked.sort_values("final_rank").head(top_n).copy()
    month_cols = [c for c in ranked.columns if c.startswith("month_")]
    month_labels = [col.replace("month_", "").replace("_", "-") for col in month_cols]
    values = data[month_cols].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for idx, (disease, series) in enumerate(zip(data["disease"], values)):
        ax.plot(month_labels, series, label=disease, linewidth=2 - idx * 0.1)
    ax.set_xlabel("Month")
    ax.set_ylabel("Monthly Cases")
    ax.set_title("Top Diseases: Monthly Trajectory")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    tick_positions = list(range(0, len(month_labels), step))
    if tick_positions[-1] != len(month_labels) - 1:
        tick_positions.append(len(month_labels) - 1)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([month_labels[idx] for idx in tick_positions], rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "top_time_series.svg", transparent=True)
    plt.close(fig)


def plot_pillar_radar(ranked: pd.DataFrame, top_diseases: Sequence[str], suffix: str) -> None:
    cols = ["metric_cases", "metric_recent", "metric_risk", "metric_trend", "metric_severity", "metric_cases_severity"]
    data = ranked.set_index("disease").loc[list(top_diseases), cols]
    labels = [c.replace("metric_", "").replace("cases_severity", "Cases x Severity").title() for c in cols]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for disease, row in data.iterrows():
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=disease)
        ax.fill(angles, values, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.set_title("Pillar Profile Radar Chart")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.05))
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / f"pillar_radar_{suffix}.svg", transparent=True)
    plt.close(fig)


def main() -> None:
    ensure_directories()
    ranked, shap_summary, entropy_weights, months = load_data()
    plot_entropy_weights(entropy_weights)
    plot_pillar_heatmap(ranked)
    plot_shap_contributions(shap_summary)
    plot_risk_vs_burden(ranked)
    plot_time_series(ranked, months)
    top5 = ranked.sort_values("final_rank").head(5)["disease"].tolist()
    top10 = ranked.sort_values("final_rank").head(10)["disease"].tolist()
    plot_pillar_radar(ranked, top5, suffix="top5")
    plot_pillar_radar(ranked, top10, suffix="top10")


if __name__ == "__main__":
    main()
