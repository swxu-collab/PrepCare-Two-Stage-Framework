#!/usr/bin/env python3
"""Generate English-only SVG figures for disease importance analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from en_label_utils import translate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
FIGURE_DIR = PROJECT_ROOT / "frontier_paper" / "figs" / "code_directly_generated_figures"

RANKED_PATH = OUTPUT_DIR / "ml_ranked_diseases.csv"
SHAP_SUMMARY_PATH = OUTPUT_DIR / "ml_shap_summary.csv"
METADATA_PATH = OUTPUT_DIR / "ml_metadata.json"

plt.style.use("seaborn-v0_8")
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
})

# Helper: scale fonts to visually match merged widths in LaTeX.
# Target panel (b) uses width=0.55\linewidth at figsize≈(8,6).
# Panels (a) and (c) use width=0.45\linewidth at figsize≈(6,4) and (6,6) respectively.
def _font_scale_for_merge(source_fig_w: float, embed_source: float) -> float:
    target_fig_w = 8.0
    embed_target = 0.5225
    # Empirical scale: account for embed width ratio and base figsize width ratio.
    # This brings smaller panels (a,c) closer to panel (b) perceived text size.
    scale = (embed_target / embed_source) * (target_fig_w / source_fig_w)
    return float(scale)

# Helper: nudge legend below axis until it no longer overlaps the axes+ticks box
def _ensure_legend_below_no_overlap(
    ax: plt.Axes,
    legend: plt.Legend,
    *,
    initial_y: float = -0.12,
    step: float = 0.02,
    min_y: float = -0.5,
    margin_px: float = 6.0,
) -> float:
    fig = ax.figure
    # Start from provided initial anchor
    y = initial_y
    legend.set_bbox_to_anchor((0.5, y), transform=ax.transAxes)
    # Must draw to populate bboxes
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Use tightbbox to include tick labels and decorations
    ax_box = ax.get_tightbbox(renderer)
    # Iteratively move legend down until top of legend is below bottom of axes box
    while True:
        leg_box = legend.get_window_extent(renderer=renderer)
        # Non-overlap if legend top <= axes bottom - margin
        if leg_box.ymax <= (ax_box.ymin - margin_px) or y <= min_y:
            break
        y -= step
        legend.set_bbox_to_anchor((0.5, y), transform=ax.transAxes)
        fig.canvas.draw()
    return y


def ensure_directories(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, list[str]]:
    ranked = pd.read_csv(RANKED_PATH)
    shap_summary = pd.read_csv(SHAP_SUMMARY_PATH)
    with METADATA_PATH.open("r", encoding="utf-8") as fp:
        metadata = json.load(fp)
    entropy_weights = pd.Series(metadata["entropy_weights"], dtype=float)
    months = metadata["months"]
    ranked["disease_en"] = ranked["disease"].map(translate)
    shap_summary["feature_en"] = shap_summary["feature"].replace({
        "total_cases": "Total Cases",
        "trend_slope": "Trend Slope",
        "risk_score": "Risk Score",
        "recent_change_pct": "Recent Change",
        "max_zscore": "Max Z-Score",
        "severity_score": "Severity Prior",
    })
    return ranked, shap_summary, entropy_weights, months


def save_figure(
    fig: plt.Figure,
    out_dir: Path,
    stem: str,
    formats: Sequence[str],
    *,
    facecolor: str = "white",
    transparent: bool = False,
    bbox_inches: Optional[str] = None,
    pad_inches: Optional[float] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fmt_norm = fmt.lower().lstrip(".")
        out_path = out_dir / f"{stem}.{fmt_norm}"
        kwargs = {}
        if fmt_norm == "png":
            kwargs["dpi"] = 300
        if bbox_inches is not None:
            kwargs["bbox_inches"] = bbox_inches
        if pad_inches is not None:
            kwargs["pad_inches"] = pad_inches
        fig.savefig(
            out_path,
            facecolor=facecolor,
            transparent=transparent if fmt_norm in {"png", "svg"} else False,
            **kwargs,
        )


def plot_entropy_weights(weights: pd.Series, out_dir: Path, formats: Sequence[str], *, transparent: bool) -> None:
    print("plot_entropy_weights weights", weights)
    display = weights.rename(index={"cases_severity": "severity"})
    display.index = [label.replace("_", " ").title() for label in display.index]
    fig_w, fig_h = 6.0, 4.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    display.sort_values().plot(kind="barh", ax=ax, color="#2b8cbe")
    # Dynamically enlarge fonts to match merged panel (b)
    base = float(plt.rcParams.get("font.size", 10.0))
    scale = _font_scale_for_merge(source_fig_w=fig_w, embed_source=0.4775)
    label_fs = base * scale * 0.95
    title_fs = base * scale * 1.05
    tick_fs = base * scale * 0.95
    ax.set_xlabel("Entropy Weight", fontsize=label_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)
    texts = ax.bar_label(ax.containers[0], fmt="{:.3f}", padding=3, fontsize=tick_fs)
    # Ensure right-side bar labels sit within the axes background by extending x-limit slightly
    try:
        # First, ensure a minimal right margin relative to data max
        xmax = float(display.max())
        left, right = ax.get_xlim()
        if right < xmax * 1.12:
            ax.set_xlim(left, xmax * 1.12)
        # Then, ensure the axes background covers the full rendered text extents
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        max_right_data = right
        for t in texts:
            bbox = t.get_window_extent(renderer=renderer)
            px = bbox.x1
            py = 0.5 * (bbox.y0 + bbox.y1)
            x_data = ax.transData.inverted().transform((px, py))[0]
            if x_data > max_right_data:
                max_right_data = x_data
        if max_right_data > right:
            left, right = ax.get_xlim()
            margin = 0.05 * (right - left)
            ax.set_xlim(left, max_right_data + margin)
    except Exception:
        pass
    plt.tight_layout()
    save_figure(fig, out_dir, "entropy_weights_en", formats, transparent=transparent)
    plt.close(fig)


def plot_pillar_heatmap(
    ranked: pd.DataFrame,
    out_dir: Path,
    formats: Sequence[str],
    *,
    transparent: bool,
    top_n: int = 15,
) -> None:
    cols = ["metric_cases", "metric_recent", "metric_risk", "metric_trend", "metric_severity", "metric_cases_severity"]
    data = ranked.sort_values("final_rank").head(top_n)
    matrix = data[cols].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="viridis")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.replace("metric_", "").replace("cases_severity", "Cases x Severity").title() for c in cols], rotation=30, ha="right")
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data["disease_en"].tolist())
    # No title (merged panel will carry overall caption)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Normalised Score")
    plt.tight_layout()
    save_figure(
        fig,
        out_dir,
        "pillar_heatmap_en",
        formats,
        transparent=transparent,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close(fig)


def plot_shap_contributions(shap_summary: pd.DataFrame, out_dir: Path, formats: Sequence[str], *, transparent: bool) -> None:
    cols = shap_summary.copy()
    cols["abs_mean"] = cols["mean_contribution"].abs()
    cols = cols.sort_values("abs_mean", ascending=True)
    total = cols["abs_mean"].sum()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(cols["feature_en"], cols["abs_mean"], color="#7fc97f")
    ax.set_xlabel("Mean |SHAP| Contribution")
    # No title (LaTeX figure provides the caption)
    max_width = cols["abs_mean"].max() if not cols.empty else 0.0
    text_objs: list[plt.Text] = []
    for bar, value in zip(bars, cols["abs_mean"]):
        pct = value / total if total else 0.0
        txt = ax.text(
            bar.get_width() + (max_width * 0.05 if max_width else 0.02),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f} ({pct:.1%})",
            va="center",
            fontsize=9,
        )
        text_objs.append(txt)
    # Ensure axes background extends far enough to cover right-side labels
    ax.set_xlim(0, max_width * 1.2 if max_width else 1)
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        left, right = ax.get_xlim()
        max_right_data = right
        for t in text_objs:
            bbox = t.get_window_extent(renderer=renderer)
            px = bbox.x1
            py = 0.5 * (bbox.y0 + bbox.y1)
            x_data = ax.transData.inverted().transform((px, py))[0]
            if x_data > max_right_data:
                max_right_data = x_data
        if max_right_data > right:
            margin = 0.05 * (right - left)
            ax.set_xlim(left, max_right_data + margin)
    except Exception:
        pass
    plt.tight_layout()
    save_figure(fig, out_dir, "shap_summary_en", formats, transparent=transparent)
    plt.close(fig)


def plot_shap_signed(shap_summary: pd.DataFrame, out_dir: Path, formats: Sequence[str], *, transparent: bool) -> None:
    cols = shap_summary.sort_values("mean_contribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#d73027" if val < 0 else "#1a9850" for val in cols["mean_contribution"]]
    bars = ax.barh(cols["feature_en"], cols["mean_contribution"], color=colors)
    ax.axvline(0, color="0.4", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Mean SHAP Contribution")
    ax.set_title("Feature Influence (Signed)")
    max_width = cols["mean_contribution"].abs().max() if not cols.empty else 1.0
    ax.set_xlim(-max_width * 1.2, max_width * 1.2)
    for bar, value in zip(bars, cols["mean_contribution"]):
        ax.text(
            bar.get_width() + (0.03 * max_width if value >= 0 else -0.03 * max_width),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=9,
        )
    plt.tight_layout()
    save_figure(fig, out_dir, "shap_summary_signed_en", formats, transparent=transparent)
    plt.close(fig)


def _distribute_offsets(values: np.ndarray, min_gap: float, lower: float, upper: float) -> np.ndarray:
    """Return adjusted y-values spaced by at least ``min_gap`` within bounds."""

    if len(values) == 0:
        return values
    sorted_idx = np.argsort(values)
    adjusted = values[sorted_idx].astype(float)
    for i in range(1, len(adjusted)):
        if adjusted[i] - adjusted[i - 1] < min_gap:
            adjusted[i] = adjusted[i - 1] + min_gap
    for i in range(len(adjusted) - 2, -1, -1):
        if adjusted[i + 1] - adjusted[i] < min_gap:
            adjusted[i] = adjusted[i + 1] - min_gap
    adjusted = np.clip(adjusted, lower, upper)
    result = np.empty_like(adjusted)
    result[sorted_idx] = adjusted
    return result


def plot_risk_vs_burden(
    ranked: pd.DataFrame,
    out_dir: Path,
    formats: Sequence[str],
    *,
    transparent: bool,
    top_n: int = 15,
) -> None:
    data = ranked.nsmallest(top_n, "final_rank")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    scatter = ax.scatter(
        np.log1p(data["total_cases"]),
        data["risk_score"],
        s=220 * data["final_target"],
        c=data["severity_score"],
        cmap="plasma",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )
    x_vals = np.log1p(data["total_cases"].to_numpy())
    y_vals = data["risk_score"].to_numpy()
    ylim = ax.get_ylim()
    min_gap = 0.08 * (ylim[1] - ylim[0])
    adjusted_y = _distribute_offsets(y_vals, min_gap, ylim[0] - min_gap, ylim[1] + min_gap)
    new_ylim = (min(adjusted_y.min(), ylim[0]), max(adjusted_y.max(), ylim[1]))
    ax.set_ylim(new_ylim)
    for x, y, ay, label in zip(x_vals, y_vals, adjusted_y, data["disease_en"]):
        ax.plot([x, x], [y, ay], color="0.6", linewidth=0.6, alpha=0.8)
        ax.text(x + 0.02, ay, label, fontsize=8, ha="left", va="center")
    ax.set_xlabel("log(1 + Total Cases)")
    ax.set_ylabel("Risk Score")
    ax.set_title("Risk vs Burden")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Severity Prior")
    plt.tight_layout()
    save_figure(fig, out_dir, "risk_vs_burden_en", formats, transparent=transparent)
    plt.close(fig)


def plot_risk_vs_burden_plain(
    ranked: pd.DataFrame,
    out_dir: Path,
    formats: Sequence[str],
    *,
    transparent: bool,
    top_n: int = 15,
) -> None:
    data = ranked.nsmallest(top_n, "final_rank")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    scatter = ax.scatter(
        np.log1p(data["total_cases"]),
        data["risk_score"],
        s=220 * data["final_target"],
        c=data["severity_score"],
        cmap="plasma",
        alpha=0.9,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlabel("log(1 + Total Cases)")
    ax.set_ylabel("Risk Score")
    ax.set_title("Risk vs Burden (Top 15)")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Severity Prior")
    plt.tight_layout()
    save_figure(fig, out_dir, "risk_vs_burden_plain_en", formats, transparent=transparent)
    plt.close(fig)


def plot_time_series(
    ranked: pd.DataFrame,
    months: Sequence[str],
    out_dir: Path,
    formats: Sequence[str],
    *,
    transparent: bool,
    top_n: int = 5,
    step: int = 4,
) -> None:
    data = ranked.sort_values("final_rank").head(top_n).copy()
    month_cols = [c for c in ranked.columns if c.startswith("month_")]
    month_labels = [col.replace("month_", "").replace("_", "-") for col in month_cols]
    values = data[month_cols].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for idx, (disease, series) in enumerate(zip(data["disease_en"], values)):
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
    save_figure(fig, out_dir, "top_time_series_en", formats, transparent=transparent)
    plt.close(fig)


def plot_pillar_radar(
    ranked: pd.DataFrame,
    top_diseases: Sequence[str],
    suffix: str,
    out_dir: Path,
    formats: Sequence[str],
    *,
    transparent: bool,
) -> None:
    cols = ["metric_cases", "metric_recent", "metric_risk", "metric_trend", "metric_severity", "metric_cases_severity"]
    data = ranked.set_index("disease").loc[list(top_diseases), cols]
    labels = [c.replace("metric_", "").replace("cases_severity", "Cases x Severity").title() for c in cols]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig_w = 6.0
    fig, ax = plt.subplots(figsize=(fig_w, 6.0), subplot_kw=dict(polar=True))
    for disease in top_diseases:
        row = data.loc[disease]
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=translate(disease))
        ax.fill(angles, values, alpha=0.15)
    # Dynamically enlarge fonts to match merged panel (b)
    base = float(plt.rcParams.get("font.size", 10.0))
    scale = _font_scale_for_merge(source_fig_w=fig_w, embed_source=0.4775)
    tick_fs = base * scale
    label_fs = base * scale * 0.95
    title_fs = base * scale * 1.05
    ax.set_xticks(angles[:-1])
    # Use custom-placed axis labels at per-axis radii to avoid overlap
    ax.set_xticklabels([])
    # Push baseline (unused) tick layer outward slightly
    ax.tick_params(axis="x", pad=10)
    ax.set_yticklabels([])
    # Compute per-axis label radius beyond the maximum polygon reach on that axis
    try:
        axis_max = data.max().to_numpy()
    except Exception:
        axis_max = np.ones(len(labels)) * 1.0
    extra = 0.12
    r_limit = 1.2
    r_positions = np.minimum(r_limit, axis_max + extra)
    ax.set_ylim(0, r_limit)
    # Place labels with sensible alignment by angle
    for ang, lab, rr in zip(angles[:-1], labels, r_positions):
        # Normalize angle to [-pi, pi]
        a = ((ang + np.pi) % (2 * np.pi)) - np.pi
        if -np.pi/2 < a < np.pi/2:
            ha = "left"
        elif a == np.pi/2 or a == -np.pi/2:
            ha = "center"
        else:
            ha = "right"
        va = "bottom" if a > 0 else ("top" if a < 0 else "center")
        ax.text(
            ang,
            rr,
            lab,
            fontsize=tick_fs * 1.2,
            ha=ha,
            va=va,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=0.6),
        )
    # No title (merged panel will carry overall caption)
    # Place legend below; then dynamically nudge further down if overlapping
    ncols = min(3, len(top_diseases))
    legend = ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=ncols,
        frameon=False,
        fontsize=label_fs,
    )
    _ensure_legend_below_no_overlap(ax, legend, initial_y=-0.12, step=0.02, min_y=-0.5, margin_px=6.0)
    plt.tight_layout()
    save_figure(
        fig,
        out_dir,
        f"pillar_radar_{suffix}_en",
        formats,
        transparent=transparent,
        bbox_inches="tight",
        pad_inches=0.35,
    )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication figures from ML ranking outputs.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=FIGURE_DIR,
        help="Directory to write figures to (default: frontier_paper/figs/code_directly_generated_figures).",
    )
    parser.add_argument(
        "--formats",
        default="pdf,png",
        help="Comma-separated output formats (e.g. 'pdf,png' or 'pdf,png,svg').",
    )
    parser.add_argument(
        "--plots",
        default="all",
        help=(
            "Comma-separated list of plots to generate, or 'all'. "
            "Options: entropy_weights,pillar_heatmap,shap_summary,shap_signed,"
            "risk_vs_burden,risk_vs_burden_plain,top_time_series,pillar_radar_top5,pillar_radar_top10."
        ),
    )
    parser.add_argument(
        "--transparent",
        action="store_true",
        help="Use transparent backgrounds when supported (SVG/PNG).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    plots = [p.strip() for p in args.plots.split(",") if p.strip()]

    ensure_directories(out_dir)
    ranked, shap_summary, entropy_weights, months = load_data()

    top5 = ranked.sort_values("final_rank").head(5)["disease"].tolist()
    top10 = ranked.sort_values("final_rank").head(10)["disease"].tolist()

    available = {
        "entropy_weights": lambda: plot_entropy_weights(entropy_weights, out_dir, formats, transparent=args.transparent),
        "pillar_heatmap": lambda: plot_pillar_heatmap(ranked, out_dir, formats, transparent=args.transparent),
        "shap_summary": lambda: plot_shap_contributions(shap_summary, out_dir, formats, transparent=args.transparent),
        "shap_signed": lambda: plot_shap_signed(shap_summary, out_dir, formats, transparent=args.transparent),
        "risk_vs_burden": lambda: plot_risk_vs_burden(ranked, out_dir, formats, transparent=args.transparent),
        "risk_vs_burden_plain": lambda: plot_risk_vs_burden_plain(ranked, out_dir, formats, transparent=args.transparent),
        "top_time_series": lambda: plot_time_series(ranked, months, out_dir, formats, transparent=args.transparent),
        "pillar_radar_top5": lambda: plot_pillar_radar(ranked, top5, "top5", out_dir, formats, transparent=args.transparent),
        "pillar_radar_top10": lambda: plot_pillar_radar(ranked, top10, "top10", out_dir, formats, transparent=args.transparent),
    }

    if args.plots.strip().lower() == "all":
        selected = list(available.keys())
    else:
        selected = plots

    missing = [name for name in selected if name not in available]
    if missing:
        raise SystemExit(f"Unknown plot(s): {', '.join(missing)}")

    for name in selected:
        available[name]()


if __name__ == "__main__":
    main()
