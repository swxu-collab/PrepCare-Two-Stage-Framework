#!/usr/bin/env python3
"""Unsupervised alert pipeline using STL residual scores and fixed alert rate.

Workflow summary
----------------
1. Load top-15 diseases and transform monthly counts via ``log1p``.
2. Split each disease timeline into calibration (first 24 months) and
   monitoring phases (remaining months).
3. Apply STL decomposition (period=12) to extract residuals; compute
   one-sided scores ``max(0, resid) / MAD`` using the calibration residuals.
4. Choose per-disease thresholds from the calibration score distribution
   based on a target alert rate ``rho`` (e.g., 5%).
5. Score the monitoring phase, raise alerts when scores exceed thresholds,
   and export detailed tables/figures for publication.

Outputs
-------
- ``outputs/unsupervised_alert_summary.csv``
- ``outputs/unsupervised_alert_scores.csv`` (all months, both phases)
- ``outputs/unsupervised_alert_monitor.csv`` (monitoring-only slice)
- SVG figures per disease and aggregate heatmap under ``figures/``.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import STL

sns.set_theme(style="whitegrid", font="DejaVu Sans")

# --- Font scaling to harmonise sizes across merged STL panels ---
CM_TO_PT = 28.3464567
# Merge layout row1 height from latex_figures/stl_diagnostics_merged.tex
MERGE_ROW1_CM = 4.0
FIG_A_IN = (10.0, 8.0)  # stl_decompose_* base figsize in inches (width, height)

def _read_merge_scale_for_a(tex_path: Path) -> float:
    """Read the LaTeX merged figure to capture panel (a) width multiplier.

    Looks for a definition like:
      \setlength{\awidth}{<scale>\wd\boxaH}
    Returns the parsed <scale> as float (default 1.50 if not found).
    """
    try:
        text = tex_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return 1.50
    import re
    m = re.search(r"\\setlength\{\\awidth\}\{\s*([0-9]*\.?[0-9]+)\\wd\\boxaH\s*\}", text)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return 1.50
    return 1.50

def _target_font_for_embed(embed_pt: float, fig_h_in: float, *, final_pt: float = 12.0) -> float:
    """Return a fontsize (pt) so that after embedding to ``embed_pt`` tall,
    the perceived font is about ``final_pt``.

    final ≈ fontsize * (embed_pt / (72 * fig_h_in)) => fontsize = final * 72 * fig_h_in / embed_pt.
    """
    return float(final_pt * 72.0 * fig_h_in / embed_pt)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "merged_disease_cases_by_month_use.csv"
RANKING_PATH = PROJECT_ROOT / "analysis_ml_ranking" / "outputs" / "ml_ranked_diseases.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
FIG_DIR = PROJECT_ROOT / "frontier_paper" / "figs" / "code_directly_generated_figures"
STL_DIR = FIG_DIR / "stl"

CALIBRATION_MONTHS = 24
PERIOD = 12
DEFAULT_RHO = 0.3  #DEFAULT_RHO = 0.05
MIN_SCORE_DEFAULT = 0.05 #0.2

import sys
sys.path.append(str(PROJECT_ROOT / "analysis_ml_ranking"))
from en_label_utils import translate  # type: ignore  # noqa: E402


@dataclass
class DiseaseAlertResult:
    disease: str
    disease_en: str
    threshold: float
    calibration_mad: float
    alerts_monitor: int
    monitor_months: int
    mean_score_monitor: float
    max_score_monitor: float


def ensure_dirs(fig_dir: Path, stl_dir: Path) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    stl_dir.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, formats: List[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fmt_norm = fmt.lower().lstrip(".")
        out_path = out_dir / f"{stem}.{fmt_norm}"
        kwargs = {}
        if fmt_norm == "png":
            kwargs["dpi"] = 300
        fig.savefig(out_path, facecolor="white", **kwargs)


def load_top_diseases(top_k: int = 15) -> List[str]:
    df = pd.read_csv(RANKING_PATH)
    return df.sort_values("final_rank").head(top_k)["disease"].tolist()


def load_series(disease_names: List[str]) -> Dict[str, pd.Series]:
    df = pd.read_csv(DATA_PATH)
    month_cols = [c for c in df.columns if c != "disease"]
    month_index = pd.to_datetime(month_cols, format="%Y_%m")
    mapping: Dict[str, pd.Series] = {}
    for _, row in df.iterrows():
        disease = row["disease"]
        if disease in disease_names:
            mapping[disease] = pd.Series(row[month_cols].to_numpy(dtype=float), index=month_index)
    return mapping


def stl_scores(
    series: pd.Series,
    rho_list: List[float],
    min_score: float,
    *,
    fig_dir: Path,
    stl_dir: Path,
    formats: List[str],
    export_decomposition: bool,
    export_scores: bool,
) -> tuple[pd.DataFrame, List[DiseaseAlertResult], Dict[float, float]]:
    log_series = np.log1p(series)
    stl = STL(log_series, period=PERIOD, robust=True).fit()

    trend = stl.trend
    seasonal = stl.seasonal
    resid = stl.resid

    calib_boundary = series.index[CALIBRATION_MONTHS - 1]
    calib_mask = series.index <= calib_boundary
    calib_mask = pd.Series(calib_mask, index=series.index)

    resid_calib = resid[calib_mask]

    median_calib = np.median(resid_calib)
    mad_calib = np.median(np.abs(resid_calib - median_calib)) + 1e-6

    score = np.maximum(0.0, resid) / mad_calib
    calibration_scores = score[calib_mask]

    records = []
    summaries: List[DiseaseAlertResult] = []
    thresholds: Dict[float, float] = {}

    for rho in rho_list:
        base_threshold = float(np.quantile(calibration_scores, 1 - rho))
        threshold = float(max(base_threshold, min_score))
        thresholds[rho] = threshold
        alert = score >= threshold

        result_df = pd.DataFrame(
            {
                "disease": series.name,
                "disease_en": translate(series.name),
                "month": series.index,
                "cases": series.values,
                "trend": np.expm1(trend.values),
                "seasonal": np.expm1(seasonal.values),
                "residual": resid.values,
                "score": score,
                "alert": alert.astype(int),
                "phase": np.where(calib_mask, "calibration", "monitor"),
                "threshold": threshold,
                "rho": rho,
            }
        )
        records.append(result_df)

        monitor_df = result_df[result_df["phase"] == "monitor"]
        alerts_monitor = int(monitor_df["alert"].sum())
        monitor_months = monitor_df.shape[0]
        mean_score_monitor = float(monitor_df["score"].mean()) if monitor_months else float("nan")
        max_score_monitor = float(monitor_df["score"].max()) if monitor_months else float("nan")

        summaries.append(
            DiseaseAlertResult(
                disease=series.name,
                disease_en=translate(series.name),
                threshold=threshold,
                calibration_mad=float(mad_calib),
                alerts_monitor=alerts_monitor,
                monitor_months=monitor_months,
                mean_score_monitor=mean_score_monitor,
                max_score_monitor=max_score_monitor,
            )
        )

    combined_df = pd.concat(records, ignore_index=True)

    disease_en = translate(series.name)
    if export_decomposition:
        plot_decomposition(
            series.index,
            series.values,
            trend.values,
            seasonal.values,
            resid.values,
            disease_en,
            out_dir=stl_dir,
            formats=formats,
        )
    if export_scores:
        plot_scores(
            combined_df[combined_df["rho"] == rho_list[0]].copy(),
            disease_en,
            calibration_boundary=calib_boundary,
            out_dir=fig_dir,
            formats=formats,
        )

    return combined_df, summaries, thresholds


def plot_decomposition(
    index: pd.Index,
    observed: np.ndarray,
    trend: np.ndarray,
    seasonal: np.ndarray,
    resid: np.ndarray,
    disease_en: str,
    *,
    out_dir: Path,
    formats: List[str],
) -> None:
    # Match LaTeX merged top-row widening for panel (a): width ≈ 1.10× natural
    # so emit a figure with width scaled by 1.10 to align aspect and avoid
    # perceived stretching when included with both width and height.
    fig_w, fig_h = 10.0 * 1.10, 8.0
    # Choose width so that after LaTeX widens by the merged scale, the net
    # apparent aspect is reasonable for a 4-row decomposition panel.
    # Start from base target (10:8 = 1.25) and slightly reduce when scale > 1
    # to counter perceived vertical squashing in the merged top row.
    merge_tex = PROJECT_ROOT / "latex_figures" / "stl_diagnostics_merged.tex"
    scale_a = _read_merge_scale_for_a(merge_tex)
    # Target a final displayed aspect ratio of 2.0 (width:height) for panel (a)
    # so it will look wider and avoid vertical squashing in the merged row.
    target_final_ra = 2.0
    # Source RA that yields the target after TeX scaling
    src_ra = target_final_ra / max(1e-6, scale_a)
    fig_h = 8.0
    fig_w_eff = src_ra * fig_h
    fig, axes = plt.subplots(4, 1, figsize=(fig_w_eff, fig_h), sharex=True)
    # Compute harmonised font sizes for merged row1 embed height
    embed_pt = MERGE_ROW1_CM * CM_TO_PT
    fs_tick = _target_font_for_embed(embed_pt, FIG_A_IN[1])
    fs_label = fs_tick * 1.0
    fs_title = fs_tick * 1.1
    # Shrink panel (a) fonts by ~3x from the previous (+10%) setting
    # Previous scale was 1.10; new net scale ≈ 1.10 / 3 ≈ 0.3667
    _a_font_scale = 0.3667
    fs_tick *= _a_font_scale
    fs_label *= _a_font_scale
    fs_title *= _a_font_scale

    time = index.strftime("%Y-%m")
    axes[0].plot(time, observed, color="#1f77b4")
    axes[0].set_title(f"{disease_en} — Observed Cases", fontsize=fs_title)
    # Remove y-axis label for a cleaner merged layout
    axes[0].set_ylabel("")
    axes[0].tick_params(axis="both", labelsize=fs_tick)
    axes[0].grid(False)

    axes[1].plot(time, np.expm1(trend), color="#2ca02c")
    axes[1].set_title("Trend", fontsize=fs_title)
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="both", labelsize=fs_tick)
    axes[1].grid(False)

    axes[2].plot(time, np.expm1(seasonal), color="#ff7f0e")
    axes[2].set_title("Seasonal", fontsize=fs_title)
    axes[2].set_ylabel("")
    axes[2].tick_params(axis="both", labelsize=fs_tick)
    axes[2].grid(False)

    axes[3].plot(time, resid, color="#d62728")
    axes[3].set_title("Residual (log-scale)", fontsize=fs_title)
    axes[3].set_ylabel("")
    axes[3].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[3].grid(False)
    # Ensure y tick labels use the same size as other subplots
    axes[3].tick_params(axis="y", labelsize=fs_tick)

    ticks = time.unique()[::3]  # 每3个取一个
    axes[3].set_xticks(ticks)

    # axes[3].set_xlabel("Month")
    axes[3].tick_params(axis="x", rotation=35, labelsize=fs_tick)

    plt.tight_layout()
    save_figure(fig, out_dir, f"stl_decompose_{disease_en.replace(' ', '_')}", formats)
    plt.close(fig)


def plot_scores(
    result_df: pd.DataFrame,
    disease_en: str,
    calibration_boundary: pd.Timestamp,
    *,
    out_dir: Path,
    formats: List[str],
) -> None:
    # Scale all fonts for panel (a) score timelines; shrink prior 1.5x by 1/1.1 ≈ 0.909 → ~1.364x
    base = float(plt.rcParams.get("font.size", 10.0))
    scale = 1.5 / 1.1
    with plt.rc_context({
        "font.size": base * scale,
        "axes.titlesize": base * scale,
        "axes.labelsize": base * scale,
        "xtick.labelsize": base * scale,
        "ytick.labelsize": base * scale,
        "legend.fontsize": base * scale,
    }):
        fig, ax = plt.subplots(figsize=(10, 4))
        df = result_df.copy()
        df["month_str"] = pd.to_datetime(df["month"]).dt.strftime("%Y-%m")
        ax.plot(df["month_str"], df["score"], color="#4c72b0", label="Score")
        ax.axhline(df["threshold"].iloc[0], color="#d62728", linestyle="--", linewidth=1, label="Threshold")
        monitor_df = df[df["phase"] == "monitor"]
        ax.scatter(
            monitor_df[monitor_df["alert"] == 1]["month_str"],
            monitor_df[monitor_df["alert"] == 1]["score"],
            color="#d62728",
            label="Alert",
            zorder=5,
        )
        boundary_str = calibration_boundary.strftime("%Y-%m")
        ax.axvline(boundary_str, color="gray", linestyle=":", linewidth=1)
        # Annotation font follows the same 1.5x scaling (8pt → 12pt if base=10)
        ax.text(
            boundary_str,
            ax.get_ylim()[1],
            "Calibration→Monitoring",
            rotation=90,
            va="top",
            ha="right",
            fontsize=8 * scale,
            color="gray",
        )


    # top_cases = df.sort_values("cases", ascending=False).head(3)
    # ax.scatter(
    #     top_cases["month_str"], top_cases["score"],
    #     color="#8c564b", marker="*", s=120, zorder=6,
    #     label="Top-3 cases"
    # )
    # for _, row in top_cases.iterrows():
    #     ax.scatter(row["month_str"], row["score"], color="#8c564b", marker="*", s=120, zorder=6)
    #     ax.text(row["month_str"], row["score"], f" {int(row['cases'])}", rotation=45, fontsize=8, color="#8c564b")


    # 只在“校准期”的前24个月里选 Top-3 cases
    calib_df = df[df["phase"] == "calibration"].copy()
    if not calib_df.empty:
        top_cases = calib_df.sort_values("cases", ascending=False).head(3)
        ax.scatter(
            top_cases["month_str"], top_cases["score"],
            color="#8c564b", marker="*", s=120, zorder=6,
            label="Top-3 cases (calibration)"
        )
        for _, row in top_cases.iterrows():
            ax.scatter(row["month_str"], row["score"], color="#8c564b", marker="*", s=120, zorder=6)
            ax.text(row["month_str"], row["score"], f" {int(row['cases'])}",
                    rotation=45, fontsize=8, color="#8c564b")

    # Explicitly set title size so it scales with the panel fonts
    ax.set_title(
        f"{disease_en} — Residual Scores and Alerts",
        fontsize=base * scale,
    )
    # Title now set unconditionally with explicit fontsize
    ax.set_ylabel("Score")
    # ax.set_xlabel("Month")
    ax.grid(False)

    # 保持 month_str 用法时：
    ticks = df["month_str"].unique()[::3]  # 每3个取一个
    ax.set_xticks(ticks)

    ax.tick_params(axis="x", rotation=35)
    ax.legend()
    plt.tight_layout()
    save_figure(fig, out_dir, f"score_timeseries_{disease_en.replace(' ', '_')}", formats)
    plt.close(fig)


def plot_heatmap_V1(monitor_df: pd.DataFrame) -> None:
    pivot = monitor_df.pivot_table(index="disease_en", columns="month", values="alert", aggfunc="max", fill_value=0)
    pivot.columns = pd.to_datetime(pivot.columns).strftime("%Y-%m")
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(pivot, cmap="Reds", linewidths=0.5, linecolor="white", cbar_kws={"label": ""}, ax=ax)
    # ax.set_title("Monitoring Alerts (Unsupervised)")
    # ax.set_xlabel("Month")
    ax.set_ylabel("")
    # Rotate month labels clockwise by 90° for readability
    plt.xticks(rotation=-90, ha="center")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "alert_heatmap_monitor.svg", facecolor="white")
    plt.close(fig)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm

def plot_heatmap(monitor_df: pd.DataFrame, *, out_dir: Path, formats: List[str]) -> None:
    pivot = monitor_df.pivot_table(
        index="disease_en",
        columns="month",
        values="alert",
        aggfunc="max",
        fill_value=0
    )

    # 列按时间顺序 & 月份标签美化
    pivot.columns = pd.to_datetime(pivot.columns).strftime("%Y-%m")

    # ❶ 二分类离散色图：0=灰、1=红（可自行换色）
    cmap = ListedColormap(["#eeeeee", "#d62728"])  # [no alert, alert]
    # ❷ 边界：[ -0.5, 0.5, 1.5 ] → 把0映射到第一格、1映射到第二格
    norm = BoundaryNorm([-0.5, 0.5, 1.5], ncolors=cmap.N)

    # Scale all fonts within panel (b) by 1.2x without changing layout/aspect
    base = float(plt.rcParams.get("font.size", 10.0))
    # Prior net scale was 0.96×; enlarge by 1.1× → 1.056× of base
    scale = 1.056
    with plt.rc_context({
        "font.size": base * scale,
        "axes.titlesize": base * scale,
        "axes.labelsize": base * scale,
        "xtick.labelsize": base * scale,
        "ytick.labelsize": base * scale,
        "legend.fontsize": base * scale,
    }):
        fig, ax = plt.subplots(figsize=(10, 5))
        hm = sns.heatmap(
            pivot,
            cmap=cmap,
            norm=norm,
            linewidths=0.5,
            linecolor="white",
            cbar=True,                 # 开启色条
            cbar_kws={"ticks": [0, 1], "label": ""},  # ❸ 指定色条刻度
            ax=ax
        )

        # ❹ 设置色条刻度标签为二分类，并放大色条字体
        cbar = hm.collections[0].colorbar
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["No alert (0)", "Alert (1)"])
        cbar.ax.tick_params(labelsize=base * scale)

        # 轴标签与排版
        ax.set_ylabel("")
        # Rotate month labels clockwise by 90° for readability
        plt.xticks(rotation=-90, ha="center")
        # Ensure axes tick font sizes reflect scaling
        ax.tick_params(axis="both", labelsize=base * scale)
        plt.tight_layout()
        save_figure(fig, out_dir, "alert_heatmap_monitor", formats)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STL-based unsupervised alert pipeline")
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=FIG_DIR,
        help="Directory to write figures to (default: frontier_paper/figs/code_directly_generated_figures).",
    )
    parser.add_argument(
        "--formats",
        default="pdf,png",
        help="Comma-separated output formats (e.g. 'pdf,png' or 'pdf,png,svg').",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Write STL decomposition figures into --fig-dir instead of a 'stl/' subfolder.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Number of top-ranked diseases to score (default: 15).",
    )
    parser.add_argument(
        "--export-decomposition-disease-en",
        action="append",
        help="Repeatable English disease name(s) to export STL decomposition plots for; omit to export all.",
    )
    parser.add_argument(
        "--export-score-disease-en",
        action="append",
        help="Repeatable English disease name(s) to export score timeline plots for; omit to export all.",
    )
    parser.add_argument(
        "--rho",
        type=float,
        action="append",
        help="Target alert rate(s) for calibration quantile; repeat to evaluate multiple rates.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=MIN_SCORE_DEFAULT,
        help="Minimum alert threshold (score units) to avoid near-zero quantiles (default 0.5)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fig_dir = args.fig_dir
    stl_dir = fig_dir if args.flat else fig_dir / "stl"
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]

    ensure_dirs(fig_dir, stl_dir)

    diseases = load_top_diseases(top_k=args.top_k)
    series_map = load_series(diseases)

    rho_list = args.rho if args.rho else [DEFAULT_RHO]

    export_decomp_set = set(args.export_decomposition_disease_en or [])
    export_score_set = set(args.export_score_disease_en or [])

    result_frames = []
    summary_records = []

    for disease, series in series_map.items():
        series.name = disease
        disease_en = translate(disease)
        export_decomposition = not export_decomp_set or disease_en in export_decomp_set
        export_scores = not export_score_set or disease_en in export_score_set
        result_df, summary_list, thresholds = stl_scores(
            series,
            rho_list,
            min_score=args.min_score,
            fig_dir=fig_dir,
            stl_dir=stl_dir,
            formats=formats,
            export_decomposition=export_decomposition,
            export_scores=export_scores,
        )
        result_frames.append(result_df)
        for rho, summary in zip(rho_list, summary_list):
            summary_records.append(
                {
                    "disease": summary.disease,
                    "disease_en": summary.disease_en,
                    "rho": rho,
                    "threshold": thresholds[rho],
                    "calibration_mad": summary.calibration_mad,
                    "alerts_monitor": summary.alerts_monitor,
                    "monitor_months": summary.monitor_months,
                    "mean_score_monitor": summary.mean_score_monitor,
                    "max_score_monitor": summary.max_score_monitor,
                }
            )

    combined = pd.concat(result_frames, ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "unsupervised_alert_scores.csv", index=False)

    monitor_df = combined[combined["phase"] == "monitor"].copy()
    monitor_df.to_csv(OUTPUT_DIR / "unsupervised_alert_monitor.csv", index=False)
    plot_heatmap(monitor_df[monitor_df["rho"] == rho_list[0]], out_dir=fig_dir, formats=formats)

    event_log = monitor_df[monitor_df["alert"] == 1].copy()
    event_log["excess_cases"] = event_log["cases"] - event_log["trend"]
    event_log.to_csv(OUTPUT_DIR / "unsupervised_alert_events.csv", index=False)

    summary_df = pd.DataFrame(summary_records).sort_values(["rho", "alerts_monitor"], ascending=[True, False])
    summary_df.to_csv(OUTPUT_DIR / "unsupervised_alert_summary.csv", index=False)

    coverage_records = []
    for rho in rho_list:
        subset = monitor_df[monitor_df["rho"] == rho]
        coverage_records.append(
            {
                "rho": rho,
                "alerts_total": int(subset["alert"].sum()),
                "months_total": subset.shape[0],
                "alert_rate": float(subset["alert"].mean()),
            }
        )
    pd.DataFrame(coverage_records).to_csv(OUTPUT_DIR / "unsupervised_alert_coverage.csv", index=False)

    print("Unsupervised alerting complete. Summary saved to outputs/unsupervised_alert_summary.csv")


if __name__ == "__main__":
    main()
