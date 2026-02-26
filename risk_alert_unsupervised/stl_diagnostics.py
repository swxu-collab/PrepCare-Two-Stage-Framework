#!/usr/bin/env python3
"""STL suitability diagnostics for the disease alert pipeline.

The script provides empirical evidence for using STL decomposition by:
1. Quantifying seasonal strength (Hyndman metric) and residual variance
   reduction between the raw log series and STL residuals.
2. Comparing first-lag autocorrelation before/after STL to show residuals
   are closer to white noise.
3. Inspecting median periodograms of log series to highlight dominant
   annual frequency components.

Outputs
-------
- outputs/stl_rationale_metrics.csv
    Table with seasonal strength, variance ratios, and autocorrelation
    reductions per disease.
- figures/stl_variance_strength.svg
    Bar charts for seasonal strength and variance reduction.
- figures/stl_residual_autocorr.svg
    Scatter plot of raw vs residual autocorrelation (lag-1).
- figures/stl_periodogram.svg
    Median periodogram across diseases, annotated at annual frequency.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from statsmodels.tsa.seasonal import STL
from statsmodels.stats.diagnostic import acorr_ljungbox

sns.set_theme(style="whitegrid", font="DejaVu Sans")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "merged_disease_cases_by_month_use.csv"
RANKING_PATH = PROJECT_ROOT / "analysis_ml_ranking" / "outputs" / "ml_ranked_diseases.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
FIG_DIR = PROJECT_ROOT / "frontier_paper" / "figs" / "code_directly_generated_figures"

CALIBRATION_MONTHS = 24
PERIOD = 12
ALT_PERIODS = [6, 12, 18]

import sys
sys.path.append(str(PROJECT_ROOT / "analysis_ml_ranking"))
from en_label_utils import translate  # type: ignore  # noqa: E402


def ensure_dirs(fig_dir: Path) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)


def save_figure(
    fig: plt.Figure,
    out_dir: Path,
    stem: str,
    formats: list[str],
    *,
    facecolor: str = "white",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fmt_norm = fmt.lower().lstrip(".")
        out_path = out_dir / f"{stem}.{fmt_norm}"
        kwargs = {}
        if fmt_norm == "png":
            kwargs["dpi"] = 300
        fig.savefig(out_path, facecolor=facecolor, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STL diagnostics figure generator")
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
        "--label-autocorr",
        action="store_true",
        help="Annotate diseases on the autocorrelation scatter and save as stl_residual_autocorr.*",
    )
    return parser.parse_args()


def load_top_diseases(top_k: int = 15) -> pd.DataFrame:
    ranking = pd.read_csv(RANKING_PATH)
    return ranking.sort_values("final_rank").head(top_k)


def decompose_series(series: pd.Series) -> STL:
    log_series = np.log1p(series)
    stl = STL(log_series, period=PERIOD, robust=True)
    return stl.fit()


def hyndman_seasonal_strength(seasonal: np.ndarray, remainder: np.ndarray) -> float:
    # Avoid divide-by-zero when seasonal+residual variance is ~0.
    denom = np.var(seasonal + remainder, ddof=0)
    if denom == 0:
        return 0.0
    strength = 1.0 - (np.var(remainder, ddof=0) / denom)
    return max(0.0, float(strength))


def periodogram(log_series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Simple FFT-based periodogram.
    n = len(log_series)
    fft_vals = np.fft.rfft(log_series - np.mean(log_series))
    power = np.abs(fft_vals) ** 2
    freq = np.fft.rfftfreq(n, d=1.0)
    return freq, power / n


def main() -> None:
    args = parse_args()
    fig_dir = args.fig_dir
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]

    ensure_dirs(fig_dir)

    diseases_df = load_top_diseases()
    full_data = pd.read_csv(DATA_PATH)
    month_cols = [c for c in full_data.columns if c != "disease"]
    month_index = pd.to_datetime(month_cols, format="%Y_%m")

    metrics = []
    periodograms = []

    seasonal_ci_records = []

    for disease in diseases_df["disease"]:
        row = full_data[full_data["disease"] == disease]
        if row.empty:
            continue
        series = pd.Series(row.iloc[0][month_cols].to_numpy(dtype=float), index=month_index)
        calib_series = series.iloc[:CALIBRATION_MONTHS]
        log_series = np.log1p(series)
        log_calib = log_series.iloc[:CALIBRATION_MONTHS]

        period_strengths = {}
        for per in ALT_PERIODS:
            stl_alt = STL(log_series, period=per, robust=True).fit()
            season_alt = stl_alt.seasonal.iloc[:CALIBRATION_MONTHS]
            resid_alt = stl_alt.resid.iloc[:CALIBRATION_MONTHS]
            strength = hyndman_seasonal_strength(season_alt.values, resid_alt.values)
            period_strengths[f"seasonal_strength_p{per}"] = strength

        result = STL(log_series, period=PERIOD, robust=True).fit()
        resid = result.resid
        trend = result.trend
        seasonal = result.seasonal

        resid_calib = resid.iloc[:CALIBRATION_MONTHS]
        seasonal_calib = seasonal.iloc[:CALIBRATION_MONTHS]

        seasonal_strength = period_strengths["seasonal_strength_p12"]
        var_ratio = float(np.var(resid_calib, ddof=0) / np.var(log_calib, ddof=0)) if np.var(log_calib, ddof=0) > 0 else 1.0

        cases_calib = calib_series.values
        monthly_means = pd.DataFrame(
            {
                "month": calib_series.index.month,
                "cases": cases_calib,
            }
        ).groupby("month")["cases"].mean()
        amp_seasonal = float(monthly_means.max() - monthly_means.min())
        mean_cases = float(monthly_means.mean())
        amp_ratio = float(amp_seasonal / mean_cases) if mean_cases > 0 else 0.0
        amp_ratio_percent = amp_ratio * 100
        autocorr_raw = float(pd.Series(log_calib).autocorr(lag=1))
        autocorr_resid = float(pd.Series(resid_calib).autocorr(lag=1))

        lb_raw = acorr_ljungbox(log_calib, lags=[12], return_df=True)
        lb_resid = acorr_ljungbox(resid_calib, lags=[12], return_df=True)

        metrics.append({
            "disease": disease,
            "disease_en": translate(disease),
            "seasonal_strength": seasonal_strength,
            "seasonal_strength_p6": period_strengths["seasonal_strength_p6"],
            "seasonal_strength_p18": period_strengths["seasonal_strength_p18"],
            "seasonal_amp_pct": amp_ratio_percent,
            "variance_ratio": var_ratio,
            "autocorr_raw_lag1": autocorr_raw,
            "autocorr_resid_lag1": autocorr_resid,
            "ljungbox_raw_p": float(lb_raw["lb_pvalue"].iloc[0]),
            "ljungbox_resid_p": float(lb_resid["lb_pvalue"].iloc[0]),
        })

        freq, power = periodogram(log_series.values)
        periodograms.append({
            "disease": disease,
            "disease_en": translate(disease),
            "frequency": freq,
            "power": power,
        })

        # Seasonal confidence intervals (calibration phase)
        calib_df = pd.DataFrame({
            "month": seasonal_calib.index.month,
            # "seasonal": np.expm1(seasonal_calib.values),
            "seasonal": seasonal_calib.values,
        })
        stats = calib_df.groupby("month")["seasonal"].agg(["mean", "std", "count"]).reset_index()
        stats["lower"] = stats["mean"] - 1.96 * stats["std"] / np.sqrt(stats["count"].clip(lower=1))
        stats["upper"] = stats["mean"] + 1.96 * stats["std"] / np.sqrt(stats["count"].clip(lower=1))
        stats["disease"] = disease
        stats["disease_en"] = translate(disease)
        seasonal_ci_records.append(stats)

    metrics_df = pd.DataFrame(metrics).sort_values("seasonal_strength", ascending=False)
    metrics_df.to_csv(OUTPUT_DIR / "stl_rationale_metrics.csv", index=False)

    if seasonal_ci_records:
        seasonal_ci_df = pd.concat(seasonal_ci_records, ignore_index=True)
        seasonal_ci_df.to_csv(OUTPUT_DIR / "stl_seasonal_ci.csv", index=False)

    # Plot seasonal strength vs variance reduction with 1.3x font scaling for panel (b)
    _base = float(plt.rcParams.get("font.size", 10.0))
    _scale_b = 1.17  # 1.3 × 0.9
    with plt.rc_context({
        "font.size": _base * _scale_b,
        "axes.titlesize": _base * _scale_b,
        "axes.labelsize": _base * _scale_b,
        "xtick.labelsize": _base * _scale_b,
        "ytick.labelsize": _base * _scale_b,
        "legend.fontsize": _base * _scale_b,
    }):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        sns.barplot(
            data=metrics_df,
            y="disease_en",
            x="seasonal_amp_pct",
            ax=axes[0],
            color="#4c72b0",
            edgecolor="black",
        )
        # Drop panel (b) subplot titles to keep merged figure clean
        # axes[0].set_title("Seasonal Amplitude vs. Mean Cases")
        axes[0].set_xlabel("Seasonal Amplitude (% of Mean)")
        # Remove left y-axis label
        axes[0].set_ylabel("")
        axes[0].set_xscale("log")
        axes[0].set_xlim(left=1e0)
        axes[0].grid(False)
        axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))

        sns.barplot(data=metrics_df, y="disease_en", x="variance_ratio", ax=axes[1], color="#dd8452", edgecolor="black")
        # axes[1].set_title("Residual Variance / Raw Variance (Calibration)")
        axes[1].set_xlabel("Variance Ratio")
        axes[1].set_ylabel("")
        plt.tight_layout()
        axes[1].grid(False)
        save_figure(fig, fig_dir, "stl_variance_strength", formats)
        plt.close(fig)

    # Autocorrelation reduction scatter plot (panel c): scale all fonts by 2x
    _base_c = float(plt.rcParams.get("font.size", 10.0))
    # Slightly enlarge panel (c): 1.5× → 1.65× (current × 1.1)
    _scale_c = 1.65
    with plt.rc_context({
        "font.size": _base_c * _scale_c,
        "axes.titlesize": _base_c * _scale_c,
        "axes.labelsize": _base_c * _scale_c,
        "xtick.labelsize": _base_c * _scale_c,
        "ytick.labelsize": _base_c * _scale_c,
        "legend.fontsize": _base_c * _scale_c,
    }):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(
            metrics_df["autocorr_raw_lag1"],
            metrics_df["autocorr_resid_lag1"],
            color="#55a868",
            edgecolor="black",
        )
        ax.plot([-1, 1], [-1, 1], linestyle="--", color="gray")

        # Dynamic, non-overlapping label placement around each point
        def _bbox_overlap(a, b):
            return not (a.x1 <= b.x0 or a.x0 >= b.x1 or a.y1 <= b.y0 or a.y0 >= b.y1)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        placed = []  # list of bboxes in display coords
        # Candidate offset points (in points units) in expanding rings
        base = 6
        rings = [1, 2, 3, 4]
        # Prefer right-side placements in dense regions
        right_first = [
            (1, 0), (1, 1), (1, -1), (0, 1), (0, -1), (-1, 0), (-1, 1), (-1, -1)
        ]
        balanced = [
            (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]
        label_fs = _base_c * _scale_c * 0.5

        xs = metrics_df["autocorr_raw_lag1"].to_numpy()
        ys = metrics_df["autocorr_resid_lag1"].to_numpy()
        # Density heuristic: neighbors within a radius in data coords
        rad = 0.08
        dense_thresh = 3

        for i, row in metrics_df.reset_index(drop=True).iterrows():
            x = row["autocorr_raw_lag1"]; y = row["autocorr_resid_lag1"]
            # Count neighbors within radius
            nbrs = np.sum((np.abs(xs - x) <= rad) & (np.abs(ys - y) <= rad)) - 1
            dirs = right_first if nbrs >= dense_thresh else balanced

            chosen = None
            # Try multiple offset candidates until no overlap with prior labels
            for r in rings:
                step = base * r
                for dx, dy in dirs:
                    trial = ax.annotate(
                        row["disease_en"],
                        (x, y),
                        textcoords="offset points",
                        xytext=(dx * step, dy * step),
                        fontsize=label_fs,
                        ha="left" if dx >= 0 else "right",
                        va="center" if dy == 0 else ("bottom" if dy > 0 else "top"),
                    )
                    fig.canvas.draw()  # update positions for bbox measurement
                    bb = trial.get_window_extent(renderer=renderer)
                    if all(not _bbox_overlap(bb, prev) for prev in placed):
                        chosen = (trial, bb, dx * step, dy * step)
                        break
                    # Overlaps: remove and try next
                    trial.remove()
                if chosen:
                    break
            if chosen:
                trial, bb, xoff, yoff = chosen
                # Special-case nudges for two ambiguous labels: bring them slightly closer vertically to their points
                name = row["disease_en"]
                if name in ("Hepatitis E", "Scarlet Fever"):
                    # Reduce vertical offset magnitude by ~3pt toward 0
                    if yoff > 0:
                        new_yoff = max(0, yoff - 3)
                    elif yoff < 0:
                        new_yoff = min(0, yoff + 3)
                    else:
                        new_yoff = yoff
                    # Replace trial with nudged annotation
                    trial.remove()
                    ann = ax.annotate(
                        name,
                        (x, y),
                        textcoords="offset points",
                        xytext=(xoff, new_yoff),
                        fontsize=label_fs,
                        ha="left" if xoff >= 0 else "right",
                        va="center" if new_yoff == 0 else ("bottom" if new_yoff > 0 else "top"),
                    )
                    fig.canvas.draw()
                    bb = ann.get_window_extent(renderer=renderer)
                else:
                    ann = trial
                placed.append(bb)
            else:
                # Fallback: place with a small right-diagonal offset
                ann = ax.annotate(
                    row["disease_en"],
                    (x, y),
                    textcoords="offset points",
                    xytext=(8, 6) if row["disease_en"] not in ("Hepatitis E", "Scarlet Fever") else (8, 3),
                    fontsize=label_fs,
                    ha="left",
                    va="bottom",
                )
                fig.canvas.draw()
                placed.append(ann.get_window_extent(renderer=renderer))
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xlabel("Lag-1 Autocorr (Log Series)")
        ax.set_ylabel("Lag-1 Autocorr (STL Residual)")
        # Drop subplot title to keep merged figure clean
        plt.tight_layout()
        plt.grid(False)
        # Always emit the file used by the merged TeX (stl_residual_autocorr.*)
        save_figure(fig, fig_dir, "stl_residual_autocorr", formats)
        # Also emit a suffix-"_clear" variant without labels for other uses
        save_figure(fig, fig_dir, "stl_residual_autocorr_clear", formats)
        plt.close(fig)

    # Median periodogram across diseases.
    all_freq = periodograms[0]["frequency"] if periodograms else np.array([])
    power_matrix = np.array([entry["power"] for entry in periodograms]) if periodograms else np.empty((0,))
    median_power = np.median(power_matrix, axis=0) if power_matrix.size else np.array([])

    if median_power.size:
        # Panel (d): increase ONLY text sizes by 3x; keep layout and figure size unchanged
        base = float(plt.rcParams.get("font.size", 10.0))
        scale = 3.0
        with plt.rc_context({
            "font.size": base * scale,
            "axes.titlesize": base * scale,
            "axes.labelsize": base * scale,
            "xtick.labelsize": base * scale,
            "ytick.labelsize": base * scale,
            "legend.fontsize": base * scale,
        }):
            # Reduce panel (d) height to 0.8x of current while keeping width unchanged
            # Further reduce panel (d) height by 0.8x (width unchanged)
            fig, ax = plt.subplots(figsize=(10, 11.52))
            # Double the line width of the red periodogram curve
            lw = float(plt.rcParams.get("lines.linewidth", 1.5)) * 2.0
            ax.plot(all_freq, median_power, color="#c44e52", linewidth=lw)
            # Highlight annual frequency (1/12 months^-1 ≈ 0.0833 cycles per month).
            annual_freq = 1 / 12
            ax.axvline(annual_freq, color="gray", linestyle="--", linewidth=1)
            ax.text(annual_freq + 0.005, max(median_power) * 0.9, "Annual frequency", color="gray")
            ax.set_xlim(0, 0.5)
            ax.set_xlabel("Frequency (cycles per month)")
            ax.set_ylabel("Median Spectral Power")
            # Intentionally omit a panel title to keep the merged figure clean
            plt.tight_layout()
            plt.grid(False)
            save_figure(fig, fig_dir, "stl_periodogram", formats)
            plt.close(fig)

    # Seasonal CI plot (aggregated across diseases) (panel e): double fonts but keep legend size unchanged
    if seasonal_ci_records:
        agg = pd.concat(seasonal_ci_records, ignore_index=True)
        _base_e = float(plt.rcParams.get("font.size", 10.0))
        # Shrink e-panel fonts to 0.8× of current 2.0× setting => 1.6× base
        _scale_e = 1.6
        # Freeze legend size to the pre-scale 'small' equivalent (~0.8 * base)
        _legend_pts = _base_e * 0.8
        with plt.rc_context({
            "font.size": _base_e * _scale_e,
            "axes.titlesize": _base_e * _scale_e,
            "axes.labelsize": _base_e * _scale_e,
            "xtick.labelsize": _base_e * _scale_e,
            "ytick.labelsize": _base_e * _scale_e,
            # Do not scale legend via rc; we will set numeric fontsize explicitly
        }):
            fig, ax = plt.subplots(figsize=(10, 6))
            for disease_en, group in agg.groupby("disease_en"):
                group = group.sort_values("month")
                ax.plot(group["month"], group["mean"], label=disease_en)
                ax.fill_between(group["month"], group["lower"], group["upper"], alpha=0.15)
            ax.set_xticks(range(1, 13))
            ax.set_xlabel("Month")
            ax.set_ylabel("Seasonal Level")
            # Drop subplot title to keep merged figure clean
            # Move legend down so it's vertically centered on the right
            ax.legend(bbox_to_anchor=(1.05, 0.5), loc="center left", fontsize=_legend_pts, borderaxespad=0.)
            plt.tight_layout()
            plt.grid(False)
            save_figure(fig, fig_dir, "stl_seasonal_ci", formats)
            plt.close(fig)

    print("STL diagnostics complete. Metrics written to outputs/stl_rationale_metrics.csv")


if __name__ == "__main__":
    main()
