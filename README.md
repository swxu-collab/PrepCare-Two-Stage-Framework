# Two-Stage Infectious Disease Prioritization and Alerting (Paper Companion Code)

This repository contains the core code used in our manuscript for:

1. **Stage 1 (Ranking):** entropy-informed, interpretable disease prioritization with a LightGBM student model.
2. **Stage 2 (Alerting):** STL-residual-based unsupervised alerts with comparator baselines and sensitivity analyses.

The code is organized into two folders:

- `analysis_ml_ranking/` (Stage 1)
- `risk_alert_unsupervised/` (Stage 2)

## 1) Environment

Recommended Python: **3.10+**

Install core dependencies:

```bash
pip install -U pandas numpy matplotlib lightgbm scipy statsmodels scikit-learn
```

Some scripts use English label utilities and plotting fonts; if figure text does not render as expected, configure local font paths in plotting scripts.

## 2) Input Data

Both stages use a monthly wide-format CSV (one row per disease, one column per month):

- default path expected by scripts: `../merged_disease_cases_by_month_use.csv` (project-root relative)

Key required column:

- `disease` (disease name)

All other columns should be monthly case counts.

## 3) Stage 1: Ranking (`analysis_ml_ranking`)

### Main scripts

- `ml_ranking_pipeline.py`  
  Trains/evaluates the ranking learner, exports ranked diseases, SHAP summaries, and LODO CV outputs.
- `stage1_single_pillar_baselines.py`  
  Computes single-pillar comparators against the entropy-weighted consensus target.
- `plot_figures_en.py`  
  Generates publication-ready Stage-1 figures from outputs.

### Typical run order

```bash
python analysis_ml_ranking/ml_ranking_pipeline.py
python analysis_ml_ranking/stage1_single_pillar_baselines.py
python analysis_ml_ranking/plot_figures_en.py
```

### Main outputs

Saved under `analysis_ml_ranking/outputs/`, including:

- `ml_ranked_diseases.csv`
- `ml_shap_summary.csv`
- `ml_lodo_cv_metrics.csv`
- `ml_lodo_cv_oof_predictions.csv`
- `ml_hyperparam_candidates.csv`
- `ml_hyperparam_search_results.csv`
- `ml_selected_hyperparams.json`
- `stage1_single_pillar_baselines.csv`

## 4) Stage 2: Alerting (`risk_alert_unsupervised`)

### Main scripts

- `unsupervised_alert_pipeline.py`  
  STL decomposition and residual-score alert generation.
- `baseline_comparators.py`  
  Comparator methods (raw threshold, seasonal-naive residual) and trade-off exports.
- `stl_diagnostics.py`  
  STL diagnostics figures/metrics used in manuscript.
- `stage2_min_score_sensitivity.py`  
  Sensitivity analysis for `min_score`.
- `stage2_window_sensitivity.py`  
  Window-length sensitivity (e.g., 24+12 vs 12+12 designs).
- `plot_stage2_tradeoff_residual_top3.py` and  
  `plot_stage2_tradeoff_residual_top3_no_rho_labels.py`  
  Trade-off figure generation.

### Typical run order

```bash
python risk_alert_unsupervised/unsupervised_alert_pipeline.py --rho 0.05 --min-score 0.0
python risk_alert_unsupervised/baseline_comparators.py --rho 0.01 --rho 0.02 --rho 0.05 --rho 0.10 --rho 0.20 --rho 0.30 --min-score 0.0
python risk_alert_unsupervised/stl_diagnostics.py
python risk_alert_unsupervised/stage2_min_score_sensitivity.py --rho 0.05
python risk_alert_unsupervised/stage2_window_sensitivity.py --rho 0.05 --min-score 0.0
python risk_alert_unsupervised/plot_stage2_tradeoff_residual_top3_no_rho_labels.py
```

### Main outputs

Saved under `risk_alert_unsupervised/outputs/`, including:

- `unsupervised_alert_scores.csv`
- `unsupervised_alert_monitor.csv`
- `unsupervised_alert_events.csv`
- `unsupervised_alert_summary.csv`
- `baseline_comparison_summary.csv`
- `baseline_tradeoff_by_rho.csv`
- `baseline_matched_burden_comparison.csv`
- `stage2_ours_min_score_sensitivity.csv`
- `stage2_window_sensitivity_comparison.csv`

## 5) Reproducibility Notes

- Random seeds are fixed where applicable (e.g., Stage-1 model training).
- Hyperparameter candidates are predefined (not an exhaustive Cartesian grid).
- Stage-1 model validation is disease-level **LODO** (leave-one-disease-out).
- Stage-2 trade-off analyses use the same calibration/monitoring design described in the manuscript.

## 6) Scope

This codebase is a **paper companion implementation** for research reproducibility and method transparency.  
For production deployment, additional work is needed for data validation, monitoring, model governance, and operational integration.

