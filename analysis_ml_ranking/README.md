# ML-Based Disease Ranking Pipeline

This module builds an interpretable importance ranking for the monthly disease
panel using three stages:

1. **Feature engineering** – summarise each disease’s time series with burden,
   trend, volatility, recent incidence, severity priors, and a normalised
   monthly profile.
2. **Entropy weighting** – convert the five macro-pillars (total burden, recent
   incidence, risk, trend, policy severity) plus a severity-weighted burden term
   into data-driven weights, avoiding hand-tuned hyperparameters.
3. **Gradient boosting** – train a LightGBM regressor on the entropy-weighted
   consensus target to capture non-linear interactions and derive TreeSHAP
   explanations.

Running the pipeline:

```bash
/home/robbie/miniconda3/envs/disease/bin/python analysis_ml_ranking/ml_ranking_pipeline.py
```

The script writes ranked scores, SHAP contributions, entropy weights, and model
metadata into `analysis_ml_ranking/outputs/` and prints the top-ranked diseases
along with the automatically derived pillar weights.
