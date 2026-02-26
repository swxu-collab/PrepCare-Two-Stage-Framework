# STL-Based Unsupervised Alerts / STL 异常告警流程

## Rationale / 方案动机
We replace supervised alerting with a label-free fallback. STL decomposition (Seasonal-Trend decomposition using Loess) isolates trend and seasonal signals before measuring residual spikes. Diagnostics are provided via `stl_diagnostics.py`, which quantifies seasonal strength, residual variance reduction, and autocorrelation collapse, and visualises the dominant annual frequency peaks. These quantitative checks demonstrate that STL delivers interpretable residuals suitable for alert scoring on our disease series.

## How to Run / 运行方式
1. **STL justification**
   ```bash
   /home/robbie/miniconda3/envs/disease/bin/python risk_alert_unsupervised/stl_diagnostics.py
   ```
   Outputs: `outputs/stl_rationale_metrics.csv`, plus SVG figures under `figures/` (variance-strength bars, autocorr scatter, periodogram).
2. **Unsupervised alert pipeline**
   ```bash
   /home/robbie/miniconda3/envs/disease/bin/python risk_alert_unsupervised/unsupervised_alert_pipeline.py --rho 0.03 --rho 0.05 --rho 0.10
   ```
   The CLI accepts multiple `--rho` values (e.g. 3%, 5%, 10%) and enforces a minimum score floor (`--min-score`, default 0.2) to prevent degenerate thresholds. The run generates per-disease decomposition plots under `figures/stl/`, score timelines (`figures/score_timeseries_*.svg` with calibration/monitor split markers and top-case annotations), a monitoring heatmap for the baseline ρ (`figures/alert_heatmap_monitor.svg`), and CSV tables in `outputs/`.

## Method Overview / 方法概述
1. **Calibration vs Monitoring**：split each disease timeline into a 24-month calibration phase and a 12-month monitoring phase. 阶段划分：前 24 个月校准、后 12 个月监测。
2. **STL decomposition**：apply STL to log-transformed counts (period=12, robust). 在对数尺度上使用 STL 提取趋势和季节项。
3. **One-sided residual scores**：keep only positive residuals and normalise by the MAD of calibration residuals. 对残差取正并以校准期的 MAD 归一化。
4. **Fixed alert rate**：choose the `(1-ρ)` quantile of calibration scores as the threshold, then cap by a minimum score floor (default 0.2 MAD) to avoid vanishing thresholds. 阈值=分位数+最小得分双重约束，使输出稳定可控。
5. **Monitoring alerts**：flag monitoring months whose score exceeds the threshold. 在监测期比较得分与阈值，输出告警矩阵。

## Outputs / 结果说明
- `outputs/stl_rationale_metrics.csv`: seasonal strength, variance ratios, and autocorrelation reduction per disease. 各病种 STL 合理性指标。
- `outputs/unsupervised_alert_summary.csv`: thresholds, calibration MAD, alert counts per disease & ρ. 含多告警率对比。
- `outputs/unsupervised_alert_coverage.csv`: overall alert coverage for each ρ. 各告警率的总体覆盖。
- `outputs/unsupervised_alert_events.csv`: monitoring alerts with trend baseline & excess cases. 告警事件明细及超额病例量。
- `outputs/unsupervised_alert_scores.csv`: full timeline scores with phase labels. 全时段得分。
- `outputs/unsupervised_alert_monitor.csv`: monitoring-only alerts for quick inspection. 监测期告警。
- `figures/stl/*.svg`: decomposition panels per disease. 分解图。
- `figures/score_timeseries_*.svg`: score view with calibration/monitor divider、阈值线与高峰月标注。 得分曲线。
- `figures/alert_heatmap_monitor.svg`: monitoring alert heatmap (white background). 监测期热力图。
- `figures/stl_*.svg`: diagnostics from `stl_diagnostics.py` supporting STL utilisation. STL 合理性诊断图。

All artefacts use white backgrounds and English disease names in figure annotations for manuscript readiness. 全部图表为白底并使用英文病种名称，便于论文引用。
