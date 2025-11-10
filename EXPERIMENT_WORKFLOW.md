# Experiment Workflow Guide

Complete guide for running all experiments and generating publication-quality visualizations.

---

## 📋 Overview

This workflow will:
1. Run comprehensive hyperparameter search (200-300 experiments)
2. Generate analysis plots comparing models and configurations
3. Create deep-dive visualizations for best models
4. Save all figures for inclusion in preliminary report

**Total time:** ~2-3 hours
**Output:** ~28 publication-quality figures at 300 DPI

---

## 🚀 Quick Start

### Option 1: Run Everything Automatically

```bash
./run_all_experiments.sh
```

This script will:
- Prompt you for each phase
- Run medium hyperparameter search (~1-2 hours)
- Generate all analysis plots (~30 seconds)
- Generate best model plots (~30 seconds)
- Provide summary of results

### Option 2: Run Phases Manually

```bash
# Phase 1: Quick test (optional)
python run_hyperparameter_search.py --mode quick

# Phase 2: Medium search (recommended)
python run_hyperparameter_search.py --mode medium

# Phase 3: Generate analysis plots
python src/generate_analysis_plots.py

# Phase 4: Generate best model plots
python src/generate_best_model_plots.py
```

---

## 📁 Output Structure

After running all experiments, you'll have:

```
figures/
├── 01_baseline_eda/          # Generated from baseline notebook
│   ├── intensity_distribution.png
│   ├── temporal_patterns.png
│   ├── user_engagement.png
│   ├── tic_type_distribution.png
│   ├── feature_importance_baseline.png
│   ├── prediction_vs_actual.png
│   ├── residuals.png
│   └── per_user_performance.png
│
├── 03_results_analysis/       # Auto-generated from results.csv
│   ├── model_comparison_mae.png
│   ├── model_comparison_f1.png
│   ├── hyperparameter_sensitivity_rf.png
│   ├── hyperparameter_sensitivity_xgb.png
│   ├── feature_set_comparison.png
│   ├── window_size_impact.png
│   ├── threshold_comparison.png
│   └── training_time_vs_performance.png
│
└── 04_best_models/            # Auto-generated for top models
    ├── feature_importance_*.png (3-6 plots)
    ├── confusion_matrix_best.png
    ├── roc_pr_curves_best.png
    └── prediction_scatter_best.png

experiments/
├── results.csv                # All experiment metrics
└── details/                   # Detailed JSON logs
    └── *.json
```

---

## 📊 Detailed Phase Guide

### Phase 1: Baseline EDA Plots

**Manual step - Run Jupyter notebook:**

```bash
jupyter notebook baseline_timeseries_model.ipynb
```

**Add to end of notebook:**

```python
import matplotlib.pyplot as plt

# Create output directory
import os
os.makedirs('figures/01_baseline_eda', exist_ok=True)

# At each plot in the notebook, add:
plt.savefig('figures/01_baseline_eda/[plot_name].png',
            dpi=300, bbox_inches='tight')
```

**Plots to save (8 total):**
1. `intensity_distribution.png` - Histogram of tic intensities
2. `temporal_patterns.png` - Time-of-day and day-of-week distributions
3. `user_engagement.png` - User engagement levels
4. `tic_type_distribution.png` - Top 15 tic types
5. `feature_importance_baseline.png` - Baseline RF feature importance
6. `prediction_vs_actual.png` - Scatter plot
7. `residuals.png` - Residual analysis
8. `per_user_performance.png` - MAE by user

**Time:** 30 minutes (manual)

---

### Phase 2: Medium Hyperparameter Search

**Command:**
```bash
python run_hyperparameter_search.py --mode medium
```

**What it does:**
- Tests 3 models: Random Forest, XGBoost, LightGBM
- All 3 targets: next intensity, future count, time-to-event
- 4 feature configurations: sequence_only, time_window_only, both, all
- Multiple windows: 3, 7, 14 days
- 50 random search iterations per config

**Expected experiments:** 200-300
**Time:** 1-2 hours
**Output:** Updated `experiments/results.csv`

**Monitor progress:**
```bash
# In another terminal
watch -n 10 "tail -n 5 experiments/results.csv | cut -d',' -f1-4"
```

---

### Phase 3: Generate Analysis Plots

**Command:**
```bash
python src/generate_analysis_plots.py
```

**Generates 8 plots:**

1. **model_comparison_mae.png**
   - Bar chart comparing best MAE for each model
   - Shows which model performs best overall

2. **model_comparison_f1.png**
   - Bar chart comparing best F1 score for each model
   - For classification tasks only

3. **hyperparameter_sensitivity_rf.png**
   - 2 subplots showing n_estimators vs MAE and max_depth vs MAE
   - Identifies optimal Random Forest hyperparameters

4. **hyperparameter_sensitivity_xgb.png**
   - 2 subplots showing learning_rate vs MAE and max_depth vs MAE
   - Identifies optimal XGBoost hyperparameters

5. **feature_set_comparison.png**
   - Box plots showing MAE distribution for each feature set
   - Answers: Do we need all features or just sequences?

6. **window_size_impact.png**
   - Line plot with error bars showing MAE vs window size
   - Answers: Does longer history improve predictions?

7. **threshold_comparison.png**
   - 2 subplots: MAE by threshold (regression) and F1 by threshold (classification)
   - Answers: Should we define high-intensity as ≥6, ≥7, or ≥8?

8. **training_time_vs_performance.png**
   - Scatter plot: training time vs MAE, colored by model
   - Shows performance/efficiency trade-offs

**Time:** ~30 seconds
**Output:** `figures/03_results_analysis/*.png`

---

### Phase 4: Generate Best Model Plots

**Command:**
```bash
python src/generate_best_model_plots.py
```

**Generates plots for top 3 models:**

1. **feature_importance_[model]_[task]_[n].png** (3-6 plots)
   - Top 15 features for each best model
   - Separate plots for regression vs classification

2. **confusion_matrix_best.png**
   - Confusion matrix for best classification model
   - Shows TP, TN, FP, FN breakdown

3. **roc_pr_curves_best.png**
   - 2 subplots: ROC-AUC and PR-AUC scores
   - Evaluates classification discriminative ability

4. **prediction_scatter_best.png**
   - Predicted vs actual intensity scatter plot
   - For best regression model
   - Note: Currently a placeholder (needs predictions saved during training)

**Time:** ~30 seconds
**Output:** `figures/04_best_models/*.png`

---

## 🔬 Advanced Options

### Full Search (Optional)

For maximum performance, run full search overnight:

```bash
python run_hyperparameter_search.py --mode full
```

**What changes:**
- All 5 models: Ridge, Lasso, Decision Tree, RF, XGBoost, LightGBM
- All hyperparameter combinations
- 100 iterations per config
- **Total: 1000+ experiments**
- **Time: 6-12 hours**

### Custom Searches

Search specific models:
```bash
python run_hyperparameter_search.py --mode medium --models xgboost lightgbm
```

Search specific targets:
```bash
python run_hyperparameter_search.py --mode quick \
    --targets target_next_intensity target_high_count_next_7d
```

Custom data path:
```bash
python run_hyperparameter_search.py --data path/to/data.csv \
    --output custom_experiments --mode medium
```

---

## 📈 Using Plots in Report

### Citing Figures in Markdown

Add to `PreliminaryReport_TicPrediction.md`:

```markdown
## Section 3.6: Data Visualization

![Intensity Distribution](figures/01_baseline_eda/intensity_distribution.png)
*Figure 1: Distribution of tic intensities across all events (n=1,367)*

![Temporal Patterns](figures/01_baseline_eda/temporal_patterns.png)
*Figure 2: Time-of-day and day-of-week reporting patterns*

## Section 4.2: Model Comparison

![Model Comparison](figures/03_results_analysis/model_comparison_mae.png)
*Figure 3: Test MAE comparison across all models after hyperparameter tuning*

![Feature Set Comparison](figures/03_results_analysis/feature_set_comparison.png)
*Figure 4: Impact of different feature configurations on prediction accuracy*

## Section 4.4: Best Model Analysis

![Confusion Matrix](figures/04_best_models/confusion_matrix_best.png)
*Figure 5: Confusion matrix for best classification model (XGBoost)*

![Feature Importance](figures/04_best_models/feature_importance_xgboost_classification_1.png)
*Figure 6: Top 15 features by importance in best XGBoost classification model*
```

### Converting to LaTeX (for paper)

```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{figures/03_results_analysis/model_comparison_mae.png}
\caption{Test MAE comparison across all models after hyperparameter tuning. Random Forest achieves the lowest MAE of 1.94.}
\label{fig:model_comparison}
\end{figure}
```

---

## 🐛 Troubleshooting

### Error: "Data file not found"
```bash
# Check file exists
ls -la "results (2).csv"

# Update path in run_hyperparameter_search.py if needed
```

### Error: "Module not found"
```bash
# Install dependencies
pip install -r requirements.txt

# Or install individually
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn pyyaml tqdm
```

### Plots not saving
```bash
# Create directories manually
mkdir -p figures/01_baseline_eda
mkdir -p figures/03_results_analysis
mkdir -p figures/04_best_models
```

### Search taking too long
```bash
# Use quick mode instead
python run_hyperparameter_search.py --mode quick

# Or limit models
python run_hyperparameter_search.py --mode medium --models random_forest
```

---

## 📊 Interpreting Results

### Key Metrics

**Regression (Intensity Prediction):**
- **MAE** (Mean Absolute Error): Lower is better. MAE=1.5 means predictions are off by 1.5 points on average
- **RMSE** (Root Mean Squared Error): Lower is better. Penalizes large errors more than MAE
- **R²**: Higher is better (max 1.0). R²=0.18 means model explains 18% of variance

**Classification (High-Intensity Detection):**
- **F1 Score**: Balance of precision and recall. F1=0.34 is moderate performance
- **Precision**: Of predictions labeled "high", what % are correct? Precision=0.66 means 66% correct
- **Recall**: Of actual high-intensity events, what % did we catch? Recall=0.23 means caught 23%
- **PR-AUC**: Area under precision-recall curve. PR-AUC=0.70 is good for imbalanced data

### What Good Looks Like

| Metric | Baseline | Good | Excellent |
|--------|----------|------|-----------|
| MAE (regression) | 2.68 | <2.0 | <1.5 |
| F1 (classification) | ~0.27 | >0.4 | >0.6 |
| PR-AUC | ~0.67 | >0.75 | >0.85 |

---

## ✅ Final Checklist

Before submitting preliminary report:

- [ ] Run medium hyperparameter search
- [ ] Generate all analysis plots (8 plots)
- [ ] Generate best model plots (4+ plots)
- [ ] Run baseline notebook and save EDA plots (8 plots)
- [ ] Update `PreliminaryReport_TicPrediction.md` with figure references
- [ ] Verify all figures render correctly in markdown preview
- [ ] Check figure quality (should be 300 DPI)
- [ ] (Optional) Run full search for final optimization
- [ ] Document best model performance in report
- [ ] Export report to PDF for submission

---

## 🎯 Expected Outcomes

After completing this workflow, you'll have:

1. **Comprehensive experiment log**: 200-300 experiments in `experiments/results.csv`
2. **~28 publication-quality figures** at 300 DPI
3. **Quantitative answers** to research questions:
   - Which model performs best? (Random Forest? XGBoost? LightGBM?)
   - Which features matter most? (Sequence? Time windows? Both?)
   - How far ahead can we predict? (1 day? 7 days? 14 days?)
   - What threshold works best? (≥6? ≥7? ≥8?)
4. **Ready-to-submit preliminary report** with all figures integrated

---

## 📚 Additional Resources

- **Hyperparameter search README**: `README_HYPERPARAMETER_SEARCH.md`
- **Baseline model README**: `README_baseline_model.md`
- **Implementation summary**: `IMPLEMENTATION_SUMMARY.md`
- **Search results summary**: `SEARCH_RESULTS_SUMMARY.md`
- **Project plan**: `Plan.md`

---

**Questions or issues?** Check the troubleshooting section or review the detailed documentation files listed above.
