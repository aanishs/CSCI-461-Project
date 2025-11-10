# Quick Start Guide - Running All Experiments

This is your **TL;DR guide** to run everything and get results fast.

---

## ⚡ 5-Minute Quick Start

### Step 1: Run All Experiments

```bash
./run_all_experiments.sh
```

Say **yes** when prompted. This will:
- Run 200-300 experiments (~1-2 hours)
- Generate all analysis plots automatically
- Save everything to `figures/` and `experiments/`

### Step 2: Generate Baseline Plots

```bash
jupyter notebook baseline_timeseries_model.ipynb
```

1. Run all cells (Cell → Run All)
2. Add this code to the **end** of the notebook:

```python
import matplotlib.pyplot as plt
import os

os.makedirs('figures/01_baseline_eda', exist_ok=True)

# Re-run your plotting cells and add after each:
plt.savefig('figures/01_baseline_eda/[plot_name].png', dpi=300, bbox_inches='tight')
```

3. Re-run the plotting cells to save figures

### Step 3: Check Results

```bash
ls -la figures/
ls -la experiments/results.csv
```

You should now have:
- `figures/01_baseline_eda/` - 8 EDA plots
- `figures/03_results_analysis/` - 8 analysis plots
- `figures/04_best_models/` - 4+ best model plots
- `experiments/results.csv` - All experiment metrics

---

## 📋 What Gets Run

### Experiments (Phase 2)
- **Models**: Random Forest, XGBoost, LightGBM
- **Targets**: Next intensity, future count, time-to-event
- **Configs**: 4 feature sets × multiple windows
- **Total**: ~200-300 experiments
- **Time**: 1-2 hours

### Plots Generated

**Baseline EDA (Manual - 8 plots):**
1. Intensity distribution
2. Temporal patterns
3. User engagement
4. Tic type distribution
5. Feature importance baseline
6. Prediction vs actual
7. Residuals
8. Per-user performance

**Analysis Plots (Auto - 8 plots):**
1. Model comparison (MAE)
2. Model comparison (F1)
3. Hyperparameter sensitivity (RF)
4. Hyperparameter sensitivity (XGBoost)
5. Feature set comparison
6. Window size impact
7. Threshold comparison
8. Training time vs performance

**Best Model Plots (Auto - 4+ plots):**
1. Feature importance (top 3 models)
2. Confusion matrix
3. ROC/PR curves
4. Prediction scatter

---

## 🎯 Expected Results

After running everything, your `experiments/results.csv` should show:

**Best Model Performance (from medium search):**
- **MAE (regression)**: ~1.6-1.9 (baseline was 1.78)
- **F1 (classification)**: ~0.35-0.50 (baseline was 0.27)
- **Training time**: <0.5 seconds per model

**Winner (likely):** XGBoost or Random Forest

---

## 🔄 If Something Fails

### Script won't run
```bash
chmod +x run_all_experiments.sh
./run_all_experiments.sh
```

### Missing dependencies
```bash
pip install -r requirements.txt
```

### Want to run manually
```bash
# Just run these in order:
python run_hyperparameter_search.py --mode medium
python src/generate_analysis_plots.py
python src/generate_best_model_plots.py
```

---

## 📊 Using Results in Report

Once plots are generated, update `PreliminaryReport_TicPrediction.md`:

### Add to Section 3.6 (Data Visualization)
```markdown
![Intensity Distribution](figures/01_baseline_eda/intensity_distribution.png)
*Figure 1: Distribution of tic intensities (n=1,367)*
```

### Add to Section 4.2 (Hyperparameter Search)
```markdown
![Model Comparison](figures/03_results_analysis/model_comparison_mae.png)
*Figure 2: Test MAE comparison across all models*
```

### Add to Section 4.4 (Best Models)
```markdown
![Confusion Matrix](figures/04_best_models/confusion_matrix_best.png)
*Figure 3: Confusion matrix for best classification model*
```

---

## ⏱️ Time Budget

| Task | Time | Priority |
|------|------|----------|
| Run medium search | 1-2 hrs | **HIGH** |
| Generate auto plots | 1 min | **HIGH** |
| Baseline plots (manual) | 30 min | **HIGH** |
| Update report with figures | 20 min | **HIGH** |
| Review results | 30 min | Medium |
| Full search (optional) | 6-12 hrs | Low |

**Total core work: ~2.5-3 hours**

---

## ✅ Done!

After completing these steps:
1. ✅ All experiments run
2. ✅ All plots generated
3. ✅ Results saved
4. ✅ Ready to update report

**Next:** Open `PreliminaryReport_TicPrediction.md` and add figure references throughout.

For detailed documentation, see `EXPERIMENT_WORKFLOW.md`.
