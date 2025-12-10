# Code Submission Packet - Tourette's Tic Prediction Project

**Author**: Aanish Sachdev
**Course**: CSCI-461
**Date**: November 24, 2025
**Project**: Machine Learning for Tourette's Syndrome Tic Episode Prediction

---

## Contents

This submission packet contains all essential code, data, results, and figures for the final project. The directory structure is organized as follows:

```
code_submission_packet/
├── README.md                    # This file
├── FINAL_REPORT.md              # Complete final report (1,100+ lines)
├── requirements.txt             # Python dependencies
├── results_2.csv                # Dataset (1,533 episodes, 89 users)
├── src/                         # Source code (23 Python files)
├── experiments/                 # Quick mode hyperparameter search results
├── experiments_medium/          # Medium mode hyperparameter search results (95 experiments)
└── figures/                     # All 38 publication-quality figures
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Important**: This project requires NumPy 1.x for matplotlib compatibility:
```bash
pip install 'numpy<2.0' --force-reinstall
```

### 2. Run All Key Analyses (⭐ RECOMMENDED)

**Execute all major analyses from the final report:**
```bash
python run_all_analyses.py
```

This script runs all 5 key analyses referenced in the final report:
1. Threshold Calibration (Section 4.4)
2. 5-Fold Cross-Validation (Section 5.4)
3. Feature Selection (Section 6.2)
4. Fairness Analysis (Section 6.6)
5. Temporal Validation (Section 4.6)

Expected runtime: **10-15 minutes**. Results saved to `report_figures/`.

### 3. Run Individual Analysis Scripts

**Core Analyses (Post-Presentation Updates):**
```bash
cd src/
python threshold_calibration.py            # User-grouped train/cal/test split (Section 5.2.1)
python temporal_threshold_calibration.py   # NEW: Temporal train/cal/test split (Section 5.2.2)
python kfold_evaluation.py                 # 5-fold user-grouped CV (Section 5.4)
python feature_selection.py                # RFE, L1, MI, Tree methods (Section 6.2)
python fairness_analysis.py                # Subgroup performance (Section 6.6)
python temporal_validation.py              # August 2025 temporal split (Section 4.6)
```

**Additional Analyses:**
```bash
python evaluate_future_targets.py   # Extended targets (Section 5.3)
python shap_analysis.py             # SHAP explainability (Section 6.3)
python feature_ablation.py          # Ablation study (Section 6.3)
python error_analysis.py            # Error stratification (Section 6.5)
```

### 4. Run Hyperparameter Search

**Quick mode (20 iterations per experiment):**
```bash
python run_hyperparameter_search.py --mode quick
```

**Medium mode (50 iterations - recommended):**
```bash
python run_hyperparameter_search.py --mode medium
```

Results are saved to `experiments/results.csv` or `experiments_medium/results.csv`.

---

## File Descriptions

### Main Scripts

- **`run_all_analyses.py`** ⭐ - Execute all 5 key analyses from final report (10-15 min runtime)
- **`CODE_DOCUMENTATION.md`** ⭐ - Comprehensive code-to-report mapping and usage guide

### Core Source Code (`src/`)

**Data Pipeline:**
- `data_loader.py` - Load and clean raw episode data
- `feature_engineering.py` - Generate 40 features (34 base + 6 interaction)
- `target_generation.py` - Create 3 prediction targets
- `validation.py` - User-grouped and temporal split strategies

**Model Training:**
- `models.py` - Model factory for Random Forest, XGBoost, LightGBM
- `hyperparameter_search.py` - Randomized search with cross-validation
- `experiment_tracker.py` - JSON-based experiment logging

**Evaluation:**
- `evaluation.py` - Regression and classification metrics calculator
- `evaluate_future_targets.py` - Evaluate extended targets (7-day forecast, time-to-event)

**Post-Presentation Analyses (⭐ NEW/UPDATED):**
- **`threshold_calibration.py`** ⭐ - Proper train/cal/test split for threshold selection (Section 4.4)
- **`kfold_evaluation.py`** ⭐ - 5-fold user-grouped cross-validation (Section 5.4)
- **`feature_selection.py`** ⭐ - RFE, L1, Mutual Information, Tree-based selection (Section 6.2)
- **`fairness_analysis.py`** ⭐ - Subgroup performance analysis (Section 6.6)
- **`temporal_validation.py`** ⭐ - August 2025 temporal split validation (Section 4.6)

**Additional Analyses:**
- `shap_analysis.py` - SHAP feature importance analysis
- `feature_ablation.py` - Systematic feature ablation study
- `error_analysis.py` - Stratified error analysis
- `statistical_tests.py` - Bootstrap CI, permutation tests
- `per_user_analysis.py` - Per-user performance metrics
- `threshold_optimization.py` - Classification threshold optimization

**Visualization:**
- `generate_analysis_plots.py` - Generate comprehensive analysis figures
- `generate_best_model_plots.py` - Best model comparison plots
- `additional_visualizations.py` - Extended visualizations

⭐ = New or significantly updated for final report

### Data

**results_2.csv** (340KB)
- 1,533 tic episodes from 89 users
- Features: userId, timestamp, intensity, type, trigger, mood, timeOfDay
- Collected from mobile app for Tourette's syndrome tracking

### Experiment Results

**experiments/results.csv** (52KB)
- Quick mode: 20 iterations per experiment
- 72 experiments total
- Models: Random Forest, XGBoost, LightGBM
- Targets: 1-4

**experiments_medium/results.csv** (67KB)
- Medium mode: 50 iterations per experiment
- 95 experiments total (exceeds planned 72)
- More thorough hyperparameter optimization
- Better convergence and final performance

### Figures (`figures/`)

**38 publication-quality figures** organized by report section:

**Section 3 - Data & Features (fig0-4):**
- fig0: Feature correlation heatmap
- fig1-4: Data distributions and temporal coverage

**Section 5 - Model Performance (fig5-11):**
- fig5-6: Model comparison (MAE, F1)
- fig7-8: Multi-metric dashboards
- fig9: Confusion matrix
- fig10-11: Baseline improvement, training time

**Section 5 - Prediction Targets (fig12-18):**
- fig12: Tic type distribution
- fig13-15: Prediction framework and architectures
- fig16: Feature importance comparison
- fig17-18: Time series predictions and dashboard

**Section 5 - Extended Targets (fig27-29):**
- fig27: Target 2 (7-day forecast) results
- fig28: Target 3 (time-to-event) results
- fig29: Temporal validation comparison

**Section 6 - Advanced Analysis (fig19-26, fig30-36):**
- fig19-20: PR and ROC curves
- fig21: Learning curves
- fig22: Calibration plot
- fig23: Per-user performance
- fig24: Statistical significance
- fig25: Residual plot
- fig26: Threshold analysis
- fig30-34: SHAP analysis (7 figures)
- fig35: Feature ablation results
- fig36: Error analysis stratification

---

## Key Results Summary

### Model Performance (Target 1 - Next Intensity)
- **Regression**: MAE = 1.94 (Random Forest), RMSE = 2.41, R² = 0.11
- **Classification**: F1 = 0.34 (XGBoost, default threshold), PR-AUC = 0.70
- **27.8% improvement** over baseline (MAE 2.69 → 1.94)

### Threshold Calibration (⭐ NEW - Section 4.4)
- **Calibrated threshold**: 0.3367 (vs 0.5 default)
- **Test F1**: 0.44 (vs 0.17 default) - **154.7% improvement**
- **Test Recall**: 0.32 (vs 0.10 default)
- **Methodology**: Proper train/cal/test split (60%/20%/20%) prevents leakage

### 5-Fold Cross-Validation (⭐ NEW - Section 5.4)
- **Regression**: MAE 1.47±0.42, R² 0.18±0.29
- **Classification**: F1 0.49±0.16, Precision 0.36±0.16, Recall 0.86±0.04, ROC-AUC 0.78±0.10
- Each user appears in test set exactly once

### Feature Selection (⭐ NEW - Section 6.2)
- **Best method**: Mutual Information → MAE 1.94 with 20 features (vs 35 total)
- **Core features**: prev_intensity_1/2/3, window_7d_mean, user_mean
- **43% feature reduction** with no performance loss

### Fairness Analysis (⭐ NEW - Section 6.6)
- **Engagement gap**: Sparse users MAE 3.08 vs Medium users 1.71 (80% worse)
- **Severity**: Low MAE 1.07 (best) vs High 2.28
- Performance varies across engagement, severity, and diversity subgroups

### Temporal Validation (⭐ UPDATED - Section 4.6)
- **Train**: May-July 2025 (566 episodes) → **Test**: Aug-Oct 2025 (708 episodes)
- **Temporal MAE**: 1.46 vs User-grouped 1.82 (19.8% better)
- Suggests tic patterns more stable over time than across users

---

## Expected Outputs

Running `run_all_analyses.py` or individual analysis scripts saves results to `report_figures/`:

**CSV Files:**
- `proper_threshold_calibration_results.csv` - Threshold calibration metrics
- `kfold_regression_results.csv`, `kfold_classification_results.csv` - 5-fold CV overall results
- `kfold_regression_user_results.csv`, `kfold_classification_user_results.csv` - Per-user results
- `fairness_regression_results.csv`, `fairness_classification_results.csv` - Subgroup performance
- `feature_selection_regression_summary.csv`, `feature_selection_classification_summary.csv` - Feature selection results

**Figures:**
- `fairness_analysis.png` - Subgroup performance bar charts
- `fig29_temporal_validation.png` - Temporal vs user-grouped comparison

---

## Project Statistics

- **Dataset**: 1,533 episodes, 89 users
- **Features**: 40 (35 engineered + 5 interaction)
- **Models**: Random Forest, XGBoost, LightGBM
- **Prediction Targets**: 5 total (next intensity, high intensity, 7-day counts, time-to-event)
- **Analysis Scripts**: 25+ Python modules
- **Figures**: 38+ publication-quality visualizations
- **Code**: 2,500+ lines across 25+ files
- **Report**: 1,100+ lines with 9 tables, 51 references

---

## Technical Notes

### Environment Requirements
- Python 3.8+
- NumPy 1.x (required for matplotlib compatibility)
- See `requirements.txt` for complete list

### Data Privacy
- All data is anonymized with hashed user IDs
- No personally identifiable information included
- Compliant with privacy guidelines

### Reproducibility
- All random seeds fixed to 42
- Deterministic train/test splits
- Exact hyperparameter configurations documented in code

---

## Report Organization

The `FINAL_REPORT.md` file contains:

1. **Introduction** - Background, motivation, research questions
2. **Related Work** - Comprehensive literature review (51 references)
3. **Methodology** - Data, features, models, validation
4. **Results** - Performance across all 3 targets
5. **Analysis** - SHAP, ablation, error analysis, statistical tests
6. **Discussion** - Clinical implications, limitations, future work
7. **Conclusion** - Key findings and contributions

Total: ~60-80 pages when rendered as PDF with figures

---

## How to Generate PDF

**Install Pandoc:**
```bash
brew install pandoc
brew install --cask basictex  # For xelatex
```

**Generate PDF:**
```bash
cd code_submission_packet/
pandoc FINAL_REPORT.md -o FINAL_REPORT.pdf --pdf-engine=xelatex --toc -N
```

Alternative: Upload `FINAL_REPORT.md` to https://markdown-pdf.com/

---

## Contact

For questions about this project:
- Review the comprehensive `FINAL_REPORT.md`
- Check code comments in `src/` files
- Examine experiment results in `experiments_medium/results.csv`

---

## Submission Checklist

✅ **All presentation feedback addressed:**
1. ✅ Research questions clarified (predicting outcomes vs factors)
2. ✅ Feature engineering distinction (time-window vs user-level)
3. ✅ Hyperparameter selection - no data leakage confirmed
4. ✅ 5-fold cross-validation implemented
5. ✅ Confusion matrix percentages added
6. ✅ AUC < 0.75 clinical threshold discussed
7. ✅ Proper threshold calibration methodology implemented
8. ✅ Temporal split by August 2025 date
9. ✅ Fairness analysis across subgroups
10. ✅ Formal feature selection methods

**Completion Status**: 100% (All analyses complete, all feedback addressed)
**Submission Ready**: Yes - all deliverables included
**Quality**: Publication-quality code, figures, and documentation

**Documentation:**
- ✅ `FINAL_REPORT.md` - Complete report with all updates
- ✅ `CODE_DOCUMENTATION.md` - Comprehensive code-to-report mapping
- ✅ `run_all_analyses.py` - One-command execution of all key analyses
- ✅ `README.md` - This file with quick start guide

---

*This submission packet contains all essential code referenced in the final report.*
*For detailed code documentation and usage examples, see CODE_DOCUMENTATION.md.*
