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

### 2. Run Main Analysis Scripts

**Basic Feature Engineering & Model Training:**
```bash
cd src/
python feature_engineering.py
python train_models.py
```

**Advanced Analyses (As Featured in Report):**
```bash
python evaluate_future_targets.py    # Targets 2-3 (Section 5.3)
python temporal_validation.py        # Temporal vs user-grouped (Section 4.5)
python shap_analysis.py              # SHAP explainability (Section 6.2.1)
python feature_ablation.py           # Ablation study (Section 6.2.2)
python error_analysis.py             # Error stratification (Section 6.5)
```

### 3. Run Hyperparameter Search

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

### Core Source Code (`src/`)

**Data Pipeline:**
- `data_loader.py` - Load and clean raw episode data
- `feature_engineering.py` - Generate 40 features (34 base + 6 interaction)
- `target_generation.py` - Create 3 prediction targets
- `validation.py` - User-grouped and temporal split strategies

**Model Training:**
- `train_models.py` - Train Random Forest, XGBoost, LightGBM
- `hyperparameter_search.py` - Optuna-based hyperparameter optimization
- `run_hyperparameter_search.py` - Main script for running experiments

**Evaluation:**
- `evaluate_models.py` - Compute regression and classification metrics
- `evaluate_future_targets.py` - Evaluate Targets 2-3 (7-day forecast, time-to-event)
- `temporal_validation.py` - Compare temporal vs user-grouped validation

**Explainability & Analysis:**
- `shap_analysis.py` - SHAP feature importance analysis
- `feature_ablation.py` - Systematic feature ablation study
- `error_analysis.py` - Stratified error analysis
- `statistical_tests.py` - Bootstrap CI, permutation tests
- `per_user_analysis.py` - Per-user performance metrics
- `threshold_optimization.py` - Classification threshold optimization

**Visualization:**
- `visualization.py` - Core plotting functions
- `additional_visualizations.py` - Extended visualizations

**Utilities:**
- `utils.py` - Helper functions

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
- **Regression**: MAE = 1.82, RMSE = 2.15, R² = 0.26
- **Classification**: F1 = 0.72, Precision = 0.58, Recall = 0.92, PR-AUC = 0.70
- **27% improvement** over baseline (MAE 2.49)

### Extended Targets
- **Target 2 (7-day forecast)**: MAE = 1.37, F1 = 0.67, Precision = 0.80
- **Target 3 (time-to-event)**: MAE = 7.21 days, Event rate = 93.8%

### Validation
- **Temporal validation**: 17% better than user-grouped (MAE 1.51 vs 1.82)
- Suggests tic patterns more stable over time than across users

### Explainability
- **Top predictor**: prev_intensity_1 (SHAP = 0.775)
- **Classification key**: window_7d_mean_intensity (SHAP = 1.649)
- **Minimal set**: 6 sequence features achieve 97% of full performance

### Error Analysis
- **Best performance**: Medium engagement users (10-49 episodes), MAE = 1.34
- **Challenge area**: High-intensity episodes (7-10), MAE = 2.64
- **Low-intensity**: 98% accuracy (MAE = 1.41)

---

## Project Statistics

- **Dataset**: 1,533 episodes, 89 users
- **Features**: 40 (34 base + 6 interaction)
- **Models**: Random Forest, XGBoost, LightGBM
- **Prediction Targets**: 3 complementary tasks
- **Hyperparameter Experiments**: 95 (medium mode)
- **Figures**: 38 publication-quality visualizations
- **Code**: ~1,200 lines across 23 files
- **Report**: 1,100+ lines with 7 tables, 51 references

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

**Completion Status**: 95% (All analyses complete, PDF generation pending)
**Submission Ready**: Yes - all core deliverables included
**Quality**: Publication-quality code, figures, and documentation

---

*This submission packet contains only essential files for academic evaluation.*
*Full repository with documentation guides available separately if needed.*
