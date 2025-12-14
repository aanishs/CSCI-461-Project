# Tic Episode Prediction - Code Submission

**Authors**: Aanish Sachdev, Aarav Monga, Arjun Bedi, Alan Yusuf
**Course**: CSCI-461
**Project**: Machine Learning for Tourette's Syndrome Tic Episode Prediction

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Important**: This project requires NumPy 1.x:
```bash
pip install 'numpy<2.0' --force-reinstall
```

### 2. Run All Key Analyses

Execute all major analyses from the final report:
```bash
python run_all_analyses.py
```

This script runs the 5 key analyses:
1. Threshold Calibration
2. 5-Fold Cross-Validation
3. Feature Selection
4. Fairness Analysis
5. Temporal Validation

**Expected runtime**: 10-15 minutes
**Results saved to**: `report_figures/`

---

## Repository Structure

```
code_submission_packet/
├── README.md                    # This file
├── FINAL_REPORT.md              # Complete project report
├── run_all_analyses.py          # Execute all key analyses
├── requirements.txt             # Python dependencies
├── results (2).csv              # Dataset (1,533 episodes, 89 users)
├── src/                         # Source code (25 Python modules)
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── models.py
│   ├── evaluation.py
│   ├── threshold_calibration.py
│   ├── kfold_evaluation.py
│   ├── feature_selection.py
│   ├── fairness_analysis.py
│   ├── temporal_validation.py
│   └── ... (16 additional modules)
├── experiments/                 # Hyperparameter search results (quick mode)
├── experiments_medium/          # Hyperparameter search results (medium mode)
└── figures/                     # All figures referenced in report
```

---

## Key Results

### Model Performance
- **Regression**: MAE = 1.94 (27.8% improvement over baseline)
- **Classification**: F1 = 0.44 with calibrated threshold (155% improvement)
- **Temporal Validation**: MAE = 1.46 (24.7% better than user-grouped)

### Dataset
- 1,533 tic episodes from 89 users
- 6-month period (April-October 2025)
- 40 engineered features

### Models Evaluated
- Random Forest (best for regression)
- XGBoost (best for classification)
- LightGBM

---

## Running Individual Analyses

From the root directory:

```bash
# Core analyses
python src/threshold_calibration.py           # Proper threshold calibration
python src/kfold_evaluation.py                # 5-fold cross-validation
python src/feature_selection.py               # Feature selection methods
python src/fairness_analysis.py               # Subgroup performance analysis
python src/temporal_validation.py             # Temporal validation

# Additional analyses
python src/shap_analysis.py                   # SHAP explainability
python src/feature_ablation.py                # Ablation study
python src/error_analysis.py                  # Error stratification
```

---

## Technical Notes

### Environment
- Python 3.8+
- NumPy 1.x (required)
- See `requirements.txt` for complete dependencies

### Data Privacy
- All data anonymized with hashed user IDs
- No personally identifiable information
- Privacy-compliant

### Reproducibility
- All random seeds fixed to 42
- Deterministic train/test splits
- Exact configurations documented in code

