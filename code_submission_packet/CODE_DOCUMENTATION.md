# Code Submission Packet - Documentation

This directory contains all code referenced in the Final Report for the Tic Episode Prediction project.

## 📁 Directory Structure

```
code_submission_packet/
├── src/                    # Source code modules
├── results (2).csv         # Dataset (1,533 episodes from 89 users)
├── requirements.txt        # Python dependencies
├── FINAL_REPORT.md         # Complete final report
└── CODE_DOCUMENTATION.md   # This file
```

---

## 🔧 Core Pipeline Modules

These modules implement the fundamental prediction framework described in **Section 4 (Experimental Design)**.

### `src/data_loader.py`
**Referenced in:** Section 3.1 (Data Collection), Section 4.1 (Models)

Loads and cleans the raw tic episode dataset.

**Key Functions:**
- `load_and_clean_data(filepath)` - Loads CSV, parses timestamps, handles missing values
- `filter_users_by_min_episodes(df, min_episodes)` - Filters users with insufficient data

**Usage:**
```python
from src.data_loader import load_and_clean_data
df = load_and_clean_data('results (2).csv')
```

---

### `src/feature_engineering.py`
**Referenced in:** Section 3.3 (Feature Engineering), Section 4.1

Creates the 40 engineered features organized into 7 categories.

**Key Classes:**
- `FeatureEngineer` - Main feature engineering pipeline

**Features Created:**
- Temporal (6): hour, day_of_week, is_weekend, day_of_month, month, timeOfDay_encoded
- Sequence (9): prev_intensity_1/2/3, time_since_prev_hours, etc.
- Time-Window (10): window_7d_mean_intensity, window_7d_std_intensity, etc.
- User-Level (5): user_mean_intensity, user_std_intensity, etc.
- Categorical (4): type_encoded, mood_encoded, trigger_encoded
- Engineered (4): intensity_trend, volatility_7d, etc.
- Interaction (6): mood×timeOfDay, trigger×type, etc.

**Usage:**
```python
from src.feature_engineering import FeatureEngineer
engineer = FeatureEngineer()
df = engineer.create_all_features(df, n_lags=3, window_days=[7], fit=True)
```

---

### `src/target_generation.py`
**Referenced in:** Section 3.4 (Target Variable Generation)

Generates prediction targets for both regression and classification tasks.

**Key Classes:**
- `TargetGenerator` - Creates all target variables

**Targets Created:**
- `target_next_intensity` - Next episode intensity (1-10) for regression
- `target_next_high_intensity` - Binary high-intensity indicator (≥7) for classification
- `target_count_next_7d` - Number of episodes in next 7 days
- `target_high_count_next_7d` - Number of high-intensity episodes in next 7 days
- `target_time_to_high_days` - Days until next high-intensity episode

**Usage:**
```python
from src.target_generation import TargetGenerator
target_gen = TargetGenerator(high_intensity_threshold=7)
df = target_gen.create_all_targets(df, k_days_list=[7])
```

---

### `src/models.py`
**Referenced in:** Section 4.1 (Model Architectures)

Implements Random Forest, XGBoost, and LightGBM model wrappers.

**Key Classes:**
- `ModelFactory` - Factory pattern for creating models

**Models:**
- Random Forest: Best for regression (MAE=1.94)
- XGBoost: Best for classification (F1=0.34, PR-AUC=0.70)
- LightGBM: Alternative gradient boosting implementation

**Usage:**
```python
from src.models import ModelFactory
factory = ModelFactory()
model = factory.get_model('random_forest', task_type='regression')
model.fit(X_train, y_train)
```

---

### `src/validation.py`
**Referenced in:** Section 4.4 (Cross-Validation), Section 4.6 (Temporal Validation)

Implements train/test splitting strategies ensuring no data leakage.

**Key Classes:**
- `UserGroupedSplit` - Splits by users (all episodes from user in same set)
- `TemporalSplit` - Splits within each user by time (first N% train, rest test)
- `LeaveOneUserOut` - LOUO cross-validation
- `UserTimeSeriesSplit` - Time-series CV within users

**Usage:**
```python
from src.validation import UserGroupedSplit
splitter = UserGroupedSplit(test_size=0.2, random_state=42)
train_df, test_df = splitter.split(df)
```

---

### `src/evaluation.py`
**Referenced in:** Section 4.3 (Evaluation Metrics)

Computes performance metrics for regression and classification.

**Key Classes:**
- `MetricsCalculator` - Calculates comprehensive metrics

**Metrics:**
- Regression: MAE, RMSE, R², MAPE
- Classification: Precision, Recall, F1, PR-AUC, ROC-AUC

**Usage:**
```python
from src.evaluation import MetricsCalculator
calc = MetricsCalculator()
metrics = calc.regression_metrics(y_true, y_pred)
```

---

### `src/hyperparameter_search.py`
**Referenced in:** Section 4.2 (Hyperparameter Search Strategy)

Orchestrates randomized hyperparameter search with cross-validation.

**Key Classes:**
- `HyperparameterSearch` - Main search coordinator

**Features:**
- RandomizedSearchCV with 20 samples (quick mode)
- User-grouped 3-fold CV
- Tracks all experiments with ExperimentTracker

**Usage:**
```python
from src.hyperparameter_search import HyperparameterSearch
search = HyperparameterSearch(data_path='results (2).csv')
df, features = search.prepare_data()
```

---

### `src/experiment_tracker.py`
**Referenced in:** Section 4.2 (Hyperparameter Search)

Logs all experiments to JSON for reproducibility.

**Key Classes:**
- `ExperimentTracker` - Saves configs, metrics, timestamps

---

## 🔬 Analysis Modules

These modules implement the analyses reported in **Section 5 (Results)** and **Section 6 (Analysis)**.

### `src/threshold_calibration.py` ⭐ **NEW**
**Referenced in:** Section 5.2.1 (User-Grouped Validation - Threshold Optimization)

Implements proper user-grouped train/calibration/test split for threshold selection.

**Key Classes:**
- `ProperThresholdCalibrator` - Calibrates threshold on separate calibration set

**Key Functions:**
- `train_calibrate_test_split()` - Splits data 60%/20%/20% by users
- `run_proper_threshold_calibration()` - Full calibration pipeline

**Results:**
- Calibrated threshold: 0.04 (vs 0.5 default)
- Test F1: 0.72, Precision: 0.59, Recall: 0.92
- Improvement: 111% F1 increase over default

**Usage:**
```python
from src.threshold_calibration import run_proper_threshold_calibration
threshold, cal_metrics, test_metrics, default_metrics, model = run_proper_threshold_calibration()
```

---

### `src/temporal_threshold_calibration.py` ⭐ **NEW**
**Referenced in:** Section 5.2.2 (Temporal Validation - Threshold Optimization)

Implements proper temporal train/calibration/test split for threshold selection using time-based splitting.

**Key Classes:**
- `ProperThresholdCalibrator` - Reused from threshold_calibration.py

**Key Functions:**
- `temporal_train_cal_test_split()` - Splits by dates: Train (May-July 15), Cal (July 16-31), Test (Aug 1-Oct 26)
- `run_temporal_threshold_calibration()` - Full temporal calibration pipeline

**Results:**
- Calibrated threshold: 0.02 (vs 0.5 default)
- Test F1: 0.43, Precision: 0.28, Recall: 0.96
- Improvement: 32.6% F1 increase, 193% recall increase

**Key Insight:**
- Temporal validation requires even lower threshold (0.02) than user-grouped (0.04)
- Despite patient history, model still conservative in probability estimates
- User-grouped with calibrated threshold (F1=0.72) outperforms temporal (F1=0.43) after optimization

**Usage:**
```python
from src.temporal_threshold_calibration import run_temporal_threshold_calibration
threshold, cal_metrics, test_metrics, default_metrics, model, metrics_df = run_temporal_threshold_calibration()
```

---

### `src/kfold_evaluation.py` ⭐ **NEW**
**Referenced in:** Section 5.4 (Robustness Evaluation: 5-Fold Cross-Validation)

5-fold user-grouped cross-validation with each user in test exactly once.

**Key Classes:**
- `UserGroupedKFold` - K-fold splitter ensuring user separation

**Key Functions:**
- `evaluate_regression_kfold()` - 5-fold CV for regression
- `evaluate_classification_kfold()` - 5-fold CV for classification
- `run_kfold_evaluation()` - Full 5-fold evaluation

**Results:**
- Regression: MAE 1.47±0.42, R² 0.18±0.29
- Classification: F1 0.49±0.16, Precision 0.36±0.16, Recall 0.86±0.04, ROC-AUC 0.78±0.10

**Usage:**
```python
from src.kfold_evaluation import run_kfold_evaluation
reg_results, clf_results = run_kfold_evaluation(n_splits=5)
```

---

### `src/feature_selection.py` ⭐ **NEW**
**Referenced in:** Section 6.2 (Formal Feature Selection Analysis)

Formal feature selection using RFE, L1, Mutual Information, and Tree Importance.

**Key Classes:**
- `FeatureSelector` - Implements 4 selection methods

**Methods:**
1. Recursive Feature Elimination (RFE)
2. L1 Regularization (Lasso/Logistic Regression)
3. Mutual Information
4. Tree-based Feature Importance

**Key Functions:**
- `rfe_selection()` - RFE with Random Forest
- `l1_selection()` - Lasso/Logistic with L1 penalty
- `mutual_info_selection()` - Information-theoretic selection
- `tree_importance_selection()` - RF importance-based selection

**Results:**
- Best: Mutual Information → MAE 1.94 with 20 features
- Core features: prev_intensity_1/2/3, window_7d_mean, user_mean

**Usage:**
```python
from src.feature_selection import run_feature_selection_analysis
reg_results, clf_results = run_feature_selection_analysis(n_features_to_select=20)
```

---

### `src/fairness_analysis.py` ⭐ **NEW**
**Referenced in:** Section 6.6 (Fairness and Subgroup Performance Analysis)

Analyzes model performance across user subgroups (engagement, severity, diversity).

**Key Classes:**
- `SubgroupAnalyzer` - Computes user characteristics and creates subgroups

**Subgroups:**
- Engagement: Sparse (1-9 episodes), Medium (10-49), High (50+)
- Severity: Low, Medium, High (by mean intensity percentile)
- Diversity: Single type, Few types, Many types

**Key Functions:**
- `evaluate_subgroup_fairness()` - Performance by subgroup
- `create_fairness_visualizations()` - Bar plots by subgroup
- `run_fairness_analysis()` - Complete fairness evaluation

**Results:**
- Engagement gap: Sparse MAE 3.08 vs Medium 1.71 (80% worse)
- Severity: Low MAE 1.07 (best) vs High 2.28

**Usage:**
```python
from src.fairness_analysis import run_fairness_analysis
reg_fairness, clf_fairness = run_fairness_analysis()
```

---

### `src/temporal_validation.py` ⭐ **UPDATED**
**Referenced in:** Section 4.6 (Temporal Validation Strategy)

Temporal validation with August 1, 2025 cutoff.

**Key Functions:**
- `temporal_split_by_date()` - Splits on specific date (August 2025)
- `compare_validation_strategies()` - Temporal vs user-grouped comparison

**Results:**
- Train: May 29 - July 31, 2025 (566 episodes)
- Test: August 1 - October 26, 2025 (708 episodes)
- Temporal MAE 1.46 vs User-grouped 1.82 (19.8% better)

**Usage:**
```python
from src.temporal_validation import temporal_split_by_date
train_df, test_df = temporal_split_by_date(df, split_date='2025-08-01')
```

---

### `src/threshold_optimization.py`
**Referenced in:** Section 5.2 (Threshold Optimization Analysis)

Optimizes classification threshold across precision-recall tradeoffs.

**Key Classes:**
- `ThresholdOptimizer` - Finds optimal threshold for given metric

**Note:** This is the original threshold optimization. The NEW `threshold_calibration.py` implements proper calibration methodology to avoid leakage.

---

### `src/statistical_tests.py`
**Referenced in:** Section 5.4 (Statistical Significance Testing)

Bootstrap confidence intervals and significance testing.

**Key Functions:**
- Bootstrap MAE improvement tests
- Paired t-tests for model comparison

---

### `src/shap_analysis.py`
**Referenced in:** Section 6.3 (Feature Importance Analysis)

SHAP (SHapley Additive exPlanations) feature importance analysis.

**Key Functions:**
- `compute_shap_values()` - SHAP importance computation
- `plot_shap_summary()` - SHAP summary plots

---

### `src/feature_ablation.py`
**Referenced in:** Section 6.3 (Feature Ablation Study)

Feature ablation experiments testing feature category importance.

**Ablation Tests:**
- Sequence-only
- Time-window-only
- User-level-only
- Temporal-only

---

### `src/per_user_analysis.py`
**Referenced in:** Section 5.4 (Per-User Performance Variability)

Per-user performance stratification and analysis.

**Key Functions:**
- Stratifies users by engagement level
- Computes per-user MAE/F1

---

### `src/error_analysis.py`
**Referenced in:** Section 6.5 (Systematic Error Analysis)

Systematic error analysis by time-of-day, tic type, intensity range.

**Key Functions:**
- Error stratification by categorical variables
- Residual analysis plots

---

### `src/evaluate_future_targets.py`
**Referenced in:** Section 5.3 (Extended Prediction Targets)

Evaluates medium-term forecasting targets (7-day counts, time-to-event).

**Extended Targets:**
- 7-day future episode count
- 7-day high-intensity count
- Time until next high-intensity episode

---

## 📊 Visualization Modules (Optional)

### `src/generate_analysis_plots.py`
Generates comprehensive analysis figures for the report.

### `src/generate_best_model_plots.py`
Generates best model comparison plots.

### `src/additional_visualizations.py`
Additional exploratory visualizations.

---

## 🚀 Running the Code

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline
```bash
python run_hyperparameter_search.py
```

### 3. Run Individual Analyses

**User-Grouped Threshold Calibration:**
```bash
python src/threshold_calibration.py
```

**Temporal Threshold Calibration:**
```bash
python src/temporal_threshold_calibration.py
```

**5-Fold Cross-Validation:**
```bash
python src/kfold_evaluation.py
```

**Feature Selection:**
```bash
python src/feature_selection.py
```

**Fairness Analysis:**
```bash
python src/fairness_analysis.py
```

**Temporal Validation:**
```bash
python src/temporal_validation.py
```

---

## 📋 Requirements

**Python Version:** 3.8+

**Key Dependencies:**
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- xgboost >= 1.5.0
- lightgbm >= 3.3.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- shap >= 0.40.0

Full list in `requirements.txt`

---

## 📈 Expected Outputs

All analysis scripts save results to `report_figures/`:

- `proper_threshold_calibration_results.csv` (user-grouped threshold calibration)
- `temporal_threshold_calibration_results.csv` (temporal threshold calibration)
- `temporal_threshold_scan.csv` (full threshold scan for temporal validation)
- `kfold_regression_results.csv`, `kfold_classification_results.csv`
- `kfold_regression_user_results.csv`, `kfold_classification_user_results.csv`
- `fairness_regression_results.csv`, `fairness_classification_results.csv`
- `feature_selection_regression_summary.csv`, `feature_selection_classification_summary.csv`
- `fairness_analysis.png`
- `fig29_temporal_validation.png`

---

## 🗺️ Code-to-Report Mapping

| Report Section | Primary Code Module |
|---------------|---------------------|
| 3.1 Data Collection | `data_loader.py` |
| 3.3 Feature Engineering | `feature_engineering.py` |
| 3.4 Target Generation | `target_generation.py` |
| 4.1 Model Architectures | `models.py` |
| 4.2 Hyperparameter Search | `hyperparameter_search.py` |
| 4.3 Evaluation Metrics | `evaluation.py` |
| 4.5 Cross-Validation | `validation.py` |
| 4.6 Temporal Validation | `temporal_validation.py` ⭐ |
| 5.1 Regression Results | All core modules |
| **5.2.1 User-Grouped Classification** | **`threshold_calibration.py`** ⭐ |
| **5.2.2 Temporal Classification** | **`temporal_threshold_calibration.py`** ⭐ |
| **5.2.3 Validation Comparison** | Both threshold calibration scripts ⭐ |
| 5.3 Extended Targets | `evaluate_future_targets.py` |
| **5.4 5-Fold CV Results** | **`kfold_evaluation.py`** ⭐ |
| 5.4 Statistical Tests | `statistical_tests.py` |
| 6.1 Model Interpretation | All core modules |
| **6.2 Feature Selection** | **`feature_selection.py`** ⭐ |
| 6.3 Feature Importance | `shap_analysis.py`, `feature_ablation.py` |
| 6.4 Clinical Deployment | All modules |
| 6.5 Error Analysis | `error_analysis.py` |
| **6.6 Fairness Analysis** | **`fairness_analysis.py`** ⭐ |

⭐ = New or significantly updated module

---

## 📝 Notes

- All modules use consistent random seeds (42) for reproducibility
- Data splits use user-grouped stratification to prevent leakage
- NEW modules (marked ⭐) implement methodologies added after presentation feedback
- All code is documented with docstrings following NumPy style
- Type hints included where applicable

---

## 📧 Contact

For questions about this code, refer to the Final Report or contact the project authors.

**Project:** Tic Episode Prediction with Machine Learning
**Dataset:** 1,533 episodes from 89 users (May-October 2025)
**Models:** Random Forest (regression), XGBoost (classification)
**Performance:** MAE 1.94 (27.8% improvement), F1 0.72 (with calibrated threshold)
