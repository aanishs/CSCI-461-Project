# Tic Episode Prediction - Hyperparameter Search Framework

A comprehensive hyperparameter search framework for predicting tic episode patterns using multiple machine learning approaches.

## Overview

This framework implements a systematic hyperparameter search to answer: **If a tic episode (especially high-intensity) occurs, what can we predict about the next few days?**

### Prediction Targets

1. **Next Single Tic Intensity** (Baseline)
   - Regression: Predict intensity (1-10) of the next tic
   - Classification: Predict if next tic is high-intensity (≥7)

2. **Future Episode Count**
   - Predict count of high-intensity episodes in next k days
   - Binary: Will there be any high-intensity episodes in next k days?

3. **Time to Event**
   - Predict time (hours/days) until next high-intensity episode

### Features

**Temporal Features:**
- Hour, day of week, day of month, month
- Weekend indicator
- Time of day category

**Sequence-Based Features (Last n Tics):**
- Previous 2-5 tic intensities
- Previous tic type and time of day
- Time since previous tic (hours)

**Time-Window Features (Past m Days):**
- Episode count, mean/max/min/std intensity
- High-intensity episode count and rate
- Weekend rate, mean hour

**User-Level Features:**
- Personal baseline (expanding mean/std/max/min intensity)
- Total tic count so far

**Engineered Features:**
- Intensity × count interaction
- Intensity trend (recent change)
- Recent volatility

**Demographics:**
- Tic type, mood, trigger (when available)

### Models

- **Linear Models:** Ridge, Lasso, Logistic Regression
- **Tree Models:** Decision Trees, Random Forest
- **Boosting:** XGBoost, LightGBM

### Hyperparameters Searched

**Data-Level:**
- High-intensity threshold: [6, 7, 8]
- Prediction window (k): [1, 3, 7, 14] days
- Feature window (m): [3, 7, 14, 30] days
- Number of lags: [2, 3, 5]
- Feature sets: sequence only, time-window only, both, all

**Model-Level:**
- Tree depth, number of estimators, learning rate
- Regularization strength
- Subsampling rates, feature sampling

## Project Structure

```
.
├── src/
│   ├── data_loader.py              # Load and clean data
│   ├── feature_engineering.py      # Create features
│   ├── target_generation.py        # Generate prediction targets
│   ├── models.py                   # Model factory
│   ├── validation.py               # Cross-validation strategies
│   ├── evaluation.py               # Metrics calculation
│   ├── experiment_tracker.py       # Track experiments
│   └── hyperparameter_search.py    # Orchestrate search
├── config/
│   └── hyperparameter_grids.yaml   # Search space configuration
├── experiments/
│   ├── results.csv                 # All experiment results
│   └── details/                    # Detailed JSON logs per experiment
├── run_hyperparameter_search.py    # Main execution script
├── requirements.txt
└── README_HYPERPARAMETER_SEARCH.md
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install individually:
pip install pandas numpy scikit-learn xgboost lightgbm pyyaml matplotlib seaborn tqdm jupyter
```

## Usage

### Quick Start (Small Search)

Run a quick test with a small parameter space:

```bash
python run_hyperparameter_search.py --mode quick
```

This runs ~20-40 experiments testing:
- Random Forest and XGBoost
- Next intensity (regression and classification)
- 7-day feature window
- Random search with 20 iterations

### Medium Search

More comprehensive search with multiple feature configurations:

```bash
python run_hyperparameter_search.py --mode medium
```

Runs ~200-300 experiments testing:
- Random Forest, XGBoost, LightGBM
- All 3 prediction targets
- Multiple feature configurations
- Random search with 50 iterations

### Full Search

Comprehensive hyperparameter search:

```bash
python run_hyperparameter_search.py --mode full
```

Runs 1000+ experiments testing:
- All models (Ridge, Decision Tree, RF, XGBoost, LightGBM)
- All prediction targets
- All hyperparameter combinations
- Random search with 100 iterations per config

### Custom Search

Search specific models or targets:

```bash
# Search only XGBoost and LightGBM
python run_hyperparameter_search.py --mode medium --models xgboost lightgbm

# Search only specific targets
python run_hyperparameter_search.py --mode quick \
    --targets target_next_intensity target_high_count_next_7d

# Custom data path and output directory
python run_hyperparameter_search.py \
    --data path/to/data.csv \
    --output custom_experiments \
    --mode medium
```

## Configuration

Edit `config/hyperparameter_grids.yaml` to customize:

### Data Parameters

```yaml
data_params:
  high_intensity_threshold: [6, 7, 8]
  prediction_window_k_days: [1, 3, 7, 14]
  feature_window_m_days:
    - [3, 7]
    - [3, 7, 14]
    - [7, 14, 30]
  n_lags: [2, 3, 5]
```

### Model Parameters

```yaml
model_params_random:
  random_forest:
    n_estimators: [50, 100, 200, 300]
    max_depth: [5, 10, 15, 20, null]
    min_samples_split: [2, 5, 10]
```

## Results Analysis

### View Experiment Summary

```python
from src.experiment_tracker import ExperimentTracker

tracker = ExperimentTracker(experiment_dir='experiments')
tracker.print_summary()
```

### Load All Results

```python
import pandas as pd

results = pd.read_csv('experiments/results.csv')
print(results.head())
```

### Get Best Models

```python
# Best regression models (by Test MAE)
best_regression = results[results['target_type'].str.contains('intensity|count')].nsmallest(10, 'metric_test_mae')

# Best classification models (by Test F1)
best_classification = results[results['target_type'].str.contains('high')].nlargest(10, 'metric_test_f1')
```

### Analyze Hyperparameter Sensitivity

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Plot MAE vs n_estimators
sns.scatterplot(data=results, x='config_n_estimators', y='metric_test_mae', hue='model_name')
plt.show()

# Plot feature importance across experiments
# (Feature importance is logged in detailed JSON files)
```

## Validation Strategy

The framework uses **user-grouped train/test split** by default:
- 80% of users for training
- 20% of users for testing
- Ensures no user appears in both train and test (prevents data leakage)

During hyperparameter search:
- TimeSeriesSplit for cross-validation (within training set)
- Maintains temporal ordering within each user

## Metrics Tracked

### Regression
- **MAE** (Mean Absolute Error) - Primary metric
- **RMSE** (Root Mean Squared Error)
- **R²** (Coefficient of determination)
- **MAPE** (Mean Absolute Percentage Error)
- Baseline comparison (improvement over predicting mean)

### Classification
- **F1-score** - Primary metric
- **Precision, Recall, Accuracy**
- **PR-AUC** (Precision-Recall Area Under Curve)
- **ROC-AUC**
- Confusion matrix (TP, TN, FP, FN)
- Baseline comparison (improvement over majority class)

## Example Workflow

```bash
# 1. Run quick test
python run_hyperparameter_search.py --mode quick

# 2. Check results
python -c "
from src.experiment_tracker import ExperimentTracker
tracker = ExperimentTracker()
tracker.print_summary()
"

# 3. If results look good, run medium search
python run_hyperparameter_search.py --mode medium

# 4. Analyze results in analysis notebook
jupyter notebook analysis_results.ipynb

# 5. Run full search on best approaches
python run_hyperparameter_search.py --mode full \
    --models xgboost lightgbm \
    --targets target_next_intensity target_high_count_next_7d
```

## Testing Individual Modules

```bash
# Test data loading
python src/data_loader.py

# Test feature engineering
python src/feature_engineering.py

# Test target generation
python src/target_generation.py

# Test models
python src/models.py

# Test validation
python src/validation.py

# Test evaluation
python src/evaluation.py

# Test experiment tracker
python src/experiment_tracker.py
```

## Expected Runtime

Approximate runtime on standard laptop:

- **Quick mode**: 5-10 minutes
- **Medium mode**: 1-2 hours
- **Full mode**: 6-12 hours

Runtime depends on:
- Dataset size
- Number of models and hyperparameters
- n_iter for random search
- CPU cores (uses n_jobs=-1 for parallelization)

## Tips for Best Results

1. **Start small**: Run quick mode first to verify everything works
2. **Feature engineering**: Experiment with different feature sets
3. **Target selection**: Focus on targets most relevant to your research question
4. **Model selection**: Boosting models (XGBoost, LightGBM) often perform best
5. **Hyperparameter tuning**: Use random search for exploration, grid search for fine-tuning
6. **Time windows**: Match prediction window (k) to your use case
7. **Validation**: Check per-user performance to identify generalization issues

## Next Steps

After hyperparameter search:

1. **Analyze results** in `analysis_results.ipynb`
2. **Select best model** for each prediction target
3. **Retrain** best models on full dataset
4. **Deploy** for predictions or further analysis
5. **Iterate** on feature engineering based on feature importance
6. **Investigate** failure cases (per-user error analysis)

## Citation

If you use this framework, please cite:

```
Tic Episode Prediction - Hyperparameter Search Framework
CSCI-461 Project
2025
```
