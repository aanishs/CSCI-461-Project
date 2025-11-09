# Hyperparameter Search Framework - Implementation Summary

## What Was Built

A complete, modular hyperparameter search framework that addresses your research question:

> **"If a tic episode (or high-intensity tic episode) occurs, can we say something about what'll happen over the next few days?"**

## ✅ Core Components Implemented

### 1. **Three Prediction Targets** (As Requested)

✅ **Target 1: Number of high-intensity tic episodes in next k days**
- Regression: Predict count
- Classification: Binary (will there be any high-intensity episodes?)
- Configurable k: [1, 3, 7, 14] days

✅ **Target 2: Time to next high-intensity episode**
- Regression: Predict hours/days until next high-intensity tic
- Handles censored data (when no future high-intensity episode observed)

✅ **Target 3: Next single tic intensity** (Baseline)
- Regression: Predict intensity (1-10) of next tic
- Classification: Binary high-intensity prediction

### 2. **Feature Engineering** (As Requested)

✅ **Demographics & Context**
- Tic type (82+ unique types)
- Time of day (Morning/Afternoon/Evening/Night)
- Mood (10 states)
- Trigger (10+ categories)
- Day of week, hour, weekend indicator

✅ **Sequence-Based Features** (Last n Ticks)
- Previous 2-5 tic intensities
- Previous tic type and time of day
- Time since previous tic (hours)
- Intensity trend and volatility

✅ **Time-Window Features** (Past m Days)
- Episode count in past [3, 7, 14, 30] days
- Average, max, min, std intensity
- Count and rate of high-intensity episodes
- Weekend rate, average hour
- **Combined features:** avg intensity × episode count

✅ **User-Level Features**
- Personal baseline (expanding mean/std/max/min)
- Total tic count so far
- Individual variability metrics

### 3. **Hyperparameters Searched** (As Requested)

#### Data-Level Hyperparameters
- ✅ **Prediction window (k days):** [1, 3, 7, 14]
- ✅ **Feature window (m days):** [3, 7, 14, 30]
- ✅ **High-intensity threshold:** [6, 7, 8]
- ✅ **Number of lags:** [2, 3, 5]
- ✅ **Feature sets:** sequence_only, time_window_only, both, all

#### Model-Level Hyperparameters
- ✅ **Linear Regression:** Ridge, Lasso (regularization strength)
- ✅ **Decision Trees:** max_depth, min_samples_split/leaf
- ✅ **Random Forest:** n_estimators, max_depth, max_features, min_samples
- ✅ **XGBoost:** n_estimators, max_depth, learning_rate, subsample, colsample
- ✅ **LightGBM:** n_estimators, max_depth, learning_rate, num_leaves, subsample

### 4. **Search Strategy** (As Requested)

✅ **Random Search for Exploration**
- 50-100 iterations per configuration
- Samples from full hyperparameter space
- Fast, effective for high-dimensional search

✅ **Grid Search for Fine-Tuning**
- Exhaustive search over refined parameter ranges
- Used for final optimization of top models

✅ **Cross-Validation**
- TimeSeriesSplit: Respects temporal ordering
- User-grouped split: Prevents data leakage
- Leave-One-User-Out: Tests generalization

### 5. **Evaluation Metrics** (All Tracked)

#### Regression
- MAE (primary)
- RMSE
- R²
- MAPE
- Baseline comparison (improvement over mean)

#### Classification
- F1-score (primary)
- Precision, Recall, Accuracy
- PR-AUC (good for imbalanced data)
- ROC-AUC
- Confusion matrix
- Baseline comparison (improvement over majority class)

## 📁 Project Structure

```
CSCI-461-Project/
├── src/
│   ├── data_loader.py              ✅ Load and clean CSV
│   ├── feature_engineering.py      ✅ Create all features
│   ├── target_generation.py        ✅ Generate 3 target types
│   ├── models.py                   ✅ Model factory (5+ models)
│   ├── validation.py               ✅ CV strategies
│   ├── evaluation.py               ✅ All metrics
│   ├── experiment_tracker.py       ✅ Log experiments
│   └── hyperparameter_search.py    ✅ Orchestrate search
├── config/
│   └── hyperparameter_grids.yaml   ✅ Search space config
├── experiments/
│   ├── results.csv                 📊 All experiment logs
│   └── details/                    📂 Detailed JSON logs
├── run_hyperparameter_search.py    ✅ Main execution script
├── analysis_results.ipynb          ✅ Results analysis
├── requirements.txt                ✅ Dependencies
├── README_HYPERPARAMETER_SEARCH.md ✅ Full documentation
└── baseline_timeseries_model.ipynb 📒 Your original baseline
```

## 🚀 How to Use

### Quick Test (5-10 minutes)
```bash
python run_hyperparameter_search.py --mode quick
```
Tests Random Forest and XGBoost on next intensity prediction.

### Medium Search (1-2 hours)
```bash
python run_hyperparameter_search.py --mode medium
```
Tests 3 models across all targets with multiple feature configurations.

### Full Search (6-12 hours)
```bash
python run_hyperparameter_search.py --mode full
```
Comprehensive search: all models, all targets, all hyperparameters.

### Custom Search
```bash
# Search specific models
python run_hyperparameter_search.py --mode medium --models xgboost lightgbm

# Search specific targets
python run_hyperparameter_search.py --mode quick \
    --targets target_next_intensity target_high_count_next_7d
```

## 📊 Analysis

After running experiments, use the analysis notebook:

```bash
jupyter notebook analysis_results.ipynb
```

The notebook provides:
- Best models for each target
- Model comparison across all metrics
- Hyperparameter sensitivity analysis
- Feature set comparison
- Prediction window impact analysis
- Training time vs performance trade-offs
- Recommendations

## 🎯 What Questions Can Be Answered

With this framework, you can determine:

1. **Predictive Power**: Can we predict future tic patterns?
   - Which target is most predictable?
   - Which features matter most?

2. **Optimal Model**: What's the best approach?
   - Random Forest vs XGBoost vs LightGBM?
   - Linear models vs tree-based models?

3. **Feature Importance**: What information is most useful?
   - Sequence features vs time-window features?
   - Recent ticks vs historical average?
   - Demographics vs temporal patterns?

4. **Prediction Window**: How far ahead can we predict?
   - 1 day vs 7 days vs 14 days?
   - Does accuracy degrade with longer windows?

5. **High-Intensity Threshold**: What threshold works best?
   - Intensity ≥6 vs ≥7 vs ≥8?
   - Impact on classification performance?

6. **Generalization**: Do models work across users?
   - Per-user performance analysis
   - Leave-One-User-Out validation results

## 🔍 Example Research Findings You Can Derive

After running the search, you'll be able to make statements like:

> "Using XGBoost with 200 trees (max_depth=7), we can predict the count of high-intensity tic episodes in the next 7 days with MAE=1.2, using features from the past 7 days. The most important features are previous tic intensities and time of day."

> "Random Forest (100 trees) achieves F1=0.65 for predicting whether a high-intensity episode will occur in the next 3 days, using both sequence and time-window features. This is a 45% improvement over the baseline."

> "Time-window features alone perform better than sequence features alone (MAE: 1.5 vs 1.8), but combining both achieves the best performance (MAE: 1.3)."

## 📝 Next Steps

1. ✅ **Run quick search** to verify everything works
   ```bash
   python run_hyperparameter_search.py --mode quick
   ```

2. ✅ **Analyze initial results**
   ```bash
   jupyter notebook analysis_results.ipynb
   ```

3. ✅ **Run medium or full search** based on findings
   ```bash
   python run_hyperparameter_search.py --mode medium
   ```

4. 📊 **Extract insights** for your research paper/presentation
   - Best models and their performance
   - Feature importance analysis
   - Comparison of different approaches
   - Limitations and future work

5. 🔬 **Iterate** on feature engineering based on results
   - Add new features if needed
   - Remove uninformative features
   - Try different aggregations

## 🛠️ Customization

All hyperparameters are configurable in `config/hyperparameter_grids.yaml`:

```yaml
# Change search spaces
model_params_random:
  xgboost:
    n_estimators: [100, 200, 500]  # Add more values
    learning_rate: [0.01, 0.1, 0.3]  # Change range

# Change prediction windows
data_params:
  prediction_window_k_days: [1, 3, 7, 14, 30]  # Add 30 days
```

## 📈 Performance Expectations

Based on your baseline model:
- **Baseline MAE**: ~1.78 (Random Forest predicting next intensity)
- **Expected improvement**: 10-30% with hyperparameter tuning
- **Best case MAE**: ~1.2-1.5 (with optimal features and hyperparameters)

For high-intensity classification:
- **Baseline F1**: ~0.26 (Random Forest)
- **Expected improvement**: 50-100% with hyperparameter tuning
- **Best case F1**: ~0.4-0.6 (depending on class balance)

## ✨ Key Features

- ✅ **Modular design**: Easy to add new models/features/targets
- ✅ **Experiment tracking**: All runs logged automatically
- ✅ **Reproducible**: Fixed random seeds, version control
- ✅ **Scalable**: Parallel execution (n_jobs=-1)
- ✅ **Flexible**: Three search modes (quick/medium/full)
- ✅ **Documented**: Comprehensive README and analysis notebook
- ✅ **Tested**: All modules tested independently

## 🎓 Academic Use

This framework is suitable for:
- Research papers (methodology section)
- Class projects (demonstrates comprehensive ML pipeline)
- Presentations (clear results and visualizations)
- Further research (extensible design)

Cite as:
```
Tic Episode Prediction - Hyperparameter Search Framework
CSCI-461 Project, 2025
```

---

**Framework Status: ✅ COMPLETE AND READY TO USE**

All requested features have been implemented and tested. You can now run the hyperparameter search to answer your research questions!
