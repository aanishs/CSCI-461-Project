# Baseline Time Series Model for Tic Intensity Prediction

## Overview
This notebook implements a baseline machine learning model that predicts the intensity of the next tic event based on a user's previous tic history.

## Quick Start

```bash
jupyter notebook baseline_timeseries_model.ipynb
```

Then run all cells (Cell → Run All)

## Model Performance (Initial Results)

### Regression Model (Predict exact intensity 1-10)
- **Test MAE: 1.778**
- **Test RMSE: 2.194**

### Classification Model (High ≥7 vs Low <7)
- **F1-score: 0.265**

## Dataset Summary
- **Total records**: 1,533 tic events
- **Users**: 89 unique users
- **Usable sequences**: 1,316 (after filtering for users with ≥4 events)
- **Train/test split**: 80/20 by user (prevents data leakage)

## Features Used (17 total)

### Temporal Features
- Hour of day
- Day of week
- Weekend indicator
- Time of day category (Morning/Afternoon/Evening/Night)

### Sequence Features (Most Important!)
- Previous 3 tic intensities (`prev_intensity_1`, `prev_intensity_2`, `prev_intensity_3`)
- Time since previous tic (in hours)
- Previous tic type

### User-Level Features
- Personal baseline intensity (expanding mean)
- Personal intensity variability (expanding std)
- Total tic count for user

### Categorical Features
- Current tic type (82 unique types)
- Previous tic type
- Mood (when available - 54.5% coverage)
- Trigger (when available - 39.4% coverage)

## Model Architecture
- **Random Forest Regressor** (100 trees, max depth 10)
- **Random Forest Classifier** (100 trees, max depth 10)
- User-grouped train/test split to prevent data leakage

## Key Insights from Analysis

### What Works
1. **Previous intensity is the strongest predictor** - Recent tic intensities are highly informative
2. **User-specific features matter** - Personal baselines help account for individual differences
3. **Sequence-based approach** - Using sliding windows (last 3 tics → predict next tic) works better than time-bound predictions

### Data Challenges
1. **Sparse users**: 52.8% of users have <10 events (hard to personalize)
2. **Batch logging**: 80% of consecutive tics occur within 1 hour (users log multiple tics at once)
3. **Retrospective logging**: 35% of events logged on different days
4. **Missing features**: Mood (45.5% missing), Trigger (60.6% missing)

### Why This Approach
Instead of predicting "will a high-intensity event occur in next 12 hours?" (which fails due to batch logging), we predict:

**"Given the last 3 tics, what will be the intensity of the NEXT tic (whenever it occurs)?"**

This is more robust to irregular logging patterns.

## Notebook Structure

1. **Load and Clean Data** - Parse timestamps, handle nulls, sort by user/time
2. **Feature Engineering** - Create 17 temporal, sequence, and user-level features
3. **Create Training Sequences** - Sliding window approach with user-grouped splits
4. **Train Baseline Models** - Random Forest for regression and classification
5. **Evaluate Models** - MAE, RMSE, F1, PR-AUC with visualizations
6. **Feature Importance Analysis** - Identify which features matter most
7. **Per-User Performance** - Analyze which users are easiest/hardest to predict

## Next Steps to Improve Performance

### Model Improvements
1. **Try XGBoost** - Often outperforms Random Forest
2. **LSTM/Transformer** - True sequence models for temporal dependencies
3. **Hyperparameter tuning** - Grid search for optimal parameters
4. **Ensemble methods** - Combine multiple models

### Feature Engineering
5. **Interaction features** - Combine mood × time of day, trigger × tic type
6. **Polynomial features** - Non-linear combinations
7. **Rolling statistics** - 24h/48h/7d aggregates (if logging patterns improve)
8. **Time to next tic** - Predict both intensity and timing

### Personalization
9. **User-specific models** - Train separate models for high-engagement users (>20 events)
10. **Cold-start strategy** - Global model → personalized as data accumulates
11. **User clustering** - Group similar users, train cluster-specific models

### Data Collection
12. **Real-time logging** - Encourage immediate tic logging to reduce batch effects
13. **More consistent features** - Improve mood/trigger completion rates
14. **Additional features** - Sleep quality, medication, stress levels, environmental factors

## Files Generated
- `baseline_timeseries_model.ipynb` - Complete analysis notebook
- `README_baseline_model.md` - This documentation

## Dependencies
```python
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```