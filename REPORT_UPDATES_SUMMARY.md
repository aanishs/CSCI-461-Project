# Report Update Summary

**Date**: November 24, 2025
**Status**: All analyses complete, ready to update report

---

## Completed Analyses - Results to Add

### 1. Interaction Features (Section 3.3)
**Features Added**: 6 new interaction features
- mood_x_timeOfDay
- trigger_x_type
- mood_x_prev_intensity
- timeOfDay_x_hour
- type_x_hour
- weekend_x_hour

**Total Features**: 34 → 40 features

---

### 2. Targets 2-3 Evaluation (Section 5.3)

#### Target 2 (Future Count - Next 7 Days):
- **Regression MAE**: 1.37 episodes
- **RMSE**: 1.52
- **R²**: 0.29
- **Classification F1**: 0.67
- **Precision**: 0.80
- **Recall**: 0.57
- **PR-AUC**: 0.77
- **Figures**: fig27_target2_future_count.png

#### Target 3 (Time to Event):
- **MAE**: 7.21 days
- **RMSE**: 8.45
- **R²**: 0.18
- **Event Rate**: 93.8%
- **Censoring Rate**: 6.2%
- **Figures**: fig28_target3_time_to_event.png

---

### 3. Temporal Validation (Section 4.5)

**User-Grouped Split** (baseline):
- Regression MAE: 1.82
- Classification F1: 0.24

**Temporal Split** (70% train, 30% test):
- Regression MAE: 1.51 (17% BETTER!)
- Classification F1: 0.49 (103% BETTER!)

**Key Finding**: Models generalize BETTER temporally than across users!
- Suggests tic patterns are more stable over time than across individuals
- Challenges assumptions about user heterogeneity
- **Figure**: fig29_temporal_validation.png

---

### 4. Medium Mode Search (Section 5.5)

**Status**: COMPLETE - 95 experiments
- Models: Random Forest, XGBoost, LightGBM
- Targets: 1-4 (next intensity, future count, time-to-event)
- Configurations: 6 data splits
- Iterations: 50 per experiment (vs 20 in quick mode)

**Compare to quick mode results** - add comparison table

---

### 5. SHAP Analysis (Section 6.2.2)

**Top 10 Features (Regression)**:
1. prev_intensity_1: 0.775
2. window_7d_mean_intensity: 0.610
3. prev_intensity_2: 0.145
4. time_since_prev_hours: 0.137
5. prev_intensity_3: 0.093
6. type_encoded: 0.086
7. user_mean_intensity: 0.076
8. trigger_encoded: 0.059
9. window_7d_mean_hour: 0.043
10. prev_type_1_encoded: 0.041

**Top 10 Features (Classification)**:
1. window_7d_mean_intensity: 1.649
2. type_encoded: 0.473
3. prev_intensity_1: 0.420
4. time_since_prev_hours: 0.315
5. prev_intensity_2: 0.273
6. user_mean_intensity: 0.238
7. mood_encoded: 0.236
8. window_7d_count: 0.211
9. month: 0.160
10. intensity_trend: 0.159

**Figures**:
- fig30_shap_regression_bar.png
- fig31_shap_regression_beeswarm.png
- fig32_shap_force_low/medium/high.png (3 files)
- fig33_shap_classification_bar.png
- fig34_shap_classification_beeswarm.png

---

### 6. Feature Ablation (Section 6.2.3)

**Results by Configuration**:

| Configuration | Features | Regression MAE | Classification F1 |
|--------------|----------|----------------|-------------------|
| window_only | 9 | 1.767 | 0.141 |
| sequence_only | 6 | 1.771 | 0.525 |
| all_features | 35 | 1.819 | 0.242 |
| no_engineered | 31 | 1.820 | 0.222 |
| user_only | 5 | 1.936 | 0.227 |
| temporal_only | 9 | 2.525 | 0.537 |
| categorical_only | 5 | 2.617 | 0.024 |

**Key Findings**:
- **Best Regression**: window_only (MAE 1.767, 2.8% better than all features)
- **Best Classification**: temporal_only (F1 0.537, 121% better than all features)
- Sequence features alone achieve 97% of full model regression performance with only 6 features
- Engineered features add minimal value (0.04% improvement)
- **Figure**: fig35_feature_ablation.png

---

### 7. Error Analysis (Section 6.4)

**By User Engagement**:
- Sparse (1-9 episodes): MAE 2.99, Accuracy 50%
- Medium (10-49 episodes): MAE 1.34, Accuracy 84%
- High (50+ episodes): MAE 1.94, Accuracy 40%

**By Intensity Range**:
- Low (1-3): MAE 1.41, Accuracy 98%
- Medium (4-6): MAE 0.90, Accuracy 89%
- High (7-10): MAE 2.64, Accuracy 15%

**By Tic Type Frequency**:
- Common (≥20 occurrences): MAE 1.74, Accuracy 64%
- Rare (<20 occurrences): MAE 1.84, Accuracy 55%

**By Time of Day**:
- Morning: MAE 1.66, Accuracy 60%
- Afternoon: MAE 1.69, Accuracy 60%
- Evening: MAE 1.87, Accuracy 65%
- Night: MAE 1.70, Accuracy 56%
- All day: MAE 2.60, Accuracy 37%

**Key Findings**:
- Medium engagement users have best performance (MAE 1.34)
- Models struggle with high-intensity episodes (MAE 2.64)
- Common tic types slightly easier to predict
- **Figure**: fig36_error_analysis.png

---

## Sections to Add to FINAL_REPORT.md

### Section 3.3 - Interaction Features
Add after Section 3.2 (Engineered Features)

### Section 4.5 - Temporal Validation
Add after Section 4.4 (Evaluation Metrics)

### Section 5.3 - Extended Prediction Targets
Add after Section 5.2 (Target 1 Results)

### Section 5.5 - Medium Mode Search Comparison
Add after Section 5.4 (Model Comparison)

### Section 6.2.2 - SHAP Explainability
Add to Section 6.2 (Explainability)

### Section 6.2.3 - Feature Ablation
Add after Section 6.2.2

### Section 6.4 - Systematic Error Analysis
Add after Section 6.3

---

## Reference Placeholders to Fill

Search for: [40], [41], [42], [43], [44], [45] etc.
Replace with proper citations from:
- SHAP: Lundberg & Lee (2017)
- Temporal validation papers
- Feature ablation studies
- Error analysis methodologies

---

## Total Progress

**New Figures Generated**: 12 (fig27-36)
**New Scripts Created**: 4
- evaluate_future_targets.py
- temporal_validation.py
- shap_analysis.py
- feature_ablation.py
- error_analysis.py

**Total Figures Now**: 26 (original) + 12 (new) = 38 figures

**Completion Status**: ~92% → Ready for report integration and PDF generation
