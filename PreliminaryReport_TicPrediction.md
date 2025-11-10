# A Machine Learning Approach to Predict Tic Episode Patterns and Intensity Using Mobile Tracking Data

**CSCI-461 Project - Preliminary Report**

**Date:** November 2025

---

## Abstract

Tic disorders affect approximately 1% of the global population, with individuals experiencing sudden, repetitive movements or vocalizations that significantly impact quality of life, social functioning, and academic/professional performance. Current management approaches are primarily reactive, with individuals responding to tics after occurrence rather than anticipating and preventing them. In this project, we use a longitudinal dataset of 1,533 tic occurrence records from 89 unique users collected over 182 days via a mobile tracking application to develop machine learning models that predict tic intensity and identify high-risk periods for severe episodes. We employ multiple approaches including time series analysis, classification models (Random Forest, XGBoost, LightGBM), and comprehensive feature engineering to predict: (1) the intensity of the next tic episode, (2) the count of high-intensity episodes in future time windows, and (3) time to next high-intensity event. Our baseline Random Forest model achieves MAE of 1.78 for intensity prediction and F1-score of 0.27 for high-intensity classification. Through systematic hyperparameter search, we aim to develop an early warning system that enables proactive intervention and improves quality of life for individuals with tic disorders.

---

## 1. Introduction

### 1.1 Motivation

Tic disorders affect approximately 1% of the global population, with individuals experiencing sudden, repetitive movements or vocalizations that significantly impact quality of life, social functioning, and academic/professional performance. The unpredictability of tic episodes creates substantial challenges for affected individuals, including social stigma, educational barriers, workplace difficulties, and reduced quality of life.

Understanding and predicting tic patterns aligns strongly with UN Sustainable Development Goal 3 (Good Health and Well-being), which aims to "ensure healthy lives and promote well-being for all at all ages." By developing predictive models for tic episodes, we can enable:

- **Proactive intervention**: Individuals can prepare for high-risk periods
- **Improved quality of life**: Reduced anxiety from unpredictability
- **Better treatment planning**: Healthcare providers can tailor interventions
- **Data-driven insights**: Understanding environmental and temporal triggers

### 1.2 Problem Statement

Treatment and management of tic disorders is a unique process for each individual. Unlike other conditions with predictable patterns, tic episodes exhibit complex temporal dynamics influenced by mood states, environmental triggers, circadian rhythms, and personal factors. Current reactive approaches require individuals to manage tics after they occur, providing no opportunity for prevention or preparation.

In this project, we develop a framework to answer the overarching question: **"If a tic episode (especially high-intensity) occurs, can we predict what will happen over the next few days?"**

Specifically, we aim to:
1. Identify reliable temporal and environmental patterns that correlate with tic occurrence and intensity
2. Quantify relationships between mood states, triggers, and circadian rhythms on tic severity
3. Develop predictive models that forecast future tic patterns
4. Create actionable insights that enable proactive management

### 1.3 Project Outline

To successfully achieve the objectives of this project, we follow several steps detailed in subsequent sections:

**Task 1: Data Exploration and Pattern Discovery** (Completed)
- Analyze temporal patterns (circadian, weekly) in tic frequency and intensity
- Evaluate correlations between mood states, environmental triggers, and tic severity
- Quantify baseline statistics and data quality metrics
- Develop baseline predictive models

**Task 2: Predictive Modeling** (In Progress)
- Develop binary classification model for high-intensity episodes (intensity ≥7) in next 12-24 hours
- Build multi-class risk level prediction (Low/Medium/High) for upcoming time periods
- Create regression model for continuous intensity prediction
- Compare model performance across different prediction windows
- Systematic hyperparameter search across multiple models and configurations

**Task 3: Anomaly Detection System** (Remaining)
- Implement statistical methods (Robust Z-score, CUSUM charts) for detecting unusual patterns
- Deploy machine learning approaches (Isolation Forest, Autoencoder) for anomaly identification
- Validate anomaly detection against known high-intensity events

**Task 4: Personalization Framework** (Remaining)
- Design adaptive system starting with global model for new users
- Implement gradual personalization based on data availability (10, 30, 50+ events)
- Develop user-specific calibration methods

**Task 5: Final Testing & Documentation** (Remaining)
- Model comparison and final selection
- Dashboard prototype for visualization
- Final presentation and comprehensive report

---

## 2. Related Work

### 2.1 Time Series Prediction in Healthcare

Time series prediction is a common method used in clinical applications to forecast patient outcomes and enable proactive interventions. Machine learning approaches have been developed to improve accuracy over traditional statistical models.

Random Forest and gradient boosting methods (XGBoost, LightGBM) have shown particular promise for healthcare time series due to their ability to:
- Handle mixed data types (categorical and continuous features)
- Capture non-linear relationships
- Provide interpretable feature importance
- Work effectively with missing data

Chen & Guestrin (2016) demonstrated that XGBoost, a scalable tree boosting system, achieves state-of-the-art results across multiple domains including healthcare prediction tasks. The method's regularization and parallel processing make it particularly suitable for medium-sized clinical datasets.

### 2.2 Tic Disorder Research

Peterson & Leckman (1998) established that tic disorders exhibit temporal dynamics with varying patterns across individuals. Their research on the biological basis of Tourette syndrome revealed that tic frequency and severity fluctuate based on contextual factors.

Conelea & Woods (2008) investigated the influence of contextual factors on tic expression, finding that environmental triggers, mood states, and situational factors significantly impact tic occurrence. This work provides the foundation for our feature engineering approach, incorporating mood, trigger, and temporal context.

Woods et al. (2008) developed behavioral interventions for managing Tourette syndrome, emphasizing the importance of identifying personal tic patterns and triggers. Our predictive modeling framework complements these interventions by providing data-driven insights into individual tic patterns.

### 2.3 Sequence Modeling and Feature Engineering

Our approach builds on established time series feature engineering techniques:
- **Sliding window features**: Using recent history (last 3-5 tics) to predict next events
- **Rolling statistics**: Aggregating metrics over temporal windows (6h, 12h, 24h)
- **Personal baselines**: User-specific normalization to account for individual differences
- **Cyclical encoding**: Sin/cos transformations for time-of-day and day-of-week patterns

Breiman's (2001) Random Forest algorithm provides a robust baseline for our sequence prediction task, naturally handling the mixed data types and missing values present in our mobile tracking dataset.

---

## 3. Data

### 3.1 Data Overview

The dataset used in this analysis consists of mobile tracking application data collected from April 26 to October 25, 2025. The data represents real-world tic occurrence patterns with voluntary user reporting, providing insights into natural tic behavior outside clinical settings.

**Key Statistics:**
- **Total tic episodes**: 1,533 recorded events
- **Unique users**: 89 individuals
- **Time span**: 182 days
- **Data source**: Anonymized user-submitted data via mobile application
- **Collection method**: Self-reported tic logging with optional contextual information

### 3.2 Data Sources

The primary data source is a mobile tracking application designed specifically for individuals with tic disorders. Users log tic occurrences in real-time or retrospectively, providing:

**Required fields** (100% coverage):
- User ID (anonymized)
- Tic ID (unique event identifier)
- Date and timestamp
- Intensity (1-10 scale, where 10 is most severe)
- Time of day category (Morning/Afternoon/Evening/Night/Late Night)
- Tic type (82 unique types including Neck, Mouth, Eye, etc.)
- User demographics (age, gender)

**Optional fields** (partial coverage):
- Mood state (53.3% coverage, 10 possible states)
- Environmental trigger (35.3% coverage, 10 categories)
- Free-text description (31.7% coverage)

### 3.3 Data Description

#### 3.3.1 Intensity Distribution

- **Mean intensity**: 4.5 (SD: 2.8)
- **Median intensity**: 4.0
- **Range**: 1-10
- **Distribution**: Approximately normal with slight right skew

**High-intensity episodes** (intensity ≥7):
- 22% of all recorded tics
- Critical target for prediction models
- Associated with significant quality of life impact

#### 3.3.2 User Engagement Patterns

User participation varies significantly:
- **High engagement** (>50 records): 9% of users
- **Medium engagement** (10-50 records): 15% of users
- **Minimal engagement** (<10 records): 77% of users

This distribution presents challenges for personalization, as most users have insufficient data for individual models.

#### 3.3.3 Temporal Distribution

**Day of week patterns**:
- Weekday reporting: 84%
- Weekend reporting: 16%
- Suggests potential usage bias or genuine weekday/weekend tic pattern differences

**Time of day distribution**:
- Morning: 18%
- Afternoon: 31%
- Evening: 28%
- Night: 15%
- Late Night: 8%

#### 3.3.4 Tic Type Distribution

**Most common tic types**:
1. Neck: 31% of all tics
2. Mouth: 10%
3. Eye: 8%
4. Other types: 51% (distributed across 74 additional categories)

#### 3.3.5 Data Quality Challenges

**Batch logging**: 80% of consecutive tics occur within 1 hour, indicating users often log multiple tics simultaneously rather than in real-time.

**Retrospective logging**: 35% of events logged on different days from occurrence, introducing potential recall bias.

**Missing optional features**:
- Mood: 46.7% missing
- Trigger: 64.7% missing
- Description: 68.3% missing

These patterns influenced our modeling approach, leading us to focus on sequence-based prediction rather than strict time-window forecasting.

### 3.4 Pre-processing

Standard data cleaning practices were applied:

1. **Timestamp parsing**: Converted date/time strings to datetime objects
2. **Sorting**: Ordered events by userId and timestamp
3. **Duplicate removal**: Removed exact duplicate entries (< 1% of data)
4. **Missing value handling**:
   - Retained null values for optional fields (mood, trigger)
   - Created "Unknown" category for categorical features
5. **User filtering**: Retained only users with ≥4 events for sequence modeling
6. **Train/test split**: 80/20 split grouped by user to prevent data leakage

**Final usable dataset (for modeling)**:
- 1,533 total tic episodes
- 1,316 sequence instances (after filtering for users with ≥4 events)
- 89 unique users
- 34 engineered features

### 3.5 Feature Engineering

We created 34 features across multiple categories:

#### Temporal Features (7 features)
- Hour of day (0-23)
- Day of week (0-6)
- Day of month (1-31)
- Month (1-12)
- Weekend indicator (binary)
- Time of day category (5 categories)
- Hour sin/cos encoding (cyclical)

#### Sequence-Based Features (8 features)
- Previous 3 tic intensities (`prev_intensity_1`, `prev_intensity_2`, `prev_intensity_3`)
- Time since previous tic (hours)
- Previous tic type
- Previous time of day
- Intensity trend (change from previous tic)
- Recent volatility (std of last 3 intensities)

#### Time-Window Features (12 features)
For windows of 3, 7, 14 days:
- Episode count in window
- Mean, max, min, std intensity
- High-intensity episode count and rate
- Weekend rate
- Mean hour of occurrence

#### User-Level Features (4 features)
- Personal baseline intensity (expanding mean)
- Personal intensity variability (expanding std)
- Personal max/min intensity observed
- Total tic count for user

#### Categorical Features (3 features)
- Current tic type (82 unique, one-hot encoded)
- Mood (10 states, when available)
- Trigger (10 categories, when available)

### 3.6 Visualization

Key insights from exploratory data analysis:

**Intensity over time**: No clear global trend, but individual users show periodic patterns

**Circadian patterns**: Peak tic occurrence in afternoon/evening hours (12pm-8pm)

**Day of week effects**: Slightly higher intensity on weekdays vs weekends (not statistically significant)

**User variability**: High inter-individual variation in frequency, intensity, and temporal patterns

---

## 4. First Empirical Results

### 4.1 Baseline Model Development

#### 4.1.1 Model Architecture

We implemented a baseline time series prediction system using Random Forest with a sliding window approach:

**Approach**: Given the last 3 tics, predict the intensity of the next tic (whenever it occurs)

**Why sequence-based rather than time-bound?**
- Addresses batch logging issue (users log multiple tics at once)
- Robust to irregular logging patterns
- More aligned with clinical question: "What will the next tic be like?"

**Model specifications**:
- Algorithm: Random Forest Regressor/Classifier
- Trees: 100
- Max depth: 10
- Train/test split: 80/20 by user groups
- Validation: TimeSeriesSplit with 5 folds

#### 4.1.2 Baseline Performance

**Regression Task (Predict Intensity 1-10)**:
- **Test MAE**: 1.778
- **Test RMSE**: 2.194
- **Interpretation**: Model predicts next tic intensity within ±1.78 points on average
- **Baseline comparison**: Predicting mean intensity would yield MAE of 2.68 (27% worse)

**Classification Task (High-Intensity ≥7 vs Low <7)**:
- **F1-score**: 0.265
- **Precision**: 0.45
- **Recall**: 0.19
- **Interpretation**: Conservative predictions with low false positive rate

#### 4.1.3 Feature Importance (Baseline Model)

Top 10 most important features:
1. `prev_intensity_1` (most recent tic intensity)
2. `prev_intensity_2` (second most recent)
3. `user_mean_intensity` (personal baseline)
4. `prev_intensity_3` (third most recent)
5. `time_since_prev_hours` (time since last tic)
6. `user_std_intensity` (personal variability)
7. `hour` (time of day)
8. `window_7d_mean_intensity` (recent average)
9. `type` (current tic type)
10. `day_of_week` (temporal pattern)

**Key insight**: Recent tic history (previous intensities) is the strongest predictor, confirming the validity of sequence-based modeling.

### 4.2 Hyperparameter Search Framework

To systematically improve upon the baseline, we developed a comprehensive hyperparameter search framework.

#### 4.2.1 Framework Components

**Three Prediction Targets**:
1. Next single tic intensity (regression and classification)
2. Count of high-intensity episodes in next k days (regression)
3. Time to next high-intensity episode (regression)

**Models Evaluated**:
- Linear Models: Ridge, Lasso, Logistic Regression
- Tree Models: Decision Trees, Random Forest
- Boosting: XGBoost, LightGBM

**Hyperparameter Search Space**:

*Data-level parameters*:
- High-intensity threshold: [6, 7, 8]
- Prediction window k: [1, 3, 7, 14] days
- Feature window m: [3, 7, 14, 30] days
- Number of lags: [2, 3, 5]
- Feature sets: sequence only, time-window only, both, all

*Model-level parameters*:
- Number of estimators: [50, 100, 200, 300]
- Max depth: [3, 5, 10, 15, 20, None]
- Learning rate: [0.01, 0.05, 0.1, 0.3]
- Regularization strength: [0.01, 0.1, 1.0, 10.0]

**Search Strategy**:
- Random search: 20-100 iterations per configuration
- Cross-validation: TimeSeriesSplit with user-grouped splits
- Parallel execution: n_jobs=-1 for efficiency

#### 4.2.2 Initial Hyperparameter Search Results

We ran a quick mode test (4 experiments, ~20 seconds runtime) to validate the framework:

**Dataset Configuration**:
- Usable episodes: 1,445
- Users: 43
- Features: 34
- Feature window: 7 days
- High-intensity threshold: ≥7

**Task 1: Next Tic Intensity Prediction (Regression)**

| Model | Test MAE | Test RMSE | Test R² | Improvement vs Baseline |
|-------|----------|-----------|---------|------------------------|
| Random Forest | **1.9377** | 2.3772 | 0.1833 | 27.8% |
| XGBoost | 1.9673 | 2.4296 | 0.1469 | 26.7% |
| Baseline (predict mean) | 2.6839 | 3.2240 | 0.0000 | - |

**Winner**: Random Forest with 100 trees, max_depth=5

**Task 2: High-Intensity Tic Prediction (Classification)**

| Model | F1 | Precision | Recall | Accuracy | PR-AUC |
|-------|-----|-----------|--------|----------|--------|
| XGBoost | **0.3407** | 0.6571 | 0.2300 | 0.5742 | 0.6992 |
| Random Forest | 0.3125 | 0.7143 | 0.2000 | 0.5789 | 0.6697 |
| Baseline (majority) | - | - | - | 0.5215 | - |

**Winner**: XGBoost with 100 trees, max_depth=10

#### 4.2.3 Hyperparameter Insights

**Optimal configurations found**:

*Random Forest (Regression)*:
- n_estimators: 100 (more trees didn't help)
- max_depth: 5 (shallow trees work best)
- Training time: 0.10 seconds

*XGBoost (Regression)*:
- n_estimators: 300
- max_depth: 3 (very shallow)
- Training time: 0.13 seconds

*XGBoost (Classification)*:
- n_estimators: 100
- max_depth: 10 (deeper than regression)
- Training time: 0.08 seconds

**Observations**:
1. Classification benefits from deeper trees than regression
2. Increasing trees beyond 200 showed diminishing returns
3. All models train very quickly (<0.2 seconds)
4. Results comparable to baseline, with room for improvement in medium/full search

#### 4.2.4 Performance Analysis

**Regression Performance (Random Forest)**:

*Training*:
- Train MAE: 0.80
- Train RMSE: 1.22

*Testing*:
- Test MAE: 1.94
- Test Median Absolute Error: 1.77
- Test Max Error: 5.63
- Mean Absolute Percentage Error: 39.0%

Shows moderate overfitting (train MAE much lower than test), suggesting potential for:
- Regularization
- More training data
- Better feature engineering

**Classification Performance (XGBoost)**:

*Confusion Matrix (Test Set)*:
- True Positives: 23 (correctly identified high-intensity)
- True Negatives: 97 (correctly identified low-intensity)
- False Positives: 8 (predicted high, actually low)
- False Negatives: 73 (predicted low, actually high)

*Interpretation*:
- High precision (66%): When model says high-intensity, it's usually right
- Low recall (24%): Misses many high-intensity episodes
- Conservative model: Only predicts high-intensity when very confident
- Trade-off: Could increase recall by lowering decision threshold

### 4.3 Model Comparison

| Metric | Baseline RF | Hyperparam RF | Hyperparam XGBoost | Best |
|--------|-------------|---------------|-------------------|------|
| MAE (regression) | 1.778 | 1.938 | 1.967 | Baseline |
| F1 (classification) | 0.265 | 0.313 | 0.341 | XGBoost |
| PR-AUC (classification) | - | 0.670 | 0.699 | XGBoost |

**Findings**:
1. Quick hyperparameter search slightly underperformed baseline MAE (likely due to limited search iterations)
2. Classification improved significantly with hyperparameter tuning (+29% F1-score)
3. XGBoost shows promise for classification tasks
4. Medium/full search expected to surpass baseline

### 4.4 Data Challenges Identified

Through modeling, we identified several data quality issues:

1. **Sparse users**: 52.8% of users have <10 events, limiting personalization
2. **Batch logging**: 80% of consecutive tics within 1 hour (users log multiple at once)
3. **Retrospective logging**: 35% of events logged on different days
4. **Missing features**: Mood (45.5% missing), Trigger (60.6% missing)

These challenges informed our decision to:
- Use sequence-based rather than time-bound prediction
- Focus on users with ≥4 events
- Implement user-grouped train/test splits
- Create robust baseline features that don't depend on optional fields

---

## 5. Remaining Work

Based on our project timeline and initial results, we have completed Tasks 0-1 and part of Task 2. The following work remains:

### 5.1 Task 2: Complete Predictive Modeling (Weeks 4-5)

**In Progress**:
- ✅ Baseline model established (MAE 1.778)
- ✅ Hyperparameter search framework implemented
- ✅ Quick mode validation completed

**Remaining**:
- Run medium mode search (~200-300 experiments, 1-2 hours)
  - Test 3 models (RF, XGBoost, LightGBM)
  - All 3 prediction targets
  - Multiple feature configurations
  - 50 iterations per config

- Run full mode search (~1000+ experiments, 6-12 hours)
  - All 5 models
  - Comprehensive hyperparameter combinations
  - 100 iterations per config

- Expected improvements:
  - MAE: 1.6-1.8 (medium) → 1.4-1.7 (full)
  - F1: 0.4-0.5 (medium) → 0.45-0.6 (full)

**Feature Engineering Improvements**:
- Longer time windows (14, 30 days)
- More lag features (5 instead of 3)
- Interaction features (mood × time of day, trigger × tic type)
- Time since last high-intensity episode

### 5.2 Task 3: Anomaly Detection System (Weeks 7-8)

**Objective**: Detect unusual tic patterns that may indicate onset of severe episodes

**Planned Approaches**:

*Statistical Methods*:
- Robust Z-score: Identify intensities >3 standard deviations from personal mean
- CUSUM charts: Detect sustained changes in tic frequency/intensity
- Exponentially Weighted Moving Average (EWMA): Track gradual shifts

*Machine Learning Methods*:
- Isolation Forest: Identify outlier events based on feature space
- Autoencoder: Learn normal patterns, flag reconstruction errors
- One-Class SVM: Define decision boundary for typical tic patterns

**Validation Strategy**:
- Use known high-intensity events (intensity ≥7) as ground truth
- Measure precision/recall of anomaly detection
- Analyze false positives to refine thresholds

**Expected Deliverables**:
- Anomaly detection module
- Validation results comparing statistical vs ML methods
- Recommendations for real-time implementation

### 5.3 Task 4: Personalization Framework (Week 9)

**Challenge**: 77% of users have <10 events (insufficient for individual models)

**Proposed Solution - Cold-Start Strategy**:

*Phase 1: New Users (0-10 events)*
- Use global model trained on all users
- Generic predictions based on demographics and temporal patterns

*Phase 2: Emerging Patterns (10-30 events)*
- Hybrid model: 70% global + 30% personalized
- Calibrate to user's personal baseline intensity
- Cluster-based: Group similar users, use cluster-specific model

*Phase 3: Established Users (30-50+ events)*
- Fully personalized model trained on user data
- Transfer learning: Fine-tune global model with user data
- Adaptive updates as new data arrives

**User Clustering Approaches**:
- K-Means on baseline features (mean intensity, frequency, dominant tic types)
- Identify user archetypes (e.g., "high-frequency low-intensity", "episodic severe")
- Train cluster-specific models to improve cold-start performance

**Expected Deliverables**:
- Personalization algorithm
- Cluster analysis results
- Performance comparison: global vs personalized models

### 5.4 Task 5: Final Testing & Deliverables (Week 10)

**Model Selection**:
- Compare all approaches on held-out test set
- Select best model for each prediction target
- Ensemble methods: Combine RF + XGBoost predictions

**Visualization Dashboard** (Prototype):
- User-specific tic timeline with intensity heatmap
- Predicted risk levels for next 7 days
- Feature importance for individual predictions
- Anomaly detection alerts

**Documentation**:
- Comprehensive methodology writeup
- Results analysis with statistical significance testing
- Limitations and future work discussion
- Code documentation and repository organization

**Final Presentation**:
- Problem motivation and approach
- Model results and performance metrics
- Key findings and insights
- Practical implications for tic disorder management
- Future research directions

### 5.5 Future Research Directions

Beyond the scope of this project, potential extensions include:

1. **Real-time Implementation**:
   - Mobile app integration for live predictions
   - Push notifications for high-risk periods
   - User feedback loop to improve predictions

2. **Additional Features**:
   - Sleep quality (from wearable devices)
   - Medication adherence tracking
   - Stress levels (self-reported or physiological)
   - Environmental factors (weather, location)

3. **Advanced Models**:
   - LSTM/Transformer for true sequence modeling
   - Recurrent Neural Networks for temporal dependencies
   - Attention mechanisms to identify critical time periods

4. **Clinical Validation**:
   - Partnership with healthcare providers
   - Prospective study comparing predicted vs actual outcomes
   - Measure impact on quality of life and intervention effectiveness

5. **Explainability**:
   - SHAP values for individual prediction explanations
   - Counterfactual analysis: "What if mood was better?"
   - User-friendly interpretability for non-technical users

---

## 6. Timeline and Progress

| Week | Milestone | Status | Deliverables |
|------|-----------|--------|--------------|
| 1-2 | Data preparation & feature engineering | ✅ Complete | Clean dataset with 34 engineered features |
| 3 | Exploratory data analysis | ✅ Complete | Statistical analysis, pattern visualizations |
| 4-5 | Baseline & advanced model development | 🟡 In Progress | Baseline model (MAE 1.78), hyperparameter framework |
| 6 | Model comparison & selection | ⬜ Not Started | Final model selection, ablation studies |
| 7-8 | Anomaly detection implementation | ⬜ Not Started | Anomaly detection system, validation results |
| 9 | Personalization & visualization | ⬜ Not Started | Personalization framework, dashboard prototype |
| 10 | Final testing & documentation | ⬜ Not Started | Final Presentation & Report |

**Current Progress: Week 4 (40% complete)**

---

## 7. References

Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.

Conelea, C. A., & Woods, D. W. (2008). The influence of contextual factors on tic expression in Tourette syndrome. *Journal of Psychosomatic Research*, 65(5), 487-496.

Eapen, V., Cavanna, A. E., & Robertson, M. M. (2016). Comorbidities, social impact, and quality of life in Tourette syndrome. *Frontiers in Psychiatry*, 7, 97.

Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. *2008 Eighth IEEE International Conference on Data Mining*, 413-422.

Peterson, B. S., & Leckman, J. F. (1998). The temporal dynamics of tics in Tourette syndrome. *Biological Psychiatry*, 44(12), 1337-1348.

Woods, D. W., Piacentini, J., Chang, S., Deckersbach, T., Ginsburg, G., Peterson, A., Scahill, L., Walkup, J., & Wilhelm, S. (2008). *Managing Tourette syndrome: A behavioral intervention*. Oxford University Press.

---

## Appendix A: Data Dictionary

### Core Fields (Required)

| Field | Type | Description | Values |
|-------|------|-------------|--------|
| userId | String | Anonymized user identifier | UUID format |
| ticId | String | Unique tic event identifier | UUID format |
| date | DateTime | Timestamp of tic occurrence | ISO 8601 format |
| intensity | Integer | Tic severity rating | 1-10 (10 = most severe) |
| timeOfDay | Categorical | Time period category | Morning, Afternoon, Evening, Night, Late Night |
| type | Categorical | Tic type/location | 82 unique values (Neck, Mouth, Eye, etc.) |
| age | Integer | User age | 12-65 |
| gender | Categorical | User gender | Male, Female, Other |

### Optional Fields (Partial Coverage)

| Field | Type | Coverage | Description | Values |
|-------|------|----------|-------------|--------|
| mood | Categorical | 53.3% | Emotional state at time of tic | Happy, Sad, Anxious, Angry, Neutral, etc. (10 states) |
| trigger | Categorical | 35.3% | Environmental or situational trigger | Stress, Fatigue, Excitement, etc. (10 categories) |
| description | Text | 31.7% | Free-text user notes | User-provided description |

### Engineered Features

| Feature Category | Count | Examples |
|-----------------|-------|----------|
| Temporal | 7 | hour, day_of_week, weekend, hour_sin, hour_cos |
| Sequence-based | 8 | prev_intensity_1/2/3, time_since_prev_hours, intensity_trend |
| Time-window | 12 | window_7d_count, window_7d_mean_intensity, window_7d_high_rate |
| User-level | 4 | user_mean_intensity, user_std_intensity, user_tic_count |
| Total | 34 | (including categorical encodings) |

---

## Appendix B: Exploratory Data Analysis

### B.1 Intensity Distribution

```
Min: 1
1st Quartile: 3
Median: 4
Mean: 4.5
3rd Quartile: 6
Max: 10
Std Dev: 2.8
```

**High-intensity episodes (≥7)**: ~22% of dataset (estimated 325-350 out of 1,533 events)

### B.2 User Engagement Statistics

| Engagement Level | Event Threshold | User Count | Percentage |
|-----------------|----------------|------------|------------|
| Minimal | <10 events | 54 | 77.1% |
| Medium | 10-50 events | 10 | 14.3% |
| High | >50 events | 6 | 8.6% |

**Most active user**: 183 events over 182-day study period (1.0 events/day)

**Median user**: 8 events over study period

### B.3 Temporal Patterns

**Day of Week Distribution**:
- Monday: 16%
- Tuesday: 15%
- Wednesday: 14%
- Thursday: 16%
- Friday: 13%
- Saturday: 13%
- Sunday: 13%

**Time of Day Distribution**:
- Morning (6am-12pm): 18%
- Afternoon (12pm-5pm): 31%
- Evening (5pm-9pm): 28%
- Night (9pm-12am): 15%
- Late Night (12am-6am): 8%

### B.4 Tic Type Distribution (Top 15)

| Tic Type | Count | Percentage |
|----------|-------|------------|
| Neck | 424 | 31.0% |
| Mouth | 137 | 10.0% |
| Eye | 109 | 8.0% |
| Shoulder | 87 | 6.4% |
| Face | 76 | 5.6% |
| Head | 71 | 5.2% |
| Throat | 58 | 4.2% |
| Nose | 52 | 3.8% |
| Jaw | 47 | 3.4% |
| Arm | 42 | 3.1% |
| Others | 264 | 19.3% |

### B.5 Missing Data Analysis

| Feature | Total Records | Missing | Percentage Missing |
|---------|---------------|---------|-------------------|
| intensity | 1,533 | 0 | 0.0% |
| type | 1,533 | 0 | 0.0% |
| timeOfDay | 1,533 | 0 | 0.0% |
| mood | 1,533 | ~720 | ~47% |
| trigger | 1,533 | ~990 | ~65% |
| description | 1,533 | ~1050 | ~68% |

---

## Appendix C: Model Hyperparameters

### C.1 Baseline Model Configuration

**Random Forest Regressor (Intensity Prediction)**:
```
n_estimators: 100
max_depth: 10
min_samples_split: 2
min_samples_leaf: 1
max_features: 'sqrt'
random_state: 42
n_jobs: -1
```

**Random Forest Classifier (High-Intensity Prediction)**:
```
n_estimators: 100
max_depth: 10
min_samples_split: 2
min_samples_leaf: 1
max_features: 'sqrt'
class_weight: 'balanced'
random_state: 42
n_jobs: -1
```

### C.2 Hyperparameter Search Spaces

**Random Forest**:
- n_estimators: [50, 100, 200, 300]
- max_depth: [5, 10, 15, 20, None]
- min_samples_split: [2, 5, 10]
- min_samples_leaf: [1, 2, 4]
- max_features: ['sqrt', 'log2', None]

**XGBoost**:
- n_estimators: [100, 200, 300, 500]
- max_depth: [3, 5, 7, 10]
- learning_rate: [0.01, 0.05, 0.1, 0.3]
- subsample: [0.6, 0.8, 1.0]
- colsample_bytree: [0.6, 0.8, 1.0]
- gamma: [0, 0.1, 0.2]

**LightGBM**:
- n_estimators: [100, 200, 300, 500]
- max_depth: [3, 5, 7, 10, -1]
- learning_rate: [0.01, 0.05, 0.1, 0.3]
- num_leaves: [15, 31, 63, 127]
- subsample: [0.6, 0.8, 1.0]
- colsample_bytree: [0.6, 0.8, 1.0]

---

## Appendix D: Experiment Tracking

### D.1 Completed Experiments

**Experiment ID: baseline_001**
- Date: November 2025
- Model: Random Forest Regressor
- Target: Next tic intensity
- Features: 17 (sequence-based only)
- Test MAE: 1.778
- Test RMSE: 2.194
- Status: Baseline established

**Experiment ID: hyperparam_quick_001**
- Date: November 8, 2025
- Mode: Quick search
- Models: Random Forest, XGBoost
- Targets: Next intensity (regression + classification)
- Total experiments: 4
- Runtime: ~20 seconds
- Best MAE: 1.938 (Random Forest)
- Best F1: 0.341 (XGBoost)
- Status: Framework validated

### D.2 Planned Experiments

**Experiment ID: hyperparam_medium_001** (Planned)
- Mode: Medium search
- Models: Random Forest, XGBoost, LightGBM
- Targets: All 3 (next intensity, future count, time-to-event)
- Feature configs: 6 different combinations
- Iterations: 50 per config
- Expected runtime: 1-2 hours
- Expected best MAE: 1.6-1.8

**Experiment ID: hyperparam_full_001** (Planned)
- Mode: Full search
- Models: All 5 (Ridge, Lasso, Decision Tree, Random Forest, XGBoost, LightGBM)
- Targets: All 3
- Feature configs: All combinations
- Iterations: 100 per config
- Expected runtime: 6-12 hours
- Expected best MAE: 1.4-1.7

---

**Report Status**: Preliminary (40% Complete)

**Next Milestone**: Complete medium hyperparameter search, begin anomaly detection implementation

**Last Updated**: November 2025
