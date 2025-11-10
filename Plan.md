1. Problem Statement

Tic disorders affect approximately 1% of the global population, with individuals experiencing sudden, repetitive movements or vocalizations that significantly impact quality of life, social functioning, and academic/professional performance. Current management approaches are primarily reactive, with individuals responding to tics after occurrence rather than anticipating and preventing them.

1.1 Social issue: The unpredictability of tic episodes creates substantial challenges for affected individuals, including social stigma, educational barriers, workplace difficulties, and reduced quality of life. This aligns with UN Sustainable Development Goal 3 (Good Health and Well-being).

1.2 Project Goals
Primary: Develop a machine learning-based early warning system that predicts high-risk periods for tic episodes, enabling proactive intervention
Subgoal 1: Identify reliable temporal and environmental patterns that correlate with tic occurrence and intensity
Subgoal 2: Quantify relationships between mood states, triggers, and circadian rhythms on tic severity
Subgoal 3: Create personalized risk models that adapt to individual user patterns over time
Subgoal 4: Design an actionable alert system that balances prediction accuracy with alert fatigue

2. Tasks/Steps

Task 1: Data Exploration and Pattern Discovery
Analyze temporal patterns (circadian, weekly) in tic frequency and intensity
Evaluate correlations between mood states, environmental triggers, and tic severity
Identify user archetypes through clustering analysis
Quantify baseline statistics and data quality metrics

Task 2: Predictive Modeling Potential Approaches
Develop binary classification model for high-intensity episodes (intensity ≥7) in next 12-24 hours
Build multi-class risk level prediction (Low/Medium/High) for upcoming time periods
Create regression model for continuous intensity prediction
Compare model performance across different prediction windows

Task 3: Anomaly Detection System
Implement statistical methods (Robust Z-score, CUSUM charts) for detecting unusual patterns
Deploy machine learning approaches (Isolation Forest, Autoencoder) for anomaly identification
Validate anomaly detection against known high-intensity events

Task 4: Personalization Framework
Design adaptive system starting with global model for new users
Implement gradual personalization based on data availability (10, 30, 50+ events)
Develop user-specific calibration methods

3. Timeline

Week
Milestones
Deliverables
1-2
Data preparation & feature engineering
Clean dataset with engineered features including temporal patterns, rolling windows, personal baselines
3
Exploratory data analysis
Statistical analysis report, pattern visualizations, user segmentation
4-5
Baseline & advanced model development
Trained models, performance metrics, Mid-term Report
6
Model comparison & selection 
Final model selection, ablation studies
7-8
Anomaly detection implementation
Anomaly detection system, validation results
9
Personalization & visualization
Personalization framework, dashboard prototype
10
Final testing & documentation
Final Presentation & Report



4. Data

Primary Dataset: Mobile tracking application data collected April-October 2025
Size: 1,533 tic occurrence records from 89 unique users across 182 days
Source: Anonymized user-submitted data via mobile application
Key Fields:
Core: userId, ticId, date, intensity (1-10 scale), timeOfDay (5 categories), type (82 unique types), age, gender
Optional: mood (53.3% coverage, 10 states), trigger (35.3% coverage, 10 categories), description (31.7% coverage)

4.1 Data Statistics:
Mean intensity: 4.5 (SD: 2.8)
User engagement: 9% high (>50 records), 15% medium (10-50), 77% minimal (<10)
Temporal distribution: 84% weekday vs 16% weekend reporting
Most common types: Neck (31%), Mouth (10%), Eye (8%)

4.2 Data Usage by Task:
Tasks 1-2: Full dataset for pattern analysis and model training
Task 3: High-intensity events (intensity ≥7) for anomaly validation
Task 4: User subsets based on engagement levels
Task 5: Full dataset with cost-benefit simulations




5. Approaches

5.1 Potential Predictive Modeling Approaches:
Random Forest (RF): Primary model for handling mixed data types and missing values naturally
Gradient Boosting (XGBoost): To address potential high bias from limited data
Logistic Regression with L2 regularization: Baseline for binary classification
Autoregressive models with potential temporal clustering to robustly identify temporal dependencies

5.2 Feature Engineering:
Temporal features: Cyclical encoding (sin/cos) for time-of-day and day-of-week
Rolling window statistics: 6h, 12h, 24h, 48h windows for tic count and intensity metrics
Personal baselines: User-specific means, variances, and deviation scores
Inter-event intervals: Time since last tic occurrence

5.3 Evaluation Metrics:
Primary: PR-AUC (handles class imbalance better than ROC-AUC)
Secondary: F1 score at multiple thresholds, Top-K precision for alert generation
Clinical utility: Number Needed to Alert (NNA), alert burden vs episodes prevented
User-level: Personalized improvement over baseline (% reduction in unpredicted high-intensity events)

5.4 Validation Strategy:
TimeSeriesSplit cross-validation with 5 folds
Leave-One-User-Out validation for generalization testing
Temporal holdout: Train on months 1-4, test on month 5
Ablation studies to quantify feature group importance


References

Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.

Conelea, C. A., & Woods, D. W. (2008). The influence of contextual factors on tic expression in Tourette syndrome. *Journal of Psychosomatic Research*, 65(5), 487-496.

Eapen, V., Cavanna, A. E., & Robertson, M. M. (2016). Comorbidities, social impact, and quality of life in Tourette syndrome. *Frontiers in Psychiatry*, 7, 97.

Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. *2008 Eighth IEEE International Conference on Data Mining*, 413-422.

Peterson, B. S., & Leckman, J. F. (1998). The temporal dynamics of tics in Tourette syndrome. *Biological Psychiatry*, 44(12), 1337-1348.

Woods, D. W., Piacentini, J., Chang, S., Deckersbach, T., Ginsburg, G., Peterson, A., Scahill, L., Walkup, J., & Wilhelm, S. (2008). *Managing Tourette syndrome: A behavioral intervention*. Oxford University Press.
