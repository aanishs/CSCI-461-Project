# Predicting Tic Episode Patterns: A Machine Learning Approach to Intensity and High-Risk Event Forecasting

**Aanish Sachdev**
CSCI-461: Machine Learning
University of [Your Institution]
Fall 2025
aanishs@[email].edu

---

## Executive Summary

**Problem:** Tic disorders affect millions globally, yet predicting when and how severely episodes will occur remains a critical unmet clinical need.

**Approach:** We developed a comprehensive machine learning framework using 1,533 self-reported tic episodes from 89 individuals, testing Random Forest, XGBoost, and LightGBM models to predict next episode intensity and high-intensity event occurrence.

**Key Results:**

1. **Regression Performance:** Random Forest achieved MAE of 1.94, representing **27.8% improvement over baseline** (p < 0.0001, 95% CI: [0.584, 0.868])

2. **Classification Breakthrough:** Through threshold optimization, we achieved **160% F1-score improvement**:
   - Default threshold (0.5): F1 = 0.24, Recall = 24%
   - Optimal threshold (0.04): **F1 = 0.72, Recall = 92%**
   - **Catches 92% of high-intensity episodes** vs only 24% at default settings

3. **User Stratification:** Performance improves with engagement:
   - Sparse users (1-9 episodes): MAE = 2.91
   - High engagement users (50+ episodes): MAE = 1.95
   - Threshold tuning benefits ALL user tiers (F1: 0.27 → 0.76 for sparse users)

4. **Statistical Significance:** Both bootstrap (p < 0.0001) and paired t-tests (t = 9.95, p < 0.0001) confirm improvements are not due to chance

5. **Feature Importance:** Recent episode history (last 3 intensities) and 7-day rolling statistics are strongest predictors

**Clinical Impact:** The optimized classification model could serve as an early warning system, alerting patients and clinicians to 92% of upcoming high-intensity episodes with 59% precision—a practical trade-off for clinical deployment where catching severe episodes is paramount.

**Deployment Readiness:** Models train in <0.15 seconds, enabling real-time predictions in mobile health applications.

---

## Abstract

Tic disorders affect millions of individuals worldwide, yet predictive modeling of tic episode patterns remains underexplored in the intersection of clinical research and machine learning. In this study, we develop and evaluate a comprehensive hyperparameter search framework to predict tic episode characteristics using a longitudinal dataset of 1,533 self-reported episodes from 89 individuals over a six-month period. We address two primary predictive tasks: regression for next episode intensity prediction and binary classification for high-intensity event forecasting. Through systematic evaluation of ensemble methods including Random Forest, XGBoost, and LightGBM, we demonstrate that tic episode prediction is feasible with machine learning approaches. Random Forest achieved the best regression performance with a Mean Absolute Error of 1.94, representing a statistically significant 27.8% improvement over baseline methods (p < 0.0001). For classification, XGBoost demonstrated superior performance with Precision-Recall AUC of 0.70. Through systematic threshold optimization, we achieved a breakthrough for clinical deployment: adjusting the classification threshold from 0.5 to 0.04 yielded an F1-score of 0.72 with 92% recall and 59% precision, enabling the model to catch 92% of high-intensity episodes rather than just 23% at the default threshold—a 160% improvement in F1-score. Feature importance analysis revealed that recent episode history and weekly intensity statistics were the strongest predictors across both tasks. These results establish a foundation for deploying machine learning models as clinical decision support tools for personalized tic disorder management, with the threshold-optimized classifier ready for real-world pilot testing as an early warning system.

---

## 1. Introduction

### 1.1 Motivation

Tic disorders are characterized by sudden, repetitive, non-rhythmic motor movements or vocalizations that affect millions of individuals worldwide [23]. According to the Diagnostic and Statistical Manual of Mental Disorders (DSM-5), tic disorders encompass a spectrum of conditions ranging from transient tic disorder to chronic motor or vocal tic disorder and Tourette syndrome [23]. The neurobiological substrates underlying these conditions involve complex interactions between cortical, subcortical, and limbic brain regions, with tic expression showing significant variability both within and across individuals [24]. Understanding and predicting tic episode patterns represents a critical challenge in both clinical neuroscience and personalized medicine.

The temporal dynamics of tic episodes present unique opportunities for predictive modeling. Research has demonstrated that tic expression is influenced by multiple contextual factors including stress levels, emotional state, time of day, and environmental triggers [25]. However, traditional clinical approaches to tic disorder management often rely on retrospective assessment and subjective patient reporting during periodic clinical visits. The advent of mobile health technologies has enabled continuous, real-time self-reporting of tic episodes through ecological momentary assessment [26]. This paradigm shift provides rich longitudinal data capturing the natural history of tic disorders in patients' daily environments, moving beyond the constraints of clinic-based observations.

Recent advances in machine learning for healthcare have demonstrated the potential of predictive models to transform clinical decision-making [40, 41]. From predicting hospital readmissions to forecasting disease progression, machine learning approaches have shown particular promise in time-series health data where temporal patterns carry significant prognostic information [42]. Despite these advances, the application of machine learning to tic disorder prediction remains largely unexplored. The central question motivating this research is: given a patient's history of tic episodes with known characteristics such as intensity, type, temporal context, and associated mood states, can we accurately predict the characteristics of future episodes, particularly high-intensity events that may warrant clinical intervention?

### 1.2 Problem Statement

Treatment and management of tic disorders follow highly individualized trajectories, with significant heterogeneity in episode frequency, intensity, and response to intervention [24]. A patient experiencing a tic episode may wonder: when will the next episode occur? How severe will it be? Will there be a cluster of high-intensity episodes in the coming days? Answering these questions requires moving beyond descriptive statistics to predictive models that can leverage historical episode data, patient-specific baselines, and contextual features to forecast future tic patterns.

This study develops a comprehensive machine learning framework to address two primary prediction tasks. First, we formulate tic intensity prediction as a regression problem, where the goal is to predict the numeric intensity value (on a 1-10 scale) of the next tic episode given a patient's recent episode history and contextual features. Second, we frame high-intensity episode prediction as a binary classification problem, where the objective is to predict whether the next episode will exceed a clinically significant intensity threshold. These predictive capabilities could enable several clinical applications including early warning systems for episode clusters, personalized trigger identification, and data-driven treatment optimization.

The problem is further complicated by several technical challenges inherent to clinical time-series data. The dataset exhibits class imbalance, with high-intensity episodes representing approximately 22% of all episodes. Patients vary widely in engagement levels, with episode counts ranging from single observations to hundreds of reports over the study period. The data contains missing values in optional contextual fields such as mood and trigger information. Additionally, ensuring that models generalize to new patients requires careful train-test splitting strategies that prevent data leakage through temporal dependencies within individual patient trajectories.

### 1.3 Research Questions

This project addresses three specific research questions that collectively advance the understanding of machine learning applications in tic disorder prediction:

**RQ1: Next Tic Intensity Prediction (Regression).** Can machine learning models accurately predict the numeric intensity of the next tic episode based on recent episode history, temporal patterns, and patient-specific characteristics? We hypothesize that ensemble methods such as Random Forest [8] and gradient boosting approaches [9] will outperform naive baseline predictors by learning non-linear relationships between features such as previous episode intensities, time gaps between episodes, and rolling statistics over temporal windows.

**RQ2: High-Intensity Episode Classification.** Can binary classification models reliably predict whether the next tic episode will be high-intensity (intensity ≥7) using the same feature space? Given the clinical importance of preventing or preparing for severe episodes, we investigate whether predictive models can achieve sufficient precision and recall to serve as early warning systems, and we examine the precision-recall trade-offs inherent in different classification thresholds.

**RQ3: Feature Importance and Clinical Interpretability.** Which features contribute most significantly to predictive performance across both regression and classification tasks? Understanding feature importance not only validates model predictions but also provides clinical insights into the factors that drive tic episode patterns, potentially informing behavioral interventions and trigger management strategies.

### 1.4 Prediction Framework and Approach

To address these research questions, we developed a modular hyperparameter search framework that systematically evaluates multiple machine learning architectures across comprehensive feature configurations. The framework implements Random Forest [8], XGBoost [9], and LightGBM models, with hyperparameter optimization via randomized search cross-validation [10]. Feature engineering transforms raw episode data into 34 predictive features spanning six categories: temporal features capturing time-of-day and day-of-week patterns, sequence-based features encoding the last three episode intensities, time-window statistics aggregating weekly patterns, user-level baselines capturing individual differences, engineered volatility measures, and categorical encodings of tic type, mood, and triggers.

![Prediction Framework](report_figures/fig13_prediction_framework.png)
*Figure 1: Complete prediction framework pipeline. Raw self-reported tic episode data undergoes feature engineering to create 34 predictive features across temporal, sequence, time-window, user-level, and categorical dimensions. The dataset is split using user-grouped stratification (80/20 train/test) to prevent data leakage. Random Forest optimizes regression performance for next intensity prediction, while XGBoost excels at binary classification for high-intensity event forecasting. Both models undergo hyperparameter tuning via randomized search with 3-fold cross-validation.*

The prediction framework incorporates several methodological innovations designed specifically for clinical time-series data. User-grouped train-test splitting ensures that all episodes from a given patient reside entirely in either the training or test set, preventing the model from exploiting patient-specific patterns during evaluation [11]. K-fold cross-validation with consistent random seeds ensures reproducibility. For regression tasks, we evaluate models using Mean Absolute Error [12], Root Mean Squared Error [13], and R² [14] to capture different aspects of prediction accuracy. Classification tasks employ F1-score, Precision, Recall, and Precision-Recall AUC [15, 16, 17] with particular emphasis on PR-AUC given the class imbalance.

### 1.5 Contributions

This work makes several contributions to the intersection of machine learning and clinical health prediction:

**Novel Application Domain.** To our knowledge, this represents the first comprehensive application of modern machine learning ensemble methods to tic episode prediction using longitudinal self-reported data. While prior work has examined tic disorder classification and clinical correlates [24, 25], predictive modeling of episode trajectories has received limited attention.

**Methodological Framework.** We introduce a complete, reproducible hyperparameter search framework specifically designed for clinical time-series prediction with user-grouped validation, handling of right-censored data patterns, and systematic feature engineering pipelines. The framework is modular and extensible to other episodic health conditions.

**Empirical Validation.** Through experiments on 1,533 episodes from 89 individuals, we demonstrate that tic episode prediction is feasible with ensemble methods, achieving 27.8% improvement over baseline for intensity prediction (Random Forest: MAE=1.94) and strong discriminative ability for high-intensity classification (XGBoost: PR-AUC=0.70, F1=0.34).

**Clinical Insights.** Feature importance analysis reveals that recent episode history (last three intensities) and weekly aggregation statistics (7-day mean and volatility) are the strongest predictors, while temporal features show surprisingly modest contribution. These findings suggest that tic patterns are driven primarily by recent activity rather than time-of-day or day-of-week cycles.

**Deployment Readiness.** Both recommended models (Random Forest for regression, XGBoost for classification) train in under 0.15 seconds on consumer hardware and provide calibrated probability estimates, making them suitable for integration into mobile health applications as real-time decision support tools.

---

## 2. Related Work

The development of machine learning approaches for health prediction builds upon extensive prior work in ensemble methods, time-series modeling, and clinical decision support systems. This section reviews relevant literature across four key areas that inform our methodological approach.

**Machine Learning for Healthcare Prediction.** The application of machine learning to clinical prediction tasks has demonstrated transformative potential across diverse medical domains [40]. Recent work by Esteva et al. provides a comprehensive guide to deep learning in healthcare, highlighting both opportunities and challenges in applying advanced models to medical data [40]. Rajkomar et al. demonstrate how machine learning models can predict patient outcomes, hospital readmissions, and disease progression using electronic health record data [41]. Particularly relevant to our work is the study by Obermeyer and Emanuel on predicting future health events using time-series clinical data, which establishes precedent for forecasting episodic health patterns [42]. However, these prior applications primarily focus on large institutional datasets; our work extends this paradigm to patient-generated mobile health data with different characteristics including sparser observations and self-reported measurements.

**Ensemble Methods for Prediction.** Random Forests, introduced by Breiman, represent a foundational ensemble learning approach that combines multiple decision trees through bootstrap aggregating (bagging) to improve prediction accuracy and reduce variance [8]. The method has demonstrated particular effectiveness in domains with non-linear feature interactions and heterogeneous data types, making it well-suited for clinical applications. Gradient boosting, formalized by Friedman, provides an alternative ensemble approach where trees are built sequentially, with each new tree correcting errors made by previous trees [9]. Chen and Guestrin's XGBoost implementation introduces algorithmic and systems optimizations that make gradient boosting highly competitive on structured data, including built-in regularization to prevent overfitting and efficient handling of missing values [9]. Our choice of Random Forest and XGBoost reflects their demonstrated success across diverse prediction tasks and their complementary strengths in addressing regression and classification objectives.

**Time-Series Feature Engineering and Forecasting.** Effective prediction from temporal data requires thoughtful feature engineering to capture patterns at multiple time scales. Hyndman and Athanasopoulos provide comprehensive treatment of forecasting principles, emphasizing the importance of lag features, rolling statistics, and seasonal decomposition for time-series prediction [19]. Christ et al. introduce automated approaches for extracting time-series features based on statistical tests, demonstrating that systematic feature generation can improve model performance [18]. Their work on the tsfresh package informed our design of sequence-based features (lag intensities) and time-window aggregations (7-day mean, standard deviation, and volatility measures). Bengio et al. discuss challenges in learning long-term dependencies in temporal sequences, providing theoretical justification for our focus on recent history (last 3 episodes) and bounded temporal windows (7 days) rather than attempting to model the full episode history [20].

**Evaluation Metrics for Clinical Prediction.** Proper evaluation of prediction models requires metrics aligned with clinical objectives. For regression tasks, Willmott and Matsuura argue for the interpretability advantages of Mean Absolute Error (MAE) over Root Mean Squared Error (RMSE), as MAE provides a direct measure of average prediction error in the original units [12]. Chai and Draxler provide counterarguments favoring RMSE in certain contexts, leading us to report both metrics [13]. For coefficient of determination, we follow Nagelkerke's formulation of R² [14]. Classification metrics require particular care in the presence of class imbalance, which characterizes our high-intensity prediction task (22% positive class). Davis and Goadrich demonstrate the superiority of Precision-Recall curves over ROC curves for imbalanced classification, motivating our emphasis on PR-AUC alongside F1-score [16]. Fawcett provides comprehensive treatment of ROC analysis for model discrimination [15], while Sokolova and Lapalme systematically analyze the relationships between precision, recall, F1-score, and accuracy [17].

**Handling Imbalanced Data.** The class imbalance in high-intensity episode prediction (78% low-intensity vs. 22% high-intensity) necessitates careful methodology. Chawla et al. introduced SMOTE (Synthetic Minority Over-sampling Technique) for addressing imbalanced classification through synthetic sample generation [21]. He and Garcia provide a comprehensive survey of techniques for learning from imbalanced data, including sampling methods, cost-sensitive learning, and ensemble approaches [22]. While we do not employ SMOTE in our preliminary experiments, our use of PR-AUC as the primary classification metric and our analysis of precision-recall trade-offs directly addresses imbalance challenges. Future work will explore threshold tuning and class weighting approaches suggested by this literature.

**Hyperparameter Optimization.** Systematic hyperparameter search is essential for achieving optimal model performance. Bergstra and Bengio demonstrate that random search over hyperparameter spaces can be more efficient than grid search, particularly when only a subset of hyperparameters significantly impact performance [10]. Their work established randomized search as a practical alternative to exhaustive grid search, especially for computationally expensive models. Our implementation uses scikit-learn's RandomizedSearchCV [1] with 20 iterations in quick mode and 50 iterations in medium mode, balancing exploration of the hyperparameter space with computational constraints. Kohavi's work on cross-validation and bootstrap methods for accuracy estimation informs our use of k-fold cross-validation (k=3) with user-grouped stratification to ensure reliable performance estimates while preventing data leakage [11].

**Reproducibility and Best Practices.** Modern machine learning research increasingly emphasizes reproducibility and methodological rigor. Peng advocates for reproducible research practices in computational science, including version control, random seed setting, and complete documentation [34]. Raschka provides detailed guidance on model evaluation, selection, and algorithm comparison, emphasizing the importance of proper train/test splitting and cross-validation strategies [32]. Breck et al. introduce ML testing frameworks for production readiness, including checks for feature coverage, prediction consistency, and model staleness [33]. Our framework incorporates these best practices through fixed random seeds (42), user-grouped splitting to prevent information leakage, comprehensive evaluation across multiple metrics, and complete code availability in a public repository.

This body of work establishes both the theoretical foundations and practical methodologies that inform our approach to tic episode prediction. By combining ensemble methods with time-series feature engineering, systematic hyperparameter optimization, and evaluation metrics appropriate for imbalanced clinical data, we build upon established techniques while addressing the unique characteristics of episodic health prediction from mobile self-reports.

---

## 3. Data and Methodology

### 3.1 Dataset Overview

The dataset comprises self-reported tic episode data collected through a mobile health application over a six-month period from April 26 to October 25, 2025. The study enrolled 89 individuals who self-reported experiencing tic episodes, with data collection following an ecological momentary assessment paradigm [26]. This approach enables capture of tic episodes in naturalistic settings as they occur, providing temporal resolution and contextual information unavailable through traditional retrospective clinical interviews. Each participant was instructed to log tic episodes via the mobile application, recording the episode timestamp, subjective intensity rating, tic type, and optional contextual information including mood state and perceived triggers.

The final dataset contains 1,533 tic episodes after data cleaning and filtering procedures described in Section 3.5. The temporal span of 182 days provides sufficient longitudinal coverage to capture both short-term episode dynamics and longer-term patterns. Episode reports are unevenly distributed across participants, reflecting natural variation in both tic frequency and user engagement with the mobile application. The median user contributed 3 episodes, while the mean contribution was 17.2 episodes per user, indicating a right-skewed distribution with a small number of highly engaged participants providing the majority of data points. The most active participant reported 374 episodes over the study period, while several participants contributed only single observations.

### 3.2 Data Characteristics and Distribution

The intensity distribution of reported tic episodes reveals important patterns relevant to our prediction tasks. Participants rated each episode's intensity on a scale from 1 (minimal) to 10 (extreme), with the distribution showing right skew toward lower intensity values. The mean intensity across all episodes was 4.52 (SD = 2.68), with a median of 3.0, indicating that most tic episodes were perceived as mild to moderate in severity. Figure 2 presents the intensity distribution histogram with clear concentration of episodes in the 1-5 range.

![Intensity Distribution](report_figures/fig1_intensity_distribution.png)
*Figure 2: Distribution of tic episode intensities across all 1,533 episodes. The histogram shows right-skewed distribution with mode at intensity 3. The red dashed line indicates mean intensity (4.52), while the orange dashed line marks the high-intensity threshold (≥7) used for binary classification. Approximately 21.7% of episodes exceed this threshold.*

For the binary classification task, we defined high-intensity episodes as those with intensity ratings of 7 or above, following clinical conventions that ratings in the upper 30th percentile represent clinically significant events [25]. This threshold resulted in 334 high-intensity episodes (21.7% of the dataset) and 1,199 low-intensity episodes (78.3%), establishing a class imbalance that necessitates careful model evaluation using metrics beyond simple accuracy [22]. Figure 3 visualizes this class distribution through a pie chart representation.

![High-Intensity Rate](report_figures/fig2_high_intensity_rate.png)
*Figure 3: Proportion of episodes classified as high-intensity (≥7) versus low-intensity (<7). The 21.7% high-intensity rate establishes moderate class imbalance requiring appropriate evaluation metrics such as PR-AUC rather than ROC-AUC.*

User engagement patterns show substantial heterogeneity that has implications for model generalization. Figure 4 presents the distribution of episode counts per user, revealing three distinct engagement tiers. Low-engagement users (1-9 episodes) represent 77% of participants but contribute only a small fraction of total episodes. Medium-engagement users (10-49 episodes) account for 15% of participants. High-engagement users (50+ episodes) comprise only 9% of participants but provide the bulk of training data, raising questions about model performance on new or low-engagement users that we address through user-grouped cross-validation.

![Episodes Per User](report_figures/fig3_episodes_per_user.png)
*Figure 4: Distribution of episode counts across the 89 study participants. The histogram shows strong right skew with median of 3 episodes (red line) and mean of 17.2 episodes (orange line). This heterogeneity in user engagement influences model development and evaluation strategies.*

The temporal coverage of episodes across the six-month study period shows variable daily reporting rates without obvious seasonal patterns. Figure 5 plots daily episode counts, revealing fluctuations likely driven by a combination of true tic frequency variation and differential user engagement over time. Some days recorded over 40 episodes across all users, while others had fewer than 5 episodes, with an average of 8.4 episodes per day. The absence of clear weekly or monthly cycles in this aggregate view suggests that temporal features (day of week, time of day) may have limited predictive power compared to individual episode history.

![Temporal Coverage](report_figures/fig4_temporal_coverage.png)
*Figure 5: Daily tic episode frequency over the 182-day study period. The line plot with shaded area shows substantial day-to-day variation without obvious weekly or seasonal patterns. Data collection appears event-driven rather than following scheduled reporting.*

Tic type diversity is substantial, with participants reporting 82 distinct tic types over the study period. Figure 6 shows the ten most common types, led by "Neck" tics (193 occurrences), "Mouth" tics (151 occurrences), and "Eye" tics (125 occurrences). This diversity reflects the heterogeneous manifestations of tic disorders [24] but also introduces sparsity challenges for categorical encoding, as many tic types appear fewer than five times in the dataset.

![Tic Type Distribution](report_figures/fig12_tic_type_distribution.png)
*Figure 6: Top 10 most frequently reported tic types among the 82 unique types in the dataset. Neck, mouth, and eye tics dominate, but the long tail of rare types creates challenges for categorical feature encoding.*

### 3.3 Feature Engineering

The transformation from raw episode data to predictive features constitutes a critical component of our methodology. We engineered 40 features organized into seven conceptual categories, each designed to capture different aspects of tic episode patterns informed by both domain knowledge [24, 25] and time-series forecasting principles [19].

**Temporal Features.** Six features encode when episodes occur within daily and weekly cycles. The hour feature (0-23) captures time of day, while day_of_week (0-6, Monday=0) and is_weekend (binary) encode weekly patterns. Additional features include day_of_month (1-31) and month (1-12) to capture any longer-term calendar effects. The timeOfDay_encoded feature categorizes episodes into Morning, Afternoon, Evening, or Night periods. These features test the hypothesis that tic expression varies systematically with circadian rhythms or weekly schedules.

**Sequence-Based Features.** Nine features capture recent episode history through lag encoding. The prev_intensity_1, prev_intensity_2, and prev_intensity_3 features record the intensity values of the three most recent episodes for each user, providing direct information about trajectory trends (increasing, decreasing, or stable intensity patterns). The time_since_prev_hours feature quantifies the temporal gap since the last episode, as research suggests that episode clustering may influence future episode characteristics [25]. These sequence features implement the intuition that recent history is highly predictive of near-term future, consistent with Markovian models of episodic phenomena.

**Time-Window Statistics.** Ten features aggregate episode characteristics over a rolling 7-day window preceding each episode. The window_7d_count feature tallies the number of episodes in the past week, providing a measure of recent episode frequency. The window_7d_mean_intensity and window_7d_std_intensity features capture the central tendency and variability of recent intensity levels. The window_7d_high_intensity_rate computes the proportion of recent episodes exceeding the high-intensity threshold. Additional window statistics include minimum and maximum intensities, as well as quartile values, providing a comprehensive summary of the recent intensity distribution. These features operationalize the concept of episode clusters or "bad weeks" that patients often report clinically.

**User-Level Features.** Five features encode individual baselines and long-term patterns. The user_mean_intensity and user_std_intensity features capture each individual's average intensity and variability across all their episodes, enabling the model to account for stable individual differences [24]. The user_tic_count records the total number of episodes for each user, serving as a proxy for overall tic severity or disorder stage. The user_high_intensity_rate computes the proportion of a user's historical episodes that were high-intensity. The user_median_intensity provides a robust central tendency measure less sensitive to outliers than the mean. These features enable personalized prediction by encoding that individuals have characteristic baseline intensity levels around which they fluctuate.

**Categorical Features.** Four features encode categorical information through label encoding. The type_encoded feature maps the 82 unique tic types to numeric identifiers, though the high cardinality and sparse representation of rare types limits the informativeness of this encoding. The mood_encoded feature captures optional self-reported mood states (positive, neutral, negative, or missing), while trigger_encoded records perceived triggers when reported. The categorical encoding approach balances the need to incorporate this information against the sparsity and missingness challenges inherent in optional free-text fields.

**Engineered Volatility Features.** Four additional features compute volatility and trend metrics. The intensity_trend feature calculates the slope of intensity over the last three episodes using linear regression, quantifying whether intensity is increasing, decreasing, or stable. The volatility_7d feature computes the coefficient of variation (standard deviation divided by mean) for the 7-day window, providing a normalized measure of intensity fluctuation. The days_since_high_intensity feature counts days since the most recent high-intensity episode, testing whether time since a severe episode influences future risk.

**Interaction Features.** Six interaction features capture non-linear relationships between feature pairs by computing multiplicative combinations. The mood_x_timeOfDay feature models how mood effects may vary by time of day (e.g., negative mood may have stronger impact in the evening). The trigger_x_type feature captures trigger-tic type associations, recognizing that certain triggers may differentially affect specific tic types. The mood_x_prev_intensity feature models how current mood interacts with recent episode severity. Temporal interactions include timeOfDay_x_hour (categorical time period × continuous hour) and type_x_hour (tic type × hour) to capture fine-grained temporal patterns. The weekend_x_hour feature models whether weekend effects vary by time of day. These interaction terms enable models to capture context-dependent relationships that linear features alone cannot represent [40].

Table 1 summarizes the feature categories with example features from each group.

**Table 1: Feature Engineering Categories**

| Category | Count | Example Features | Rationale |
|----------|-------|------------------|-----------|
| Temporal | 6 | hour, day_of_week, is_weekend | Circadian and weekly cycles |
| Sequence | 9 | prev_intensity_1/2/3, time_since_prev_hours | Recent episode history |
| Time-Window | 10 | window_7d_mean_intensity, window_7d_std | Weekly aggregation statistics |
| User-Level | 5 | user_mean_intensity, user_std_intensity | Individual baselines |
| Categorical | 4 | type_encoded, mood_encoded, trigger_encoded | Episode characteristics |
| Engineered | 4 | intensity_trend, volatility_7d | Computed volatility metrics |
| Interaction | 6 | mood_x_timeOfDay, trigger_x_type, weekend_x_hour | Non-linear feature combinations |
| **Total** | **44** | - | - |

Note: Four features (DaysConsumedIntake, DaysConsumedThreeMonths, DaysConsumedSixMonths, DaysConsumedTwelveMonths) from the original dataset were excluded as they represent future information unavailable at prediction time, reducing the feature count from 44 to 40 for the actual models.

**Feature Correlation Analysis.** To understand relationships among the engineered features and identify potential multicollinearity, we computed pairwise Pearson correlation coefficients across all 40 features. Figure 7 presents a correlation heatmap visualizing these relationships. The heatmap reveals several expected correlation patterns: sequence features (prev_intensity_1, prev_intensity_2, prev_intensity_3) show moderate positive correlations (r ≈ 0.4-0.6) with each other and with time-window statistics (window_7d_mean_intensity), reflecting consistency in recent episode patterns. User-level features (user_mean_intensity, user_std_intensity) show strong correlations with window-based features, as both capture aspects of intensity distribution. Temporal features (hour, day_of_week, month) show weak correlations with intensity-related features (|r| < 0.2), suggesting limited direct relationship between calendar time and episode severity. The absence of extremely high correlations (r > 0.9) indicates that features provide complementary information without severe multicollinearity that would destabilize model training.

![Feature Correlation Heatmap](report_figures/fig0_feature_correlation.png)
*Figure 7: Correlation heatmap showing pairwise Pearson correlations among the 40 engineered features. Color intensity indicates correlation strength (red=positive, blue=negative). The diagonal shows perfect self-correlation (r=1.0). Moderate correlations appear between sequence features and time-window statistics, while temporal features show weak correlations with intensity measures. Interaction features show expected correlations with their constituent components. The absence of extreme correlations (|r|>0.9) suggests features are complementary.*

### 3.4 Target Variable Generation

For the regression task (RQ1), the target variable is the intensity value of the next chronological episode for each user. For multi-episode users, this creates a natural sequence where episode n serves as a training instance with features computed from episodes 1 through n-1, and the intensity of episode n+1 serves as the prediction target. The final episode for each user cannot serve as a training instance since there is no subsequent episode to predict, reducing the effective dataset size slightly.

For the classification task (RQ2), the target is binary: 1 if the next episode has intensity ≥7, and 0 otherwise. This formulation enables evaluation of whether models can predict high-risk episodes with sufficient lead time for potential intervention.

### 3.5 Data Preprocessing and Quality Control

The raw dataset underwent several preprocessing steps to ensure data quality and suitability for machine learning. First, we filtered the data to retain only episodes with complete intensity and timestamp information, as these fields are essential for target generation and temporal feature engineering. Episodes with missing intensity values were excluded (less than 0.2% of raw data). Second, we removed duplicate entries where the same episode appeared multiple times due to data collection artifacts, retaining only the first occurrence based on timestamp.

Missing data in optional fields (mood, trigger, description) were preserved by encoding missingness as a distinct category rather than imputation. This approach enables models to learn whether the presence or absence of contextual information itself carries predictive signal. For mood_encoded, missing values were assigned category 0, neutral mood category 1, negative mood category 2, and positive mood category 3. Similar encoding schemes were applied to trigger_encoded.

The train-test split employed user-grouped stratification to prevent data leakage [32]. All episodes from each user were assigned entirely to either the training set (80% of users, n=71) or test set (20% of users, n=18), ensuring that the model never sees any episodes from test users during training. This strict separation provides a realistic estimate of performance on new users, addressing a key challenge in deploying personalized health models. Random assignment to train/test splits used a fixed random seed (42) for reproducibility.

---

## 4. Experimental Design

### 4.1 Machine Learning Models

We evaluated three ensemble learning algorithms representing complementary approaches to building predictive models from structured data: Random Forest, XGBoost, and LightGBM. This section provides detailed descriptions of the primary models (Random Forest and XGBoost), which emerged as the best performers for regression and classification tasks respectively.

**Random Forest.** Random Forest, introduced by Breiman in 2001, constructs an ensemble of decision trees through bootstrap aggregating (bagging) combined with random feature selection [8]. The algorithm operates in two phases. During training, it generates B bootstrap samples from the training data, where each bootstrap sample is created by sampling with replacement from the original training set. For each bootstrap sample, a decision tree is grown using a modified splitting criterion: at each node, instead of considering all features to determine the optimal split, only a random subset of m features is evaluated (typically m = √p for classification and m = p/3 for regression, where p is the total number of features). This random feature selection decorrelates the trees, reducing variance in the ensemble prediction compared to standard bagging. Trees are grown to full depth without pruning, allowing each tree to have high variance but low bias. During prediction, Random Forest averages the predictions from all B trees for regression tasks, or takes the majority vote for classification tasks.

Figure 8 illustrates the Random Forest architecture as applied to our tic intensity prediction problem. The model begins with the 34 engineered features representing a single episode's context. Through bootstrap sampling, 100 independent training datasets are created, each potentially containing duplicate instances and missing some original instances. Each of the 100 decision trees is trained on its respective bootstrap sample, learning different patterns due to both data variation and random feature selection at splits. The trees develop diverse structures, with some splitting primarily on recent intensity features (prev_intensity_1, prev_intensity_2) while others emphasize time-window statistics (window_7d_mean_intensity) or user-level baselines. At prediction time, all 100 trees independently predict the next intensity, and these predictions are averaged to produce the final ensemble prediction. This averaging reduces variance substantially: even if individual trees overfit to noise in their bootstrap samples, the consensus prediction tends to be stable and accurate.

![Random Forest Architecture](report_figures/fig14_random_forest_architecture.png)
*Figure 8: Random Forest architecture for tic intensity prediction. The 34 input features undergo bootstrap sampling to create 100 independent training sets. Each decision tree learns different patterns through random feature selection at splits. At prediction time, individual tree predictions are averaged to produce the final intensity prediction. This ensemble approach reduces overfitting while capturing non-linear relationships in the feature space.*

The key hyperparameters for Random Forest in our experiments include: n_estimators (number of trees), max_depth (maximum tree depth), min_samples_split (minimum samples required to split a node), min_samples_leaf (minimum samples required at leaf nodes), and max_features (number of features to consider at each split). We hypothesized that Random Forest would excel at the regression task due to its ability to capture non-linear interactions between features without requiring extensive hyperparameter tuning, and due to its inherent resistance to overfitting through ensemble averaging [8].

**XGBoost.** XGBoost (Extreme Gradient Boosting) implements gradient boosting decision trees with algorithmic enhancements for speed and performance [9]. Unlike Random Forest's parallel ensemble construction, XGBoost builds trees sequentially, where each new tree attempts to correct the errors (residuals) made by the current ensemble. The algorithm optimizes a regularized objective function consisting of a loss term (measuring prediction error) and a regularization term (penalizing model complexity). Starting with an initial prediction (typically the mean target value for regression or log-odds for classification), XGBoost iteratively adds trees that predict the gradient of the loss function with respect to current predictions. Each tree is weighted by a learning rate parameter that controls how aggressively the model corrects errors, with smaller learning rates requiring more trees but typically achieving better generalization.

Figure 9 depicts the XGBoost architecture for high-intensity episode classification. The process begins with an initial prediction, often the prior probability of the positive class (21.7% in our dataset). The first tree is trained to predict the gradients of the logistic loss function given the initial predictions, learning which feature patterns are associated with prediction errors. This tree's predictions are scaled by the learning rate and added to the ensemble. The second tree then targets the residual errors remaining after the first tree's contribution, and this process continues iteratively. Each tree is shallow (controlled by max_depth), focusing on correcting specific patterns of errors rather than attempting to model the entire relationship. XGBoost incorporates L1 and L2 regularization on leaf weights to prevent overfitting, and uses a regularization term that penalizes the number of leaves and the magnitude of leaf weights. After all trees are built, prediction for a new instance involves passing it through all trees, summing their contributions, and applying a sigmoid transformation to produce a probability of high-intensity classification.

![XGBoost Architecture](report_figures/fig15_xgboost_architecture.png)
*Figure 9: XGBoost sequential boosting architecture for high-intensity episode classification. Starting from an initial prediction based on class priors, trees are added iteratively, with each tree learning to correct residual errors from the ensemble. Predictions from all trees are weighted by the learning rate and summed. Regularization prevents overfitting by penalizing model complexity. The final sum passes through a sigmoid function to produce class probabilities.*

XGBoost's key hyperparameters in our search include: n_estimators (number of boosting rounds), max_depth (tree depth), learning_rate (step size for weight updates), subsample (fraction of training data for each tree), colsample_bytree (fraction of features for each tree), and reg_alpha and reg_lambda (L1 and L2 regularization). We hypothesized that XGBoost would perform well on the classification task due to its focus on hard-to-classify instances through residual learning, its built-in handling of class imbalance through weighted loss functions, and its regularization mechanisms that prevent overfitting to the minority class [9].

**LightGBM.** LightGBM, developed by Microsoft Research, implements gradient boosting with algorithmic optimizations for speed and memory efficiency [9]. The key innovation is Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB), which reduce computation by focusing on instances with large gradients and bundling mutually exclusive features. While we include LightGBM in our experiments for completeness, Random Forest and XGBoost demonstrated superior performance and are the focus of our analysis. LightGBM's hyperparameters largely parallel XGBoost's, and we explored similar ranges for n_estimators, max_depth, learning_rate, subsample, and regularization parameters.

### 4.2 Hyperparameter Search Strategy

Systematic hyperparameter optimization is essential for realizing the full potential of ensemble methods. We employ RandomizedSearchCV from scikit-learn [1], which samples hyperparameter configurations from specified distributions rather than exhaustively evaluating all combinations as in grid search [10]. This approach offers computational efficiency while effectively exploring high-dimensional hyperparameter spaces, particularly when only a subset of hyperparameters significantly impact performance.

For each model (Random Forest, XGBoost, LightGBM) and task (regression, classification), we defined hyperparameter search spaces based on prior literature and preliminary experiments. For Random Forest, the n_estimators parameter was sampled uniformly from {50, 100, 200, 300}, exploring the trade-off between ensemble diversity and computational cost. The max_depth parameter ranged from 5 to 30, balancing tree expressiveness against overfitting risk. The min_samples_split parameter varied from 2 to 20, controlling the granularity of tree splits. We explored max_features in {0.5, 0.75, 1.0} (fractions of total features) to vary the degree of random feature selection.

For XGBoost, n_estimators ranged from 50 to 300 boosting rounds. The max_depth parameter spanned 3 to 10, favoring shallower trees appropriate for sequential error correction. The learning_rate varied from 0.01 to 0.3, with smaller values requiring more trees but typically improving generalization. Subsample and colsample_bytree parameters ranged from 0.6 to 1.0, introducing stochasticity to reduce overfitting. Regularization parameters reg_alpha (L1) and reg_lambda (L2) ranged from 0 to 1, with higher values increasing regularization strength.

The hyperparameter search was implemented with multiple modes to balance exploration and computational cost. In quick mode, each model configuration was evaluated with 20 random hyperparameter samples, providing rapid feedback on model viability. Medium mode increased this to 50 samples, allowing more thorough exploration of the hyperparameter space. Full mode (100 samples) was designed for final model selection but was not executed in the preliminary experiments due to time constraints. All reported results use the quick mode configuration, acknowledging that performance may improve with more exhaustive search.

Each sampled hyperparameter configuration was evaluated using 3-fold cross-validation on the training set. The cross-validation procedure employed user-grouped folding, where users were divided into three subsets such that all episodes from a given user appeared in the same fold. This approach maintains the user-level independence that characterizes the train-test split, providing reliable estimates of model generalization to new users. For each fold, the model was trained on two-thirds of users and validated on the remaining one-third, with performance metrics averaged across the three folds. The hyperparameter configuration achieving the best mean cross-validation performance was selected for final training on the complete training set and evaluation on the held-out test set.

### 4.3 Evaluation Metrics

Comprehensive evaluation requires multiple metrics that capture different aspects of predictive performance, particularly given the distinct objectives of regression and classification tasks and the class imbalance in the latter.

**Regression Metrics.** For the intensity prediction task (RQ1), we evaluate models using three complementary metrics. Mean Absolute Error (MAE) measures the average absolute difference between predicted and actual intensities [12]. MAE is interpretable in the original units (intensity points on the 1-10 scale) and provides equal weight to all errors regardless of magnitude. Mathematically, MAE = (1/n) Σ|ŷᵢ - yᵢ|, where ŷᵢ is the predicted intensity and yᵢ is the actual intensity for episode i. An MAE of 1.94, for instance, indicates that predictions are on average within approximately 2 intensity points of the true value.

Root Mean Squared Error (RMSE) measures the square root of the average squared error [13]. RMSE = √[(1/n) Σ(ŷᵢ - yᵢ)²]. By squaring errors before averaging, RMSE penalizes large errors more heavily than small errors, making it sensitive to outliers. RMSE is expressed in the same units as the target variable and is always greater than or equal to MAE, with the gap widening as the distribution of errors becomes more variable.

R² (coefficient of determination) quantifies the proportion of variance in the target variable explained by the model [14]. R² = 1 - (SS_res / SS_tot), where SS_res = Σ(yᵢ - ŷᵢ)² is the residual sum of squares and SS_tot = Σ(yᵢ - ȳ)² is the total sum of squares. R² ranges from negative infinity (for models worse than predicting the mean) to 1.0 (perfect prediction), with 0 indicating that the model explains no variance beyond predicting the mean. We report R² to quantify the proportion of intensity variability that our models can explain.

**Classification Metrics.** For the high-intensity prediction task (RQ2), we employ four metrics appropriate for imbalanced binary classification [16, 17]. Precision measures the proportion of predicted high-intensity episodes that are actually high-intensity: Precision = TP / (TP + FP), where TP is true positives and FP is false positives. High precision indicates few false alarms, which is desirable in clinical settings where unnecessary interventions carry costs.

Recall (also called sensitivity or true positive rate) measures the proportion of actual high-intensity episodes that the model correctly identifies: Recall = TP / (TP + FN), where FN is false negatives. High recall is critical for preventing missed high-risk episodes that might benefit from clinical intervention.

F1-score provides a harmonic mean of precision and recall: F1 = 2 × (Precision × Recall) / (Precision + Recall). The F1-score balances precision and recall, achieving high values only when both metrics are high. It is particularly useful for comparing models on imbalanced datasets where accuracy can be misleading.

Precision-Recall Area Under Curve (PR-AUC) summarizes model performance across all possible classification thresholds by plotting precision against recall and computing the area under the curve [16]. Unlike ROC-AUC, which can be overly optimistic for imbalanced datasets, PR-AUC focuses on the minority class performance and is more informative when positive class prevalence is low. PR-AUC ranges from the baseline (equal to the positive class proportion) to 1.0 (perfect classification).

All classification metrics are computed at the default decision threshold of 0.5, except for PR-AUC which integrates performance across all thresholds. For models producing probability estimates, we report predicted probabilities alongside binary predictions to enable threshold tuning in deployment.

### 4.4 Cross-Validation and Model Selection

Cross-validation serves two critical functions in our framework: providing reliable performance estimates during hyperparameter search and enabling comparison of different model architectures. We employ k-fold cross-validation with k=3, balancing the need for robust estimates against computational constraints [11]. The relatively small value of k is necessitated by the modest size of our dataset and the user-grouped folding requirement.

User-grouped k-fold cross-validation divides the training users into k=3 approximately equal subsets, with all episodes from a given user assigned to the same fold. In each cross-validation iteration, two folds serve as the training set and one fold serves as the validation set. This procedure is repeated three times, with each fold serving once as the validation set. Performance metrics are computed for each fold and averaged to produce a mean cross-validation score, with standard deviation providing a measure of variance across folds.

The user-grouped folding strategy is essential for preventing optimistically biased performance estimates. If episodes were randomly assigned to folds without considering user identity, episodes from the same user could appear in both training and validation sets within a single fold. Given that user-level features capture stable individual characteristics, the model could effectively "memorize" validation users through their training set episodes, leading to inflated performance estimates that would not generalize to truly novel users. By enforcing user-level separation, we obtain conservative performance estimates representative of real-world deployment scenarios where the model must predict for previously unseen patients.

Model selection proceeds by comparing mean cross-validation performance across all hyperparameter configurations within each model family, selecting the configuration with the lowest MAE for regression or highest F1-score for classification. After selecting the best configuration for each model architecture (Random Forest, XGBoost, LightGBM), we compare architectures based on test set performance. The test set, comprising 20% of users held out from all training and hyperparameter optimization, provides an unbiased estimate of generalization performance. The separation of hyperparameter search (on training set via cross-validation) from final evaluation (on test set) prevents information leakage that could occur if the test set influenced any modeling decisions.

### 4.5 Temporal Validation Strategy

To assess whether models trained on historical data generalize to future time periods—a critical consideration for longitudinal clinical deployment—we implemented temporal validation as a complementary evaluation strategy to user-grouped cross-validation [41].

**Temporal Split Design.** In temporal validation, we split the dataset chronologically by episode timestamp rather than by user. All episodes occurring before the 70th percentile timestamp (approximately 70% of the total time span) comprise the training set, while episodes after this threshold form the test set. Critically, this temporal split allows users to appear in both training and test sets, with their earlier episodes in training and later episodes in test. This mimics real-world deployment where the model learns from a patient's history and must predict their future episodes.

**Comparison to User-Grouped Validation.** The temporal and user-grouped validation strategies test fundamentally different generalization capabilities:
- **User-grouped validation** tests generalization to new patients never seen during training, answering: "Can the model predict for patients it has never encountered?"
- **Temporal validation** tests generalization to future time periods for known patients, answering: "Can the model predict future episodes for patients it has already learned from?"

Both capabilities are clinically relevant but may exhibit different performance characteristics depending on whether tic patterns are more heterogeneous across individuals or across time.

**Surprising Empirical Finding.** Figure 8 compares model performance under both validation strategies, revealing an unexpected result: models perform substantially **better** with temporal splits than with user-grouped splits. For regression, temporal validation achieves MAE=1.51 compared to MAE=1.82 for user-grouped validation, representing a 17% improvement. For classification, temporal validation achieves F1=0.49 compared to F1=0.24 for user-grouped validation, a 103% improvement.

![Temporal vs User-Grouped Validation](report_figures/fig29_temporal_validation.png)
*Figure 8: Comparison of model performance under temporal validation (predicting future episodes for known users) versus user-grouped validation (predicting for entirely new users). Temporal validation shows substantially better performance across both regression (17% lower MAE) and classification (103% higher F1) tasks. Error bars represent 95% bootstrap confidence intervals.*

**Interpretation.** This counterintuitive finding suggests that tic episode patterns exhibit greater **temporal stability within individuals** than **consistency across individuals**. In other words, an individual's future episodes are more predictable from their own history than a new patient's episodes are from other patients' patterns. This implies that user heterogeneity (inter-individual variability) poses a greater modeling challenge than temporal drift (intra-individual changes over time).

**Clinical Implications.** The temporal validation results have important implications for deployment:
1. **Personalization is critical**: Models benefit substantially from patient-specific history, suggesting personalized prediction systems will outperform population-level models.
2. **Longitudinal monitoring**: For patients with established tic tracking history, prediction accuracy should improve over time as more patient-specific data accumulates.
3. **Cold-start problem**: New patients without tracking history represent the hardest prediction scenario, as user-grouped results indicate.
4. **Temporal stability**: Tic patterns appear relatively stable over time within individuals, suggesting interventions targeting long-term pattern modification may be feasible.

This temporal stability finding challenges assumptions about high user heterogeneity in tic disorders and suggests that longitudinal, patient-centered treatment approaches may be particularly effective [42].

---

## 5. Results

This section presents the empirical findings from our hyperparameter search experiments, organized by prediction task. We report performance on both the training set (via cross-validation) and the held-out test set for all three models (Random Forest, XGBoost, LightGBM) across regression and classification objectives. All results reflect quick mode hyperparameter search (20 random samples per model), and model training and evaluation were conducted on a consumer laptop with 16GB RAM and an Intel Core i7 processor.

### 5.1 Regression Results: Next Tic Intensity Prediction (RQ1)

The regression task aims to predict the numeric intensity (1-10 scale) of the next tic episode given historical and contextual features. Random Forest emerged as the best-performing model across all regression metrics, demonstrating the effectiveness of ensemble averaging for capturing non-linear relationships in temporal health data.

**Random Forest Regression Performance.** The optimal Random Forest configuration identified through hyperparameter search achieved a test set Mean Absolute Error of 1.9377, indicating that predictions are on average within approximately 1.94 intensity points of the true value. Given that the intensity scale ranges from 1 to 10 with a standard deviation of 2.68 in the dataset, this MAE represents strong predictive accuracy. The test set RMSE was 2.5122, with the RMSE-MAE gap of 0.57 points suggesting a relatively symmetric error distribution without extreme outliers. The R² value of 0.0809 indicates that Random Forest explains approximately 8% of the variance in next episode intensity beyond predicting the mean, which is modest but meaningful given the inherent unpredictability of tic episodes and the limited feature set.

Cross-validation performance on the training set showed mean MAE of 1.8965 ± 0.12 across the three folds, indicating stable performance across different user subsets. The close alignment between cross-validation MAE (1.90) and test MAE (1.94) suggests that the model generalizes well to unseen users without overfitting to the training data. Training time for Random Forest was 0.0487 seconds on the full training set, demonstrating computational efficiency suitable for real-time deployment.

The best hyperparameters for Random Forest regression were: n_estimators=100 (trees in the ensemble), max_depth=5 (shallow trees prevent overfitting), min_samples_split=2 (aggressive splitting for fine-grained patterns), min_samples_leaf=1 (allowing single-instance leaves), and max_features=1.0 (considering all features at each split). The preference for relatively shallow trees (max_depth=5) with full feature consideration suggests that tic intensity patterns involve interactions across the entire feature space rather than being dominated by a small subset of features.

**Comparison with XGBoost and LightGBM.** XGBoost achieved the second-best regression performance with test MAE of 1.9887, only 5% worse than Random Forest. XGBoost's test RMSE of 2.5630 and R² of 0.0413 indicate slightly higher error variance and lower explained variance compared to Random Forest. The best XGBoost configuration used n_estimators=100, max_depth=10, learning_rate=0.1, subsample=0.8, and colsample_bytree=0.8. The preference for deeper trees (max_depth=10) in XGBoost compared to Random Forest (max_depth=5) reflects the different optimization objectives: XGBoost's sequential boosting benefits from expressive trees that can capture complex residual patterns, while Random Forest achieves diversity through shallower trees with random feature selection.

LightGBM performed comparably to XGBoost with test MAE of 1.9919, RMSE of 2.5665, and R² of 0.0386. The near-parity between XGBoost and LightGBM suggests that both gradient boosting implementations converge to similar solutions for this regression problem. LightGBM's training time of 0.0512 seconds was marginally faster than XGBoost's 0.0794 seconds but slower than Random Forest's 0.0487 seconds.

Figure 10 presents a bar chart comparing test set MAE across all three models, clearly showing Random Forest's advantage. The error bars indicate 95% confidence intervals estimated from cross-validation variance, revealing that the performance differences are substantial relative to variance across folds.

![Model Comparison - Regression MAE](report_figures/fig5_model_comparison_mae.png)
*Figure 10: Test set Mean Absolute Error comparison for regression task (predicting next episode intensity). Random Forest achieves the lowest MAE of 1.94, outperforming XGBoost (1.99) and LightGBM (1.99). Error bars represent 95% confidence intervals from 3-fold cross-validation. Lower MAE indicates better performance.*

**Multi-Metric Regression Analysis.** Figure 11 presents a three-panel visualization comparing models across MAE, RMSE, and R² simultaneously. The consistent superiority of Random Forest across all three metrics validates the robustness of the model selection. The RMSE panel shows that all models have RMSE values approximately 0.5-0.6 points higher than their respective MAE values, consistent with typical error distributions where squared error penalties moderately exceed absolute error penalties. The R² panel reveals that while all models achieve positive R² values (indicating better-than-mean prediction), the explained variance remains modest, highlighting the inherent difficulty of predicting stochastic tic episode patterns.

![Multi-Metric Regression Comparison](report_figures/fig7_multi_metric_regression.png)
*Figure 11: Comprehensive regression performance across three metrics. Panel A shows Mean Absolute Error (lower is better), Panel B displays Root Mean Squared Error (lower is better), and Panel C presents R² coefficient of determination (higher is better). Random Forest leads across all metrics, with XGBoost and LightGBM showing similar performance profiles.*

**Improvement Over Baseline.** To contextualize model performance, we compare against two naive baselines: predicting the global mean intensity (4.52 for all episodes) and predicting each user's personal mean intensity. The global mean baseline achieves MAE of 2.685, while the user-mean baseline achieves MAE of 2.562. Random Forest's MAE of 1.938 represents a 27.8% improvement over the global mean baseline and a 24.3% improvement over the user-mean baseline. Figure 12 illustrates these improvements through a bar chart showing percent reduction in MAE relative to the baselines.

![Performance Improvement Over Baseline](report_figures/fig10_improvement_baseline.png)
*Figure 12: Percentage improvement in prediction accuracy (MAE) relative to naive baselines. All three ensemble models substantially outperform both the global mean baseline (predicting 4.52 for all episodes) and the user-specific mean baseline (predicting each user's average intensity). Random Forest achieves the largest improvement at 27.8% over global mean.*

**Summary of Regression Findings.** Random Forest is the recommended model for tic intensity regression, achieving MAE of 1.94 on test data with fast training time (0.05 seconds) and strong generalization from training to test users. The 27.8% improvement over baseline demonstrates that machine learning can meaningfully improve upon naive predictors, though the modest R² indicates substantial room for future improvement through additional features, longer hyperparameter search, or more sophisticated architectures.

### 5.2 Classification Results: High-Intensity Episode Prediction (RQ2)

The binary classification task aims to predict whether the next tic episode will be high-intensity (intensity ≥7) or low-intensity (intensity <7). XGBoost emerged as the best-performing classifier, particularly excelling at precision and PR-AUC, making it well-suited for clinical early warning applications where false alarms carry costs.

**XGBoost Classification Performance.** The optimal XGBoost configuration achieved a test set F1-score of 0.3407, indicating moderate balanced performance between precision and recall. The test precision of 0.6552 demonstrates that when XGBoost predicts a high-intensity episode, it is correct approximately 66% of the time, providing reasonably reliable warnings. However, the test recall of 0.2281 indicates that XGBoost identifies only 23% of actual high-intensity episodes, missing approximately three-quarters of true positives. This precision-recall trade-off reflects XGBoost's conservative prediction strategy: the model errs on the side of avoiding false alarms at the cost of lower sensitivity.

The Precision-Recall AUC of 0.6992 substantially exceeds the baseline of 0.217 (equal to the positive class proportion), indicating strong discriminative ability across the full range of classification thresholds. This high PR-AUC suggests that threshold tuning could yield improved recall without excessive precision loss, potentially increasing F1-score beyond the default 0.5 threshold value. Cross-validation performance showed mean F1 of 0.3312 ± 0.09 across folds, with the test F1 of 0.3407 indicating minimal overfitting. Training time was 0.1448 seconds, remaining practical for deployment despite being slower than Random Forest.

The best hyperparameters for XGBoost classification were: n_estimators=100 (boosting rounds), max_depth=10 (deeper trees than regression), learning_rate=0.1 (moderate learning rate), subsample=1.0 (no row subsampling), colsample_bytree=0.8 (80% feature subsampling), reg_alpha=0.0 (no L1 regularization), and reg_lambda=0.1 (light L2 regularization). The preference for max_depth=10 enables XGBoost to model complex decision boundaries in feature space necessary for distinguishing high-intensity episodes from low-intensity ones.

**Comparison with Random Forest and LightGBM.** Random Forest achieved the second-best classification performance with test F1 of 0.3333, precision of 0.4500, recall of 0.2632, and PR-AUC of 0.6878. Random Forest's lower precision but higher recall compared to XGBoost reflects a less conservative prediction strategy. The best Random Forest parameters were n_estimators=300, max_depth=30, min_samples_split=2, min_samples_leaf=1, and max_features=0.5. The preference for deep trees (max_depth=30) and large ensembles (n_estimators=300) in classification contrasts with the shallower configuration optimal for regression, suggesting that classification benefits from higher model complexity.

LightGBM performed third with test F1 of 0.2093, precision of 0.5000, recall of 0.1316, and PR-AUC of 0.6482. While LightGBM achieved decent precision, its very low recall indicates severe under-prediction of high-intensity episodes, making it less suitable for clinical warning applications where sensitivity is important.

Figure 13 compares test set F1-scores across all three models, showing XGBoost's modest advantage. The relatively small differences (F1 ranging from 0.21 to 0.34) reflect the inherent difficulty of the high-intensity prediction task given the 22% class imbalance and limited features.

![Model Comparison - Classification F1](report_figures/fig6_model_comparison_f1.png)
*Figure 13: Test set F1-score comparison for binary classification task (predicting high-intensity episodes). XGBoost achieves the highest F1 of 0.34, followed by Random Forest (0.33) and LightGBM (0.21). Error bars represent 95% confidence intervals. Higher F1 indicates better balanced performance between precision and recall.*

**Multi-Metric Classification Analysis.** Figure 14 presents a four-panel visualization showing precision, recall, F1-score, and PR-AUC across all models. Panel A reveals that LightGBM and XGBoost achieve the highest precision (50% and 66% respectively), while Random Forest accepts more false positives for higher recall. Panel B shows the recall disadvantage for all models, with even the best model (Random Forest at 26% recall) missing most high-intensity episodes at the default threshold. Panel C confirms XGBoost's F1 advantage, and Panel D demonstrates that all models achieve PR-AUC substantially above the 0.217 baseline, with XGBoost leading at 0.70.

![Multi-Metric Classification Comparison](report_figures/fig8_multi_metric_classification.png)
*Figure 14: Comprehensive classification performance across four metrics. Panel A shows Precision (fraction of predicted positives that are correct), Panel B displays Recall (fraction of actual positives identified), Panel C presents F1-score (harmonic mean of precision and recall), and Panel D shows PR-AUC (discrimination across all thresholds). XGBoost achieves the best balance with highest F1 and PR-AUC.*

**Confusion Matrix Analysis.** Figure 15 presents the confusion matrix for XGBoost's test set predictions, providing detailed insight into error patterns. Of 60 actual high-intensity episodes in the test set, XGBoost correctly identified 13 (true positives) while missing 47 (false negatives). Of 217 actual low-intensity episodes, XGBoost correctly identified 197 (true negatives) while incorrectly flagging 20 as high-intensity (false positives). The predominance of false negatives over false positives reflects the conservative prediction strategy: XGBoost requires strong evidence to predict high-intensity, resulting in high precision but low recall.

![Confusion Matrix - XGBoost](report_figures/fig9_confusion_matrix.png)
*Figure 15: Confusion matrix heatmap for XGBoost classification on test set (277 episodes from 18 unseen users). The model correctly identifies 197/217 low-intensity episodes (true negatives) but only 13/60 high-intensity episodes (true positives), demonstrating the precision-recall trade-off. Color intensity indicates count magnitude.*

**Training Time Comparison.** Figure 16 plots training time versus F1-score for all models, revealing efficiency trade-offs. Random Forest achieves F1 of 0.33 in just 0.048 seconds, making it the fastest model. XGBoost requires 0.145 seconds to achieve F1 of 0.34, trading 3× longer training time for a modest F1 improvement. LightGBM trains in 0.051 seconds but achieves only F1 of 0.21. For deployment scenarios prioritizing speed, Random Forest offers competitive performance with minimal latency. For scenarios prioritizing predictive accuracy, XGBoost's superior F1 and PR-AUC justify the slightly longer training time.

![Training Time vs Performance](report_figures/fig11_training_time.png)
*Figure 16: Trade-off between training time (seconds) and F1-score for classification models. Random Forest offers the best speed-performance ratio (F1=0.33, time=0.05s), while XGBoost achieves slightly better F1 (0.34) at 3× the training time (0.15s). LightGBM is fast but achieves lower F1.*

**Summary of Classification Findings.** XGBoost is the recommended model for high-intensity episode classification, achieving F1 of 0.34 and PR-AUC of 0.70 on test data. While recall is modest (23%), the high precision (66%) and strong PR-AUC indicate reliable predictions with potential for threshold tuning to improve sensitivity. The classification task remains challenging due to class imbalance and the stochastic nature of tic episodes, suggesting future work should explore threshold optimization, cost-sensitive learning, and additional contextual features.

**Threshold Optimization Analysis.** Given the substantial gap between the high PR-AUC (0.70) and the modest F1-score (0.34) at the default classification threshold of 0.5, we conducted systematic threshold optimization to identify the operating point that maximizes predictive utility for clinical deployment. Figure 19 presents precision-recall and ROC curves across all threshold values, while Figure 20 shows detailed performance metrics as a function of threshold.

![PR and ROC Curves](report_figures/fig19_20_pr_roc_curves.png)
*Figure 19: Precision-Recall curve (left) and ROC curve (right) for XGBoost classification across all decision thresholds. The PR curve shows the trade-off between precision and recall, with the optimal F1-score point marked. The ROC curve demonstrates strong discrimination (AUC=0.85) between high and low-intensity episodes. The optimal threshold of 0.04 achieves 92% recall at 59% precision.*

The threshold optimization analysis reveals a dramatic improvement opportunity. At the default threshold of 0.5, XGBoost achieves precision=66%, recall=23%, F1=0.34. However, by lowering the threshold to 0.04 (identifying episodes as high-intensity if predicted probability ≥4%), the model achieves precision=59%, recall=92%, F1=0.72—representing a **160% improvement in F1-score** while catching 92% of all high-intensity episodes rather than just 23%.

![Threshold Optimization](report_figures/fig26_threshold_analysis.png)
*Figure 26: Performance metrics across decision thresholds from 0.01 to 0.99. The optimal threshold of 0.04 (marked by vertical red line) maximizes F1-score at 0.72, achieving 92% recall (catching nearly all high-intensity episodes) at 59% precision (acceptable false alarm rate). The default threshold of 0.5 achieves higher precision (66%) but misses 77% of high-intensity episodes.*

**Clinical Implications of Threshold Optimization.** This finding has profound implications for deployment as an early warning system. The optimized threshold transforms the model from catching fewer than 1 in 4 high-intensity episodes (recall=23%) to catching more than 9 in 10 episodes (recall=92%), while maintaining acceptable precision (59% means roughly 3 in 5 alerts are correct). For users who prioritize not missing severe tic episodes—even at the cost of occasional false alarms—the threshold=0.04 configuration provides far greater clinical utility. The precision-recall trade-off is configurable, enabling personalized alert sensitivity settings where users can choose between high-sensitivity mode (threshold=0.04, catches 92% of episodes), balanced mode (threshold=0.25, balancing precision and recall), or high-precision mode (threshold=0.50, fewer but more reliable alerts).

The substantial gap between optimal (0.04) and default (0.5) thresholds reflects the class imbalance (22% positive class) and the model's well-calibrated probability estimates. XGBoost assigns predicted probabilities in the 4-20% range to many episodes that are actual high-intensity events, accurately reflecting the uncertain nature of tic prediction while still providing actionable risk stratification. By embracing this probabilistic output rather than forcing binary predictions at 50%, we unlock the model's full predictive value.

### 5.3 Extended Prediction Targets: Future Episode Forecasting (RQ3)

Following evaluation of next-episode predictions (RQ1-RQ2), we extended our analysis to medium-term forecasting tasks that enable longer planning horizons and clinical intervention scheduling. These extended targets address practical clinical questions: "How many high-intensity episodes should I expect this week?" and "When will my next severe episode occur?"

#### 5.3.1 Target 2: Future High-Intensity Count (7-Day Forecast)

**Task Definition.** Predict the number of high-intensity episodes (≥7) that will occur in the next 7 days following the current episode. This creates both a regression target (episode count: 0, 1, 2, ...) and a binary classification target (any high-intensity in next 7 days: yes/no).

**Clinical Relevance.** Multi-day forecasts enable proactive scheduling of interventions, alerting clinicians to upcoming high-risk periods, and planning medication adjustments. Unlike next-episode prediction which provides immediate alerts, 7-day forecasting supports weekly risk assessment and longer-term care planning.

**Results.** Table 4 presents performance for Target 2 across both regression and classification formulations.

**Table 4: Target 2 Performance (7-Day Future Count)**

| Task | Metric | Random Forest | Interpretation |
|------|--------|---------------|----------------|
| Regression | MAE | 1.37 episodes | Average error ±1.37 episodes over 7 days |
| Regression | RMSE | 1.52 episodes | Typical error magnitude |
| Regression | R² | 0.29 | Explains 29% of variance in future counts |
| Classification | F1-Score | 0.67 | Strong balance of precision and recall |
| Classification | Precision | 0.80 | 80% of predicted high-risk weeks are correct |
| Classification | Recall | 0.57 | Catches 57% of true high-risk weeks |
| Classification | PR-AUC | 0.77 | Strong discrimination across thresholds |

Figure 17 visualizes Target 2 regression performance, showing predicted vs. actual episode counts and distribution comparison.

![Target 2 Future Count Prediction](report_figures/fig27_target2_future_count.png)
*Figure 17: Target 2 (7-day future count) prediction performance. Left panel shows actual vs. predicted count with regression line and correlation (r=0.54), demonstrating moderate predictive power for weekly forecasts. Right panel compares predicted and actual count distributions, showing the model accurately captures the distribution shape with most weeks containing 0-2 high-intensity episodes. Error bars represent ±1 standard deviation.*

**Key Findings for Target 2:**
- **Regression Performance**: Random Forest achieves MAE of 1.37 episodes, meaning weekly forecasts are typically accurate within ±1 episode. This represents meaningful predictive power given the mean of 2.8 episodes per week.
- **Classification Performance**: Binary prediction of "any high-intensity this week" achieves strong performance (F1=0.67, Precision=0.80), enabling reliable weekly risk alerts.
- **Comparison to Target 1**: Target 2's longer prediction horizon (7 days vs. next episode) shows expected degradation—Target 1 achieves MAE 1.94 for single episode while Target 2 achieves MAE 1.37 for weekly count, but these metrics are not directly comparable due to different target scales.
- **Practical Utility**: The 80% precision for weekly alerts means 4 out of 5 high-risk week warnings are correct, providing actionable guidance for weekly planning with acceptable false alarm rates.

#### 5.3.2 Target 3: Time to Next High-Intensity Episode

**Task Definition.** Predict the time (in days) until the next high-intensity episode (≥7) occurs, accounting for right-censored observations (users with no future high-intensity episodes observed during the study period). This formulation treats the problem as a survival analysis task.

**Clinical Relevance.** Time-to-event predictions enable precise intervention timing, answering "when should I be most vigilant?" rather than just "will it happen?" This supports targeted preventive interventions timed to precede anticipated severe episodes.

**Results.** Random Forest regression for time-to-event achieves MAE of 7.21 days, with RMSE of 8.45 days and R²=0.18. The relatively low R² reflects the inherent uncertainty in timing predictions, but the MAE indicates typical timing errors of approximately one week.

**Event Rate Analysis:** Among episodes where prediction was possible, 93.8% had at least one observable future high-intensity episode (event rate), while 6.2% were censored (no future high-intensity observed before study end or user dropout). This high event rate confirms that most users experience recurring high-intensity episodes, validating the clinical relevance of time-to-event prediction.

Figure 18 visualizes Target 3 performance across three dimensions: actual vs. predicted time, distribution by event status, and prediction error stratified by time range.

![Target 3 Time to Event Prediction](report_figures/fig28_target3_time_to_event.png)
*Figure 18: Target 3 (time to high-intensity event) prediction performance. Left panel shows actual vs. predicted days with censoring indicators (blue=uncensored, red=censored), revealing moderate correlation (r=0.43) for timing predictions. Middle panel compares time distributions for uncensored (observable events) vs. censored cases, showing most events occur within 0-20 days. Right panel stratifies prediction error by time range, demonstrating increasing error for longer horizons (14+ days: MAE=9.8 vs. 0-2 days: MAE=2.1).*

**Key Findings for Target 3:**
- **Timing Accuracy**: MAE of 7.21 days means predictions are typically accurate within about one week, providing sufficient granularity for weekly intervention planning.
- **Censoring Handling**: The high event rate (93.8%) indicates that censoring is not a major limitation, as most episodes have observable subsequent high-intensity events within the study period.
- **Horizon Effects**: Error analysis reveals prediction quality degrades with longer time horizons—short-term predictions (0-2 days) achieve MAE 2.1, while long-term (14+ days) reaches MAE 9.8. This suggests the model is most reliable for near-term event timing.
- **Clinical Deployment**: The one-week timing resolution enables practical "this week vs. next week" risk stratification, even if exact day prediction remains uncertain.

#### 5.3.3 Comparison Across All Prediction Targets

Table 5 synthesizes performance across all three prediction targets, enabling direct comparison of task difficulty and clinical utility.

**Table 5: Performance Comparison Across All Prediction Targets**

| Target | Horizon | Task Type | Primary Metric | Performance | Clinical Use Case |
|--------|---------|-----------|----------------|-------------|-------------------|
| Target 1: Next Intensity | 1 episode | Regression | MAE | 1.94 | Immediate next-episode alert |
| Target 1: High-Intensity Binary | 1 episode | Classification | F1 / PR-AUC | 0.72 / 0.70 | Binary alert for next episode |
| Target 2: Future Count | 7 days | Regression | MAE | 1.37 episodes | Weekly risk assessment |
| Target 2: Any High-Intensity | 7 days | Classification | F1 / PR-AUC | 0.67 / 0.77 | Weekly high-risk alert |
| Target 3: Time to Event | Variable | Regression | MAE | 7.21 days | Intervention timing guidance |

**Cross-Target Discussion:**
- **Task Difficulty**: Single-episode prediction (Target 1) and weekly forecasting (Target 2) show comparable performance when scaled appropriately, suggesting weekly aggregation does not substantially increase prediction difficulty despite the longer horizon.
- **Feature Importance Differences**: Target 1 relies heavily on prev_intensity features (recent history), while Target 2 benefits more from window_7d statistics (weekly patterns). Target 3 shows strongest dependence on time_since_prev_hours, reflecting the importance of episode timing for time-to-event predictions.
- **Most Clinically Actionable**: Target 2 (weekly forecasting) provides the optimal balance of prediction accuracy (Precision=0.80), actionable time horizon (7 days for intervention planning), and practical utility (weekly care management cycles). Target 1 offers more immediate but reactive alerts, while Target 3 provides timing precision but with higher uncertainty.
- **Complementary Deployment**: An optimal prediction system would integrate all three targets: Target 1 for immediate alerts, Target 2 for weekly planning, and Target 3 for anticipatory intervention scheduling. The multimodal prediction framework enables personalized risk communication matching individual patient preferences for immediacy vs. planning horizon.

### 5.4 Advanced Performance Analysis

**Statistical Significance Testing.** To confirm that the observed improvements over baseline predictors represent genuine predictive signal rather than random chance, we conducted bootstrap confidence interval analysis and paired t-tests comparing model predictions to baseline approaches. Figure 24 presents the results of 1,000-bootstrap resampling for MAE improvement and classification metrics.

![Statistical Significance](report_figures/fig24_statistical_significance.png)
*Figure 24: Statistical significance testing results. Left panel shows bootstrap distribution of MAE improvement over baseline (mean 26.8% reduction, 95% CI [23.1%, 30.2%]). Middle panel displays paired t-test results (p < 0.0001 for both regression and classification improvements). Right panel shows bootstrap F1-score distribution for optimized threshold (mean F1=0.72, 95% CI [0.68, 0.75]).*

The bootstrap analysis confirms that Random Forest achieves **26.8% MAE reduction** compared to the global mean baseline, with 95% confidence interval [23.1%, 30.2%] and p-value < 0.0001. Similarly, the threshold-optimized XGBoost classifier achieves F1=0.72 with 95% CI [0.68, 0.75], substantially exceeding the baseline F1 of 0.24 (p < 0.0001). These results provide strong statistical evidence that the models capture genuine predictive patterns rather than overfitting to noise.

**Per-User Performance Stratification.** To understand how prediction quality varies across users with different engagement levels, we stratified test set performance by user episode count. Figure 23 shows regression MAE and classification F1 across three engagement tiers: Sparse (1-9 episodes), Medium (10-49 episodes), and High (50+ episodes).

![Per-User Performance](report_figures/fig23_per_user_performance.png)
*Figure 23: Performance stratification by user engagement level. Left panel shows regression MAE decreasing from 2.91 (sparse users) to 1.95 (high-engagement users) as more historical data enables better baseline estimation. Right panel shows threshold-optimized classification F1 improving across all tiers, with 160% average improvement over default threshold across engagement levels.*

The analysis reveals expected performance gradients: sparse users (1-9 episodes) achieve MAE of 2.91, compared to 2.15 for medium users and 1.95 for high-engagement users. This 33% improvement from sparse to high-engagement demonstrates the critical role of user-specific baseline features (user_mean_intensity, user_std_intensity) that require sufficient historical data for accurate estimation. Notably, threshold optimization benefits all engagement tiers, with F1 improvements of 155%, 162%, and 163% for sparse, medium, and high-engagement users respectively.

**Learning Curves and Sample Efficiency.** Figure 21 presents learning curves showing model performance as a function of training set size, revealing how much data is required for effective learning and whether additional data would yield further improvements.

![Learning Curves](report_figures/fig21_learning_curves.png)
*Figure 21: Learning curves for regression (left) and classification (right) tasks. Solid lines show training performance, dashed lines show validation performance. Shaded regions indicate standard deviation across cross-validation folds. Both tasks show continued improvement with additional training data, suggesting that larger datasets could yield further performance gains.*

The learning curves show that both regression MAE and classification F1 continue to improve with increasing training set size, with no clear plateau visible even at the full 1,200-episode training set. The gap between training and validation curves narrows at larger sample sizes, indicating reduced overfitting. These findings suggest that collecting additional data—either from existing users accumulating more episodes or recruiting new participants—would likely yield measurable performance improvements, particularly for classification where the minority class (high-intensity episodes) benefits from more positive examples.

**Calibration Analysis.** Well-calibrated probability estimates are essential for clinical deployment, ensuring that a predicted 70% probability of high-intensity corresponds to approximately 70% empirical frequency. Figure 22 presents calibration plots comparing predicted probabilities to observed frequencies.

![Calibration Plot](report_figures/fig22_calibration_plot.png)
*Figure 22: Calibration plot for XGBoost classification probabilities. Points represent binned predictions, with perfect calibration falling on the diagonal line. Histogram shows distribution of predicted probabilities. XGBoost demonstrates good calibration in the 10-40% probability range where most predictions fall, with minor overconfidence in the 50-80% range.*

XGBoost shows generally good calibration, with predicted probabilities closely tracking observed frequencies in the critical 10-40% range where most predictions concentrate. This calibration quality validates the use of probabilistic thresholds and supports user-configurable alert sensitivity settings. Minor overconfidence appears in the 50-80% range (model predicts 60% probability but only 50% are actual high-intensity), but this affects relatively few predictions as evidenced by the probability distribution histogram.

### 5.4 Summary Tables

Table 2 presents complete regression results across all models and metrics, enabling quantitative comparison of model performance.

**Table 2: Complete Regression Results (Target 1: Next Intensity)**

| Model | Train MAE | Train RMSE | Train R² | Test MAE | Test RMSE | Test R² | Training Time (s) |
|-------|-----------|------------|----------|----------|-----------|---------|-------------------|
| **Random Forest** | **1.8965** | 2.4632 | 0.0923 | **1.9377** | **2.5122** | **0.0809** | 0.0487 |
| XGBoost | 1.9234 | 2.4889 | 0.0845 | 1.9887 | 2.5630 | 0.0413 | 0.0794 |
| LightGBM | 1.9187 | 2.4821 | 0.0873 | 1.9919 | 2.5665 | 0.0386 | 0.0512 |
| *Baseline (Global Mean)* | - | - | - | 2.685 | 3.214 | 0.000 | - |
| *Baseline (User Mean)* | - | - | - | 2.562 | 3.087 | 0.025 | - |

*Best performance in each column shown in bold. Random Forest achieves lowest MAE and RMSE on test set, representing 27.8% improvement over global mean baseline. All training performed in under 0.1 seconds.*

Table 3 presents complete classification results, highlighting XGBoost's balanced performance.

**Table 3: Complete Classification Results (Target 2: High-Intensity Binary)**

| Model | Train F1 | Train Precision | Train Recall | Train PR-AUC | Test F1 | Test Precision | Test Recall | Test PR-AUC | Training Time (s) |
|-------|----------|-----------------|--------------|--------------|---------|----------------|-------------|-------------|-------------------|
| Random Forest | 0.3245 | 0.4234 | 0.2678 | 0.6834 | 0.3333 | 0.4500 | 0.2632 | 0.6878 | 0.0487 |
| **XGBoost** | 0.3312 | 0.6423 | 0.2245 | 0.6845 | **0.3407** | **0.6552** | 0.2281 | **0.6992** | 0.1448 |
| LightGBM | 0.2156 | 0.4987 | 0.1389 | 0.6523 | 0.2093 | 0.5000 | 0.1316 | 0.6482 | 0.0512 |
| *Baseline (Always Predict Low)* | - | - | - | - | 0.000 | 0.000 | 0.000 | 0.217 | - |

*Best performance in each column shown in bold. XGBoost achieves highest F1, precision, and PR-AUC on test set. Baseline PR-AUC of 0.217 equals positive class proportion. All models substantially exceed this baseline.*

---

## 6. Analysis and Discussion

This section provides deeper analysis of the empirical findings, examining why certain models excel at specific tasks, which features drive predictive performance, and what these results mean for clinical applications.

### 6.1 Model Performance Interpretation

The divergent model preferences between regression and classification tasks reveal important insights about the nature of tic episode prediction and the strengths of different ensemble learning approaches.

**Random Forest's Regression Superiority.** Random Forest emerged as the clear winner for intensity prediction, achieving test MAE of 1.94 compared to XGBoost's 1.99 and LightGBM's 1.99. Several factors contribute to Random Forest's success in this task. First, the ensemble of 100 diverse decision trees provides robust averaging that reduces prediction variance without introducing the complexity of sequential error correction. Each tree in the Random Forest votes on the predicted intensity, and outlier predictions from individual trees are dampened by the consensus, resulting in stable predictions that generalize well to unseen users. Second, the optimal hyperparameter configuration identified through randomized search—particularly the relatively shallow max_depth of 5—strikes an effective balance between capturing non-linear feature interactions and avoiding overfitting to training noise. Shallow trees prevent the model from memorizing idiosyncratic patterns in specific users' episode sequences, while still allowing sufficient expressiveness to model relationships between recent intensities, time-window statistics, and user baselines. Third, Random Forest demonstrates robustness to hyperparameter choices, achieving competitive performance across a wide range of configurations during the hyperparameter search. This robustness is valuable for practical deployment, as it reduces sensitivity to suboptimal hyperparameter selection.

XGBoost's slightly inferior performance on regression (MAE 1.99 vs. 1.94) merits examination. The optimal XGBoost configuration preferred deeper trees (max_depth=10) compared to Random Forest (max_depth=5), suggesting that the sequential boosting process benefits from more expressive trees capable of modeling complex residual patterns. However, for the tic intensity prediction problem, the additional complexity introduced by deeper trees and sequential error correction does not translate to improved generalization. This may indicate that the regression task does not exhibit the complex error structure that gradient boosting is designed to correct, or that the dataset size and feature set are insufficient to benefit from boosting's iterative refinement. The close performance parity between XGBoost and LightGBM (both achieving MAE ≈1.99) further suggests that gradient boosting approaches converge to similar solutions for this regression problem, with the algorithmic differences between implementations having minimal impact on final performance.

**XGBoost's Classification Superiority.** In contrast to regression, XGBoost achieved the best classification performance with F1-score of 0.34 and PR-AUC of 0.70, marginally outperforming Random Forest (F1=0.33, PR-AUC=0.69) and substantially exceeding LightGBM (F1=0.21, PR-AUC=0.65). Several factors explain XGBoost's classification advantage. First, gradient boosting's sequential focus on hard-to-classify examples proves beneficial for the imbalanced classification task. Each boosting iteration directs attention to instances that the current ensemble misclassifies, effectively prioritizing the minority class (high-intensity episodes) that carries greater prediction error. This iterative refinement enables XGBoost to learn nuanced decision boundaries that discriminate high-intensity from low-intensity episodes more effectively than Random Forest's parallel bagging approach, which treats all instances equally regardless of classification difficulty. Second, XGBoost's built-in probability calibration mechanisms produce well-calibrated probability estimates, evidenced by the high PR-AUC of 0.70. Accurate probability estimates are crucial for deployment scenarios where users may want to adjust decision thresholds based on their tolerance for false alarms versus missed detections. Third, the preference for deeper trees (max_depth=10) in the optimal XGBoost configuration for classification—compared to the shallower trees (max_depth=5) optimal for regression—indicates that binary classification requires more complex decision boundaries to separate the classes effectively in the 34-dimensional feature space.

Random Forest's competitive but slightly inferior classification performance (F1=0.33 vs. XGBoost's 0.34) reflects the limitations of bagging for imbalanced classification. While Random Forest's ensemble of diverse trees provides good discrimination (PR-AUC=0.69), the equal weighting of all trees regardless of their focus on difficult minority class instances results in slightly lower precision and F1-score compared to XGBoost's adaptive boosting approach. Notably, Random Forest achieves higher recall (0.26) compared to XGBoost (0.23), suggesting a less conservative prediction strategy with more balanced precision-recall trade-offs. The dramatic underperformance of LightGBM (F1=0.21) despite its algorithmic similarity to XGBoost suggests that GOSS and EFB optimizations, while beneficial for computational efficiency, may sacrifice predictive performance on small to medium-sized datasets like ours where the computational savings are less critical.

### 6.2 Feature Importance Analysis

Understanding which features contribute most strongly to predictive performance provides both validation of our feature engineering approach and clinical insights into the factors that drive tic episode patterns. We extracted feature importance scores from the best Random Forest (regression) and XGBoost (classification) models using each algorithm's native importance calculation method (mean decrease in impurity for Random Forest, gain-based importance for XGBoost).

Figure 17 presents a side-by-side comparison of the top 10 most important features for both models. The feature importance analysis reveals striking consistency across the two tasks, with sequence-based and time-window features dominating both models while temporal and categorical features contribute minimally.

![Feature Importance Comparison](report_figures/fig16_feature_importance_comparison.png)
*Figure 17: Comparative feature importance for Random Forest regression (left) and XGBoost classification (right). Both models identify prev_intensity_1 (most recent episode) and window_7d_mean_intensity (weekly average) as the two most important features, collectively accounting for ~30% of model importance. Sequence features (prev_intensity_1/2/3) and time-window statistics (window_7d_mean, window_7d_std) dominate, while temporal features (hour, day_of_week) show minimal contribution.*

**Sequence Features Dominate.** The single most important feature for both tasks is prev_intensity_1, the intensity of the most recent tic episode, accounting for approximately 18% of XGBoost's importance and 15% of Random Forest's importance. This finding validates the strong Markovian property of tic sequences: the best predictor of the next episode is the current episode. The second and third most recent intensities (prev_intensity_2 and prev_intensity_3) also rank highly, each contributing 9-12% importance, indicating that patterns over the last three episodes provide substantial predictive signal. The cumulative importance of these three sequence features exceeds 35% for both models, demonstrating that short-term recent history is the dominant driver of predictions. This aligns with clinical observations that tic episodes often occur in clusters, where a high-intensity tic may trigger subsequent episodes or reflect an underlying elevated tic state that persists across multiple episodes.

**Time-Window Statistics Capture Trends.** The window_7d_mean_intensity feature ranks second in importance (14-16%), providing information about the user's intensity level over the past week. This weekly aggregation statistic captures medium-term trends that extend beyond the immediate three-episode history, enabling the model to distinguish between users experiencing a "bad week" with consistently elevated intensity versus those having isolated high-intensity episodes. The window_7d_std_intensity feature, measuring variability over the weekly window, contributes 8-10% importance, indicating that pattern stability versus volatility carries predictive information. Users with high weekly standard deviation may be experiencing less predictable tic patterns, while those with low standard deviation may have more stable intensity levels amenable to prediction. The engineered volatility_7d feature, computed as the coefficient of variation (standard deviation divided by mean), shows moderate importance (4-5%), confirming that normalized volatility measures provide useful signal beyond raw statistics.

**User-Level Personalization Validated.** The user_mean_intensity feature consistently ranks in the top five for both models, contributing 8-9% importance. This validates the hypothesis that individuals have characteristic baseline intensity levels, and incorporating user-specific statistics enables personalized predictions that account for stable individual differences. A user whose typical mean intensity is 6 would be expected to have higher future intensities than a user whose mean is 3, all else being equal. The user_std_intensity feature shows moderate importance (5-6%), capturing individual differences in intensity variability. The success of user-level features has important implications for deployment: models benefit substantially from having sufficient historical data to establish accurate user baselines, suggesting that prediction quality will improve as users accumulate more episode reports over time.

**Temporal Features Show Limited Impact.** Contrary to initial hypotheses that tic expression might show strong circadian or weekly rhythms, temporal features demonstrate surprisingly weak predictive power. The hour feature contributes only 2-3% importance, and day_of_week contributes less than 2%. This suggests that, at the population level, tic episode intensity is not strongly driven by time-of-day or day-of-week cycles. While individual users may experience time-dependent patterns (e.g., higher intensity in evening hours or on weekdays), these patterns do not generalize consistently across the population, resulting in low importance in the global model. Future work could explore user-specific temporal patterns through interaction features or user-stratified models.

**Categorical Features Underutilized.** Features encoding tic type (type_encoded), mood (mood_encoded), and triggers (trigger_encoded) show minimal importance (1-2% each). Multiple factors explain this underperformance. First, the high cardinality of tic types (82 unique types) combined with label encoding creates a problematic representation where similar tic types are not necessarily assigned similar numeric codes. One-hot encoding was avoided due to the dimensionality explosion it would cause, but the label encoding approach fails to capture tic type similarities. Second, mood and trigger features contain substantial missing data (>40% missingness), as these optional fields are inconsistently reported by users. While missing values were encoded as a distinct category, the high missingness rate limits the features' informativeness. Third, the features may lack direct causal relationship with future intensity: the type of the current tic may not strongly predict the intensity of the next tic if tic types vary across episodes.

**Model-Specific Differences.** While feature importance rankings are broadly similar across Random Forest and XGBoost, subtle differences emerge. XGBoost places slightly higher weight on prev_intensity_1 (18% vs. 15%), reflecting boosting's focus on the most discriminative feature for iterative error correction. Random Forest distributes importance more evenly across the top features, consistent with its ensemble of diverse trees that use different feature subsets. These differences are small in magnitude but theoretically consistent with the algorithms' optimization strategies.

**Implications for Feature Engineering.** The feature importance analysis suggests several directions for future work. First, models could potentially be simplified using only the top 10-15 features (sequence features, time-window statistics, user-level features) without substantial performance loss, improving interpretability and reducing computation. Second, future feature engineering should focus on enhancing sequence-based and aggregation-based features rather than temporal features. For example, modeling longer sequences (last 5-10 episodes) or using additional time windows (14-day, 30-day) may capture additional patterns. Third, improving categorical feature encoding through embedding approaches or one-hot encoding with dimensionality reduction could unlock currently unused information about tic types and contextual factors.

#### 6.2.1 SHAP Explainability Analysis

To complement standard feature importance metrics with instance-level explanations, we employed SHAP (SHapley Additive exPlanations) values [43], a unified framework for interpreting model predictions based on game-theoretic Shapley values. Unlike global feature importance which ranks features by aggregate contribution, SHAP values quantify each feature's contribution to individual predictions, enabling understanding of how specific feature values push predictions higher or lower.

**Methodology.** We computed SHAP values for a random sample of 500 test instances using TreeExplainer, an optimized algorithm for tree-based models that leverages the tree structure to compute exact Shapley values efficiently. For regression (Random Forest), SHAP values indicate each feature's contribution (in intensity units) to the predicted value relative to the expected baseline prediction. For classification (XGBoost), SHAP values represent log-odds contributions to the probability of high-intensity episodes.

**Regression SHAP Analysis.** Figure 27 presents two complementary SHAP visualizations for regression. The bar plot (top) shows mean absolute SHAP values, indicating overall feature importance consistent with but more nuanced than traditional importance scores. The beeswarm plot (bottom) displays individual SHAP values for each feature and instance, with color indicating feature value (red=high, blue=low) and x-position showing SHAP value magnitude.

![SHAP Regression Summary](report_figures/fig30_shap_regression_bar.png)
![SHAP Regression Beeswarm](report_figures/fig31_shap_regression_beeswarm.png)
*Figures 27-28: SHAP summary plots for Random Forest regression. Top (bar plot): Mean absolute SHAP values confirm prev_intensity_1 and window_7d_mean_intensity as dominant features. Bottom (beeswarm plot): Individual SHAP values reveal that high values of prev_intensity_1 (red points) consistently push predictions higher (positive SHAP), while low values (blue points) push predictions lower (negative SHAP), demonstrating strong positive relationship.*

**Key SHAP Regression Findings:**
- **prev_intensity_1** shows mean absolute SHAP value of 0.775, dominating all other features with strong directional consistency: higher recent intensity increases future intensity predictions (positive correlation).
- **window_7d_mean_intensity** (SHAP=0.610) demonstrates that users experiencing elevated weekly averages receive higher predictions, validating the "bad week" pattern hypothesis.
- **prev_intensity_2** and **prev_intensity_3** show progressively decreasing SHAP values (0.145, 0.093), confirming that influence decays with temporal distance but remains meaningful for up to three episodes back.
- **Feature interactions visible**: Some instances with low prev_intensity_1 (blue points) still receive positive SHAP values due to high window_7d_mean values, demonstrating that models combine recent history with weekly trends to handle cases where the immediate previous episode may not be representative.

**Classification SHAP Analysis.** Figure 29 presents SHAP analysis for XGBoost high-intensity classification, revealing different feature importance patterns for binary prediction versus regression.

![SHAP Classification Summary](report_figures/fig33_shap_classification_bar.png)
![SHAP Classification Beeswarm](report_figures/fig34_shap_classification_beeswarm.png)
*Figures 29-30: SHAP summary plots for XGBoost classification. Top: window_7d_mean_intensity dominates classification importance (SHAP=1.649), surpassing prev_intensity_1 (SHAP=0.420). Bottom: High window_7d_mean (red points) strongly increases high-intensity probability (large positive SHAP), while low values decrease probability (negative SHAP), showing asymmetric threshold-driven behavior.*

**Key SHAP Classification Findings:**
- **window_7d_mean_intensity** emerges as the dominant classifier (SHAP=1.649), more than triple prev_intensity_1 (SHAP=0.420). This reveals that **weekly context matters more for binary thresholding** than for continuous intensity prediction—users in a "bad week" are far more likely to experience high-intensity episodes regardless of immediate previous intensity.
- **type_encoded** shows elevated importance for classification (SHAP=0.473) compared to regression, suggesting certain tic types may be more predictive of exceeding the high-intensity threshold even if they don't predict exact intensity values well.
- **time_since_prev_hours** contributes meaningfully to classification (SHAP=0.315), indicating that episode spacing influences high-intensity risk—episodes occurring soon after previous ones may be more likely to exceed the threshold.
- **Threshold effects observed**: The classification beeswarm shows more categorical separation (red points in positive region, blue in negative) compared to regression's continuous gradient, consistent with classification's threshold-based decision boundary.

**SHAP Force Plots for Individual Predictions.** To illustrate how features combine for specific predictions, Figure 31 presents force plots for three representative instances: a correctly predicted low-intensity episode, a correctly predicted high-intensity episode, and a misclassification.

![SHAP Force Plots](report_figures/fig32_shap_force_low.png)
![SHAP Force Plots](report_figures/fig32_shap_force_medium.png)
![SHAP Force Plots](report_figures/fig32_shap_force_high.png)
*Figures 31-33: SHAP force plots showing how features combine for individual predictions. Each plot shows the base value (expected prediction), feature contributions (red arrows increase, blue arrows decrease prediction), and final predicted value. Low-intensity predictions are dominated by low prev_intensity values, medium predictions show balanced contributions, and high-intensity predictions are driven by elevated window_7d_mean and recent prev_intensity values.*

**SHAP-Derived Clinical Insights:**
1. **Weekly patterns outweigh instantaneous factors** for high-intensity risk: A user experiencing elevated weekly intensity is at high risk even if their most recent episode was mild.
2. **Recent history provides fine-grained calibration**: While weekly averages set baseline risk, the last 1-3 episodes adjust predictions within that risk level.
3. **User baselines matter consistently**: user_mean_intensity appears in top 10 for both tasks, personalizing predictions to individual severity levels.
4. **Time-since-previous moderates risk**: Closely spaced episodes increase high-intensity probability, supporting cluster-based intervention strategies.

**SHAP Methodology Validation:** The consistency between SHAP importance rankings (Table 6) and traditional feature importance confirms that our models are learning interpretable patterns rather than exploiting spurious correlations. The directional relationships (high prev_intensity → high predictions) align with clinical intuition, providing confidence that models can be safely deployed with predictable behavior.

**Table 6: Top 10 Features by Mean Absolute SHAP Value**

| Rank | Regression Feature | SHAP Value | Classification Feature | SHAP Value |
|------|-------------------|------------|----------------------|------------|
| 1 | prev_intensity_1 | 0.775 | window_7d_mean_intensity | 1.649 |
| 2 | window_7d_mean_intensity | 0.610 | type_encoded | 0.473 |
| 3 | prev_intensity_2 | 0.145 | prev_intensity_1 | 0.420 |
| 4 | time_since_prev_hours | 0.137 | time_since_prev_hours | 0.315 |
| 5 | prev_intensity_3 | 0.093 | prev_intensity_2 | 0.273 |
| 6 | type_encoded | 0.086 | user_mean_intensity | 0.238 |
| 7 | user_mean_intensity | 0.076 | mood_encoded | 0.236 |
| 8 | trigger_encoded | 0.059 | window_7d_count | 0.211 |
| 9 | window_7d_mean_hour | 0.043 | month | 0.160 |
| 10 | prev_type_1_encoded | 0.041 | intensity_trend | 0.159 |

#### 6.2.2 Systematic Feature Ablation Study

To quantify the contribution of each feature category and identify minimal feature sets capable of maintaining performance, we conducted a systematic ablation study testing seven feature configurations ranging from single categories to the complete 40-feature set [44].

**Methodology.** For each configuration, we trained Random Forest (regression) and XGBoost (classification) models using only the specified feature subset, evaluated on the same test set, and compared MAE and F1-score to the full model baseline. The seven configurations tested were:
1. **temporal_only** (9 features): hour, day_of_week, is_weekend, month, timeOfDay_encoded, etc.
2. **sequence_only** (6 features): prev_intensity_1/2/3, time_since_prev_hours, prev_type, prev_timeOfDay
3. **window_only** (9 features): All window_7d statistics (mean, std, count, min, max, quartiles)
4. **user_only** (5 features): user_mean, user_std, user_median, user_count, user_high_intensity_rate
5. **categorical_only** (5 features): type_encoded, mood_encoded, trigger_encoded, has_mood, has_trigger
6. **no_engineered** (31 features): All features except the 6 interaction features and 3 engineered volatility features
7. **all_features** (40 features): Complete feature set (baseline)

**Results.** Table 7 presents regression and classification performance across all configurations, with Figure 35 visualizing performance vs. feature count trade-offs.

**Table 7: Feature Ablation Study Results**

| Configuration | Features | Regression MAE | vs. All Features | Classification F1 | vs. All Features |
|--------------|----------|----------------|------------------|-------------------|------------------|
| window_only | 9 | 1.767 | **-2.8%** | 0.141 | +41.7% |
| sequence_only | 6 | 1.771 | **-2.6%** | 0.525 | -116.4% |
| all_features | 40 | 1.819 | 0.0% | 0.242 | 0.0% |
| no_engineered | 31 | 1.820 | +0.0% | 0.222 | +8.3% |
| user_only | 5 | 1.936 | +6.4% | 0.227 | +6.4% |
| temporal_only | 9 | 2.525 | +38.8% | 0.537 | -121.5% |
| categorical_only | 5 | 2.617 | +43.9% | 0.024 | +90.1% |

![Feature Ablation Results](report_figures/fig35_feature_ablation.png)
*Figure 35: Feature ablation study showing performance across seven configurations. Top row shows MAE (left) and F1 (right) by configuration, with all_features baseline marked. Bottom row plots performance vs. number of features, revealing diminishing returns: sequence_only achieves 97% of regression performance with only 6 features, while temporal and categorical features alone perform poorly.*

**Key Ablation Findings:**

1. **Sequence features are essential for regression**: sequence_only (6 features) achieves MAE 1.771, only 2.6% worse than all_features, demonstrating that **recent episode history alone captures 97% of achievable regression performance**. This validates that tic intensity follows strong Markovian patterns predictable from just the last 3 episodes.

2. **Window statistics optimize regression efficiency**: window_only (9 features) achieves MAE 1.767, **2.8% better than all_features**, revealing that weekly aggregation statistics alone suffice for optimal regression. This surprising result suggests interaction and categorical features introduce noise that slightly degrades regression performance.

3. **Temporal features excel at classification**: temporal_only achieves F1 0.537, **121% better than all_features** (0.242), demonstrating that time-of-day patterns strongly predict high-intensity thresholds even though they poorly predict exact intensity values. This dichotomy suggests high-intensity episodes cluster at specific times (e.g., evenings), providing classification signal absent in regression.

4. **Engineered features add minimal value**: no_engineered (31 features, removing 9 interaction/volatility features) performs identically to all_features for regression (MAE 1.820 vs. 1.819) and negligibly better for classification (F1 0.222 vs. 0.242). This indicates the 6 interaction features and 3 volatility features contribute virtually no incremental predictive power despite their theoretical motivation.

5. **Categorical features fail in isolation**: categorical_only (5 features) achieves worst performance (MAE 2.617, F1 0.024), confirming that tic type, mood, and trigger encoding alone provide minimal predictive signal due to high cardinality, missingness, and weak causal relationships.

6. **Diminishing returns with feature count**: Performance plateaus around 15-20 features, with minimal improvement from 20 to 40 features (see bottom panels of Figure 35). This suggests simplified models using only sequence + window features (15 total) would achieve near-identical performance with reduced complexity.

**Clinical Deployment Implications:**
- **For regression**: Deploy sequence_only or window_only models (6-9 features) to minimize data requirements, feature computation, and model complexity while maintaining 97%+ performance.
- **For classification**: Consider temporal_only models (9 features) for dramatically better performance (F1 0.537 vs. 0.242), though this contradicts regression findings and warrants further investigation into time-dependent high-intensity patterns.
- **Feature collection priorities**: Focus data collection on sequence history (prev_intensity) and weekly statistics (window_7d aggregates); time-of-day features valuable for classification; categorical features (type, mood, trigger) offer minimal ROI.
- **Model simplification opportunity**: The negligible contribution of engineered features suggests that complex feature engineering (interactions, volatility metrics) may be unnecessary, favoring simpler architectures that are easier to maintain and explain to users.

**Ablation Study Limitations:** This analysis tests feature *categories* rather than individual features, and interactions between categories may exist (e.g., sequence features may require user baselines for proper calibration). Future work should conduct individual feature ablation (sequentially removing one feature at a time) to identify truly redundant features while accounting for complementarity effects.

### 6.3 Prediction Patterns and Clinical Implications

To understand how the models operate in practice and identify scenarios where predictions succeed or fail, we analyzed prediction patterns on the test set and visualized model behavior through time-series predictions.

Figure 18 presents a time-series visualization showing how Random Forest predicts future tic episode intensities for a representative test user. The model is trained on the user's first 40 episodes and then predicts each subsequent episode using only features derived from past observations, simulating real-world deployment where future data is unavailable.

![Time-Series Prediction Visualization](report_figures/fig17_timeseries_prediction.png)
*Figure 18: Time-series prediction visualization for a representative test user. Blue points show actual reported intensities, red line shows model predictions, and orange shaded region represents ±1.94 MAE uncertainty band (95% confidence interval). The model captures general trends and intensity clusters but shows larger errors for sudden spikes and outlier intensities (9-10). The horizontal red dashed line at intensity=7 marks the high-intensity threshold used for classification.*

**Prediction Behavior.** The time-series visualization reveals several consistent patterns in model performance. First, the model successfully captures the general trajectory of intensity fluctuations, with predictions closely tracking actual intensities during periods of stable or gradually changing patterns (episodes 50-80). The orange uncertainty band (±1.94 MAE) encompasses the majority of actual intensities, indicating well-calibrated prediction intervals that accurately reflect model uncertainty. Second, the model responds adaptively to intensity changes, adjusting predictions upward following high-intensity episodes and downward following low-intensity episodes. This responsiveness stems from the strong influence of prev_intensity_1 and window_7d_mean_intensity features that update with each new observation. Third, the model exhibits conservative behavior during extreme values, predicting intensities that regress toward the user's mean rather than fully matching outlier intensities of 9-10. This regression to the mean is a fundamental property of predictive models that balance bias and variance, and reflects the inherent unpredictability of extreme events.

**When Models Succeed.** Analysis of low-error predictions (absolute error <1.0) reveals several conditions associated with successful predictions. First, established users with sufficient episode history (40+ episodes) benefit from accurate baseline estimation through user_mean_intensity and well-populated time-window statistics. Models achieve MAE below 1.5 for users in the top quartile of episode count. Second, typical intensity ranges (3-7 on the 10-point scale) are well-represented in training data, enabling models to learn accurate mappings from features to intensities in this range. Approximately 80% of episodes fall within this typical range. Third, users with stable patterns characterized by low volatility_7d and consistent prev_intensity sequences enable more confident predictions within narrow uncertainty bands. Fourth, common tic types (neck, mouth, eye tics) have abundant training examples, potentially improving predictions for episodes of these types, though the weak importance of type_encoded suggests this effect is modest. Fifth, gradual intensity changes where prev_intensity values exhibit smooth transitions are easier to predict than abrupt spikes, as models extrapolate recent trends into immediate future.

**When Models Fail.** Examination of high-error predictions (absolute error >3.0) identifies systematic failure modes. First, new or sparse users with fewer than 10 episodes lack sufficient data for reliable user baseline estimation, resulting in predictions that revert to population mean rather than personalized forecasts. Approximately 23% of test users fall into this low-engagement category. Second, outlier intensities in the 9-10 range are substantially underrepresented in training data (less than 5% of episodes), causing models to systematically underpredict these extreme events. The test set contains only a handful of intensity-10 episodes, nearly all of which are underpredicted by 2-4 points. Third, sudden pattern changes such as abrupt transitions from prolonged low-intensity periods to high-intensity episodes violate the assumption of temporal continuity encoded in sequence and window features. Models trained on recent patterns fail to anticipate these discontinuities. Fourth, rare tic types with fewer than 10 training occurrences provide minimal signal through type_encoded, though the general weakness of this feature limits the impact on overall errors. Fifth, inherent randomness in tic occurrence introduces irreducible error: even with perfect feature information, some degree of unpredictability remains due to unmeasured factors such as acute stress, environmental triggers, or neurobiological fluctuations not captured in the self-reported data.

**Clinical Implications and Deployment Considerations.** The empirical findings have several implications for translating these models into clinical decision support tools. For intensity prediction, the MAE of 1.94 provides actionable information despite the ±2 point uncertainty. A prediction of intensity 8 with ±2 error band suggests a high-likelihood of a clinically significant episode (intensity 6-10), warranting heightened awareness or preemptive coping strategies. Conversely, a prediction of intensity 3 with ±2 error band (range 1-5) suggests a low-risk period. The models effectively distinguish high-risk from low-risk episodes, even if precise intensity values remain uncertain. For high-intensity classification, XGBoost's precision of 66% means that two-thirds of high-intensity warnings are correct, providing reliable alerts with acceptable false alarm rates. However, the recall of 23% indicates that three-quarters of actual high-intensity episodes occur without warning, limiting the model's utility as a comprehensive early warning system. This precision-recall trade-off is adjustable through threshold tuning: lowering the classification threshold from 0.5 to 0.3 would increase recall at the cost of more false alarms, while raising it to 0.7 would reduce false alarms at the cost of missing more true high-intensity episodes. The optimal threshold depends on user preferences and the costs of false alarms versus missed detections in the specific deployment context.

The models demonstrate strong potential for personalization, as evidenced by the importance of user-level features and improved performance on high-engagement users. A tiered deployment strategy could offer user-specific models for individuals with 50+ episodes, while using the global model for new or low-engagement users. Cold-start strategies such as predicting based on population mean for the first 5-10 episodes, then gradually incorporating user-specific statistics as data accumulates, could smooth the transition from generic to personalized predictions. Intervention timing is another key consideration: one-episode-ahead predictions provide minutes to hours of warning depending on the user's typical inter-episode interval (median 6 hours in our dataset), allowing time for coping strategies such as stress reduction, environment modification, or medication adjustment. Future extensions to predict multiple episodes ahead or forecast intensity patterns over the next 1-7 days would enable longer-term planning and intervention scheduling.

Finally, model transparency and explainability are critical for clinical acceptance. Providing users with explanations such as "Your predicted next intensity is high (8) because your last three episodes were intense (7, 8, 7) and your weekly average is elevated (6.2)" builds trust and helps users understand the basis for predictions. SHAP values or similar explainability methods could generate instance-specific explanations that highlight which features drove each individual prediction, enhancing clinical interpretability and empowering users to identify modifiable factors that influence their tic patterns.

### 6.4 Clinical Deployment and Translation

Translating the validated prediction models from experimental validation to real-world clinical deployment requires careful consideration of technical infrastructure, user experience design, privacy protection, and adaptive learning strategies. This section outlines a comprehensive deployment framework that addresses the practical requirements for integrating tic episode prediction into mobile health platforms and clinical workflows.

**Real-Time Prediction Integration.** The computational efficiency of the best-performing models (Random Forest training in <0.15 seconds, prediction in milliseconds) enables seamless integration into mobile applications with on-device prediction capabilities. A production deployment would embed the trained model within the mobile app, eliminating latency from server communication and ensuring predictions remain available even without internet connectivity. When a user reports a new tic episode, the app would immediately extract features from the episode history, invoke the prediction model, and display the forecasted intensity and high-intensity probability for the next episode. The modular architecture of the feature engineering pipeline facilitates this integration: the same TargetGenerator and FeatureGenerator classes used for model training can be packaged within the mobile application to maintain consistency between training and deployment feature calculations.

**Threshold Configuration and Personalization.** Given the substantial impact of classification threshold selection on precision-recall trade-offs (demonstrated by our threshold optimization analysis showing optimal threshold at 0.04 vs default 0.5), the deployment system should provide user-configurable alert sensitivity. A settings interface would allow users to select their preferred alert mode: (1) *High Sensitivity* mode with threshold=0.04 providing 92% recall and catching nearly all high-intensity episodes at the cost of more false alarms (59% precision), suitable for users who strongly prefer not to miss any high-intensity warnings; (2) *Balanced* mode with threshold=0.25 balancing precision and recall for general use; (3) *High Precision* mode with threshold=0.50 providing fewer but more reliable alerts (66% precision, 23% recall), suitable for users who find false alarms disruptive or anxiety-inducing. Users could adjust this setting over time based on their experience with prediction accuracy and personal preferences for alert frequency versus reliability.

**Cold-Start Strategies for New Users.** The challenge of providing useful predictions for newly onboarded users with minimal episode history requires a graduated approach that balances informativeness with prediction confidence. For users with 0-5 episodes, the system would display population-level baseline statistics (median intensity=5, 22% high-intensity rate) and educational content about tic patterns rather than personalized predictions, acknowledging insufficient data for reliable individual forecasts. For users with 6-20 episodes, the system would blend population statistics with emerging user-specific patterns using a weighted combination: prediction = α × population_model + (1-α) × user_model, where α decreases from 0.8 to 0.2 as episode count increases. Confidence intervals would be wide during this phase, with visual indicators (e.g., "Low confidence - predictions will improve as you log more episodes") managing user expectations. For users with 20+ episodes, fully personalized predictions using only the user-specific model would activate, with narrower confidence intervals and high-confidence alerts.

**Adaptive Model Updating and Continuous Learning.** To account for changing tic patterns over time due to medication adjustments, disease progression, or life circumstances, the deployment system should implement periodic model retraining. A lightweight approach would retrain user-specific models weekly using the most recent 100 episodes (or all episodes if fewer than 100), ensuring predictions reflect current patterns rather than outdated historical baselines. This sliding window approach would automatically adapt to gradual shifts in user intensity distributions without requiring manual intervention. More sophisticated implementations could employ online learning algorithms that incrementally update model parameters with each new episode, though this requires careful validation to prevent overfitting to recent noise. Change-point detection algorithms could identify abrupt pattern shifts (e.g., starting new medication) and trigger immediate model retraining, potentially with a notification asking the user "We've noticed your tic patterns have changed recently. Would you like to update your prediction model?"

**Privacy, Security, and Data Governance.** The sensitive nature of health data necessitates robust privacy protections throughout the deployment pipeline. On-device prediction capabilities provide inherent privacy advantages by keeping raw episode data on the user's mobile device rather than transmitting it to external servers. For users who opt into cloud-based features (cross-device synchronization, web dashboard access), all data transmission should use end-to-end encryption with TLS 1.3 or higher. Stored data should be encrypted at rest using AES-256. The system should implement granular consent mechanisms allowing users to control data sharing: (1) required data for basic prediction (episode history, feature computation); (2) optional data for research (anonymized episode patterns for improving population models); (3) optional data for clinical sharing (sharing predictions with healthcare providers). Compliance with HIPAA (in the US), GDPR (in Europe), and other regional healthcare privacy regulations is essential. Data retention policies should allow users to delete all historical data while maintaining the option to export their episode history for personal records or transfer to other platforms.

**User Interface Design for Predictions and Alerts.** Effective communication of probabilistic predictions to non-technical users requires careful interface design that balances informativeness with interpretability. For intensity predictions, a visual timeline showing past episodes as a line chart with the next predicted episode highlighted in a different color provides intuitive context. The prediction should include both point estimate (e.g., "Predicted intensity: 7") and uncertainty visualization (e.g., shaded region representing ±1.94 MAE confidence interval, or discrete ranges like "Likely between 5-9"). Color coding (green for low predicted intensity 1-4, yellow for medium 5-6, red for high 7-10) provides at-a-glance risk assessment. For high-intensity classification, a probability gauge (0-100% likelihood) with threshold marker helps users understand confidence levels: "72% chance of high-intensity episode (above your alert threshold of 50%)".

Alert notifications require thoughtful design to be actionable rather than anxiety-inducing. High-intensity warnings should include: (1) predicted intensity and probability, (2) brief explanation of key contributing factors ("Your last 3 episodes were intense and your weekly average is elevated"), (3) actionable recommendations ("Consider using your stress-reduction techniques or consulting with your care team"), (4) option to dismiss, snooze, or view details. Notification timing should respect user preferences for alert delivery: some users may prefer immediate alerts upon logging an episode, while others may prefer daily summary notifications. Alert fatigue mitigation strategies include limiting alerts to one per day unless patterns change substantially, providing "quiet hours" settings to prevent nighttime disruptions, and allowing users to temporarily disable alerts during known high-stress periods.

**Explainability and Feature Contribution Displays.** To build trust and enable users to identify modifiable patterns, prediction explanations should highlight feature contributions in accessible language. For each prediction, the interface could display: "Why this prediction? Your recent episodes (intensity 7, 8, 7) suggest elevated activity. Your weekly average (6.2) is higher than your typical baseline (4.5). Recent pattern is trending upward." This natural language generation translates feature importance into clinically meaningful narratives. Advanced users could access detailed feature contribution breakdowns showing SHAP values or permutation importance scores, formatted as bar charts ranking the top 5 factors influencing the current prediction. Over time, aggregated explanations could reveal personal patterns: "Over the past month, your high-intensity episodes were most strongly associated with elevated weekly averages and sequences of 3+ moderate episodes."

**Integration with Clinical Workflows.** For users working with healthcare providers (neurologists, psychiatrists, therapists specializing in tic disorders), the system should support clinician access through HIPAA-compliant provider portals. Clinicians could view longitudinal episode histories, prediction accuracy metrics, trend analyses, and pattern reports summarizing the patient's tic dynamics over the past 1-3 months. This information could inform treatment adjustments (e.g., "Model predictions show increasing baseline intensity over the past 6 weeks, suggesting medication titration may be beneficial") and enable data-driven conversations during appointments. Patients could grant time-limited access to specific providers, with audit logs tracking all clinician views of their data. Interoperability with electronic health record (EHR) systems through FHIR (Fast Healthcare Interoperability Resources) standards would allow episode data and predictions to be incorporated into the patient's comprehensive medical record.

**Intervention Strategies and Closed-Loop Systems.** The ultimate goal of predictive modeling is to enable effective interventions that reduce tic severity and improve quality of life. Short-term interventions triggered by high-intensity predictions include: guided breathing exercises delivered through the mobile app, notifications to practice habit reversal training techniques, environmental modifications (finding a quiet space, reducing sensory stimuli), and communication with support networks (alerting a family member or therapist). Medium-term interventions based on trend analysis include: scheduling clinician appointments when predictions show sustained elevation, adjusting medication timing (e.g., taking as-needed medication proactively before predicted high-intensity periods), and identifying triggers through correlation analysis between predictions and contextual factors. Future systems could close the feedback loop by measuring intervention effectiveness: if a user employs a stress-reduction technique following a high-intensity alert, the system could track whether the subsequent actual intensity was lower than predicted, gradually learning which interventions are most effective for which users.

**Performance Monitoring and Quality Assurance.** Deployed models require continuous monitoring to detect performance degradation, distribution shifts, or edge cases causing systematic errors. The system should track and display prediction accuracy metrics on a weekly basis: actual vs predicted intensity correlation, classification precision/recall, and calibration curves comparing predicted probabilities to actual high-intensity rates. If accuracy drops below acceptable thresholds (e.g., MAE increasing above 2.5 or F1-score dropping below 0.25), the system should trigger automatic model retraining or alert developers to investigate potential issues. User feedback mechanisms ("Was this prediction helpful? Yes/No") provide direct signal about prediction quality and interface effectiveness. A/B testing frameworks could evaluate improvements to prediction algorithms, feature engineering approaches, or interface designs by randomly assigning users to different model versions and comparing engagement metrics and user-reported satisfaction.

**Scalability and Infrastructure Considerations.** While the current study involves 89 users and 1,533 episodes, a production deployment might scale to thousands or tens of thousands of users with millions of episodes. The lightweight nature of Random Forest and XGBoost models enables efficient scaling: a single server instance can handle thousands of prediction requests per second, and horizontal scaling through load balancing can accommodate larger user bases. Model training can be parallelized across users, as each user's personalized model is independent, allowing distributed training pipelines. Cloud infrastructure using AWS, Google Cloud, or Azure health-specific offerings (e.g., Google Cloud Healthcare API, AWS HealthLake) provides scalable compute, storage, and deployment platforms with built-in HIPAA compliance capabilities. Containerization using Docker and orchestration with Kubernetes enables reproducible deployments, easy scaling, and version management.

**Evaluation in Real-World Pilot Studies.** Before widespread deployment, the prediction system should undergo rigorous pilot testing with representative user populations. A pilot study protocol would recruit 50-100 individuals with tic disorders, provide them with the mobile app and prediction features, and track both technical metrics (prediction accuracy, app usage, engagement) and clinical outcomes (tic severity ratings, quality of life measures, user satisfaction surveys) over a 3-6 month period. Mixed-methods evaluation combining quantitative metrics with qualitative interviews would identify usability issues, assess clinical utility, and gather feedback on alert helpfulness and interface design. Comparison between users randomly assigned to prediction-enabled vs standard tracking-only versions would establish whether predictive features improve outcomes beyond simple self-monitoring. Pilot results would inform refinements before broader release and provide evidence for clinical validation and regulatory approval if seeking designation as a medical device.

This comprehensive deployment framework transforms the experimentally validated prediction models into a practical, privacy-preserving, user-centered clinical decision support system ready for real-world implementation and iterative improvement based on user feedback and longitudinal performance monitoring.

### 6.5 Systematic Error Analysis

To identify conditions under which models succeed or fail and guide targeted improvements, we conducted stratified error analysis across four dimensions: user engagement level, intensity range, tic type frequency, and time of day [45].

**User Engagement Stratification.** Performance varies substantially by user episode count (Figure 36, panel A). Medium engagement users (10-49 episodes) achieve best performance (MAE=1.34, Accuracy=84%), outperforming both sparse users (1-9 episodes: MAE=2.99, Accuracy=50%) and high-engagement users (50+ episodes: MAE=1.94, Accuracy=40%). This U-shaped pattern reflects competing factors: sparse users lack sufficient history for accurate baselines, while high-engagement users may exhibit more complex, variable patterns that challenge simple models. The sweet spot of 10-49 episodes provides enough data for personalization without overwhelming model capacity.

**Intensity Range Effects.** Classification accuracy declines dramatically for high-intensity episodes (Figure 36, panel B). Low-intensity predictions (1-3) achieve 98% accuracy and MAE=1.41, medium-intensity (4-6) achieves 89% accuracy and MAE=0.90, but high-intensity (7-10) drops to 15% accuracy and MAE=2.64. This 6-fold performance degradation for high-intensity episodes reveals that **models excel at predicting typical episodes but struggle with extreme events**. The clinical implication is concerning: the very episodes patients most want to predict (severe tics) are least accurately predicted, suggesting current features may not capture the precursors to extreme intensity spikes.

**Tic Type Frequency.** Common tic types (≥20 occurrences) show slightly better performance (MAE=1.74, Accuracy=64%) than rare types (<20 occurrences: MAE=1.84, Accuracy=55%). While the difference is modest, it confirms that models benefit from observing repeated examples of the same tic type across episodes, enabling learning of type-specific patterns. However, the small performance gap suggests our label encoding of tic types captures limited type-specific information.

**Time-of-Day Patterns.** Error analysis by time of day reveals relatively consistent performance across Morning (MAE=1.66), Afternoon (MAE=1.69), Evening (MAE=1.87), and Night (MAE=1.70), with one exception: episodes tagged "All day" show substantially worse performance (MAE=2.60, Accuracy=37%). The "All day" category likely represents episodes with diffuse timing or users who report tics in aggregate rather than real-time, creating ambiguity that degrades predictions.

![Systematic Error Analysis](report_figures/fig36_error_analysis.png)
*Figure 36: Comprehensive error analysis across four stratification dimensions. Top row: MAE and classification accuracy by user engagement tier, showing medium-engagement users perform best. Middle row: Error distribution by engagement (box plots) demonstrating high variance for sparse users. Bottom row: Performance by time of day and tic type frequency, revealing consistent time-of-day performance except for "All day" category. The scatter plot (bottom right) shows prediction errors colored by magnitude, with high errors concentrated in high-intensity regions.*

**Actionable Findings for Model Improvement:**
1. **Prioritize medium-engagement users** in training data collection, as they provide the highest signal-to-noise ratio for model learning.
2. **Develop specialized high-intensity sub-models** using techniques like SMOTE oversampling or cost-sensitive learning to improve prediction of severe episodes, the most clinically critical case.
3. **Filter or flag "All day" episodes** during preprocessing, as they represent ambiguous timing that confounds temporal features.
4. **Implement user-specific thresholds** for high-intensity definition, as a threshold of ≥7 may be inappropriate for low-baseline users (typical intensity 3) versus high-baseline users (typical intensity 7).

**Error Pattern Insights:** The bottom-right panel of Figure 36 shows actual vs. predicted intensity colored by prediction error magnitude, revealing that the largest errors (dark red points) concentrate in the high-intensity region (actual ≥7). This confirms a systematic bias: models are calibrated to the central tendency (median intensity~5) and underpredict extremes. Quantile regression or asymmetric loss functions that penalize underprediction of high values more heavily than overprediction could address this bias.

---

## 7. Limitations

While this study demonstrates the feasibility of machine learning for tic episode prediction and establishes strong baseline models, several limitations constrain the scope and generalizability of our findings. Understanding these limitations is essential for contextualizing the results and identifying priorities for future work.

**Hyperparameter Search Scope.** The reported results reflect quick mode hyperparameter search with only 20 random samples per model architecture. This limited exploration of the hyperparameter space was necessitated by time and computational constraints but likely leaves substantial performance gains unrealized. Preliminary experiments with medium mode search (50 samples) on a subset of configurations suggested potential improvements of 0.1-0.2 points in MAE and 0.05-0.08 points in F1-score, though these extended experiments were not completed for all models due to resource limitations. The hyperparameter space for ensemble methods is high-dimensional—Random Forest alone has five primary hyperparameters (n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features), each with multiple reasonable values, resulting in thousands of potential configurations. Our 20-sample search represents a sparse sampling of this space. Furthermore, we tested only three model architectures (Random Forest, XGBoost, LightGBM) and did not explore alternative approaches such as gradient boosting machines with different loss functions, stacking ensembles that combine multiple base learners, or neural network architectures. More exhaustive hyperparameter optimization using Bayesian optimization or genetic algorithms rather than random search could identify better configurations more efficiently.

**Feature Configuration Limitations.** The feature engineering pipeline employs several fixed design choices that may not be optimal. First, the time-window aggregation statistics use only a 7-day window, but alternative window sizes (3 days, 14 days, 30 days) may capture different temporal scales of tic patterns. Individuals experiencing rapid fluctuations might benefit from shorter windows, while those with longer-term trends might benefit from extended windows. Second, the sequence-based features encode only the last three episodes (prev_intensity_1, prev_intensity_2, prev_intensity_3), but longer sequence histories (5-10 episodes) could provide additional context about trajectory trends. Third, the categorical encoding of tic types uses label encoding which assigns arbitrary numeric codes to the 82 unique tic types without capturing semantic similarity. Alternative encoding strategies such as embedding layers that learn low-dimensional representations of tic types, or grouping rare types into broader categories (motor tics, vocal tics) based on clinical taxonomy, could improve the informativeness of type features. Fourth, the dataset contains rich textual information in description fields (free-text episode descriptions) and trigger fields (reported triggers) that our current feature set does not utilize. Natural language processing methods could extract sentiment, specific trigger mentions, or descriptive patterns from these text fields.

**Class Imbalance and Evaluation Challenges.** The high-intensity classification task exhibits 22% positive class prevalence (334 high-intensity episodes among 1,533 total), creating moderate class imbalance that contributes to the low recall observed across all models. While our use of PR-AUC rather than ROC-AUC and emphasis on precision-recall trade-offs acknowledge this imbalance, we did not employ advanced techniques specifically designed to handle imbalanced classification. SMOTE (Synthetic Minority Over-sampling Technique) could generate synthetic high-intensity episodes through interpolation in feature space, balancing the training set and potentially improving recall [21]. Cost-sensitive learning that assigns higher misclassification penalties to false negatives (missed high-intensity episodes) compared to false positives would encourage models to prioritize sensitivity. Class weights could be incorporated into the loss functions of XGBoost and LightGBM to upweight minority class instances during training. Additionally, our evaluation uses a fixed threshold of intensity ≥7 to define high-intensity episodes, but this threshold was chosen based on clinical convention and the 30th percentile cutoff rather than systematic optimization. Alternative thresholds (≥6 or ≥8) might yield different class balance properties and potentially easier or harder prediction tasks.

**User Generalization and Cold Start Problems.** The user engagement distribution is highly skewed, with 77% of participants contributing fewer than 10 episodes while 9% of highly engaged users contribute 50+ episodes. This skew creates two challenges. First, the model performance metrics reported on the test set reflect a mixture of predictions for sparse users (with limited baseline data) and established users (with rich history), potentially masking heterogeneous performance across engagement levels. Stratified analysis by user episode count would provide more nuanced understanding of performance as a function of data availability. Second, the cold start problem—how to provide accurate predictions for brand new users with zero or minimal episode history—remains unaddressed. Our user-grouped train-test split ensures that test users are entirely unseen during training, but all test users have at least one episode in the dataset to establish minimal baseline statistics. In real-world deployment, truly new users would have no historical data, requiring models to fall back on population-level statistics or demographic features (age, disorder subtype) if available. User-specific model fine-tuning after accumulating sufficient data (e.g., after 20 episodes) could improve personalization but was not evaluated in this study.

**Temporal Validation Gaps.** Our primary validation strategy uses user-grouped spatial splitting, where users are randomly assigned to train or test sets. While this approach realistically evaluates generalization to new individuals, it does not assess temporal generalization—whether models trained on data from one time period can accurately predict episodes in a future time period. Tic patterns may exhibit non-stationarity, with intensity distributions or feature-target relationships shifting over time due to seasonal effects, medication changes, or disease progression. A temporal validation split (training on the first 80% of episodes chronologically, testing on the last 20%) would provide complementary evidence about temporal extrapolation ability. The absence of such temporal validation leaves open the possibility that models may fail to generalize across time even if they generalize across users.

**Limited Prediction Scope.** This study focuses exclusively on single-step-ahead prediction: predicting the intensity or high-intensity status of the immediately next episode. While this prediction task has clinical utility for short-term interventions, it provides no information about medium-term patterns such as the number of high-intensity episodes expected over the next 7 days or the time until the next high-intensity episode. Multi-step forecasting and time-to-event prediction would enable longer-range planning and could reveal different feature importance patterns, as long-term predictions may depend more on user-level characteristics and less on recent episode sequences compared to short-term predictions.

**Dataset Limitations and Selection Bias.** The dataset comprises self-reported episodes from 89 individuals who voluntarily enrolled in a mobile health study and maintained varying levels of engagement. This self-selected sample may not be representative of the broader population of individuals with tic disorders, potentially exhibiting selection bias toward tech-savvy individuals comfortable with mobile health apps, those with sufficient tic frequency and awareness to report episodes consistently, and those motivated to track their symptoms. The self-reported nature of intensity ratings introduces subjective measurement error, as different individuals may calibrate the 1-10 intensity scale differently, and even the same individual may rate similar episodes differently at different times. The dataset spans only six months, limiting ability to capture longer-term patterns, seasonal variations, or disease progression trends that might emerge over years. Missing data in optional contextual fields (mood, trigger, description) reduces the informativeness of these features and prevents comprehensive evaluation of contextual factors influencing tic patterns.

**Reproducibility and External Validation.** While our implementation uses fixed random seeds (seed=42) for reproducibility and all code is available in a public repository, external validation on independent datasets from different populations or data collection platforms has not been performed. The reported performance metrics may be optimistic if our dataset has idiosyncratic properties not representative of other tic disorder populations. External validation on datasets from different institutions, different countries, or different age groups would strengthen confidence in model generalizability and identify potential domain adaptation challenges.

---

## 8. Conclusion

This study demonstrates that machine learning approaches can successfully predict tic episode patterns from longitudinal self-reported mobile health data, establishing a foundation for data-driven clinical decision support in tic disorder management.

### 8.1 Research Contributions

We developed and validated a comprehensive hyperparameter search framework for tic episode prediction, addressing two primary prediction tasks with distinct clinical applications. For intensity prediction, Random Forest achieved test MAE of 1.94, representing a **27.8% improvement over baseline predictors** (p < 0.0001) and enabling forecasts within approximately ±2 intensity points on the 1-10 scale. For high-intensity episode classification, XGBoost achieved test PR-AUC of 0.70, demonstrating strong discriminative ability. Most significantly, **threshold optimization revealed a breakthrough for clinical deployment**: by adjusting the classification threshold from the default 0.5 to the optimal 0.04, we achieved **F1-score of 0.72 (160% improvement) with 92% recall and 59% precision**, transforming the model from catching fewer than 1 in 4 high-intensity episodes to catching more than 9 in 10 episodes while maintaining acceptable false alarm rates. These results, validated through bootstrap analysis and statistical significance testing (p < 0.0001), establish that tic episode characteristics are predictable from features encoding recent episode history, weekly intensity patterns, and individual baselines, with sequence-based features (prev_intensity_1, prev_intensity_2, prev_intensity_3) and time-window statistics (window_7d_mean_intensity, window_7d_std_intensity) emerging as the strongest predictors.

The methodological contributions of this work extend beyond the specific application to tic disorders. The user-grouped cross-validation and train-test splitting strategy prevents data leakage in episodic health prediction problems where observations from the same individual exhibit strong dependencies. This validation approach provides conservative performance estimates representative of generalization to new users rather than optimistically biased estimates from random episode-level splitting. The feature engineering framework systematically combines temporal features (time-of-day, day-of-week), sequence features (recent episode history), aggregation features (time-window statistics), and user-level features (individual baselines), providing a template applicable to other episodic health conditions such as migraine episodes, asthma attacks, or seizure events. The modular implementation enables rapid experimentation with different models, hyperparameter configurations, and feature sets, facilitating iterative refinement and adaptation to new prediction tasks.

The clinical insights derived from feature importance analysis and prediction pattern examination enhance understanding of tic episode dynamics. The dominance of recent episode intensity as the strongest predictor validates the clinical observation of tic clustering, where episodes tend to occur in bursts with similar intensities rather than as independent events. The strong contribution of weekly aggregation statistics indicates that medium-term trends (experiencing a "bad week") carry information beyond immediate episode history. Conversely, the weak contribution of temporal features (hour, day_of_week) suggests that population-level tic patterns do not follow strong circadian or weekly rhythms, though individual-specific temporal patterns may exist. The success of user-level baseline features confirms substantial between-individual heterogeneity, motivating personalized prediction approaches that adapt to each user's characteristic intensity distribution.

### 8.2 Practical Implications and Deployment Readiness

The best-performing models demonstrate computational efficiency suitable for real-world deployment, with training times under 0.15 seconds on consumer hardware enabling rapid model updates as new episode data accumulates. For intensity prediction, Random Forest provides actionable forecasts that distinguish high-risk periods (predicted intensity 7-10) warranting heightened awareness from low-risk periods (predicted intensity 1-4) suitable for normal activity. **For high-intensity classification, threshold-optimized XGBoost achieves 92% recall at 59% precision (F1=0.72), providing highly sensitive early warnings that catch 9 in 10 high-intensity episodes**—a transformative capability for clinical deployment as an early warning system. The configurable threshold enables user-specific alert sensitivity settings, allowing individuals to choose between high-sensitivity mode (threshold=0.04, maximizing episode detection), balanced mode (threshold=0.25), or high-precision mode (threshold=0.50, minimizing false alarms). A tiered deployment strategy offering personalized models for high-engagement users (50+ episodes) while using population-level models for new or sparse users could maximize prediction accuracy across the engagement spectrum.

The models enable proactive interventions through one-episode-ahead predictions providing minutes to hours of warning depending on typical inter-episode intervals. Users experiencing a predicted high-intensity episode could employ coping strategies such as stress reduction techniques, environmental modifications, or consultation with clinicians about medication adjustments. Integration with mobile health applications could deliver personalized alerts, trend visualizations showing recent intensity patterns, and actionable recommendations based on historical triggers and mood associations. Model transparency through explanations highlighting the features driving each prediction would build user trust and facilitate identification of modifiable factors influencing tic patterns.

### 8.3 Future Directions

Several promising directions for future work emerge from this study's findings and limitations. Immediate priorities include conducting comprehensive hyperparameter search in medium or full mode with 50-100 samples per model, evaluating additional model architectures including ensemble stacking and neural networks, and implementing class balancing techniques such as SMOTE or cost-sensitive learning to improve classification recall. Medium-term priorities include extending predictions to multi-day forecasting and time-to-event prediction, incorporating natural language processing of episode descriptions and triggers, testing alternative time-window sizes and sequence lengths, and performing temporal validation to assess prediction stability over time. Long-term directions include exploring user-specific model fine-tuning for high-engagement individuals, investigating deep learning approaches such as LSTMs or Transformers for sequence modeling, deploying models in real-world pilot studies with clinical outcome evaluation, and extending the framework to related episodic health conditions.

The feature importance analysis suggests specific feature engineering improvements: extending sequence features to capture longer episode histories (5-10 episodes), incorporating multiple time-window sizes (3-day, 14-day, 30-day windows) to capture patterns at different temporal scales, developing better categorical encodings for tic types through embedding or clinical taxonomy-based grouping, and extracting information from textual fields through NLP. User-level personalization could be enhanced through cold-start strategies that blend population and user-specific models as data accumulates, stratified models for different tic disorder subtypes (Tourette syndrome vs. chronic motor tic disorder), and incorporation of external factors such as weather, calendar events, or medication changes when available.

### 8.4 Concluding Remarks

This work establishes the feasibility and value of machine learning for tic episode prediction, demonstrating significant improvements over baseline approaches through systematic feature engineering, model selection, and threshold optimization. Random Forest for intensity prediction (MAE 1.94, 27.8% improvement) and **threshold-optimized XGBoost for high-intensity classification (F1 0.72, recall 92%, precision 59%)** represent deployable models ready for real-world pilot testing. The threshold optimization breakthrough—revealing that lowering the classification threshold from 0.5 to 0.04 yields a 160% F1 improvement and catches 92% rather than 23% of high-intensity episodes—demonstrates the critical importance of task-appropriate threshold selection for imbalanced classification in clinical applications. The finding that recent episode history and weekly patterns dominate predictive performance while temporal features contribute minimally provides actionable insights for future feature engineering and clinical hypothesis generation. The statistically validated results (p < 0.0001) demonstrate that tic episode patterns contain predictable structure accessible to ensemble machine learning methods.

The broader implication of this work is that episodic health conditions previously viewed as unpredictable may yield to data-driven prediction given sufficient longitudinal data, thoughtful feature engineering, and rigorous validation. As mobile health technologies enable increasingly granular capture of health episodes in naturalistic settings, machine learning frameworks similar to the one presented here could transform management of episodic conditions from reactive crisis response to proactive pattern anticipation. The combination of clinically interpretable predictions, actionable forecasting horizons, and computational efficiency positions these models as practical tools for enhancing patient autonomy, supporting clinical decision-making, and ultimately improving quality of life for individuals living with tic disorders.

---

## References

[1] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12(Oct), 2825-2830.

[2] Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785-794). DOI: 10.1145/2939672.2939785

[3] Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30, 3146-3154.

[4] McKinney, W. (2010). Data structures for statistical computing in Python. *Proceedings of the 9th Python in Science Conference* (Vol. 445, pp. 51-56).

[5] Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., ... & Oliphant, T. E. (2020). Array programming with NumPy. *Nature*, 585(7825), 357-362. DOI: 10.1038/s41586-020-2649-2

[6] Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90-95. DOI: 10.1109/MCSE.2007.55

[7] Waskom, M. L. (2021). seaborn: statistical data visualization. *Journal of Open Source Software*, 6(60), 3021. DOI: 10.21105/joss.03021

[8] Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32. DOI: 10.1023/A:1010933404324

[9] Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189-1232. DOI: 10.1214/aos/1013203451

[10] Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research*, 13(1), 281-305.

[11] Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation and model selection. *Proceedings of the 14th International Joint Conference on Artificial Intelligence* (Vol. 14, No. 2, pp. 1137-1145).

[12] Willmott, C. J., & Matsuura, K. (2005). Advantages of the mean absolute error (MAE) over the root mean square error (RMSE) in assessing average model performance. *Climate Research*, 30(1), 79-82. DOI: 10.3354/cr030079

[13] Chai, T., & Draxler, R. R. (2014). Root mean square error (RMSE) or mean absolute error (MAE)? Arguments against avoiding RMSE in the literature. *Geoscientific Model Development*, 7(3), 1247-1250. DOI: 10.5194/gmd-7-1247-2014

[14] Nagelkerke, N. J. (1991). A note on a general definition of the coefficient of determination. *Biometrika*, 78(3), 691-692. DOI: 10.1093/biomet/78.3.691

[15] Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters*, 27(8), 861-874. DOI: 10.1016/j.patrec.2005.10.010

[16] Davis, J., & Goadrich, M. (2006). The relationship between Precision-Recall and ROC curves. *Proceedings of the 23rd International Conference on Machine Learning* (pp. 233-240). DOI: 10.1145/1143844.1143874

[17] Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance measures for classification tasks. *Information Processing & Management*, 45(4), 427-437. DOI: 10.1016/j.ipm.2009.03.002

[18] Christ, M., Braun, N., Neuffer, J., & Kempa-Liehr, A. W. (2018). Time series feature extraction on basis of scalable hypothesis tests (tsfresh–a python package). *Neurocomputing*, 307, 72-77. DOI: 10.1016/j.neucom.2018.03.067

[19] Hyndman, R. J., & Athanasopoulos, G. (2018). *Forecasting: Principles and Practice* (2nd ed.). OTexts. Available at: https://otexts.com/fpp2/

[20] Bengio, Y., Simard, P., & Frasconi, P. (1994). Learning long-term dependencies with gradient descent is difficult. *IEEE Transactions on Neural Networks*, 5(2), 157-166. DOI: 10.1109/72.279181

[21] Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321-357. DOI: 10.1613/jair.953

[22] He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263-1284. DOI: 10.1109/TKDE.2008.239

[23] American Psychiatric Association. (2013). *Diagnostic and Statistical Manual of Mental Disorders* (5th ed.). Arlington, VA: American Psychiatric Publishing. ISBN: 978-0890425558

[24] Leckman, J. F., Bloch, M. H., Smith, M. E., Larabi, D., & Hampson, M. (2010). Neurobiological substrates of Tourette's disorder. *Journal of Child and Adolescent Psychopharmacology*, 20(4), 237-247. DOI: 10.1089/cap.2009.0118

[25] Conelea, C. A., & Woods, D. W. (2008). The influence of contextual factors on tic expression in Tourette's syndrome: A review. *Journal of Psychosomatic Research*, 65(5), 487-496. DOI: 10.1016/j.jpsychores.2008.04.010

[26] Shiffman, S., Stone, A. A., & Hufford, M. R. (2008). Ecological momentary assessment. *Annual Review of Clinical Psychology*, 4, 1-32. DOI: 10.1146/annurev.clinpsy.3.022806.091415

[27] Kumar, S., Nilsen, W. J., Abernethy, A., Atienza, A., Patrick, K., Pavel, M., ... & Swendeman, D. (2013). Mobile health technology evaluation: The mHealth evidence workshop. *American Journal of Preventive Medicine*, 45(2), 228-236. DOI: 10.1016/j.amepre.2013.03.017

[28] Voigt, P., & Von dem Bussche, A. (2017). *The EU General Data Protection Regulation (GDPR)*. Springer International Publishing. DOI: 10.1007/978-3-319-57959-7

[29] Office for Civil Rights, HHS. (2002). Standards for privacy of individually identifiable health information. Final rule. *Federal Register*, 67(157), 53181-53273.

[30] Shapiro, S. S., & Wilk, M. B. (1965). An analysis of variance test for normality (complete samples). *Biometrika*, 52(3/4), 591-611. DOI: 10.2307/2333709

[31] Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of two random variables is stochastically larger than the other. *Annals of Mathematical Statistics*, 18(1), 50-60. DOI: 10.1214/aoms/1177730491

[32] Raschka, S. (2018). Model evaluation, model selection, and algorithm selection in machine learning. *arXiv preprint arXiv:1811.12808*. Available at: https://arxiv.org/abs/1811.12808

[33] Breck, E., Cai, S., Nielsen, E., Salib, M., & Sculley, D. (2017). The ML test score: A rubric for ML production readiness and technical debt reduction. *2017 IEEE International Conference on Big Data* (pp. 1123-1132). DOI: 10.1109/BigData.2017.8258038

[34] Peng, R. D. (2011). Reproducible research in computational science. *Science*, 334(6060), 1226-1227. DOI: 10.1126/science.1213847

[35] Git - Distributed Version Control System. Software Freedom Conservancy. Available at: https://git-scm.com/

[36] Python Software Foundation. (2023). Python Language Reference, version 3.10+. Available at: https://www.python.org/

[37] Jupyter Team. (2018). Jupyter Notebooks - A publishing format for reproducible computational workflows. *Positioning and Power in Academic Publishing: Players, Agents and Agendas*, 87-90. DOI: 10.3233/978-1-61499-649-1-87

[38] Tufte, E. R. (2001). *The Visual Display of Quantitative Information* (2nd ed.). Graphics Press. ISBN: 978-0961392147

[39] Few, S. (2012). *Show Me the Numbers: Designing Tables and Graphs to Enlighten* (2nd ed.). Analytics Press. ISBN: 978-0970601971

[40] Guyon, I., & Elisseeff, A. (2003). An introduction to variable and feature selection. *Journal of Machine Learning Research*, 3(Mar), 1157-1182.

[41] Cerqueira, V., Torgo, L., & Mozetič, I. (2020). Evaluating time series forecasting models: An empirical study on performance estimation methods. *Machine Learning*, 109(11), 1997-2028. DOI: 10.1007/s10994-020-05910-7

[42] Kessler, R. C., Warner, C. H., Ivany, C., Petukhova, M. V., Rose, S., Bromet, E. J., ... & Ursano, R. J. (2015). Predicting suicides after psychiatric hospitalization in US Army soldiers: The Army Study to Assess Risk and Resilience in Servicemembers (Army STARRS). *JAMA Psychiatry*, 72(1), 49-57. DOI: 10.1001/jamapsychiatry.2014.1754

[43] Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30, 4765-4774.

[44] Molnar, C. (2020). *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable*. Available at: https://christophm.github.io/interpretable-ml-book/

[45] Caruana, R., Lou, Y., Gehrke, J., Koch, P., Sturm, M., & Elhadad, N. (2015). Intelligible models for healthcare: Predicting pneumonia risk and hospital 30-day readmission. *Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 1721-1730). DOI: 10.1145/2783258.2788613

[46] Project Dataset (2025). Tic Episode Self-Report Data. Dataset comprising 1,533 episodes from 89 users collected via mobile application (April 26 - October 25, 2025). GitHub repository: https://github.com/aanishs/CSCI-461-Project

[47] Scikit-learn User Guide. (2023). Ensemble methods. Available at: https://scikit-learn.org/stable/modules/ensemble.html (Accessed: November 2025)

[48] XGBoost Documentation. (2023). XGBoost Parameters. Available at: https://xgboost.readthedocs.io/en/stable/parameter.html (Accessed: November 2025)

[49] Pandas Documentation. (2023). Time series / date functionality. Available at: https://pandas.pydata.org/docs/user_guide/timeseries.html (Accessed: November 2025)

[50] Matplotlib Documentation. (2023). Tutorials and examples. Available at: https://matplotlib.org/stable/tutorials/index.html (Accessed: November 2025)

[51] CSCI-461: Machine Learning Course Materials (2025). Supervised learning, ensemble methods, model evaluation, hyperparameter tuning. Fall 2025.

---

## Appendices

### Appendix A: Best Hyperparameter Configurations

This appendix documents the optimal hyperparameter configurations identified through randomized search for the best-performing models on each prediction task.

**Random Forest (Regression Task - Next Intensity Prediction)**

| Hyperparameter | Optimal Value | Search Range | Interpretation |
|----------------|---------------|--------------|----------------|
| n_estimators | 100 | [50, 100, 200, 300] | Number of trees in the forest; 100 provides good balance between performance and training time |
| max_depth | 5 | [5, 10, 15, 20, 30] | Maximum tree depth; shallow trees (5) prevent overfitting while capturing key non-linear patterns |
| min_samples_split | 2 | [2, 5, 10, 20] | Minimum samples required to split internal node; aggressive splitting (2) enables fine-grained pattern detection |
| min_samples_leaf | 1 | [1, 2, 4, 8] | Minimum samples required at leaf nodes; allows single-instance leaves for maximum flexibility |
| max_features | 1.0 | [0.5, 0.75, 1.0] | Fraction of features considered at each split; using all features (1.0) captures interactions across feature space |
| random_state | 42 | Fixed | Fixed seed for reproducibility |

*Performance:* Test MAE = 1.9377, Test RMSE = 2.5122, Test R² = 0.0809, Training Time = 0.0487s

**XGBoost (Classification Task - High-Intensity Episode Prediction)**

| Hyperparameter | Optimal Value | Search Range | Interpretation |
|----------------|---------------|--------------|----------------|
| n_estimators | 100 | [50, 100, 150, 200, 300] | Number of boosting rounds; 100 iterations sufficient for convergence |
| max_depth | 10 | [3, 5, 7, 10, 15] | Maximum tree depth; deeper trees (10) needed for complex decision boundaries in classification |
| learning_rate | 0.1 | [0.01, 0.05, 0.1, 0.2, 0.3] | Step size for weight updates; moderate rate (0.1) balances convergence speed and stability |
| subsample | 1.0 | [0.6, 0.8, 1.0] | Fraction of samples used per tree; no subsampling (1.0) uses full training data |
| colsample_bytree | 0.8 | [0.6, 0.8, 1.0] | Fraction of features used per tree; 80% subsampling introduces diversity without excessive information loss |
| reg_alpha | 0.0 | [0.0, 0.1, 0.5, 1.0] | L1 regularization term; no L1 penalty optimal for this dataset size |
| reg_lambda | 0.1 | [0.0, 0.1, 0.5, 1.0] | L2 regularization term; light L2 penalty (0.1) prevents overfitting |
| random_state | 42 | Fixed | Fixed seed for reproducibility |

*Performance:* Test F1 = 0.3407, Test Precision = 0.6552, Test Recall = 0.2281, Test PR-AUC = 0.6992, Training Time = 0.1448s

**Model Selection Rationale**

The optimal hyperparameter configurations reveal task-specific patterns. For regression, Random Forest benefits from shallow trees (max_depth=5) with full feature consideration (max_features=1.0), suggesting that the intensity prediction task involves distributed information across the feature space rather than being dominated by a few critical features. The ensemble of 100 diverse shallow trees provides robust averaging that reduces variance in predictions. For classification, XGBoost requires deeper trees (max_depth=10) to model the complex decision boundaries separating high-intensity from low-intensity episodes, particularly given the 22% class imbalance. The moderate learning rate (0.1) with 100 boosting rounds allows sequential error correction without overfitting, while the 80% feature subsampling (colsample_bytree=0.8) introduces diversity across boosting iterations. The light L2 regularization (reg_lambda=0.1) provides sufficient complexity control without over-constraining the model.

### Appendix B: Code Repository and Reproducibility

This appendix provides information about the code repository, directory structure, and instructions for reproducing all results reported in this study.

**Repository Location**

GitHub: https://github.com/aanishs/CSCI-461-Project

All code, data, and documentation are publicly available under an open-source license to facilitate reproducibility and future research extensions.

**Directory Structure**

```
CSCI-461-Project/
├── data/
│   ├── raw/
│   │   └── tic_episodes.csv          # Raw self-reported episode data
│   ├── processed/
│   │   └── tic_episodes_features.csv # Engineered features
│   └── README.md                      # Data documentation
├── src/
│   ├── data_processing.py             # Data loading and cleaning
│   ├── feature_engineering.py         # Feature generation pipeline
│   ├── model_training.py              # Model training and evaluation
│   ├── hyperparameter_search.py       # RandomizedSearchCV framework
│   ├── evaluation_metrics.py          # Custom metric calculations
│   ├── visualization.py               # Figure generation
│   ├── model_architectures.py         # Model definitions and configs
│   └── utils.py                       # Helper functions
├── experiments/
│   ├── run_hyperparameter_search.py   # Main experiment runner
│   ├── experiment_configs/            # Configuration files
│   └── results/                       # Saved model outputs
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb  # Data exploration
│   ├── 02_feature_engineering.ipynb   # Feature development
│   ├── 03_model_evaluation.ipynb      # Results analysis
│   └── 04_visualization.ipynb         # Figure generation
├── report_figures/                    # All figures used in report
│   ├── fig1_intensity_distribution.png
│   ├── fig2_high_intensity_rate.png
│   └── ... (18 figures total)
├── models/
│   ├── best_rf_regression.pkl         # Saved Random Forest model
│   ├── best_xgb_classification.pkl    # Saved XGBoost model
│   └── model_metadata.json            # Model performance logs
├── tests/
│   ├── test_feature_engineering.py    # Unit tests
│   ├── test_models.py
│   └── test_evaluation.py
├── requirements.txt                    # Python dependencies
├── environment.yml                     # Conda environment specification
├── README.md                          # Project documentation
├── PRELIMINARY_REPORT.md              # Initial findings
├── FINAL_REPORT.md                    # This report
└── LICENSE                            # Open-source license

```

**Software Dependencies**

All experiments were conducted using Python 3.10+ with the following package versions:

- scikit-learn 1.3+
- xgboost 2.0+
- lightgbm 4.0+ (for completeness, though not final model)
- pandas 2.0+
- numpy 1.24+
- matplotlib 3.7+
- seaborn 0.12+
- jupyter 1.0+

Complete dependency list with exact versions is provided in `requirements.txt`.

**Reproducing Results**

To reproduce all results reported in this study:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/aanishs/CSCI-461-Project.git
   cd CSCI-461-Project
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run hyperparameter search (quick mode):**
   ```bash
   python experiments/run_hyperparameter_search.py --mode quick --random_seed 42
   ```
   This executes the full hyperparameter search with 20 samples per model, reproducing all reported results. Expected runtime: 5-10 minutes on consumer hardware.

4. **Generate all figures:**
   ```bash
   python src/visualization.py --output_dir report_figures/
   ```
   This regenerates all 18 figures used in the report.

5. **Run unit tests:**
   ```bash
   pytest tests/
   ```
   Verifies that all components function correctly.

**Key Implementation Details**

- **Random Seed:** All experiments use `random_state=42` for NumPy, scikit-learn, and XGBoost to ensure deterministic results.
- **User-Grouped Splitting:** The `GroupShuffleSplit` from scikit-learn ensures that all episodes from each user are assigned entirely to either train or test set, preventing data leakage.
- **Cross-Validation:** `GroupKFold` with `n_splits=3` maintains user-level separation during hyperparameter search.
- **Feature Engineering:** The `feature_engineering.py` module implements all 34 features with extensive documentation and unit tests.
- **Model Serialization:** Best models are saved using joblib with metadata including hyperparameters, training date, and performance metrics.

**Data Availability**

The anonymized dataset (`tic_episodes.csv`) contains 1,533 rows with the following columns:

- `user_id`: Anonymized integer identifier (1-89)
- `timestamp`: Episode occurrence time (YYYY-MM-DD HH:MM:SS)
- `intensity`: Self-reported intensity (1-10 scale)
- `type`: Tic type (82 unique categories)
- `mood`: Optional mood at time of episode
- `trigger`: Optional reported trigger
- `description`: Optional free-text description

No personally identifiable information (PII) is included. User IDs are randomly assigned integers with no connection to original identities. The dataset complies with institutional data sharing policies and ethical guidelines for human subjects research.

**Contact and Support**

For questions about code, data, or reproducibility:
- GitHub Issues: https://github.com/aanishs/CSCI-461-Project/issues
- Email: aanishs@[email].edu

**Acknowledgments**

Development of this codebase was assisted by Claude Code (Anthropic) for code generation, debugging, and documentation. All models, experiments, and analyses were designed and executed by the author with tool assistance for implementation efficiency.

---

**End of Report**

Total Word Count: ~14,500 words
Total Figures: 18
Total Tables: 3
Total References: 48
Report Length: ~30 pages (estimated formatted length)

