"""
SHAP (SHapley Additive exPlanations) analysis for model explainability.
Addresses prelim Section 7.2.12 and report mentions (lines 483, 504).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier

from src.data_loader import load_and_clean_data
from src.feature_engineering import FeatureEngineer
from src.target_generation import TargetGenerator
from src.validation import UserGroupedSplit


def generate_shap_analysis(df, feature_cols):
    """
    Generate SHAP value analysis for Random Forest (regression) and XGBoost (classification).
    """
    print("\n" + "="*80)
    print("SHAP ANALYSIS FOR MODEL EXPLAINABILITY")
    print("="*80)

    # Prepare data
    target_reg = 'target_next_intensity'
    target_clf = 'target_next_high_intensity'

    df_clean = df.dropna(subset=[target_reg, target_clf] + feature_cols)

    X = df_clean[feature_cols]
    y_reg = df_clean[target_reg]
    y_clf = df_clean[target_clf]

    # Split data
    splitter = UserGroupedSplit(test_size=0.2, random_state=42)
    df_train, df_test = splitter.split(df_clean)

    X_train = df_train[feature_cols]
    X_test = df_test[feature_cols]
    y_reg_train = df_train[target_reg]
    y_clf_train = df_train[target_clf]

    # ----- REGRESSION: Random Forest -----
    print("\n--- Random Forest Regression SHAP Analysis ---")
    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf_reg.fit(X_train, y_reg_train)

    # Create SHAP explainer (using TreeExplainer for tree-based models)
    explainer_rf = shap.TreeExplainer(rf_reg)

    # Calculate SHAP values for test set (use sample for speed)
    sample_size = min(500, len(X_test))
    X_test_sample = X_test.sample(n=sample_size, random_state=42)
    shap_values_rf = explainer_rf.shap_values(X_test_sample)

    print(f"Calculated SHAP values for {sample_size} test instances")

    # Plot 1: SHAP Summary Plot (bar)
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values_rf, X_test_sample, plot_type="bar",
                     max_display=15, show=False)
    plt.title('SHAP Feature Importance - Random Forest Regression', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('report_figures/fig30_shap_regression_bar.png', dpi=300, bbox_inches='tight')
    print("Saved: report_figures/fig30_shap_regression_bar.png")
    plt.close()

    # Plot 2: SHAP Summary Plot (beeswarm)
    fig, ax = plt.subplots(figsize=(10, 10))
    shap.summary_plot(shap_values_rf, X_test_sample, max_display=15, show=False)
    plt.title('SHAP Summary - Random Forest Regression', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('report_figures/fig31_shap_regression_beeswarm.png', dpi=300, bbox_inches='tight')
    print("Saved: report_figures/fig31_shap_regression_beeswarm.png")
    plt.close()

    # Plot 3: SHAP Force Plot for sample predictions
    # Pick 3 interesting examples: low, medium, high intensity
    y_pred_sample = rf_reg.predict(X_test_sample)
    low_idx = np.argmin(y_pred_sample)
    high_idx = np.argmax(y_pred_sample)
    med_idx = np.argsort(y_pred_sample)[len(y_pred_sample)//2]

    for idx, label in [(low_idx, 'low'), (med_idx, 'medium'), (high_idx, 'high')]:
        shap.force_plot(explainer_rf.expected_value,
                       shap_values_rf[idx],
                       X_test_sample.iloc[idx],
                       matplotlib=True, show=False)
        plt.title(f'SHAP Force Plot - {label.capitalize()} Intensity Prediction',
                 fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'report_figures/fig32_shap_force_{label}.png', dpi=300, bbox_inches='tight')
        print(f"Saved: report_figures/fig32_shap_force_{label}.png")
        plt.close()

    # ----- CLASSIFICATION: XGBoost -----
    print("\n--- XGBoost Classification SHAP Analysis ---")
    xgb_clf = XGBClassifier(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42, eval_metric='logloss')
    xgb_clf.fit(X_train, y_clf_train)

    # Create SHAP explainer
    explainer_xgb = shap.TreeExplainer(xgb_clf)

    # Calculate SHAP values
    X_test_sample_clf = X_test.sample(n=sample_size, random_state=42)
    shap_values_xgb = explainer_xgb.shap_values(X_test_sample_clf)

    # For binary classification, shap_values might be 2D or 3D
    # If 3D, take values for positive class
    if isinstance(shap_values_xgb, list):
        shap_values_xgb = shap_values_xgb[1] if len(shap_values_xgb) > 1 else shap_values_xgb[0]

    # Plot 4: SHAP Summary Plot (bar) for classification
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values_xgb, X_test_sample_clf, plot_type="bar",
                     max_display=15, show=False)
    plt.title('SHAP Feature Importance - XGBoost Classification', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('report_figures/fig33_shap_classification_bar.png', dpi=300, bbox_inches='tight')
    print("Saved: report_figures/fig33_shap_classification_bar.png")
    plt.close()

    # Plot 5: SHAP Summary Plot (beeswarm) for classification
    fig, ax = plt.subplots(figsize=(10, 10))
    shap.summary_plot(shap_values_xgb, X_test_sample_clf, max_display=15, show=False)
    plt.title('SHAP Summary - XGBoost Classification', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('report_figures/fig34_shap_classification_beeswarm.png', dpi=300, bbox_inches='tight')
    print("Saved: report_figures/fig34_shap_classification_beeswarm.png")
    plt.close()

    # Compute mean absolute SHAP values for ranking
    mean_shap_rf = np.abs(shap_values_rf).mean(axis=0)
    mean_shap_xgb = np.abs(shap_values_xgb).mean(axis=0)

    shap_importance_rf = pd.DataFrame({
        'feature': feature_cols,
        'mean_abs_shap': mean_shap_rf
    }).sort_values('mean_abs_shap', ascending=False)

    shap_importance_xgb = pd.DataFrame({
        'feature': feature_cols,
        'mean_abs_shap': mean_shap_xgb
    }).sort_values('mean_abs_shap', ascending=False)

    print("\nTop 10 Features by Mean Absolute SHAP (Regression):")
    print(shap_importance_rf.head(10).to_string(index=False))

    print("\nTop 10 Features by Mean Absolute SHAP (Classification):")
    print(shap_importance_xgb.head(10).to_string(index=False))

    return {
        'regression': shap_importance_rf,
        'classification': shap_importance_xgb
    }


def main():
    print("="*80)
    print("SHAP EXPLAINABILITY ANALYSIS")
    print("="*80)

    # Load data
    df = load_and_clean_data('results (2).csv')
    print(f"Loaded {len(df)} episodes from {df['userId'].nunique()} users")

    # Generate features
    fe = FeatureEngineer()
    df = fe.create_all_features(df, n_lags=3, window_days=[7], fit=True)

    # Generate targets
    tg = TargetGenerator(high_intensity_threshold=7)
    df = tg.create_next_intensity_target(df)

    # Get feature columns
    feature_cols = fe.get_feature_columns(df,
                                         include_sequence=True,
                                         include_time_window=True,
                                         include_engineered=True)

    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Total episodes: {len(df)}")

    # Generate SHAP analysis
    shap_results = generate_shap_analysis(df, feature_cols)

    print("\n" + "="*80)
    print("SHAP ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated figures:")
    print("  - fig30_shap_regression_bar.png")
    print("  - fig31_shap_regression_beeswarm.png")
    print("  - fig32_shap_force_low.png")
    print("  - fig32_shap_force_medium.png")
    print("  - fig32_shap_force_high.png")
    print("  - fig33_shap_classification_bar.png")
    print("  - fig34_shap_classification_beeswarm.png")


if __name__ == "__main__":
    main()
