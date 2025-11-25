#!/usr/bin/env python3
"""
Additional visualizations for the final report.

Generates:
1. Learning curves (performance vs training set size)
2. Precision-Recall curves
3. ROC curves
4. Calibration plots
5. Residual plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (mean_absolute_error, precision_recall_curve, roc_curve,
                             auc, precision_score, recall_score, f1_score)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import learning_curve as sklearn_learning_curve
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_and_clean_data, filter_users_by_min_episodes
from feature_engineering import FeatureEngineer
from target_generation import TargetGenerator
from validation import UserGroupedSplit
from models import ModelFactory


def generate_all_visualizations(
    data_path: str = 'results (2).csv',
    output_dir: str = 'report_figures',
    high_intensity_threshold: int = 7,
    n_lags: int = 3,
    window_days: list = [7],
):
    """Generate all additional visualizations."""

    print("="*80)
    print("GENERATING ADDITIONAL VISUALIZATIONS")
    print("="*80)
    print()

    # Load and prepare data
    print("Loading and preparing data...")
    df = load_and_clean_data(data_path)
    df = filter_users_by_min_episodes(df, min_episodes=4)

    engineer = FeatureEngineer()
    df = engineer.create_all_features(df, n_lags=n_lags, window_days=window_days, fit=True)

    target_gen = TargetGenerator(high_intensity_threshold=high_intensity_threshold)
    df = target_gen.create_all_targets(df, k_days_list=[7])

    # Get feature columns
    exclude_cols = ['userId', 'timestamp', 'date', 'intensity', 'type', 'mood', 'trigger', 'description']
    exclude_cols += [col for col in df.columns if col.startswith('target_')]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    numeric_feature_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    # Split data
    splitter = UserGroupedSplit(test_size=0.2, random_state=42)
    train_df, test_df = splitter.split(df, user_col='userId')

    X_train = train_df[numeric_feature_cols]
    y_train_reg = train_df['target_next_intensity']
    y_train_clf = train_df['target_next_high_intensity']
    X_test = test_df[numeric_feature_cols]
    y_test_reg = test_df['target_next_intensity']
    y_test_clf = test_df['target_next_high_intensity']

    # Train models
    print("Training models...")
    factory = ModelFactory()

    rf_reg = factory.get_model('random_forest', task_type='regression')
    rf_reg.set_params(n_estimators=100, max_depth=5, random_state=42)
    rf_reg.fit(X_train, y_train_reg)

    xgb_clf = factory.get_model('xgboost', task_type='classification')
    xgb_clf.set_params(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42)
    xgb_clf.fit(X_train, y_train_clf)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Learning Curves
    print("\n1. Generating learning curves...")
    generate_learning_curves(rf_reg, xgb_clf, X_train, y_train_reg, y_train_clf, output_dir)

    # 2. PR and ROC Curves
    print("2. Generating PR and ROC curves...")
    if hasattr(xgb_clf, 'predict_proba'):
        y_proba = xgb_clf.predict_proba(X_test)[:, 1]
    else:
        y_proba = xgb_clf.predict(X_test)
    generate_pr_roc_curves(y_test_clf, y_proba, output_dir)

    # 3. Calibration Plot
    print("3. Generating calibration plot...")
    generate_calibration_plot(y_test_clf, y_proba, output_dir)

    # 4. Residual Plot
    print("4. Generating residual plot...")
    y_pred_reg = rf_reg.predict(X_test)
    generate_residual_plot(y_test_reg, y_pred_reg, output_dir)

    print("\n" + "="*80)
    print("ALL VISUALIZATIONS COMPLETE")
    print("="*80)


def generate_learning_curves(rf_reg, xgb_clf, X_train, y_train_reg, y_train_clf, output_dir):
    """Generate learning curves showing performance vs training set size."""

    factory = ModelFactory()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Regression learning curve
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_sizes_abs, train_scores, val_scores = sklearn_learning_curve(
        rf_reg, X_train, y_train_reg,
        train_sizes=train_sizes,
        cv=3,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        random_state=42
    )

    train_mae = -train_scores.mean(axis=1)
    val_mae = -val_scores.mean(axis=1)
    train_mae_std = train_scores.std(axis=1)
    val_mae_std = val_scores.std(axis=1)

    ax = axes[0]
    ax.plot(train_sizes_abs, train_mae, 'o-', color='blue', label='Training MAE', linewidth=2)
    ax.fill_between(train_sizes_abs, train_mae - train_mae_std, train_mae + train_mae_std,
                     alpha=0.2, color='blue')
    ax.plot(train_sizes_abs, val_mae, 'o-', color='red', label='Validation MAE', linewidth=2)
    ax.fill_between(train_sizes_abs, val_mae - val_mae_std, val_mae + val_mae_std,
                     alpha=0.2, color='red')
    ax.set_xlabel('Training Set Size', fontsize=12)
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title('Learning Curve: Regression (Random Forest)', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Classification learning curve (F1-score)
    # Manually compute F1 scores since sklearn doesn't provide F1 for learning_curve
    train_sizes_clf = [int(len(X_train) * p) for p in [0.2, 0.4, 0.6, 0.8, 1.0]]
    train_f1_scores = []
    val_f1_scores = []

    for size in train_sizes_clf:
        X_train_subset = X_train.iloc[:size]
        y_train_subset = y_train_clf.iloc[:size]

        xgb_temp = factory.get_model('xgboost', task_type='classification')
        xgb_temp.set_params(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42)
        xgb_temp.fit(X_train_subset, y_train_subset)

        # Training F1
        y_train_pred = xgb_temp.predict(X_train_subset)
        if hasattr(y_train_pred, 'shape') and len(y_train_pred.shape) > 1:
            y_train_pred = (y_train_pred[:, 1] >= 0.5).astype(int)
        train_f1_scores.append(f1_score(y_train_subset, y_train_pred, zero_division=0))

        # Validation F1 (on full validation set)
        y_val_pred = xgb_temp.predict(X_train.iloc[size:])
        if hasattr(y_val_pred, 'shape') and len(y_val_pred.shape) > 1:
            y_val_pred = (y_val_pred[:, 1] >= 0.5).astype(int)
        if len(y_train_clf.iloc[size:]) > 0:
            val_f1_scores.append(f1_score(y_train_clf.iloc[size:], y_val_pred, zero_division=0))
        else:
            val_f1_scores.append(0)

    ax = axes[1]
    ax.plot(train_sizes_clf, train_f1_scores, 'o-', color='blue', label='Training F1', linewidth=2)
    ax.plot(train_sizes_clf, val_f1_scores, 'o-', color='red', label='Validation F1', linewidth=2)
    ax.set_xlabel('Training Set Size', fontsize=12)
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('Learning Curve: Classification (XGBoost)', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = Path(output_dir) / 'fig21_learning_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   Saved: {save_path}")
    plt.close()


def generate_pr_roc_curves(y_true, y_proba, output_dir):
    """Generate Precision-Recall and ROC curves."""

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Precision-Recall Curve
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)

    ax = axes[0]
    ax.plot(recall, precision, color='blue', linewidth=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
    ax.fill_between(recall, precision, alpha=0.2, color='blue')

    # Mark some key thresholds
    for threshold in [0.04, 0.3, 0.5, 0.7]:
        idx = np.argmin(np.abs(thresholds_pr - threshold))
        if idx < len(recall):
            ax.scatter([recall[idx]], [precision[idx]], s=150, zorder=5,
                      label=f'θ={threshold:.2f}')

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # ROC Curve
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    ax = axes[1]
    ax.plot(fpr, tpr, color='red', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax.fill_between(fpr, tpr, alpha=0.2, color='red')

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()

    save_path_pr = Path(output_dir) / 'fig19_precision_recall_curve.png'
    save_path_roc = Path(output_dir) / 'fig20_roc_curve.png'

    # Save combined
    plt.savefig(Path(output_dir) / 'fig19_20_pr_roc_curves.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: {Path(output_dir) / 'fig19_20_pr_roc_curves.png'}")
    plt.close()


def generate_calibration_plot(y_true, y_proba, output_dir):
    """Generate calibration plot showing reliability of predicted probabilities."""

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_proba, n_bins=10, strategy='uniform'
    )

    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfectly Calibrated')
    ax.plot(mean_predicted_value, fraction_of_positives, 's-',
            linewidth=2, markersize=8, color='blue', label='XGBoost')
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title('Calibration Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Histogram of predicted probabilities
    ax = axes[1]
    ax.hist(y_proba[y_true == 0], bins=20, alpha=0.5, label='Negative Class', color='blue')
    ax.hist(y_proba[y_true == 1], bins=20, alpha=0.5, label='Positive Class', color='red')
    ax.axvline(x=0.5, color='black', linestyle='--', label='Default Threshold (0.5)')
    ax.axvline(x=0.04, color='green', linestyle='--', label='Optimal Threshold (0.04)')
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Predicted Probabilities', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = Path(output_dir) / 'fig22_calibration_plot.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   Saved: {save_path}")
    plt.close()


def generate_residual_plot(y_true, y_pred, output_dir):
    """Generate residual plot for regression model."""

    residuals = y_true - y_pred

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Residuals vs Predicted
    ax = axes[0, 0]
    ax.scatter(y_pred, residuals, alpha=0.5, s=30)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Predicted Intensity', fontsize=12)
    ax.set_ylabel('Residual (True - Predicted)', fontsize=12)
    ax.set_title('Residuals vs Predicted Values', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 2: Residuals vs True
    ax = axes[0, 1]
    ax.scatter(y_true, residuals, alpha=0.5, s=30)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('True Intensity', fontsize=12)
    ax.set_ylabel('Residual (True - Predicted)', fontsize=12)
    ax.set_title('Residuals vs True Values', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 3: Histogram of Residuals
    ax = axes[1, 0]
    ax.hist(residuals, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Residual', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Residuals', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add summary statistics
    ax.text(0.05, 0.95, f'Mean: {residuals.mean():.3f}\nStd: {residuals.std():.3f}\nMedian: {residuals.median():.3f}',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 4: Q-Q Plot
    ax = axes[1, 1]
    from scipy import stats as sp_stats
    sp_stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (Normality Check)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = Path(output_dir) / 'fig25_residual_plot.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   Saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    generate_all_visualizations()

    print("\nAll additional visualizations have been generated successfully!")
    print("Check the report_figures/ directory for the new plots.")
