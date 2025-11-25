"""
Temporal validation: Train on early time period, test on late time period.
Addresses limitation in FINAL_REPORT.md:532.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.metrics import mean_absolute_error, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_and_clean_data
from src.feature_engineering import FeatureEngineer
from src.target_generation import TargetGenerator
from src.validation import UserGroupedSplit


def temporal_split(df, train_pct=0.70):
    """
    Split data chronologically by timestamp.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'timestamp' column
    train_pct : float
        Proportion of time period for training

    Returns
    -------
    df_train, df_test : tuple of DataFrames
    """
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)

    # Get time range
    min_time = df_sorted['timestamp'].min()
    max_time = df_sorted['timestamp'].max()
    total_duration = (max_time - min_time).total_seconds()

    # Calculate split time
    split_time = min_time + pd.Timedelta(seconds=total_duration * train_pct)

    # Split
    df_train = df_sorted[df_sorted['timestamp'] <= split_time].copy()
    df_test = df_sorted[df_sorted['timestamp'] > split_time].copy()

    print(f"\nTemporal Split:")
    print(f"  Training period: {df_train['timestamp'].min().date()} to {df_train['timestamp'].max().date()}")
    print(f"  Test period: {df_test['timestamp'].min().date()} to {df_test['timestamp'].max().date()}")
    print(f"  Train episodes: {len(df_train)} ({len(df_train)/len(df)*100:.1f}%)")
    print(f"  Test episodes: {len(df_test)} ({len(df_test)/len(df)*100:.1f}%)")
    print(f"  Train users: {df_train['userId'].nunique()}")
    print(f"  Test users: {df_test['userId'].nunique()}")

    # Check overlap
    train_users = set(df_train['userId'].unique())
    test_users = set(df_test['userId'].unique())
    overlap_users = train_users & test_users
    print(f"  Overlapping users: {len(overlap_users)} ({len(overlap_users)/len(test_users)*100:.1f}% of test users)")

    return df_train, df_test


def compare_validation_strategies(df, feature_cols):
    """
    Compare temporal vs user-grouped validation.
    """
    print("\n" + "="*80)
    print("COMPARING VALIDATION STRATEGIES")
    print("="*80)

    # Prepare targets
    target_reg = 'target_next_intensity'
    target_clf = 'target_next_high_intensity'

    df_clean = df.dropna(subset=[target_reg, target_clf] + feature_cols)

    results = {}

    # STRATEGY 1: User-Grouped (current approach)
    print("\n--- Strategy 1: User-Grouped Split ---")
    splitter = UserGroupedSplit(test_size=0.2, random_state=42)
    df_train, df_test = splitter.split(df_clean)

    X_train = df_train[feature_cols]
    X_test = df_test[feature_cols]
    y_reg_train = df_train[target_reg]
    y_reg_test = df_test[target_reg]
    y_clf_train = df_train[target_clf]
    y_clf_test = df_test[target_clf]

    # Train models
    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf_reg.fit(X_train, y_reg_train)
    xgb_clf = XGBClassifier(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42, eval_metric='logloss')
    xgb_clf.fit(X_train, y_clf_train)

    # Evaluate
    reg_mae_user = mean_absolute_error(y_reg_test, rf_reg.predict(X_test))
    clf_f1_user = f1_score(y_clf_test, xgb_clf.predict(X_test))
    clf_prec_user = precision_score(y_clf_test, xgb_clf.predict(X_test), zero_division=0)
    clf_rec_user = recall_score(y_clf_test, xgb_clf.predict(X_test), zero_division=0)

    print(f"Regression MAE: {reg_mae_user:.4f}")
    print(f"Classification F1: {clf_f1_user:.4f}, Precision: {clf_prec_user:.4f}, Recall: {clf_rec_user:.4f}")

    results['user_grouped'] = {
        'regression_mae': reg_mae_user,
        'classification_f1': clf_f1_user,
        'classification_precision': clf_prec_user,
        'classification_recall': clf_rec_user
    }

    # STRATEGY 2: Temporal Split
    print("\n--- Strategy 2: Temporal Split ---")
    df_train_temp, df_test_temp = temporal_split(df_clean, train_pct=0.70)

    # Remove rows with NaN in features
    df_train_temp = df_train_temp.dropna(subset=feature_cols)
    df_test_temp = df_test_temp.dropna(subset=feature_cols)

    X_train_temp = df_train_temp[feature_cols]
    X_test_temp = df_test_temp[feature_cols]
    y_reg_train_temp = df_train_temp[target_reg]
    y_reg_test_temp = df_test_temp[target_reg]
    y_clf_train_temp = df_train_temp[target_clf]
    y_clf_test_temp = df_test_temp[target_clf]

    # Train models
    rf_reg_temp = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf_reg_temp.fit(X_train_temp, y_reg_train_temp)
    xgb_clf_temp = XGBClassifier(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42, eval_metric='logloss')
    xgb_clf_temp.fit(X_train_temp, y_clf_train_temp)

    # Evaluate
    reg_mae_temp = mean_absolute_error(y_reg_test_temp, rf_reg_temp.predict(X_test_temp))
    clf_f1_temp = f1_score(y_clf_test_temp, xgb_clf_temp.predict(X_test_temp))
    clf_prec_temp = precision_score(y_clf_test_temp, xgb_clf_temp.predict(X_test_temp), zero_division=0)
    clf_rec_temp = recall_score(y_clf_test_temp, xgb_clf_temp.predict(X_test_temp), zero_division=0)

    print(f"Regression MAE: {reg_mae_temp:.4f}")
    print(f"Classification F1: {clf_f1_temp:.4f}, Precision: {clf_prec_temp:.4f}, Recall: {clf_rec_temp:.4f}")

    results['temporal'] = {
        'regression_mae': reg_mae_temp,
        'classification_f1': clf_f1_temp,
        'classification_precision': clf_prec_temp,
        'classification_recall': clf_rec_temp
    }

    # Visualize comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: MAE comparison
    strategies = ['User-Grouped', 'Temporal']
    mae_values = [reg_mae_user, reg_mae_temp]

    axes[0].bar(strategies, mae_values, color=['#1f77b4', '#ff7f0e'], edgecolor='black')
    axes[0].set_ylabel('Mean Absolute Error', fontsize=11, fontweight='bold')
    axes[0].set_title('Regression Performance by Validation Strategy', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(mae_values):
        axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Plot 2: F1 comparison
    metrics = ['F1-Score', 'Precision', 'Recall']
    user_grouped_vals = [clf_f1_user, clf_prec_user, clf_rec_user]
    temporal_vals = [clf_f1_temp, clf_prec_temp, clf_rec_temp]

    x = np.arange(len(metrics))
    width = 0.35

    axes[1].bar(x - width/2, user_grouped_vals, width, label='User-Grouped',
                color='#1f77b4', edgecolor='black')
    axes[1].bar(x + width/2, temporal_vals, width, label='Temporal',
                color='#ff7f0e', edgecolor='black')
    axes[1].set_ylabel('Score', fontsize=11, fontweight='bold')
    axes[1].set_title('Classification Performance by Validation Strategy', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('report_figures/fig29_temporal_validation.png', dpi=300, bbox_inches='tight')
    print("\nSaved: report_figures/fig29_temporal_validation.png")
    plt.close()

    return results


def main():
    print("="*80)
    print("TEMPORAL VALIDATION ANALYSIS")
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

    # Compare validation strategies
    results = compare_validation_strategies(df, feature_cols)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY: VALIDATION STRATEGY COMPARISON")
    print("="*80)
    print("\nRegression (MAE):")
    print(f"  User-Grouped: {results['user_grouped']['regression_mae']:.4f}")
    print(f"  Temporal: {results['temporal']['regression_mae']:.4f}")
    diff_mae = abs(results['temporal']['regression_mae'] - results['user_grouped']['regression_mae'])
    print(f"  Difference: {diff_mae:.4f}")

    print("\nClassification (F1):")
    print(f"  User-Grouped: {results['user_grouped']['classification_f1']:.4f}")
    print(f"  Temporal: {results['temporal']['classification_f1']:.4f}")
    diff_f1 = abs(results['temporal']['classification_f1'] - results['user_grouped']['classification_f1'])
    print(f"  Difference: {diff_f1:.4f}")

    print("\nInterpretation:")
    if diff_mae < 0.1 and diff_f1 < 0.05:
        print("  ✓ Temporal and user-grouped validation yield similar performance")
        print("  ✓ Models generalize well across both space (users) and time")
    else:
        print("  ⚠ Performance differs between validation strategies")
        print("  ⚠ Models may not generalize equally well across time vs users")

    print("\n" + "="*80)
    print("TEMPORAL VALIDATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
