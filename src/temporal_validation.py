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
from sklearn.utils import resample
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


def temporal_split_by_date(df, split_date='2025-08-01'):
    """
    Split data by a specific date cutoff.
    Train on everything before split_date, test on everything after.
    Same users can appear in both train and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'timestamp' column
    split_date : str
        Date cutoff in 'YYYY-MM-DD' format (default: '2025-08-01')

    Returns
    -------
    df_train, df_test : tuple of DataFrames
    """
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)

    # Convert split_date to timestamp, matching timezone of timestamp column
    split_timestamp = pd.Timestamp(split_date, tz='UTC')

    # Split
    df_train = df_sorted[df_sorted['timestamp'] < split_timestamp].copy()
    df_test = df_sorted[df_sorted['timestamp'] >= split_timestamp].copy()

    print(f"\nTemporal Split by Date (cutoff: {split_date}):")
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
    print(f"  Users only in train: {len(train_users - test_users)}")
    print(f"  Users only in test: {len(test_users - train_users)}")

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

    # STRATEGY 2: Temporal Split (by date - August 2025 cutoff)
    print("\n--- Strategy 2: Temporal Split (August 2025 Cutoff) ---")
    df_train_temp, df_test_temp = temporal_split_by_date(df_clean, split_date='2025-08-01')

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

    # Calculate bootstrap confidence intervals
    def bootstrap_ci(y_true, y_pred, metric_func, n_bootstrap=1000, ci=0.95):
        """Calculate bootstrap confidence interval for a metric."""
        bootstrap_scores = []
        n_samples = len(y_true)

        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = resample(range(n_samples), n_samples=n_samples, replace=True)
            y_true_boot = [y_true.iloc[i] if hasattr(y_true, 'iloc') else y_true[i] for i in indices]
            y_pred_boot = [y_pred[i] for i in indices]

            # Calculate metric
            score = metric_func(y_true_boot, y_pred_boot)
            bootstrap_scores.append(score)

        # Calculate confidence interval
        alpha = 1 - ci
        lower = np.percentile(bootstrap_scores, (alpha/2) * 100)
        upper = np.percentile(bootstrap_scores, (1 - alpha/2) * 100)
        return upper - lower  # Return the total interval width

    # Calculate bootstrap CIs for regression
    reg_user_ci = bootstrap_ci(y_reg_test, rf_reg.predict(X_test), mean_absolute_error)
    reg_temp_ci = bootstrap_ci(y_reg_test_temp, rf_reg_temp.predict(X_test_temp), mean_absolute_error)

    # Calculate bootstrap CIs for classification
    clf_f1_user_ci = bootstrap_ci(y_clf_test, xgb_clf.predict(X_test), f1_score)
    clf_f1_temp_ci = bootstrap_ci(y_clf_test_temp, xgb_clf_temp.predict(X_test_temp), f1_score)
    clf_prec_user_ci = bootstrap_ci(y_clf_test, xgb_clf.predict(X_test),
                                     lambda y_t, y_p: precision_score(y_t, y_p, zero_division=0))
    clf_prec_temp_ci = bootstrap_ci(y_clf_test_temp, xgb_clf_temp.predict(X_test_temp),
                                     lambda y_t, y_p: precision_score(y_t, y_p, zero_division=0))
    clf_rec_user_ci = bootstrap_ci(y_clf_test, xgb_clf.predict(X_test),
                                    lambda y_t, y_p: recall_score(y_t, y_p, zero_division=0))
    clf_rec_temp_ci = bootstrap_ci(y_clf_test_temp, xgb_clf_temp.predict(X_test_temp),
                                    lambda y_t, y_p: recall_score(y_t, y_p, zero_division=0))

    # Visualize comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: MAE comparison with bootstrap error bars
    strategies = ['User-Grouped', 'Temporal']
    mae_values = [reg_mae_user, reg_mae_temp]
    mae_errors = [reg_user_ci / 2, reg_temp_ci / 2]  # Half interval for +/- error bars

    axes[0].bar(strategies, mae_values, yerr=mae_errors, capsize=8,
                color=['#1f77b4', '#ff7f0e'], edgecolor='black',
                error_kw={'linewidth': 2, 'ecolor': 'black'})
    axes[0].set_ylabel('Mean Absolute Error', fontsize=11, fontweight='bold')
    axes[0].set_title('Regression Performance by Validation Strategy', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, (v, err) in enumerate(zip(mae_values, mae_errors)):
        axes[0].text(i, v + err + 0.02, f'{v:.3f}', ha='center', va='bottom',
                     fontsize=12, fontweight='bold')

    # Plot 2: F1 comparison with bootstrap error bars
    metrics = ['F1-Score', 'Precision', 'Recall']
    user_grouped_vals = [clf_f1_user, clf_prec_user, clf_rec_user]
    temporal_vals = [clf_f1_temp, clf_prec_temp, clf_rec_temp]
    user_grouped_errors = [clf_f1_user_ci / 2, clf_prec_user_ci / 2, clf_rec_user_ci / 2]
    temporal_errors = [clf_f1_temp_ci / 2, clf_prec_temp_ci / 2, clf_rec_temp_ci / 2]

    x = np.arange(len(metrics))
    width = 0.35

    axes[1].bar(x - width/2, user_grouped_vals, width, yerr=user_grouped_errors,
                capsize=5, label='User-Grouped', color='#1f77b4', edgecolor='black',
                error_kw={'linewidth': 1.5, 'ecolor': 'black'})
    axes[1].bar(x + width/2, temporal_vals, width, yerr=temporal_errors,
                capsize=5, label='Temporal', color='#ff7f0e', edgecolor='black',
                error_kw={'linewidth': 1.5, 'ecolor': 'black'})
    axes[1].set_ylabel('Score', fontsize=11, fontweight='bold')
    axes[1].set_title('Classification Performance by Validation Strategy', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('report_figures/fig29_temporal_validation.png', dpi=300, bbox_inches='tight')
    print("\nSaved: report_figures/fig29_temporal_validation.png (with bootstrap error bars)")
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
