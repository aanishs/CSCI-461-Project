"""
Evaluate Target 2 (future count) and Target 3 (time-to-event) predictions.
Addresses RQ3 from preliminary report.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_and_clean_data
from src.feature_engineering import FeatureEngineer
from src.target_generation import TargetGenerator
from src.validation import UserGroupedSplit

def evaluate_target_2_future_count(df, feature_cols):
    """Evaluate Target 2: Count of high-intensity episodes in next 7 days."""

    print("\n" + "="*80)
    print("TARGET 2: FUTURE HIGH-INTENSITY COUNT (Next 7 Days)")
    print("="*80)

    # Target columns
    regression_target = 'target_high_count_next_7d'
    classification_target = 'target_has_high_next_7d'

    # Remove rows with missing targets
    df_clean = df.dropna(subset=[regression_target, classification_target] + feature_cols)

    print(f"\nDataset: {len(df_clean)} episodes")
    print(f"Mean high-intensity count (next 7d): {df_clean[regression_target].mean():.2f}")
    print(f"Rate of having high-intensity in next 7d: {df_clean[classification_target].mean()*100:.1f}%")

    # Split data
    X = df_clean[feature_cols]
    y_reg = df_clean[regression_target]
    y_clf = df_clean[classification_target]

    # User-grouped split
    splitter = UserGroupedSplit(test_size=0.2, random_state=42)
    df_train, df_test = splitter.split(df_clean)

    X_train = df_train[feature_cols]
    X_test = df_test[feature_cols]
    y_reg_train = df_train[regression_target]
    y_reg_test = df_test[regression_target]
    y_clf_train = df_train[classification_target]
    y_clf_test = df_test[classification_target]

    # REGRESSION: Predict count
    print("\n--- Regression: High-Intensity Count ---")
    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf_reg.fit(X_train, y_reg_train)

    y_pred_train = rf_reg.predict(X_train)
    y_pred_test = rf_reg.predict(X_test)

    train_mae = mean_absolute_error(y_reg_train, y_pred_train)
    test_mae = mean_absolute_error(y_reg_test, y_pred_test)
    train_rmse = mean_squared_error(y_reg_train, y_pred_train, squared=False)
    test_rmse = mean_squared_error(y_reg_test, y_pred_test, squared=False)
    train_r2 = r2_score(y_reg_train, y_pred_train)
    test_r2 = r2_score(y_reg_test, y_pred_test)

    print(f"Train MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
    print(f"Test  MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")

    # CLASSIFICATION: Will there be any high-intensity episode?
    print("\n--- Classification: Any High-Intensity in Next 7 Days ---")
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_clf.fit(X_train, y_clf_train)

    y_pred_train_clf = rf_clf.predict(X_train)
    y_pred_test_clf = rf_clf.predict(X_test)
    y_prob_test = rf_clf.predict_proba(X_test)[:, 1]

    train_f1 = f1_score(y_clf_train, y_pred_train_clf)
    test_f1 = f1_score(y_clf_test, y_pred_test_clf)
    test_precision = precision_score(y_clf_test, y_pred_test_clf, zero_division=0)
    test_recall = recall_score(y_clf_test, y_pred_test_clf, zero_division=0)
    test_pr_auc = average_precision_score(y_clf_test, y_prob_test)

    print(f"Train F1: {train_f1:.4f}")
    print(f"Test F1: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, PR-AUC: {test_pr_auc:.4f}")

    # Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Actual vs Predicted Count
    axes[0].scatter(y_reg_test, y_pred_test, alpha=0.5)
    max_val = max(y_reg_test.max(), y_pred_test.max())
    axes[0].plot([0, max_val], [0, max_val], 'r--', lw=2)
    axes[0].set_xlabel('Actual High-Intensity Count (Next 7 Days)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Predicted Count', fontsize=11, fontweight='bold')
    axes[0].set_title(f'Target 2 Regression: MAE={test_mae:.2f}', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Distribution of predicted vs actual
    axes[1].hist(y_reg_test, bins=20, alpha=0.5, label='Actual', edgecolor='black')
    axes[1].hist(y_pred_test, bins=20, alpha=0.5, label='Predicted', edgecolor='black')
    axes[1].set_xlabel('High-Intensity Count (Next 7 Days)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[1].set_title('Distribution: Actual vs Predicted', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('report_figures/fig27_target2_future_count.png', dpi=300, bbox_inches='tight')
    print("\nSaved: report_figures/fig27_target2_future_count.png")
    plt.close()

    return {
        'regression': {'test_mae': test_mae, 'test_rmse': test_rmse, 'test_r2': test_r2},
        'classification': {'test_f1': test_f1, 'test_precision': test_precision,
                          'test_recall': test_recall, 'test_pr_auc': test_pr_auc}
    }


def evaluate_target_3_time_to_event(df, feature_cols):
    """Evaluate Target 3: Time until next high-intensity episode."""

    print("\n" + "="*80)
    print("TARGET 3: TIME TO NEXT HIGH-INTENSITY EPISODE")
    print("="*80)

    # Target columns
    target_time = 'target_time_to_high_days'
    target_event = 'target_event_occurred'

    # Remove rows with missing targets
    df_clean = df.dropna(subset=[target_time, target_event] + feature_cols)

    print(f"\nDataset: {len(df_clean)} episodes")
    print(f"Mean time to next high-intensity: {df_clean[target_time].mean():.2f} days")
    print(f"Event occurrence rate: {df_clean[target_event].mean()*100:.1f}%")
    print(f"Censored rate: {(1 - df_clean[target_event].mean())*100:.1f}%")

    # Split data
    splitter = UserGroupedSplit(test_size=0.2, random_state=42)
    df_train, df_test = splitter.split(df_clean)

    X_train = df_train[feature_cols]
    X_test = df_test[feature_cols]
    y_time_train = df_train[target_time]
    y_time_test = df_test[target_time]
    y_event_train = df_train[target_event]
    y_event_test = df_test[target_event]

    # REGRESSION: Predict time to event (treating censored as observed for now)
    print("\n--- Regression: Time to High-Intensity (Days) ---")
    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf_reg.fit(X_train, y_time_train)

    y_pred_train = rf_reg.predict(X_train)
    y_pred_test = rf_reg.predict(X_test)

    train_mae = mean_absolute_error(y_time_train, y_pred_train)
    test_mae = mean_absolute_error(y_time_test, y_pred_test)
    train_rmse = mean_squared_error(y_time_train, y_pred_train, squared=False)
    test_rmse = mean_squared_error(y_time_test, y_pred_test, squared=False)
    train_r2 = r2_score(y_time_train, y_pred_train)
    test_r2 = r2_score(y_time_test, y_pred_test)

    print(f"Train MAE: {train_mae:.4f} days, RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
    print(f"Test  MAE: {test_mae:.4f} days, RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")

    # Stratify by censoring status
    uncensored_mask = y_event_test == 1
    if uncensored_mask.sum() > 0:
        uncensored_mae = mean_absolute_error(y_time_test[uncensored_mask],
                                              y_pred_test[uncensored_mask])
        print(f"Test MAE (uncensored only): {uncensored_mae:.4f} days")

    # Visualizations
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Actual vs Predicted Time
    axes[0].scatter(y_time_test[y_event_test==1], y_pred_test[y_event_test==1],
                    alpha=0.5, label='Uncensored', color='blue')
    axes[0].scatter(y_time_test[y_event_test==0], y_pred_test[y_event_test==0],
                    alpha=0.3, label='Censored', color='red', marker='x')
    max_val = max(y_time_test.max(), y_pred_test.max())
    axes[0].plot([0, max_val], [0, max_val], 'k--', lw=2)
    axes[0].set_xlabel('Actual Time to High-Intensity (Days)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Predicted Time (Days)', fontsize=11, fontweight='bold')
    axes[0].set_title(f'Target 3 Regression: MAE={test_mae:.2f} days', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Distribution by censoring status
    axes[1].hist(y_time_test[y_event_test==1], bins=30, alpha=0.5,
                 label='Uncensored', edgecolor='black', color='blue')
    axes[1].hist(y_time_test[y_event_test==0], bins=30, alpha=0.5,
                 label='Censored', edgecolor='black', color='red')
    axes[1].set_xlabel('Time to High-Intensity (Days)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[1].set_title('Time Distribution by Event Status', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Prediction error by time range
    errors = np.abs(y_time_test - y_pred_test)
    time_bins = pd.cut(y_time_test, bins=[0, 2, 7, 14, 100],
                       labels=['0-2d', '2-7d', '7-14d', '14+d'])
    error_by_bin = pd.DataFrame({'error': errors, 'time_bin': time_bins})
    error_by_bin.boxplot(column='error', by='time_bin', ax=axes[2])
    axes[2].set_xlabel('Time Range', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('Absolute Error (Days)', fontsize=11, fontweight='bold')
    axes[2].set_title('Prediction Error by Time Range', fontsize=12, fontweight='bold')
    axes[2].get_figure().suptitle('')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('report_figures/fig28_target3_time_to_event.png', dpi=300, bbox_inches='tight')
    print("\nSaved: report_figures/fig28_target3_time_to_event.png")
    plt.close()

    return {
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'event_rate': y_event_test.mean()
    }


def main():
    print("="*80)
    print("EVALUATING TARGETS 2-3: FUTURE PREDICTIONS")
    print("="*80)

    # Load data
    df = load_and_clean_data('results (2).csv')
    print(f"Loaded {len(df)} episodes from {df['userId'].nunique()} users")

    # Generate features
    fe = FeatureEngineer()
    df = fe.create_all_features(df, n_lags=3, window_days=[3, 7, 14], fit=True)

    # Generate all targets
    tg = TargetGenerator(high_intensity_threshold=7)
    df = tg.create_all_targets(df, k_days_list=[1, 3, 7, 14])

    # Get feature columns
    feature_cols = fe.get_feature_columns(df,
                                         include_sequence=True,
                                         include_time_window=True,
                                         include_engineered=True)

    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Total episodes: {len(df)}")

    # Evaluate Target 2
    target2_results = evaluate_target_2_future_count(df, feature_cols)

    # Evaluate Target 3
    target3_results = evaluate_target_3_time_to_event(df, feature_cols)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY OF FUTURE PREDICTION RESULTS")
    print("="*80)
    print("\nTarget 2 (Future Count - Next 7 Days):")
    print(f"  Regression MAE: {target2_results['regression']['test_mae']:.4f} episodes")
    print(f"  Regression RMSE: {target2_results['regression']['test_rmse']:.4f}")
    print(f"  Regression R²: {target2_results['regression']['test_r2']:.4f}")
    print(f"  Classification F1: {target2_results['classification']['test_f1']:.4f}")
    print(f"  Classification Precision: {target2_results['classification']['test_precision']:.4f}")
    print(f"  Classification Recall: {target2_results['classification']['test_recall']:.4f}")
    print(f"  PR-AUC: {target2_results['classification']['test_pr_auc']:.4f}")

    print("\nTarget 3 (Time to Event):")
    print(f"  MAE: {target3_results['test_mae']:.4f} days")
    print(f"  RMSE: {target3_results['test_rmse']:.4f} days")
    print(f"  R²: {target3_results['test_r2']:.4f}")
    print(f"  Event Rate: {target3_results['event_rate']*100:.1f}%")

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print("\nGenerated figures:")
    print("  - report_figures/fig27_target2_future_count.png")
    print("  - report_figures/fig28_target3_time_to_event.png")


if __name__ == "__main__":
    main()
