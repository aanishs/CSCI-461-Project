"""
Feature ablation study: Test performance with different feature subsets.
Addresses prelim Section 7.2.2 and report limitation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.metrics import mean_absolute_error, f1_score, precision_score, recall_score

from data_loader import load_and_clean_data
from feature_engineering import FeatureEngineer
from target_generation import TargetGenerator
from validation import UserGroupedSplit

def define_feature_configurations(all_features):
    """
    Define 7 feature configurations for ablation study.
    """
    configs = {}

    # Config 1: Temporal only
    configs['temporal_only'] = [f for f in all_features if any(
        x in f for x in ['hour', 'day_of_week', 'is_weekend', 'day_of_month', 'month', 'timeOfDay_encoded']
    )]

    # Config 2: Sequence only
    configs['sequence_only'] = [f for f in all_features if any(
        x in f for x in ['prev_intensity', 'time_since_prev', 'prev_type', 'prev_timeOfDay']
    )]

    # Config 3: Time-window only
    configs['window_only'] = [f for f in all_features if f.startswith('window_')]

    # Config 4: User-level only
    configs['user_only'] = [f for f in all_features if f.startswith('user_')]

    # Config 5: Categorical only
    configs['categorical_only'] = [f for f in all_features if any(
        x in f for x in ['type_encoded', 'mood_encoded', 'trigger_encoded', 'has_mood', 'has_trigger']
    )]

    # Config 6: All except engineered
    engineered = [f for f in all_features if any(
        x in f for x in ['intensity_x_count', 'intensity_trend', 'volatility', 'mood_x', 'trigger_x', 'weekend_x']
    )]
    configs['no_engineered'] = [f for f in all_features if f not in engineered]

    # Config 7: All features
    configs['all_features'] = all_features

    return configs

def evaluate_configuration(X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test,
                          features, config_name):
    """
    Evaluate a single feature configuration.
    """
    print(f"\n--- Configuration: {config_name} ({len(features)} features) ---")

    # Select features
    X_train_subset = X_train[features]
    X_test_subset = X_test[features]

    # REGRESSION
    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    rf_reg.fit(X_train_subset, y_reg_train)
    reg_mae = mean_absolute_error(y_reg_test, rf_reg.predict(X_test_subset))

    # CLASSIFICATION
    xgb_clf = XGBClassifier(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42)
    xgb_clf.fit(X_train_subset, y_clf_train)
    clf_pred = xgb_clf.predict(X_test_subset)
    clf_f1 = f1_score(y_clf_test, clf_pred)
    clf_prec = precision_score(y_clf_test, clf_pred)
    clf_rec = recall_score(y_clf_test, clf_pred)

    print(f"Regression MAE: {reg_mae:.4f}")
    print(f"Classification F1: {clf_f1:.4f}, Precision: {clf_prec:.4f}, Recall: {clf_rec:.4f}")

    return {
        'config': config_name,
        'num_features': len(features),
        'regression_mae': reg_mae,
        'classification_f1': clf_f1,
        'classification_precision': clf_prec,
        'classification_recall': clf_rec
    }

def run_ablation_study(df, all_feature_cols):
    """
    Run complete ablation study across all configurations.
    """
    print("\n" + "="*80)
    print("FEATURE ABLATION STUDY")
    print("="*80)

    # Prepare data
    target_reg = 'target_next_intensity'
    target_clf = 'target_next_high_intensity'

    df_clean = df.dropna(subset=[target_reg, target_clf] + all_feature_cols)

    X = df_clean[all_feature_cols]
    y_reg = df_clean[target_reg]
    y_clf = df_clean[target_clf]

    # Split data using UserGroupedSplit
    splitter = UserGroupedSplit(test_size=0.2, random_state=42)
    df_train, df_test = splitter.split(df_clean)

    X_train = df_train[all_feature_cols]
    X_test = df_test[all_feature_cols]
    y_reg_train = df_train[target_reg]
    y_reg_test = df_test[target_reg]
    y_clf_train = df_train[target_clf]
    y_clf_test = df_test[target_clf]

    # Define configurations
    configs = define_feature_configurations(all_feature_cols)

    print(f"\nTotal features available: {len(all_feature_cols)}")
    print(f"Testing {len(configs)} configurations:")
    for name, features in configs.items():
        print(f"  - {name}: {len(features)} features")

    # Evaluate each configuration
    results = []
    for config_name, features in configs.items():
        result = evaluate_configuration(
            X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test,
            features, config_name
        )
        results.append(result)

    results_df = pd.DataFrame(results)

    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: MAE by configuration
    ax1 = axes[0, 0]
    results_sorted = results_df.sort_values('regression_mae')
    colors = ['#2ecc71' if x == 'all_features' else '#3498db' for x in results_sorted['config']]
    ax1.barh(results_sorted['config'], results_sorted['regression_mae'], color=colors, edgecolor='black')
    ax1.set_xlabel('Mean Absolute Error (Lower is Better)', fontsize=12, fontweight='bold')
    ax1.set_title('Regression Performance by Feature Configuration', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.axvline(results_df[results_df['config'] == 'all_features']['regression_mae'].values[0],
                color='red', linestyle='--', label='All Features Baseline')
    ax1.legend()

    # Plot 2: F1 by configuration
    ax2 = axes[0, 1]
    results_sorted = results_df.sort_values('classification_f1', ascending=False)
    colors = ['#2ecc71' if x == 'all_features' else '#e74c3c' for x in results_sorted['config']]
    ax2.barh(results_sorted['config'], results_sorted['classification_f1'], color=colors, edgecolor='black')
    ax2.set_xlabel('F1-Score (Higher is Better)', fontsize=12, fontweight='bold')
    ax2.set_title('Classification Performance by Feature Configuration', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.axvline(results_df[results_df['config'] == 'all_features']['classification_f1'].values[0],
                color='red', linestyle='--', label='All Features Baseline')
    ax2.legend()

    # Plot 3: Performance vs number of features (regression)
    ax3 = axes[1, 0]
    ax3.scatter(results_df['num_features'], results_df['regression_mae'], s=150, alpha=0.7, edgecolor='black')
    for idx, row in results_df.iterrows():
        ax3.annotate(row['config'], (row['num_features'], row['regression_mae']),
                    fontsize=9, ha='left', va='bottom')
    ax3.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
    ax3.set_ylabel('MAE (Lower is Better)', fontsize=12, fontweight='bold')
    ax3.set_title('Regression: Performance vs Feature Count', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Performance vs number of features (classification)
    ax4 = axes[1, 1]
    ax4.scatter(results_df['num_features'], results_df['classification_f1'], s=150, alpha=0.7, edgecolor='black')
    for idx, row in results_df.iterrows():
        ax4.annotate(row['config'], (row['num_features'], row['classification_f1']),
                    fontsize=9, ha='left', va='bottom')
    ax4.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
    ax4.set_ylabel('F1-Score (Higher is Better)', fontsize=12, fontweight='bold')
    ax4.set_title('Classification: Performance vs Feature Count', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('report_figures/fig35_feature_ablation.png', dpi=300, bbox_inches='tight')
    print("\nSaved: report_figures/fig35_feature_ablation.png")

    return results_df

def main():
    print("="*80)
    print("FEATURE ABLATION STUDY")
    print("="*80)

    # Load data
    df = load_and_clean_data('results (2).csv')

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

    print(f"\nTotal features: {len(feature_cols)}")

    # Run ablation study
    results = run_ablation_study(df, feature_cols)

    # Print summary
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)
    print("\n" + results.to_string(index=False))

    # Identify best configuration
    best_reg_config = results.loc[results['regression_mae'].idxmin()]
    best_clf_config = results.loc[results['classification_f1'].idxmax()]

    print(f"\nBest Regression Configuration: {best_reg_config['config']}")
    print(f"  Features: {best_reg_config['num_features']}, MAE: {best_reg_config['regression_mae']:.4f}")

    print(f"\nBest Classification Configuration: {best_clf_config['config']}")
    print(f"  Features: {best_clf_config['num_features']}, F1: {best_clf_config['classification_f1']:.4f}")

    # Calculate performance drop for simplified models
    all_features_mae = results[results['config'] == 'all_features']['regression_mae'].values[0]
    all_features_f1 = results[results['config'] == 'all_features']['classification_f1'].values[0]

    print(f"\nPerformance vs All Features:")
    for _, row in results.iterrows():
        if row['config'] != 'all_features':
            mae_drop = ((row['regression_mae'] - all_features_mae) / all_features_mae) * 100
            f1_drop = ((all_features_f1 - row['classification_f1']) / all_features_f1) * 100
            print(f"  {row['config']}: MAE {mae_drop:+.1f}%, F1 {f1_drop:+.1f}%")

if __name__ == "__main__":
    main()
