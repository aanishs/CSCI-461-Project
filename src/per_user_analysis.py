#!/usr/bin/env python3
"""
Per-user performance analysis.

Analyze model performance stratified by user engagement level.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, f1_score, precision_score, recall_score
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_and_clean_data, filter_users_by_min_episodes
from feature_engineering import FeatureEngineer
from target_generation import TargetGenerator
from validation import UserGroupedSplit
from models import ModelFactory


def analyze_per_user_performance(
    data_path: str = 'results (2).csv',
    output_dir: str = 'report_figures',
    high_intensity_threshold: int = 7,
    n_lags: int = 3,
    window_days: list = [7],
):
    """
    Analyze model performance stratified by user engagement level.

    Creates visualizations showing:
    1. Performance vs episodes per user
    2. Performance by user tier (sparse, medium, high engagement)
    3. User-level prediction error distribution
    """
    print("="*80)
    print("PER-USER PERFORMANCE ANALYSIS")
    print("="*80)
    print()

    # Load and prepare data
    print("Loading data...")
    df = load_and_clean_data(data_path)
    df = filter_users_by_min_episodes(df, min_episodes=4)

    print("Engineering features...")
    engineer = FeatureEngineer()
    df = engineer.create_all_features(
        df,
        n_lags=n_lags,
        window_days=window_days,
        fit=True
    )

    print("Generating targets...")
    target_gen = TargetGenerator(high_intensity_threshold=high_intensity_threshold)
    df = target_gen.create_all_targets(df, k_days_list=[7])

    # Get feature columns
    exclude_cols = ['userId', 'timestamp', 'date', 'intensity', 'type', 'mood', 'trigger', 'description',
                    'target_next_intensity', 'target_next_high_intensity',
                    'target_count_next_7d', 'target_high_count_next_7d', 'target_has_high_next_7d',
                    'target_time_to_high_days', 'target_time_to_high_censored']
    feature_cols = [col for col in df.columns if col not in exclude_cols and not col.startswith('target_')]
    numeric_feature_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    # Split data
    print("Splitting data (user-grouped)...")
    splitter = UserGroupedSplit(test_size=0.2, random_state=42)
    train_df, test_df = splitter.split(df, user_col='userId')

    # Calculate episodes per user
    episodes_per_user = df.groupby('userId').size().to_dict()

    # Categorize users
    def categorize_user(n_episodes):
        if n_episodes < 10:
            return 'Sparse (1-9)'
        elif n_episodes < 50:
            return 'Medium (10-49)'
        else:
            return 'High (50+)'

    df['user_tier'] = df['userId'].map(lambda x: categorize_user(episodes_per_user[x]))
    train_df['user_tier'] = train_df['userId'].map(lambda x: categorize_user(episodes_per_user[x]))
    test_df['user_tier'] = test_df['userId'].map(lambda x: categorize_user(episodes_per_user[x]))

    print(f"\nUser distribution:")
    print(df.groupby('user_tier')['userId'].nunique())
    print(f"\nEpisode distribution by tier:")
    print(df.groupby('user_tier').size())

    # Train models
    print("\n" + "="*60)
    print("TRAINING MODELS")
    print("="*60)

    factory = ModelFactory()

    # Random Forest for regression
    print("\nTraining Random Forest (regression)...")
    rf_reg = factory.get_model('random_forest', task_type='regression')
    rf_reg.set_params(n_estimators=100, max_depth=5, random_state=42)

    X_train_reg = train_df[numeric_feature_cols]
    y_train_reg = train_df['target_next_intensity']
    X_test_reg = test_df[numeric_feature_cols]
    y_test_reg = test_df['target_next_intensity']

    rf_reg.fit(X_train_reg, y_train_reg)
    y_pred_reg = rf_reg.predict(X_test_reg)

    # XGBoost for classification
    print("Training XGBoost (classification)...")
    xgb_clf = factory.get_model('xgboost', task_type='classification')
    xgb_clf.set_params(
        n_estimators=100,
        max_depth=10,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=0.8,
        random_state=42
    )

    X_train_clf = train_df[numeric_feature_cols]
    y_train_clf = train_df['target_next_high_intensity']
    X_test_clf = test_df[numeric_feature_cols]
    y_test_clf = test_df['target_next_high_intensity']

    xgb_clf.fit(X_train_clf, y_train_clf)

    # Get probabilities and predictions
    if hasattr(xgb_clf, 'predict_proba'):
        y_proba_clf = xgb_clf.predict_proba(X_test_clf)[:, 1]
    else:
        y_proba_clf = xgb_clf.predict(X_test_clf)

    # Use default threshold
    y_pred_clf_default = (y_proba_clf >= 0.5).astype(int)

    # Use optimal threshold from threshold optimization
    y_pred_clf_optimal = (y_proba_clf >= 0.04).astype(int)

    print("Models trained successfully!")

    # Add predictions to test dataframe
    test_df = test_df.copy()
    test_df['pred_intensity'] = y_pred_reg
    test_df['pred_high_default'] = y_pred_clf_default
    test_df['pred_high_optimal'] = y_pred_clf_optimal
    test_df['error_intensity'] = np.abs(test_df['target_next_intensity'] - test_df['pred_intensity'])

    # Analyze per-user performance
    print("\n" + "="*60)
    print("ANALYZING PER-USER PERFORMANCE")
    print("="*60)

    user_performance = []

    for user_id in test_df['userId'].unique():
        user_test = test_df[test_df['userId'] == user_id]
        n_episodes = len(user_test)
        tier = user_test['user_tier'].iloc[0]

        # Regression metrics
        mae = mean_absolute_error(user_test['target_next_intensity'], user_test['pred_intensity'])

        # Classification metrics (if user has positive class)
        if user_test['target_next_high_intensity'].sum() > 0:
            f1_default = f1_score(user_test['target_next_high_intensity'], user_test['pred_high_default'])
            f1_optimal = f1_score(user_test['target_next_high_intensity'], user_test['pred_high_optimal'])
            recall_default = recall_score(user_test['target_next_high_intensity'], user_test['pred_high_default'])
            recall_optimal = recall_score(user_test['target_next_high_intensity'], user_test['pred_high_optimal'])
        else:
            f1_default = np.nan
            f1_optimal = np.nan
            recall_default = np.nan
            recall_optimal = np.nan

        user_performance.append({
            'user_id': user_id,
            'n_episodes': n_episodes,
            'user_tier': tier,
            'mae': mae,
            'f1_default': f1_default,
            'f1_optimal': f1_optimal,
            'recall_default': recall_default,
            'recall_optimal': recall_optimal
        })

    user_perf_df = pd.DataFrame(user_performance)

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY BY USER TIER")
    print("="*60)

    print("\nRegression (MAE):")
    print(user_perf_df.groupby('user_tier')['mae'].describe()[['count', 'mean', 'std', 'min', 'max']])

    print("\nClassification (F1-Score, Default Threshold=0.5):")
    print(user_perf_df.groupby('user_tier')['f1_default'].describe()[['count', 'mean', 'std', 'min', 'max']])

    print("\nClassification (F1-Score, Optimal Threshold=0.04):")
    print(user_perf_df.groupby('user_tier')['f1_optimal'].describe()[['count', 'mean', 'std', 'min', 'max']])

    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Figure 1: MAE vs Episodes Per User (Scatter)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: MAE vs Episodes
    ax = axes[0, 0]
    for tier in ['Sparse (1-9)', 'Medium (10-49)', 'High (50+)']:
        tier_data = user_perf_df[user_perf_df['user_tier'] == tier]
        ax.scatter(tier_data['n_episodes'], tier_data['mae'],
                  label=tier, s=100, alpha=0.6)
    ax.set_xlabel('Episodes Per User', fontsize=12)
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title('Regression Performance vs User Engagement', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: MAE by Tier (Box Plot)
    ax = axes[0, 1]
    tier_order = ['Sparse (1-9)', 'Medium (10-49)', 'High (50+)']
    user_perf_df_sorted = user_perf_df.copy()
    user_perf_df_sorted['user_tier'] = pd.Categorical(
        user_perf_df_sorted['user_tier'],
        categories=tier_order,
        ordered=True
    )
    sns.boxplot(data=user_perf_df_sorted, x='user_tier', y='mae', ax=ax)
    ax.set_xlabel('User Engagement Tier', fontsize=12)
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title('MAE Distribution by User Tier', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=15)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: F1 by Tier (Default Threshold)
    ax = axes[1, 0]
    sns.boxplot(data=user_perf_df_sorted, x='user_tier', y='f1_default', ax=ax)
    ax.set_xlabel('User Engagement Tier', fontsize=12)
    ax.set_ylabel('F1-Score (Threshold=0.5)', fontsize=12)
    ax.set_title('Classification F1 by User Tier (Default)', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=15)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: F1 by Tier (Optimal Threshold)
    ax = axes[1, 1]
    sns.boxplot(data=user_perf_df_sorted, x='user_tier', y='f1_optimal', ax=ax)
    ax.set_xlabel('User Engagement Tier', fontsize=12)
    ax.set_ylabel('F1-Score (Threshold=0.04)', fontsize=12)
    ax.set_title('Classification F1 by User Tier (Optimized)', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=15)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = Path(output_dir) / 'fig23_per_user_performance.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {save_path}")

    # Save results to CSV
    results_path = Path(output_dir) / 'per_user_performance_results.csv'
    user_perf_df.to_csv(results_path, index=False)
    print(f"Results saved: {results_path}")

    # Print key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    print("\n1. REGRESSION PERFORMANCE (MAE):")
    for tier in tier_order:
        tier_mae = user_perf_df[user_perf_df['user_tier'] == tier]['mae']
        print(f"   {tier}: {tier_mae.mean():.3f} ± {tier_mae.std():.3f}")

    print("\n2. CLASSIFICATION PERFORMANCE (F1, Default):")
    for tier in tier_order:
        tier_f1 = user_perf_df[user_perf_df['user_tier'] == tier]['f1_default'].dropna()
        if len(tier_f1) > 0:
            print(f"   {tier}: {tier_f1.mean():.3f} ± {tier_f1.std():.3f} (n={len(tier_f1)})")

    print("\n3. CLASSIFICATION PERFORMANCE (F1, Optimal Threshold=0.04):")
    for tier in tier_order:
        tier_f1 = user_perf_df[user_perf_df['user_tier'] == tier]['f1_optimal'].dropna()
        if len(tier_f1) > 0:
            print(f"   {tier}: {tier_f1.mean():.3f} ± {tier_f1.std():.3f} (n={len(tier_f1)})")

    print("\n4. THRESHOLD OPTIMIZATION IMPACT:")
    default_f1 = user_perf_df['f1_default'].mean()
    optimal_f1 = user_perf_df['f1_optimal'].mean()
    print(f"   Average F1 improvement: {optimal_f1:.3f} vs {default_f1:.3f} "
          f"(+{((optimal_f1/default_f1 - 1)*100):.1f}%)")

    return user_perf_df


if __name__ == "__main__":
    user_perf_df = analyze_per_user_performance()

    print("\n" + "="*80)
    print("PER-USER PERFORMANCE ANALYSIS COMPLETE")
    print("="*80)
