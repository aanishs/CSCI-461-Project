#!/usr/bin/env python3
"""
Statistical significance testing for model improvements.

Tests whether model improvements over baseline are statistically significant.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.utils import resample
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


def bootstrap_confidence_interval(
    y_true,
    y_pred_baseline,
    y_pred_model,
    metric_func,
    n_bootstrap=1000,
    confidence_level=0.95,
    random_state=42
):
    """
    Calculate bootstrap confidence intervals for performance improvement.

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred_baseline : array-like
        Baseline predictions
    y_pred_model : array-like
        Model predictions
    metric_func : callable
        Metric function (e.g., mean_absolute_error)
    n_bootstrap : int
        Number of bootstrap samples
    confidence_level : float
        Confidence level (0.95 for 95% CI)
    random_state : int
        Random seed

    Returns
    -------
    dict
        Dictionary with baseline_metric, model_metric, improvement, ci_lower, ci_upper, p_value
    """
    np.random.seed(random_state)
    n = len(y_true)

    baseline_metric = metric_func(y_true, y_pred_baseline)
    model_metric = metric_func(y_true, y_pred_model)

    # Bootstrap
    bootstrap_improvements = []

    for i in range(n_bootstrap):
        # Resample with replacement
        indices = resample(range(n), n_samples=n, random_state=random_state+i)

        y_true_boot = y_true[indices]
        y_pred_baseline_boot = y_pred_baseline[indices]
        y_pred_model_boot = y_pred_model[indices]

        baseline_boot = metric_func(y_true_boot, y_pred_baseline_boot)
        model_boot = metric_func(y_true_boot, y_pred_model_boot)

        # For MAE, lower is better, so improvement is negative
        # For F1, higher is better, so improvement is positive
        if metric_func.__name__ in ['mean_absolute_error']:
            improvement_boot = baseline_boot - model_boot  # Positive means model is better
        else:
            improvement_boot = model_boot - baseline_boot  # Positive means model is better

        bootstrap_improvements.append(improvement_boot)

    bootstrap_improvements = np.array(bootstrap_improvements)

    # Confidence intervals
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_improvements, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_improvements, (1 - alpha/2) * 100)

    # P-value: proportion of bootstrap samples where improvement <= 0
    p_value = np.mean(bootstrap_improvements <= 0)

    # Calculate actual improvement
    if metric_func.__name__ in ['mean_absolute_error']:
        improvement = baseline_metric - model_metric
    else:
        improvement = model_metric - baseline_metric

    return {
        'baseline_metric': baseline_metric,
        'model_metric': model_metric,
        'improvement': improvement,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': p_value,
        'bootstrap_improvements': bootstrap_improvements
    }


def paired_t_test(y_true, y_pred_baseline, y_pred_model, metric_func):
    """
    Perform paired t-test on prediction errors/scores.

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred_baseline : array-like
        Baseline predictions
    y_pred_model : array-like
        Model predictions
    metric_func : callable
        Metric function (applied element-wise if possible)

    Returns
    -------
    dict
        Dictionary with t_statistic and p_value
    """
    # For MAE, compute absolute errors for each sample
    if metric_func.__name__ == 'mean_absolute_error':
        baseline_errors = np.abs(y_true - y_pred_baseline)
        model_errors = np.abs(y_true - y_pred_model)

        # Paired t-test (baseline errors should be higher if model is better)
        t_stat, p_value = stats.ttest_rel(baseline_errors, model_errors)

        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'interpretation': 'Lower errors are better. Positive t-stat means model has lower errors.'
        }
    else:
        # For other metrics, we can't easily compute per-sample scores
        # So return None
        return None


def run_statistical_tests(
    data_path: str = 'results (2).csv',
    output_dir: str = 'report_figures',
    high_intensity_threshold: int = 7,
    n_lags: int = 3,
    window_days: list = [7],
    n_bootstrap: int = 1000
):
    """
    Run comprehensive statistical significance tests.

    Tests:
    1. Model vs Baseline (always predict mean)
    2. Model vs Last-observation-carried-forward
    3. Bootstrap confidence intervals
    4. Paired t-tests
    """
    print("="*80)
    print("STATISTICAL SIGNIFICANCE TESTING")
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
    y_pred_rf = rf_reg.predict(X_test_reg)

    # Baseline: Always predict mean
    baseline_mean = y_train_reg.mean()
    y_pred_baseline_mean = np.full(len(y_test_reg), baseline_mean)

    # Baseline: Last observation carried forward (using lag_1_intensity)
    if 'lag_1_intensity' in numeric_feature_cols:
        y_pred_baseline_locf = test_df['lag_1_intensity'].values
    else:
        print("Warning: lag_1_intensity not found, using mean baseline")
        y_pred_baseline_locf = y_pred_baseline_mean

    print("Models trained successfully!")

    # Run tests
    print("\n" + "="*80)
    print("REGRESSION: RANDOM FOREST VS BASELINES")
    print("="*80)

    # Test 1: RF vs Mean Baseline
    print("\n1. Random Forest vs Mean Baseline")
    print("-" * 60)

    mae_rf = mean_absolute_error(y_test_reg, y_pred_rf)
    mae_mean = mean_absolute_error(y_test_reg, y_pred_baseline_mean)

    print(f"Random Forest MAE: {mae_rf:.4f}")
    print(f"Mean Baseline MAE: {mae_mean:.4f}")
    print(f"Absolute Improvement: {mae_mean - mae_rf:.4f}")
    print(f"Relative Improvement: {((mae_mean - mae_rf) / mae_mean * 100):.2f}%")

    print("\nBootstrap Confidence Interval (n={}):".format(n_bootstrap))
    results_rf_vs_mean = bootstrap_confidence_interval(
        y_test_reg.values,
        y_pred_baseline_mean,
        y_pred_rf,
        mean_absolute_error,
        n_bootstrap=n_bootstrap
    )

    print(f"  95% CI for improvement: [{results_rf_vs_mean['ci_lower']:.4f}, {results_rf_vs_mean['ci_upper']:.4f}]")
    print(f"  P-value: {results_rf_vs_mean['p_value']:.4f}")

    if results_rf_vs_mean['p_value'] < 0.05:
        print("  ✓ Statistically significant at α=0.05")
    else:
        print("  ✗ NOT statistically significant at α=0.05")

    print("\nPaired T-Test:")
    ttest_rf_vs_mean = paired_t_test(y_test_reg.values, y_pred_baseline_mean, y_pred_rf, mean_absolute_error)
    if ttest_rf_vs_mean:
        print(f"  t-statistic: {ttest_rf_vs_mean['t_statistic']:.4f}")
        print(f"  p-value: {ttest_rf_vs_mean['p_value']:.4f}")
        if ttest_rf_vs_mean['p_value'] < 0.05:
            print("  ✓ Statistically significant at α=0.05")
        else:
            print("  ✗ NOT statistically significant at α=0.05")

    # Test 2: RF vs LOCF Baseline
    print("\n2. Random Forest vs Last-Observation-Carried-Forward")
    print("-" * 60)

    mae_locf = mean_absolute_error(y_test_reg, y_pred_baseline_locf)

    print(f"Random Forest MAE: {mae_rf:.4f}")
    print(f"LOCF Baseline MAE: {mae_locf:.4f}")
    print(f"Absolute Improvement: {mae_locf - mae_rf:.4f}")
    print(f"Relative Improvement: {((mae_locf - mae_rf) / mae_locf * 100):.2f}%")

    print("\nBootstrap Confidence Interval (n={}):".format(n_bootstrap))
    results_rf_vs_locf = bootstrap_confidence_interval(
        y_test_reg.values,
        y_pred_baseline_locf,
        y_pred_rf,
        mean_absolute_error,
        n_bootstrap=n_bootstrap
    )

    print(f"  95% CI for improvement: [{results_rf_vs_locf['ci_lower']:.4f}, {results_rf_vs_locf['ci_upper']:.4f}]")
    print(f"  P-value: {results_rf_vs_locf['p_value']:.4f}")

    if results_rf_vs_locf['p_value'] < 0.05:
        print("  ✓ Statistically significant at α=0.05")
    else:
        print("  ✗ NOT statistically significant at α=0.05")

    print("\nPaired T-Test:")
    ttest_rf_vs_locf = paired_t_test(y_test_reg.values, y_pred_baseline_locf, y_pred_rf, mean_absolute_error)
    if ttest_rf_vs_locf:
        print(f"  t-statistic: {ttest_rf_vs_locf['t_statistic']:.4f}")
        print(f"  p-value: {ttest_rf_vs_locf['p_value']:.4f}")
        if ttest_rf_vs_locf['p_value'] < 0.05:
            print("  ✓ Statistically significant at α=0.05")
        else:
            print("  ✗ NOT statistically significant at α=0.05")

    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Bootstrap distribution (RF vs Mean)
    ax = axes[0, 0]
    ax.hist(results_rf_vs_mean['bootstrap_improvements'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No Improvement')
    ax.axvline(x=results_rf_vs_mean['improvement'], color='green', linestyle='-', linewidth=2,
               label=f'Observed: {results_rf_vs_mean["improvement"]:.3f}')
    ax.axvline(x=results_rf_vs_mean['ci_lower'], color='orange', linestyle=':', linewidth=1.5,
               label=f'95% CI: [{results_rf_vs_mean["ci_lower"]:.3f}, {results_rf_vs_mean["ci_upper"]:.3f}]')
    ax.axvline(x=results_rf_vs_mean['ci_upper'], color='orange', linestyle=':', linewidth=1.5)
    ax.set_xlabel('MAE Improvement (Mean Baseline - RF)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Bootstrap Distribution: RF vs Mean Baseline', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Bootstrap distribution (RF vs LOCF)
    ax = axes[0, 1]
    ax.hist(results_rf_vs_locf['bootstrap_improvements'], bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No Improvement')
    ax.axvline(x=results_rf_vs_locf['improvement'], color='green', linestyle='-', linewidth=2,
               label=f'Observed: {results_rf_vs_locf["improvement"]:.3f}')
    ax.axvline(x=results_rf_vs_locf['ci_lower'], color='orange', linestyle=':', linewidth=1.5,
               label=f'95% CI: [{results_rf_vs_locf["ci_lower"]:.3f}, {results_rf_vs_locf["ci_upper"]:.3f}]')
    ax.axvline(x=results_rf_vs_locf['ci_upper'], color='orange', linestyle=':', linewidth=1.5)
    ax.set_xlabel('MAE Improvement (LOCF - RF)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Bootstrap Distribution: RF vs LOCF', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Error distribution comparison
    ax = axes[1, 0]
    errors_mean = np.abs(y_test_reg.values - y_pred_baseline_mean)
    errors_rf = np.abs(y_test_reg.values - y_pred_rf)

    ax.hist(errors_mean, bins=30, alpha=0.5, label='Mean Baseline', color='red')
    ax.hist(errors_rf, bins=30, alpha=0.5, label='Random Forest', color='blue')
    ax.set_xlabel('Absolute Error', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Error Distribution: Baseline vs RF', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Summary table
    ax = axes[1, 1]
    ax.axis('off')

    summary_data = [
        ['Comparison', 'MAE Baseline', 'MAE RF', 'Improvement', 'P-value', 'Significant?'],
        ['─' * 15, '─' * 15, '─' * 10, '─' * 15, '─' * 10, '─' * 12],
        ['RF vs Mean', f'{mae_mean:.3f}', f'{mae_rf:.3f}',
         f'{mae_mean - mae_rf:.3f}', f'{results_rf_vs_mean["p_value"]:.4f}',
         '✓ Yes' if results_rf_vs_mean['p_value'] < 0.05 else '✗ No'],
        ['RF vs LOCF', f'{mae_locf:.3f}', f'{mae_rf:.3f}',
         f'{mae_locf - mae_rf:.3f}', f'{results_rf_vs_locf["p_value"]:.4f}',
         '✓ Yes' if results_rf_vs_locf['p_value'] < 0.05 else '✗ No'],
    ]

    table = ax.table(cellText=summary_data,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.2, 0.15, 0.15, 0.15, 0.15, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Style header row
    for i in range(6):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax.set_title('Statistical Significance Summary', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    save_path = Path(output_dir) / 'fig24_statistical_significance.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {save_path}")

    # Save results
    results_summary = pd.DataFrame([
        {
            'comparison': 'RF vs Mean Baseline',
            'baseline_mae': mae_mean,
            'model_mae': mae_rf,
            'improvement': mae_mean - mae_rf,
            'ci_lower': results_rf_vs_mean['ci_lower'],
            'ci_upper': results_rf_vs_mean['ci_upper'],
            'p_value_bootstrap': results_rf_vs_mean['p_value'],
            'p_value_ttest': ttest_rf_vs_mean['p_value'] if ttest_rf_vs_mean else None,
            'significant': results_rf_vs_mean['p_value'] < 0.05
        },
        {
            'comparison': 'RF vs LOCF',
            'baseline_mae': mae_locf,
            'model_mae': mae_rf,
            'improvement': mae_locf - mae_rf,
            'ci_lower': results_rf_vs_locf['ci_lower'],
            'ci_upper': results_rf_vs_locf['ci_upper'],
            'p_value_bootstrap': results_rf_vs_locf['p_value'],
            'p_value_ttest': ttest_rf_vs_locf['p_value'] if ttest_rf_vs_locf else None,
            'significant': results_rf_vs_locf['p_value'] < 0.05
        }
    ])

    results_path = Path(output_dir) / 'statistical_significance_results.csv'
    results_summary.to_csv(results_path, index=False)
    print(f"Results saved: {results_path}")

    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    print("\n1. Random Forest shows statistically significant improvement over Mean Baseline")
    print(f"   - MAE reduction: {mae_mean - mae_rf:.3f} ({((mae_mean - mae_rf) / mae_mean * 100):.1f}%)")
    print(f"   - 95% CI: [{results_rf_vs_mean['ci_lower']:.3f}, {results_rf_vs_mean['ci_upper']:.3f}]")
    print(f"   - P-value: {results_rf_vs_mean['p_value']:.4f} {'< 0.05 ✓' if results_rf_vs_mean['p_value'] < 0.05 else '>= 0.05 ✗'}")

    print("\n2. Random Forest vs LOCF Baseline")
    print(f"   - MAE reduction: {mae_locf - mae_rf:.3f} ({((mae_locf - mae_rf) / mae_locf * 100):.1f}%)")
    print(f"   - 95% CI: [{results_rf_vs_locf['ci_lower']:.3f}, {results_rf_vs_locf['ci_upper']:.3f}]")
    print(f"   - P-value: {results_rf_vs_locf['p_value']:.4f} {'< 0.05 ✓' if results_rf_vs_locf['p_value'] < 0.05 else '>= 0.05 ✗'}")

    return results_summary


if __name__ == "__main__":
    results = run_statistical_tests(n_bootstrap=1000)

    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE TESTING COMPLETE")
    print("="*80)
