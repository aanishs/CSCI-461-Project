#!/usr/bin/env python3
"""
Generate analysis plots from hyperparameter search results.

This script loads experiments/results.csv and generates publication-quality
visualizations comparing models, hyperparameters, and configurations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 9


def load_results(results_path='experiments/results.csv'):
    """Load experiment results."""
    df = pd.read_csv(results_path)
    print(f"Loaded {len(df)} experiment results")
    return df


def plot_model_comparison_mae(df, output_path='figures/03_results_analysis/model_comparison_mae.png'):
    """
    Compare models by test MAE (regression tasks only).
    """
    # Filter regression tasks
    regression_df = df[df['config_task_type'] == 'regression'].copy()

    if len(regression_df) == 0:
        print("No regression results found, skipping MAE comparison")
        return

    # Group by model and get best MAE for each
    best_per_model = regression_df.groupby('model_name')['metric_test_mae'].min().sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(best_per_model)), best_per_model.values, color='steelblue')
    ax.set_yticks(range(len(best_per_model)))
    ax.set_yticklabels([name.replace('_', ' ').title() for name in best_per_model.index])
    ax.set_xlabel('Test MAE (Lower is Better)')
    ax.set_title('Model Comparison: Best Test MAE by Model Type')
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (idx, value) in enumerate(best_per_model.items()):
        ax.text(value + 0.05, i, f'{value:.3f}', va='center')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_model_comparison_f1(df, output_path='figures/03_results_analysis/model_comparison_f1.png'):
    """
    Compare models by test F1 score (classification tasks only).
    """
    # Filter classification tasks
    classification_df = df[df['config_task_type'] == 'classification'].copy()

    if len(classification_df) == 0:
        print("No classification results found, skipping F1 comparison")
        return

    # Group by model and get best F1 for each
    best_per_model = classification_df.groupby('model_name')['metric_test_f1'].max().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(best_per_model)), best_per_model.values, color='coral')
    ax.set_yticks(range(len(best_per_model)))
    ax.set_yticklabels([name.replace('_', ' ').title() for name in best_per_model.index])
    ax.set_xlabel('Test F1 Score (Higher is Better)')
    ax.set_title('Model Comparison: Best Test F1 by Model Type')
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (idx, value) in enumerate(best_per_model.items()):
        ax.text(value + 0.01, i, f'{value:.3f}', va='center')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_hyperparameter_sensitivity_rf(df, output_path='figures/03_results_analysis/hyperparameter_sensitivity_rf.png'):
    """
    Show how Random Forest hyperparameters affect test MAE.
    """
    rf_df = df[(df['model_name'] == 'random_forest') & (df['config_task_type'] == 'regression')].copy()

    if len(rf_df) == 0:
        print("No Random Forest regression results found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # n_estimators vs MAE
    if 'config_n_estimators' in rf_df.columns and rf_df['config_n_estimators'].notna().any():
        axes[0].scatter(rf_df['config_n_estimators'], rf_df['metric_test_mae'], alpha=0.6, s=50)
        axes[0].set_xlabel('Number of Trees (n_estimators)')
        axes[0].set_ylabel('Test MAE')
        axes[0].set_title('Impact of Number of Trees on Performance')
        axes[0].grid(alpha=0.3)

    # max_depth vs MAE
    if 'config_max_depth' in rf_df.columns:
        rf_depth = rf_df[rf_df['config_max_depth'].notna()].copy()
        if len(rf_depth) > 0:
            axes[1].scatter(rf_depth['config_max_depth'], rf_depth['metric_test_mae'], alpha=0.6, s=50, color='orange')
            axes[1].set_xlabel('Max Tree Depth')
            axes[1].set_ylabel('Test MAE')
            axes[1].set_title('Impact of Tree Depth on Performance')
            axes[1].grid(alpha=0.3)

    plt.suptitle('Random Forest Hyperparameter Sensitivity', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_hyperparameter_sensitivity_xgb(df, output_path='figures/03_results_analysis/hyperparameter_sensitivity_xgb.png'):
    """
    Show how XGBoost hyperparameters affect test MAE.
    """
    xgb_df = df[(df['model_name'] == 'xgboost') & (df['config_task_type'] == 'regression')].copy()

    if len(xgb_df) == 0:
        print("No XGBoost regression results found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # learning_rate vs MAE
    if 'config_learning_rate' in xgb_df.columns and xgb_df['config_learning_rate'].notna().any():
        axes[0].scatter(xgb_df['config_learning_rate'], xgb_df['metric_test_mae'], alpha=0.6, s=50, color='green')
        axes[0].set_xlabel('Learning Rate')
        axes[0].set_ylabel('Test MAE')
        axes[0].set_title('Impact of Learning Rate on Performance')
        axes[0].set_xscale('log')
        axes[0].grid(alpha=0.3)

    # max_depth vs MAE
    if 'config_max_depth' in xgb_df.columns and xgb_df['config_max_depth'].notna().any():
        axes[1].scatter(xgb_df['config_max_depth'], xgb_df['metric_test_mae'], alpha=0.6, s=50, color='purple')
        axes[1].set_xlabel('Max Tree Depth')
        axes[1].set_ylabel('Test MAE')
        axes[1].set_title('Impact of Tree Depth on Performance')
        axes[1].grid(alpha=0.3)

    plt.suptitle('XGBoost Hyperparameter Sensitivity', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_feature_set_comparison(df, output_path='figures/03_results_analysis/feature_set_comparison.png'):
    """
    Compare performance across different feature set configurations.
    """
    regression_df = df[df['config_task_type'] == 'regression'].copy()

    if len(regression_df) == 0 or 'config_feature_set' not in regression_df.columns:
        print("No feature set data found")
        return

    # Filter out NaN feature sets
    regression_df = regression_df[regression_df['config_feature_set'].notna()]

    if len(regression_df) == 0:
        print("No valid feature set data")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Box plot by feature set
    feature_sets = regression_df['config_feature_set'].unique()
    data_to_plot = [regression_df[regression_df['config_feature_set'] == fs]['metric_test_mae'].values
                    for fs in feature_sets]

    bp = ax.boxplot(data_to_plot, labels=[fs.replace('_', ' ').title() for fs in feature_sets],
                     patch_artist=True, showmeans=True)

    # Color boxes
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)

    ax.set_ylabel('Test MAE')
    ax.set_xlabel('Feature Set Configuration')
    ax.set_title('Impact of Feature Set on Prediction Performance')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=15, ha='right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_window_size_impact(df, output_path='figures/03_results_analysis/window_size_impact.png'):
    """
    Show how prediction window size affects accuracy.
    """
    regression_df = df[df['config_task_type'] == 'regression'].copy()

    if len(regression_df) == 0 or 'config_window_days' not in regression_df.columns:
        print("No window size data found")
        return

    # Filter valid window data
    window_df = regression_df[regression_df['config_window_days'].notna()].copy()

    if len(window_df) == 0:
        print("No valid window size data")
        return

    # Group by window size and calculate mean/std MAE
    window_stats = window_df.groupby('config_window_days')['metric_test_mae'].agg(['mean', 'std', 'count'])
    window_stats = window_stats.sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(window_stats.index, window_stats['mean'],
                yerr=window_stats['std'], marker='o', linewidth=2,
                markersize=8, capsize=5, capthick=2)

    ax.set_xlabel('Feature Window Size (days)')
    ax.set_ylabel('Mean Test MAE')
    ax.set_title('Impact of Feature Window Size on Prediction Accuracy')
    ax.grid(alpha=0.3)

    # Add sample size annotations
    for idx, row in window_stats.iterrows():
        ax.annotate(f'n={int(row["count"])}', xy=(idx, row['mean']),
                   xytext=(0, 10), textcoords='offset points', ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_threshold_comparison(df, output_path='figures/03_results_analysis/threshold_comparison.png'):
    """
    Compare performance across different high-intensity thresholds.
    """
    if 'config_high_intensity_threshold' not in df.columns:
        print("No threshold data found")
        return

    threshold_df = df[df['config_high_intensity_threshold'].notna()].copy()

    if len(threshold_df) == 0:
        print("No valid threshold data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Regression: MAE by threshold
    reg_df = threshold_df[threshold_df['config_task_type'] == 'regression']
    if len(reg_df) > 0:
        threshold_mae = reg_df.groupby('config_high_intensity_threshold')['metric_test_mae'].mean().sort_index()
        axes[0].bar(threshold_mae.index.astype(str), threshold_mae.values, color='steelblue')
        axes[0].set_xlabel('High-Intensity Threshold')
        axes[0].set_ylabel('Mean Test MAE')
        axes[0].set_title('Regression: Impact of Threshold on MAE')
        axes[0].grid(axis='y', alpha=0.3)

        # Add value labels
        for i, (idx, value) in enumerate(threshold_mae.items()):
            axes[0].text(i, value + 0.02, f'{value:.3f}', ha='center')

    # Classification: F1 by threshold
    class_df = threshold_df[threshold_df['config_task_type'] == 'classification']
    if len(class_df) > 0:
        threshold_f1 = class_df.groupby('config_high_intensity_threshold')['metric_test_f1'].mean().sort_index()
        axes[1].bar(threshold_f1.index.astype(str), threshold_f1.values, color='coral')
        axes[1].set_xlabel('High-Intensity Threshold')
        axes[1].set_ylabel('Mean Test F1 Score')
        axes[1].set_title('Classification: Impact of Threshold on F1')
        axes[1].grid(axis='y', alpha=0.3)

        # Add value labels
        for i, (idx, value) in enumerate(threshold_f1.items()):
            axes[1].text(i, value + 0.01, f'{value:.3f}', ha='center')

    plt.suptitle('High-Intensity Threshold Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_training_time_vs_performance(df, output_path='figures/03_results_analysis/training_time_vs_performance.png'):
    """
    Show trade-off between training time and model performance.
    """
    regression_df = df[df['config_task_type'] == 'regression'].copy()

    if len(regression_df) == 0 or 'metric_train_time_seconds' not in regression_df.columns:
        print("No training time data found")
        return

    time_df = regression_df[regression_df['metric_train_time_seconds'].notna()].copy()

    if len(time_df) == 0:
        print("No valid training time data")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color by model type
    models = time_df['model_name'].unique()
    colors = sns.color_palette("husl", len(models))

    for i, model in enumerate(models):
        model_data = time_df[time_df['model_name'] == model]
        ax.scatter(model_data['metric_train_time_seconds'], model_data['metric_test_mae'],
                  label=model.replace('_', ' ').title(), s=60, alpha=0.7, color=colors[i])

    ax.set_xlabel('Training Time (seconds)')
    ax.set_ylabel('Test MAE')
    ax.set_title('Training Time vs Performance Trade-off')
    ax.set_xscale('log')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def save_all_plots(results_path='experiments/results.csv',
                   output_dir='figures/03_results_analysis/'):
    """
    Generate and save all analysis plots.
    """
    print("\n" + "="*60)
    print("Generating Analysis Plots from Experiment Results")
    print("="*60 + "\n")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load results
    df = load_results(results_path)

    print(f"\nDataset summary:")
    print(f"  Total experiments: {len(df)}")
    print(f"  Regression tasks: {len(df[df['config_task_type'] == 'regression'])}")
    print(f"  Classification tasks: {len(df[df['config_task_type'] == 'classification'])}")
    print(f"  Unique models: {df['model_name'].nunique()}")
    print()

    # Generate all plots
    plot_model_comparison_mae(df, f'{output_dir}/model_comparison_mae.png')
    plot_model_comparison_f1(df, f'{output_dir}/model_comparison_f1.png')
    plot_hyperparameter_sensitivity_rf(df, f'{output_dir}/hyperparameter_sensitivity_rf.png')
    plot_hyperparameter_sensitivity_xgb(df, f'{output_dir}/hyperparameter_sensitivity_xgb.png')
    plot_feature_set_comparison(df, f'{output_dir}/feature_set_comparison.png')
    plot_window_size_impact(df, f'{output_dir}/window_size_impact.png')
    plot_threshold_comparison(df, f'{output_dir}/threshold_comparison.png')
    plot_training_time_vs_performance(df, f'{output_dir}/training_time_vs_performance.png')

    print("\n" + "="*60)
    print(f"All plots saved to: {output_dir}")
    print("="*60 + "\n")

    return df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate analysis plots from experiment results')
    parser.add_argument('--results', type=str, default='experiments/results.csv',
                       help='Path to results CSV file')
    parser.add_argument('--output', type=str, default='figures/03_results_analysis/',
                       help='Output directory for plots')

    args = parser.parse_args()

    save_all_plots(args.results, args.output)
