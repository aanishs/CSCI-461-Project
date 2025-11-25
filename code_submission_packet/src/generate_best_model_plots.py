#!/usr/bin/env python3
"""
Generate deep-dive plots for best performing models.

This script loads the best models from experiment results and generates
detailed visualizations including feature importance, confusion matrices,
ROC/PR curves, and prediction timelines.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def load_experiment_details(run_id, details_dir='experiments/details'):
    """Load detailed experiment results from JSON."""
    json_path = Path(details_dir) / f"{run_id}.json"
    if json_path.exists():
        with open(json_path, 'r') as f:
            return json.load(f)
    return None


def get_best_models(results_path='experiments/results.csv', top_n=3):
    """
    Identify best models for regression and classification.

    Returns
    -------
    dict : {'regression': list of run_ids, 'classification': list of run_ids}
    """
    df = pd.read_csv(results_path)

    best_models = {}

    # Best regression models (lowest MAE)
    reg_df = df[df['config_task_type'] == 'regression'].copy()
    if len(reg_df) > 0:
        best_reg = reg_df.nsmallest(top_n, 'metric_test_mae')
        best_models['regression'] = best_reg['run_id'].tolist()
        print(f"\nBest {top_n} Regression Models:")
        for idx, row in best_reg.iterrows():
            print(f"  {row['model_name']}: MAE={row['metric_test_mae']:.3f}")

    # Best classification models (highest F1)
    class_df = df[df['config_task_type'] == 'classification'].copy()
    if len(class_df) > 0:
        best_class = class_df.nlargest(top_n, 'metric_test_f1')
        best_models['classification'] = best_class['run_id'].tolist()
        print(f"\nBest {top_n} Classification Models:")
        for idx, row in best_class.iterrows():
            print(f"  {row['model_name']}: F1={row['metric_test_f1']:.3f}")

    return best_models, df


def plot_feature_importance_from_details(run_id, model_name, output_path):
    """
    Plot feature importance from experiment details if available.
    """
    details = load_experiment_details(run_id)

    if details is None or 'feature_importance' not in details:
        print(f"No feature importance data found for {run_id}")
        return

    importance = details['feature_importance']

    # Convert to DataFrame and sort
    if isinstance(importance, dict):
        imp_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance'])
    else:
        print(f"Unexpected feature importance format for {run_id}")
        return

    imp_df = imp_df.sort_values('Importance', ascending=False).head(15)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(len(imp_df)), imp_df['Importance'].values, color='steelblue')
    ax.set_yticks(range(len(imp_df)))
    ax.set_yticklabels(imp_df['Feature'].values)
    ax.set_xlabel('Importance Score')
    ax.set_title(f'Top 15 Features: {model_name.replace("_", " ").title()}')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, value in enumerate(imp_df['Importance'].values):
        ax.text(value + 0.001, i, f'{value:.3f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_confusion_matrix(run_id, model_name, df_results, output_path):
    """
    Plot confusion matrix for classification model.
    """
    row = df_results[df_results['run_id'] == run_id].iloc[0]

    # Extract confusion matrix values
    tp = row.get('metric_test_true_positives', 0)
    tn = row.get('metric_test_true_negatives', 0)
    fp = row.get('metric_test_false_positives', 0)
    fn = row.get('metric_test_false_negatives', 0)

    cm = np.array([[tn, fp], [fn, tp]])

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Low Intensity', 'High Intensity'],
                yticklabels=['Low Intensity', 'High Intensity'],
                ax=ax, annot_kws={'size': 14})

    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title(f'Confusion Matrix: {model_name.replace("_", " ").title()}\n'
                 f'F1={row["metric_test_f1"]:.3f}, Precision={row["metric_test_precision"]:.3f}, '
                 f'Recall={row["metric_test_recall"]:.3f}')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_roc_pr_curves(run_id, model_name, df_results, output_path):
    """
    Plot ROC and PR curves side by side (using available metrics).
    """
    row = df_results[df_results['run_id'] == run_id].iloc[0]

    roc_auc = row.get('metric_test_roc_auc', None)
    pr_auc = row.get('metric_test_pr_auc', None)
    precision = row.get('metric_test_precision', None)
    recall = row.get('metric_test_recall', None)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ROC Curve (simplified - just show AUC score)
    if roc_auc is not None:
        axes[0].text(0.5, 0.5, f'ROC-AUC\n\n{roc_auc:.3f}',
                    ha='center', va='center', fontsize=24, fontweight='bold')
        axes[0].set_xlim([0, 1])
        axes[0].set_ylim([0, 1])
        axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('ROC Curve Score')
        axes[0].grid(alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, 'ROC-AUC\nNot Available',
                    ha='center', va='center', fontsize=16)
        axes[0].set_xlim([0, 1])
        axes[0].set_ylim([0, 1])

    # PR Curve (simplified - just show AUC score and point)
    if pr_auc is not None and precision is not None and recall is not None:
        axes[1].scatter([recall], [precision], s=200, c='red', marker='o', zorder=5)
        axes[1].text(0.5, 0.8, f'PR-AUC: {pr_auc:.3f}',
                    ha='center', va='center', fontsize=14, fontweight='bold')
        axes[1].text(recall, precision - 0.1, f'({recall:.2f}, {precision:.2f})',
                    ha='center', fontsize=10)
        axes[1].set_xlim([0, 1])
        axes[1].set_ylim([0, 1])
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title('Precision-Recall Score')
        axes[1].grid(alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'PR-AUC\nNot Available',
                    ha='center', va='center', fontsize=16)
        axes[1].set_xlim([0, 1])
        axes[1].set_ylim([0, 1])

    plt.suptitle(f'{model_name.replace("_", " ").title()} Performance', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_prediction_scatter(run_id, model_name, output_path):
    """
    Plot predicted vs actual values for regression (simulated data).

    Note: Actual predictions would need to be saved during model training.
    This creates a placeholder plot showing expected format.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Placeholder text
    ax.text(0.5, 0.5,
           'Prediction Scatter Plot\n\n'
           'To generate this plot:\n'
           '1. Save predictions during model training\n'
           '2. Load predictions from JSON details\n'
           '3. Plot y_true vs y_pred\n\n'
           f'Model: {model_name}\n'
           f'Run ID: {run_id[:20]}...',
           ha='center', va='center', fontsize=12,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    ax.set_xlabel('Actual Intensity')
    ax.set_ylabel('Predicted Intensity')
    ax.set_title('Predicted vs Actual Tic Intensity (Placeholder)')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved (placeholder): {output_path}")
    plt.close()


def generate_best_model_plots(results_path='experiments/results.csv',
                              output_dir='figures/04_best_models/',
                              top_n=3):
    """
    Generate all deep-dive plots for best models.
    """
    print("\n" + "="*60)
    print("Generating Best Model Deep-Dive Plots")
    print("="*60)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get best models
    best_models, df_results = get_best_models(results_path, top_n)

    # Feature importance for top 3 models
    all_run_ids = []
    if 'regression' in best_models:
        all_run_ids.extend(best_models['regression'][:3])
    if 'classification' in best_models:
        all_run_ids.extend(best_models['classification'][:3])

    print("\n" + "-"*60)
    print("Generating Feature Importance Plots...")
    print("-"*60)

    for i, run_id in enumerate(all_run_ids):
        row = df_results[df_results['run_id'] == run_id].iloc[0]
        model_name = row['model_name']
        task_type = row['config_task_type']

        output_path = f'{output_dir}/feature_importance_{model_name}_{task_type}_{i+1}.png'
        plot_feature_importance_from_details(run_id, model_name, output_path)

    # Confusion matrix for best classification model
    if 'classification' in best_models and len(best_models['classification']) > 0:
        print("\n" + "-"*60)
        print("Generating Confusion Matrix...")
        print("-"*60)

        best_class_id = best_models['classification'][0]
        row = df_results[df_results['run_id'] == best_class_id].iloc[0]
        model_name = row['model_name']

        output_path = f'{output_dir}/confusion_matrix_best.png'
        plot_confusion_matrix(best_class_id, model_name, df_results, output_path)

        # ROC/PR curves
        print("\n" + "-"*60)
        print("Generating ROC/PR Curves...")
        print("-"*60)

        output_path = f'{output_dir}/roc_pr_curves_best.png'
        plot_roc_pr_curves(best_class_id, model_name, df_results, output_path)

    # Prediction scatter for best regression model
    if 'regression' in best_models and len(best_models['regression']) > 0:
        print("\n" + "-"*60)
        print("Generating Prediction Scatter Plot...")
        print("-"*60)

        best_reg_id = best_models['regression'][0]
        row = df_results[df_results['run_id'] == best_reg_id].iloc[0]
        model_name = row['model_name']

        output_path = f'{output_dir}/prediction_scatter_best.png'
        plot_prediction_scatter(best_reg_id, model_name, output_path)

    print("\n" + "="*60)
    print(f"All best model plots saved to: {output_dir}")
    print("="*60 + "\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate deep-dive plots for best models')
    parser.add_argument('--results', type=str, default='experiments/results.csv',
                       help='Path to results CSV file')
    parser.add_argument('--output', type=str, default='figures/04_best_models/',
                       help='Output directory for plots')
    parser.add_argument('--top-n', type=int, default=3,
                       help='Number of top models to analyze')

    args = parser.parse_args()

    generate_best_model_plots(args.results, args.output, args.top_n)
