#!/usr/bin/env python3
"""
Generate all figures for the preliminary report.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')
from data_loader import load_and_clean_data

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# Create output directory
output_dir = Path('report_figures')
output_dir.mkdir(exist_ok=True)

print("Generating report figures...")

# Load data
df = load_and_clean_data('results (2).csv')
results = pd.read_csv('experiments/results.csv')

print(f"\nDataset: {len(df)} episodes, {df['userId'].nunique()} users")
print(f"Experiments: {len(results)} runs")

# ============================================================================
# Figure 1: Intensity Distribution
# ============================================================================
print("\n[1/12] Intensity distribution...")
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df['intensity'].dropna(), bins=10, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(df['intensity'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["intensity"].mean():.2f}')
ax.axvline(7, color='orange', linestyle='--', linewidth=2, label='High-Intensity Threshold (≥7)')
ax.set_xlabel('Intensity (1-10)', fontsize=12, fontweight='bold')
ax.set_ylabel('Count', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Tic Episode Intensities', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'fig1_intensity_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# Figure 2: High-Intensity Rate Pie Chart
# ============================================================================
print("[2/12] High-intensity rate pie chart...")
high_count = (df['intensity'] >= 7).sum()
low_count = len(df) - high_count
fig, ax = plt.subplots(figsize=(8, 8))
colors = ['#ff9999', '#66b3ff']
explode = (0.05, 0)
ax.pie([low_count, high_count], labels=['Low Intensity (<7)', 'High Intensity (≥7)'],
       autopct='%1.1f%%', startangle=90, colors=colors, explode=explode,
       textprops={'fontsize': 12, 'fontweight': 'bold'})
ax.set_title('Proportion of High-Intensity Episodes', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / 'fig2_high_intensity_rate.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# Figure 3: Episodes Per User Distribution
# ============================================================================
print("[3/12] Episodes per user distribution...")
user_counts = df.groupby('userId').size()
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(user_counts, bins=30, edgecolor='black', alpha=0.7, color='mediumseagreen')
ax.axvline(user_counts.median(), color='red', linestyle='--', linewidth=2, label=f'Median: {user_counts.median():.0f}')
ax.axvline(user_counts.mean(), color='orange', linestyle='--', linewidth=2, label=f'Mean: {user_counts.mean():.1f}')
ax.set_xlabel('Number of Episodes per User', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Users', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Episodes per User', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'fig3_episodes_per_user.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# Figure 4: Temporal Coverage
# ============================================================================
print("[4/12] Temporal coverage timeline...")
df['date'] = pd.to_datetime(df['date'])
daily_counts = df.groupby(df['date'].dt.date).size()
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(daily_counts.index, daily_counts.values, linewidth=2, color='steelblue')
ax.fill_between(daily_counts.index, daily_counts.values, alpha=0.3, color='steelblue')
ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Episodes', fontsize=12, fontweight='bold')
ax.set_title('Tic Episode Frequency Over Time', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_dir / 'fig4_temporal_coverage.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# Figure 5: Model Comparison - Regression (MAE)
# ============================================================================
print("[5/12] Model comparison - regression MAE...")
reg_results = results[results['target_type'] == 'target_next_intensity'].copy()
if len(reg_results) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))
    models = reg_results['model_name'].values
    mae_values = reg_results['metric_test_mae'].values
    baseline = reg_results['metric_test_baseline_mae'].iloc[0]

    x = np.arange(len(models))
    bars = ax.bar(x, mae_values, color=['steelblue', 'coral'], edgecolor='black', linewidth=1.5)
    ax.axhline(baseline, color='red', linestyle='--', linewidth=2, label=f'Baseline: {baseline:.2f}')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, mae_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test MAE', fontsize=12, fontweight='bold')
    ax.set_title('Regression Performance: Mean Absolute Error by Model', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in models], rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_model_comparison_mae.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# Figure 6: Model Comparison - Classification (F1)
# ============================================================================
print("[6/12] Model comparison - classification F1...")
clf_results = results[results['target_type'] == 'target_next_high_intensity'].copy()
if len(clf_results) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))
    models = clf_results['model_name'].values
    f1_values = clf_results['metric_test_f1'].values

    x = np.arange(len(models))
    bars = ax.bar(x, f1_values, color=['mediumseagreen', 'gold'], edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, val in zip(bars, f1_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test F1-Score', fontsize=12, fontweight='bold')
    ax.set_title('Classification Performance: F1-Score by Model', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in models], rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, max(f1_values) * 1.2)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig6_model_comparison_f1.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# Figure 7: Multi-Metric Comparison (Regression)
# ============================================================================
print("[7/12] Multi-metric comparison - regression...")
if len(reg_results) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=False)

    metrics = ['metric_test_mae', 'metric_test_rmse', 'metric_test_r2']
    titles = ['Mean Absolute Error', 'Root Mean Squared Error', 'R² Score']
    colors = ['steelblue', 'coral']

    for ax, metric, title in zip(axes, metrics, titles):
        values = reg_results[metric].values
        models = reg_results['model_name'].values
        x = np.arange(len(models))

        bars = ax.bar(x, values, color=colors[:len(models)], edgecolor='black', linewidth=1.5)

        # Add labels with a small offset and ensure vertical margins to avoid overlap
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2.0, height),
                        xytext=(0, 4),
                        textcoords='offset points',
                        ha='center',
                        va='bottom',
                        fontsize=8,
                        fontweight='bold')

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in models], rotation=30, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        # Add y-margins to provide headroom for labels
        y_min, y_max = float(values.min()), float(values.max())
        y_range = max(1e-6, y_max - y_min)
        ax.set_ylim(y_min - 0.15 * y_range, y_max + 0.25 * y_range)
        ax.margins(y=0.1)

    fig.suptitle('Regression Performance: Multiple Metrics', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_dir / 'fig7_multi_metric_regression.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# Figure 8: Multi-Metric Comparison (Classification)
# ============================================================================
print("[8/12] Multi-metric comparison - classification...")
if len(clf_results) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=False)
    axes = axes.flatten()

    metrics = ['metric_test_f1', 'metric_test_precision', 'metric_test_recall', 'metric_test_pr_auc']
    titles = ['F1-Score', 'Precision', 'Recall', 'PR-AUC']
    colors = ['mediumseagreen', 'gold']

    for ax, metric, title in zip(axes, metrics, titles):
        values = clf_results[metric].values
        models = clf_results['model_name'].values
        x = np.arange(len(models))

        bars = ax.bar(x, values, color=colors[:len(models)], edgecolor='black', linewidth=1.5)

        # Add labels with offset points to prevent overlap with axes/top
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2.0, height),
                        xytext=(0, 4),
                        textcoords='offset points',
                        ha='center',
                        va='bottom',
                        fontsize=8,
                        fontweight='bold')

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in models], rotation=30, ha='right', fontsize=9)
        # Provide vertical margins and headroom for text labels
        y_min, y_max = float(values.min()), float(values.max())
        y_range = max(1e-6, y_max - y_min)
        ax.set_ylim(max(0.0, y_min - 0.15 * y_range), y_max + 0.25 * y_range)
        ax.margins(y=0.1)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Classification Performance: Multiple Metrics', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_dir / 'fig8_multi_metric_classification.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# Figure 9: Confusion Matrix (Best Classification Model)
# ============================================================================
print("[9/12] Confusion matrix...")
if len(clf_results) > 0:
    best_clf = clf_results.loc[clf_results['metric_test_f1'].idxmax()]

    # Extract confusion matrix values
    tp = int(best_clf['metric_test_true_positives'])
    tn = int(best_clf['metric_test_true_negatives'])
    fp = int(best_clf['metric_test_false_positives'])
    fn = int(best_clf['metric_test_false_negatives'])

    cm = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Predicted Low', 'Predicted High'],
                yticklabels=['Actual Low', 'Actual High'],
                annot_kws={'fontsize': 14, 'fontweight': 'bold'},
                ax=ax)
    ax.set_title(f'Confusion Matrix: {best_clf["model_name"].replace("_", " ").title()}',
                fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig9_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# Figure 10: Performance Improvement Over Baseline
# ============================================================================
print("[10/12] Performance improvement...")
if len(reg_results) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))
    models = reg_results['model_name'].values
    improvements = reg_results['metric_test_mae_improvement'].values

    x = np.arange(len(models))
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    bars = ax.bar(x, improvements, color=colors, edgecolor='black', linewidth=1.5, alpha=0.7)

    # Add value labels
    for bar, val in zip(bars, improvements):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -1),
                f'{val:.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                fontweight='bold')

    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('MAE Improvement over Baseline (%)', fontsize=12, fontweight='bold')
    ax.set_title('Performance Improvement Over Baseline (Predicting Mean)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in models], rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig10_improvement_baseline.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# Figure 11: Training Time Comparison
# ============================================================================
print("[11/12] Training time comparison...")
if 'metric_train_time_seconds' in results.columns:
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by model and target
    for target in results['target_type'].unique():
        target_data = results[results['target_type'] == target]
        models = target_data['model_name'].values
        times = target_data['metric_train_time_seconds'].values

        ax.scatter(models, times, s=100, alpha=0.6, label=target.replace('target_', '').replace('_', ' '))

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Model Training Time Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig11_training_time.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# Figure 12: Tic Type Distribution (Top 10)
# ============================================================================
print("[12/12] Tic type distribution...")
type_counts = df['type'].value_counts().head(10)
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(range(len(type_counts)), type_counts.values, color='orchid', edgecolor='black', linewidth=1.5)
ax.set_yticks(range(len(type_counts)))
ax.set_yticklabels(type_counts.index)
ax.set_xlabel('Count', fontsize=12, fontweight='bold')
ax.set_ylabel('Tic Type', fontsize=12, fontweight='bold')
ax.set_title('Top 10 Most Common Tic Types', fontsize=14, fontweight='bold')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, val in enumerate(type_counts.values):
    ax.text(val + 1, i, str(val), va='center', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'fig12_tic_type_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "="*80)
print("All figures generated successfully!")
print(f"Saved to: {output_dir}/")
print("="*80)
