#!/usr/bin/env python3
"""
Generate figures using EXACT values from Final Final Report.md text.
This ensures figures match the report descriptions precisely.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# Create output directory
output_dir = Path('report_figures')
output_dir.mkdir(exist_ok=True)

print("Generating figures from report text values...")

# ============================================================================
# Figure 13 (fig5): Regression MAE Comparison
# Values from report text lines 663, 669
# ============================================================================
print("\n[1/4] Figure 13 - Regression MAE comparison...")

model_names = ['Random Forest', 'XGBoost', 'LightGBM']
mae_values = [1.9377, 1.9887, 1.9919]  # From report text
baseline = 2.68385825552407  # From CSV

# Error bars from k-fold CV
kfold_results = np.array([1.981609802765026, 0.892232900137555, 1.5349393629960064,
                          1.2310641806524578, 1.7225441293556614])
mae_std = kfold_results.std()
mae_ci = 1.96 * mae_std  # 95% CI

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(model_names))
error_bars = [mae_ci] * len(model_names)

bars = ax.bar(x, mae_values, yerr=error_bars, capsize=5,
               color=['steelblue', 'coral', 'mediumseagreen'],
               edgecolor='black', linewidth=1.5,
               error_kw={'linewidth': 2, 'ecolor': 'black'})
ax.axhline(baseline, color='red', linestyle='--', linewidth=2,
           label=f'Baseline: {baseline:.2f}')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, mae_values)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + mae_ci + 0.05,
            f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Test MAE', fontsize=12, fontweight='bold')
ax.set_title('Regression Performance: Mean Absolute Error by Model', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=10)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(output_dir / 'fig5_model_comparison_mae.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Generated fig5 with RF=1.94, XGB=1.99, LightGBM=1.99")

# ============================================================================
# Figure 17 (fig6): Classification F1 Comparison
# Values from report text lines 714, 720, 722
# ============================================================================
print("\n[2/4] Figure 17 - Classification F1 comparison...")

model_names_clf = ['XGBoost', 'Random Forest', 'LightGBM']
f1_values = [0.3407, 0.3333, 0.2093]  # From report text

# Error bars from k-fold CV
kfold_clf_results = np.array([0.6907630522088354, 0.5953488372093023, 0.37777777777777777,
                               0.5031446540880503, 0.3003003003003003])
f1_std = kfold_clf_results.std()
f1_ci = 1.96 * f1_std

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(model_names_clf))
error_bars = [f1_ci] * len(model_names_clf)

bars = ax.bar(x, f1_values, yerr=error_bars, capsize=5,
               color=['gold', 'mediumseagreen', 'coral'],
               edgecolor='black', linewidth=1.5,
               error_kw={'linewidth': 2, 'ecolor': 'black'})

# Add value labels
for bar, val in zip(bars, f1_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + f1_ci + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Test F1-Score', fontsize=12, fontweight='bold')
ax.set_title('Classification Performance: F1-Score by Model', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names_clf, fontsize=10)
ax.set_ylim(0, max(f1_values) * 1.6)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(output_dir / 'fig6_model_comparison_f1.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Generated fig6 with XGB=0.34, RF=0.33, LightGBM=0.21")

# ============================================================================
# Figure 18 (fig8): Multi-Metric Classification Comparison
# Values from report text lines 714, 720, 722, 729
# ============================================================================
print("\n[3/4] Figure 18 - Multi-metric classification comparison...")

# Values from report text
models = ['XGBoost', 'Random Forest', 'LightGBM']
precision_vals = [0.6552, 0.4500, 0.5000]  # Line 714, 720, 722
recall_vals = [0.2281, 0.2632, 0.1316]     # Line 714, 720, 722
f1_vals = [0.3407, 0.3333, 0.2093]          # Line 714, 720, 722
pr_auc_vals = [0.6992, 0.6878, 0.6482]      # Line 714, 720, 722

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

metrics = [precision_vals, recall_vals, f1_vals, pr_auc_vals]
titles = ['Precision', 'Recall', 'F1-Score', 'PR-AUC']
colors = ['gold', 'mediumseagreen', 'coral']

for ax, metric, title in zip(axes, metrics, titles):
    x = np.arange(len(models))
    bars = ax.bar(x, metric, color=colors, edgecolor='black', linewidth=1.5)

    # Add labels
    for bar, val in zip(bars, metric):
        height = bar.get_height()
        ax.annotate(f'{val:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2.0, height),
                    xytext=(0, 4),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha='right', fontsize=9)
    y_min, y_max = float(min(metric)), float(max(metric))
    y_range = max(1e-6, y_max - y_min)
    ax.set_ylim(max(0.0, y_min - 0.15 * y_range), y_max + 0.25 * y_range)
    ax.grid(True, alpha=0.3, axis='y')

fig.suptitle('Classification Performance: Multiple Metrics', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(output_dir / 'fig8_multi_metric_classification.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Generated fig8 with all metrics from report")

# ============================================================================
# Figure 19 (fig9): Confusion Matrix - XGBoost
# Values from report text line 737
# ============================================================================
print("\n[4/4] Figure 19 - Confusion matrix...")

# Values from report text (277 episodes total)
tp = 13   # Line 737
tn = 197  # Line 737
fp = 20   # Line 737
fn = 47   # Line 737

cm = np.array([[tn, fp], [fn, tp]])

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Predicted Low', 'Predicted High'],
            yticklabels=['Actual Low', 'Actual High'],
            annot_kws={'fontsize': 14, 'fontweight': 'bold'},
            ax=ax)
ax.set_title('Confusion Matrix: XGBoost', fontsize=14, fontweight='bold')
ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / 'fig9_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Generated fig9 with TP={tp}, TN={tn}, FP={fp}, FN={fn} (Total={tp+tn+fp+fn} episodes)")

print("\n" + "="*80)
print("All figures generated successfully from report text values!")
print(f"Saved to: {output_dir}/")
print("="*80)
