#!/usr/bin/env python3
"""
Threshold optimization for classification models.

This module provides functions to find optimal classification thresholds
by analyzing precision, recall, and F1-score across different threshold values.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from typing import Tuple, Dict
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_and_clean_data, filter_users_by_min_episodes
from feature_engineering import FeatureEngineer
from target_generation import TargetGenerator
from validation import UserGroupedSplit
from models import ModelFactory


class ThresholdOptimizer:
    """
    Optimize classification threshold for best F1-score or custom metric.
    """

    def __init__(self, model, X_test, y_test, y_proba=None):
        """
        Initialize optimizer.

        Parameters
        ----------
        model : sklearn model
            Trained classification model
        X_test : array-like
            Test features
        y_test : array-like
            True test labels
        y_proba : array-like, optional
            Predicted probabilities (if None, will use model.predict_proba)
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

        # Get probability predictions if not provided
        if y_proba is None:
            if hasattr(model, 'predict_proba'):
                # For models with predict_proba (RF, XGB with binary:logistic)
                self.y_proba = model.predict_proba(X_test)[:, 1]
            else:
                # For models without predict_proba (some XGB configs)
                self.y_proba = model.predict(X_test)
        else:
            self.y_proba = y_proba

    def find_optimal_threshold(self,
                              metric='f1',
                              thresholds=None) -> Tuple[float, Dict[str, float]]:
        """
        Find optimal threshold for given metric.

        Parameters
        ----------
        metric : str, default='f1'
            Metric to optimize: 'f1', 'precision', 'recall', 'accuracy'
        thresholds : array-like, optional
            Custom thresholds to test. If None, uses 100 values from 0.01 to 0.99

        Returns
        -------
        best_threshold : float
            Optimal threshold value
        best_metrics : dict
            Metrics at optimal threshold
        """
        if thresholds is None:
            thresholds = np.linspace(0.01, 0.99, 100)

        metrics_at_thresholds = {
            'threshold': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'accuracy': []
        }

        for threshold in thresholds:
            y_pred = (self.y_proba >= threshold).astype(int)

            # Calculate metrics
            precision = precision_score(self.y_test, y_pred, zero_division=0)
            recall = recall_score(self.y_test, y_pred, zero_division=0)
            f1 = f1_score(self.y_test, y_pred, zero_division=0)
            accuracy = accuracy_score(self.y_test, y_pred)

            metrics_at_thresholds['threshold'].append(threshold)
            metrics_at_thresholds['precision'].append(precision)
            metrics_at_thresholds['recall'].append(recall)
            metrics_at_thresholds['f1'].append(f1)
            metrics_at_thresholds['accuracy'].append(accuracy)

        # Convert to DataFrame
        df = pd.DataFrame(metrics_at_thresholds)

        # Find optimal threshold
        best_idx = df[metric].idxmax()
        best_threshold = df.loc[best_idx, 'threshold']
        best_metrics = {
            'threshold': best_threshold,
            'precision': df.loc[best_idx, 'precision'],
            'recall': df.loc[best_idx, 'recall'],
            'f1': df.loc[best_idx, 'f1'],
            'accuracy': df.loc[best_idx, 'accuracy']
        }

        return best_threshold, best_metrics, df

    def plot_threshold_analysis(self,
                                metrics_df: pd.DataFrame,
                                save_path: str = None):
        """
        Plot precision, recall, F1 vs threshold.

        Parameters
        ----------
        metrics_df : pd.DataFrame
            DataFrame with threshold, precision, recall, f1
        save_path : str, optional
            Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Precision, Recall, F1 vs Threshold
        ax = axes[0, 0]
        ax.plot(metrics_df['threshold'], metrics_df['precision'],
                label='Precision', linewidth=2, color='blue')
        ax.plot(metrics_df['threshold'], metrics_df['recall'],
                label='Recall', linewidth=2, color='green')
        ax.plot(metrics_df['threshold'], metrics_df['f1'],
                label='F1-Score', linewidth=2, color='red')
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Default (0.5)')
        ax.set_xlabel('Classification Threshold', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Precision, Recall, F1 vs Threshold', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Plot 2: F1-Score zoomed in
        ax = axes[0, 1]
        ax.plot(metrics_df['threshold'], metrics_df['f1'],
                linewidth=2, color='red')
        best_f1_idx = metrics_df['f1'].idxmax()
        best_threshold = metrics_df.loc[best_f1_idx, 'threshold']
        best_f1 = metrics_df.loc[best_f1_idx, 'f1']
        ax.scatter([best_threshold], [best_f1],
                  color='darkred', s=200, zorder=5, marker='*',
                  label=f'Best: {best_threshold:.3f} (F1={best_f1:.3f})')
        ax.axvline(x=best_threshold, color='darkred', linestyle='--', alpha=0.5)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Default (0.5)')
        ax.set_xlabel('Classification Threshold', fontsize=12)
        ax.set_ylabel('F1-Score', fontsize=12)
        ax.set_title('F1-Score Optimization', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Plot 3: Precision-Recall Tradeoff
        ax = axes[1, 0]
        ax.plot(metrics_df['recall'], metrics_df['precision'],
                linewidth=2, color='purple')
        ax.scatter([metrics_df.loc[best_f1_idx, 'recall']],
                  [metrics_df.loc[best_f1_idx, 'precision']],
                  color='darkred', s=200, zorder=5, marker='*',
                  label=f'Best F1: {best_f1:.3f}')
        # Mark default threshold point
        default_idx = (metrics_df['threshold'] - 0.5).abs().idxmin()
        ax.scatter([metrics_df.loc[default_idx, 'recall']],
                  [metrics_df.loc[default_idx, 'precision']],
                  color='gray', s=150, zorder=5, marker='o',
                  label='Default (0.5)', alpha=0.7)
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Tradeoff', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Plot 4: Summary Table
        ax = axes[1, 1]
        ax.axis('off')

        # Create summary table
        summary_data = [
            ['Metric', 'Default (0.5)', f'Optimal ({best_threshold:.3f})'],
            ['─' * 15, '─' * 15, '─' * 20],
        ]

        default_precision = metrics_df.loc[default_idx, 'precision']
        default_recall = metrics_df.loc[default_idx, 'recall']
        default_f1 = metrics_df.loc[default_idx, 'f1']

        best_precision = metrics_df.loc[best_f1_idx, 'precision']
        best_recall = metrics_df.loc[best_f1_idx, 'recall']

        summary_data.extend([
            ['Precision', f'{default_precision:.4f}', f'{best_precision:.4f}'],
            ['Recall', f'{default_recall:.4f}', f'{best_recall:.4f}'],
            ['F1-Score', f'{default_f1:.4f}', f'{best_f1:.4f}'],
            ['', '', ''],
            ['Improvement', f'(Baseline)', f'+{((best_f1/default_f1-1)*100):.1f}%']
        ])

        table = ax.table(cellText=summary_data,
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.3, 0.35, 0.35])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)

        # Style header row
        for i in range(3):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(weight='bold', color='white')

        ax.set_title('Threshold Comparison Summary',
                    fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Threshold analysis plot saved to: {save_path}")

        return fig


def run_threshold_optimization_for_best_model(
    data_path: str = 'results (2).csv',
    experiment_dir: str = 'experiments',
    output_dir: str = 'report_figures',
    high_intensity_threshold: int = 7,
    n_lags: int = 3,
    window_days: list = [7],
    model_name: str = 'xgboost',
    target_col: str = 'target_next_high_intensity'
):
    """
    Run threshold optimization for the best classification model.

    Parameters
    ----------
    data_path : str
        Path to data CSV
    experiment_dir : str
        Directory with experiment results
    output_dir : str
        Directory to save plots
    high_intensity_threshold : int
        Threshold for high-intensity classification
    n_lags : int
        Number of lag features
    window_days : list
        Window sizes for aggregation features
    model_name : str
        Name of model to train
    target_col : str
        Target column name
    """
    print("="*80)
    print("THRESHOLD OPTIMIZATION FOR HIGH-INTENSITY CLASSIFICATION")
    print("="*80)
    print(f"\nModel: {model_name}")
    print(f"Target: {target_col}")
    print(f"High-intensity threshold: ≥{high_intensity_threshold}")
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

    # Get feature columns (exclude targets and system columns)
    exclude_cols = ['userId', 'timestamp', 'date', 'intensity', 'type', 'mood', 'trigger', 'description',
                    'target_next_intensity', 'target_next_high_intensity',
                    'target_count_next_7d', 'target_high_count_next_7d', 'target_has_high_next_7d',
                    'target_time_to_high_days', 'target_time_to_high_censored']
    feature_cols = [col for col in df.columns if col not in exclude_cols and not col.startswith('target_')]

    # Split data
    print("Splitting data (user-grouped)...")
    splitter = UserGroupedSplit(test_size=0.2, random_state=42)
    train_df, test_df = splitter.split(df, user_col='userId')

    # Select only numeric features (XGBoost doesn't work well with object dtype)
    numeric_feature_cols = train_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    X_train = train_df[numeric_feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[numeric_feature_cols]
    y_test = test_df[target_col]

    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Positive class rate (train): {y_train.mean():.2%}")
    print(f"Positive class rate (test): {y_test.mean():.2%}")

    # Train model
    print(f"\nTraining {model_name} model...")
    factory = ModelFactory()
    model = factory.get_model(model_name, task_type='classification')

    # Use best hyperparameters from previous search
    if model_name == 'xgboost':
        model.set_params(
            n_estimators=100,
            max_depth=10,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=0.8,
            random_state=42
        )
    elif model_name == 'random_forest':
        model.set_params(
            n_estimators=100,
            max_depth=15,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42
        )

    model.fit(X_train, y_train)
    print("Model trained successfully!")

    # Run threshold optimization
    print("\nOptimizing classification threshold...")
    optimizer = ThresholdOptimizer(model, X_test, y_test)
    best_threshold, best_metrics, metrics_df = optimizer.find_optimal_threshold(metric='f1')

    print(f"\n{'='*60}")
    print("THRESHOLD OPTIMIZATION RESULTS")
    print(f"{'='*60}")
    print(f"\nOptimal Threshold: {best_threshold:.4f}")
    print(f"  Precision: {best_metrics['precision']:.4f}")
    print(f"  Recall:    {best_metrics['recall']:.4f}")
    print(f"  F1-Score:  {best_metrics['f1']:.4f}")
    print(f"  Accuracy:  {best_metrics['accuracy']:.4f}")

    # Compare with default threshold (0.5)
    default_idx = (metrics_df['threshold'] - 0.5).abs().idxmin()
    default_f1 = metrics_df.loc[default_idx, 'f1']
    default_precision = metrics_df.loc[default_idx, 'precision']
    default_recall = metrics_df.loc[default_idx, 'recall']

    print(f"\nDefault Threshold (0.5):")
    print(f"  Precision: {default_precision:.4f}")
    print(f"  Recall:    {default_recall:.4f}")
    print(f"  F1-Score:  {default_f1:.4f}")

    print(f"\nImprovement:")
    print(f"  F1-Score:  +{(best_metrics['f1'] - default_f1):.4f} "
          f"({((best_metrics['f1']/default_f1 - 1)*100):.1f}%)")
    print(f"  Recall:    +{(best_metrics['recall'] - default_recall):.4f} "
          f"({((best_metrics['recall']/default_recall - 1)*100 if default_recall > 0 else 0):.1f}%)")

    # Generate plots
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(output_dir) / 'fig26_threshold_analysis.png'

    print(f"\nGenerating threshold analysis plot...")
    optimizer.plot_threshold_analysis(metrics_df, save_path=str(save_path))

    # Save results to CSV
    results_path = Path(output_dir) / 'threshold_optimization_results.csv'
    metrics_df.to_csv(results_path, index=False)
    print(f"Metrics saved to: {results_path}")

    return best_threshold, best_metrics, metrics_df, model


if __name__ == "__main__":
    # Run threshold optimization
    best_threshold, best_metrics, metrics_df, model = run_threshold_optimization_for_best_model()

    print("\n" + "="*80)
    print("THRESHOLD OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"\nRecommendation: Use threshold = {best_threshold:.4f} for deployment")
    print(f"Expected F1-Score: {best_metrics['f1']:.4f}")
    print(f"Expected Recall: {best_metrics['recall']:.4f} "
          f"(catches {best_metrics['recall']*100:.1f}% of high-intensity episodes)")
