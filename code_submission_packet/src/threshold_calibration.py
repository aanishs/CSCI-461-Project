#!/usr/bin/env python3
"""
Proper threshold calibration using a held-out calibration set.

This module implements the correct methodology for threshold selection:
1. Split data into train/calibration/test sets
2. Train model on train set
3. Select optimal threshold on calibration set
4. Evaluate with that threshold on test set

This avoids data leakage from selecting threshold on test data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
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


class ProperThresholdCalibrator:
    """
    Calibrate classification threshold using proper train/cal/test split.
    """

    def __init__(self, model, X_cal, y_cal):
        """
        Initialize calibrator.

        Parameters
        ----------
        model : sklearn model
            Trained classification model
        X_cal : array-like
            Calibration features
        y_cal : array-like
            Calibration labels
        """
        self.model = model
        self.X_cal = X_cal
        self.y_cal = y_cal

        # Get probability predictions on calibration set
        if hasattr(model, 'predict_proba'):
            self.y_cal_proba = model.predict_proba(X_cal)[:, 1]
        else:
            self.y_cal_proba = model.predict(X_cal)

    def find_optimal_threshold(self,
                              metric='f1',
                              thresholds=None,
                              min_precision=None,
                              min_recall=None) -> Tuple[float, Dict[str, float]]:
        """
        Find optimal threshold on calibration set.

        Parameters
        ----------
        metric : str, default='f1'
            Metric to optimize: 'f1', 'precision', 'recall', 'accuracy'
        thresholds : array-like, optional
            Custom thresholds to test. If None, uses 100 values from 0.01 to 0.99
        min_precision : float, optional
            Minimum precision constraint
        min_recall : float, optional
            Minimum recall constraint

        Returns
        -------
        best_threshold : float
            Optimal threshold value
        best_metrics : dict
            Metrics at optimal threshold
        metrics_df : pd.DataFrame
            Metrics at all tested thresholds
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
            y_pred = (self.y_cal_proba >= threshold).astype(int)

            # Calculate metrics
            precision = precision_score(self.y_cal, y_pred, zero_division=0)
            recall = recall_score(self.y_cal, y_pred, zero_division=0)
            f1 = f1_score(self.y_cal, y_pred, zero_division=0)
            accuracy = accuracy_score(self.y_cal, y_pred)

            metrics_at_thresholds['threshold'].append(threshold)
            metrics_at_thresholds['precision'].append(precision)
            metrics_at_thresholds['recall'].append(recall)
            metrics_at_thresholds['f1'].append(f1)
            metrics_at_thresholds['accuracy'].append(accuracy)

        # Convert to DataFrame
        df = pd.DataFrame(metrics_at_thresholds)

        # Apply constraints if specified
        valid_df = df.copy()
        if min_precision is not None:
            valid_df = valid_df[valid_df['precision'] >= min_precision]
        if min_recall is not None:
            valid_df = valid_df[valid_df['recall'] >= min_recall]

        if len(valid_df) == 0:
            print("WARNING: No threshold satisfies constraints, using best unconstrained")
            valid_df = df

        # Find optimal threshold
        best_idx = valid_df[metric].idxmax()
        best_threshold = valid_df.loc[best_idx, 'threshold']
        best_metrics = {
            'threshold': best_threshold,
            'precision': valid_df.loc[best_idx, 'precision'],
            'recall': valid_df.loc[best_idx, 'recall'],
            'f1': valid_df.loc[best_idx, 'f1'],
            'accuracy': valid_df.loc[best_idx, 'accuracy']
        }

        return best_threshold, best_metrics, df

    def evaluate_on_test(self, X_test, y_test, threshold):
        """
        Evaluate model on test set using calibrated threshold.

        Parameters
        ----------
        X_test : array-like
            Test features
        y_test : array-like
            Test labels
        threshold : float
            Calibrated threshold

        Returns
        -------
        dict
            Test metrics
        """
        # Get test probabilities
        if hasattr(self.model, 'predict_proba'):
            y_test_proba = self.model.predict_proba(X_test)[:, 1]
        else:
            y_test_proba = self.model.predict(X_test)

        # Apply calibrated threshold
        y_test_pred = (y_test_proba >= threshold).astype(int)

        # Calculate metrics
        test_metrics = {
            'precision': precision_score(y_test, y_test_pred, zero_division=0),
            'recall': recall_score(y_test, y_test_pred, zero_division=0),
            'f1': f1_score(y_test, y_test_pred, zero_division=0),
            'accuracy': accuracy_score(y_test, y_test_pred)
        }

        # Also calculate default threshold (0.5) metrics for comparison
        y_test_pred_default = (y_test_proba >= 0.5).astype(int)
        default_metrics = {
            'precision': precision_score(y_test, y_test_pred_default, zero_division=0),
            'recall': recall_score(y_test, y_test_pred_default, zero_division=0),
            'f1': f1_score(y_test, y_test_pred_default, zero_division=0),
            'accuracy': accuracy_score(y_test, y_test_pred_default)
        }

        return test_metrics, default_metrics, y_test_proba, y_test_pred


def train_calibrate_test_split(df, user_col='userId',
                               train_size=0.6, cal_size=0.2, test_size=0.2,
                               random_state=42):
    """
    Split data into train/calibration/test sets by users.

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    user_col : str
        Column containing user IDs
    train_size : float
        Proportion for training (default 0.6)
    cal_size : float
        Proportion for calibration (default 0.2)
    test_size : float
        Proportion for testing (default 0.2)
    random_state : int
        Random seed

    Returns
    -------
    train_df, cal_df, test_df : tuple of DataFrames
    """
    assert abs((train_size + cal_size + test_size) - 1.0) < 1e-6, "Sizes must sum to 1.0"

    # Get unique users
    users = df[user_col].unique()

    # First split: train+cal vs test
    temp_users, test_users = train_test_split(
        users,
        test_size=test_size,
        random_state=random_state
    )

    # Second split: train vs cal
    train_users, cal_users = train_test_split(
        temp_users,
        test_size=cal_size/(train_size + cal_size),
        random_state=random_state
    )

    # Create dataframes
    train_df = df[df[user_col].isin(train_users)].copy()
    cal_df = df[df[user_col].isin(cal_users)].copy()
    test_df = df[df[user_col].isin(test_users)].copy()

    print(f"\nTrain/Calibration/Test Split:")
    print(f"  Train: {len(train_df)} episodes from {len(train_users)} users ({train_size*100:.0f}%)")
    print(f"  Calibration: {len(cal_df)} episodes from {len(cal_users)} users ({cal_size*100:.0f}%)")
    print(f"  Test: {len(test_df)} episodes from {len(test_users)} users ({test_size*100:.0f}%)")

    return train_df, cal_df, test_df


def run_proper_threshold_calibration(
    data_path: str = 'results (2).csv',
    output_dir: str = 'report_figures',
    high_intensity_threshold: int = 7,
    n_lags: int = 3,
    window_days: list = [7],
    model_name: str = 'xgboost',
    target_col: str = 'target_next_high_intensity'
):
    """
    Run proper threshold calibration with train/cal/test split.

    Parameters
    ----------
    data_path : str
        Path to data CSV
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
    print("PROPER THRESHOLD CALIBRATION WITH TRAIN/CAL/TEST SPLIT")
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

    # Get feature columns
    exclude_cols = ['userId', 'timestamp', 'date', 'intensity', 'type', 'mood', 'trigger', 'description',
                    'target_next_intensity', 'target_next_high_intensity',
                    'target_count_next_7d', 'target_high_count_next_7d', 'target_has_high_next_7d',
                    'target_time_to_high_days', 'target_time_to_high_censored']
    feature_cols = [col for col in df.columns if col not in exclude_cols and not col.startswith('target_')]
    numeric_feature_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    # Split into train/cal/test
    print("\nSplitting data into train/calibration/test...")
    train_df, cal_df, test_df = train_calibrate_test_split(
        df.dropna(subset=[target_col]),
        train_size=0.6,
        cal_size=0.2,
        test_size=0.2,
        random_state=42
    )

    X_train = train_df[numeric_feature_cols]
    y_train = train_df[target_col]
    X_cal = cal_df[numeric_feature_cols]
    y_cal = cal_df[target_col]
    X_test = test_df[numeric_feature_cols]
    y_test = test_df[target_col]

    print(f"\nPositive class rate:")
    print(f"  Train: {y_train.mean():.2%}")
    print(f"  Calibration: {y_cal.mean():.2%}")
    print(f"  Test: {y_test.mean():.2%}")

    # Train model on train set only
    print(f"\nTraining {model_name} model on training set...")
    factory = ModelFactory()
    model = factory.get_model(model_name, task_type='classification')

    # Use best hyperparameters
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

    # Calibrate threshold on calibration set
    print("\nCalibrating threshold on calibration set...")
    calibrator = ProperThresholdCalibrator(model, X_cal, y_cal)
    best_threshold, cal_metrics, metrics_df = calibrator.find_optimal_threshold(metric='f1')

    print(f"\n{'='*60}")
    print("CALIBRATION RESULTS (on calibration set)")
    print(f"{'='*60}")
    print(f"\nOptimal Threshold: {best_threshold:.4f}")
    print(f"  Precision: {cal_metrics['precision']:.4f}")
    print(f"  Recall:    {cal_metrics['recall']:.4f}")
    print(f"  F1-Score:  {cal_metrics['f1']:.4f}")
    print(f"  Accuracy:  {cal_metrics['accuracy']:.4f}")

    # Evaluate on test set
    print(f"\n{'='*60}")
    print("TEST SET EVALUATION (using calibrated threshold)")
    print(f"{'='*60}")
    test_metrics, default_metrics, y_test_proba, y_test_pred = calibrator.evaluate_on_test(
        X_test, y_test, best_threshold
    )

    print(f"\nWith Calibrated Threshold ({best_threshold:.4f}):")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1-Score:  {test_metrics['f1']:.4f}")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")

    print(f"\nWith Default Threshold (0.5):")
    print(f"  Precision: {default_metrics['precision']:.4f}")
    print(f"  Recall:    {default_metrics['recall']:.4f}")
    print(f"  F1-Score:  {default_metrics['f1']:.4f}")
    print(f"  Accuracy:  {default_metrics['accuracy']:.4f}")

    print(f"\nImprovement from Calibration:")
    print(f"  F1-Score:  +{(test_metrics['f1'] - default_metrics['f1']):.4f} "
          f"({((test_metrics['f1']/default_metrics['f1'] - 1)*100 if default_metrics['f1'] > 0 else 0):.1f}%)")
    print(f"  Recall:    +{(test_metrics['recall'] - default_metrics['recall']):.4f} "
          f"({((test_metrics['recall']/default_metrics['recall'] - 1)*100 if default_metrics['recall'] > 0 else 0):.1f}%)")

    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results_path = Path(output_dir) / 'proper_threshold_calibration_results.csv'

    results_summary = pd.DataFrame({
        'threshold': [best_threshold, 0.5],
        'dataset': ['calibrated', 'default'],
        'test_precision': [test_metrics['precision'], default_metrics['precision']],
        'test_recall': [test_metrics['recall'], default_metrics['recall']],
        'test_f1': [test_metrics['f1'], default_metrics['f1']],
        'test_accuracy': [test_metrics['accuracy'], default_metrics['accuracy']]
    })
    results_summary.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")

    return best_threshold, cal_metrics, test_metrics, default_metrics, model


if __name__ == "__main__":
    # Run proper threshold calibration
    best_threshold, cal_metrics, test_metrics, default_metrics, model = run_proper_threshold_calibration()

    print("\n" + "="*80)
    print("PROPER THRESHOLD CALIBRATION COMPLETE")
    print("="*80)
    print(f"\nRecommendation: Use threshold = {best_threshold:.4f} for deployment")
    print(f"Expected Test F1-Score: {test_metrics['f1']:.4f}")
    print(f"Expected Test Recall: {test_metrics['recall']:.4f} "
          f"(catches {test_metrics['recall']*100:.1f}% of high-intensity episodes)")
