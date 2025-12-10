#!/usr/bin/env python3
"""
Threshold calibration for temporal validation.

This module performs threshold calibration using temporal splits:
1. Train on early period (May 29 - July 15, 2025)
2. Calibrate threshold on mid period (July 16-31, 2025)
3. Evaluate on late period (August 1 - October 26, 2025)

This tests whether optimal thresholds calibrated on mid-period data
generalize to future time periods.
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
from models import ModelFactory
from threshold_calibration import ProperThresholdCalibrator


def temporal_train_cal_test_split(df,
                                   train_end_date='2025-07-15',
                                   cal_end_date='2025-07-31',
                                   test_start_date='2025-08-01'):
    """
    Split data temporally into train/calibration/test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with 'timestamp' column
    train_end_date : str
        End date for training period (inclusive)
    cal_end_date : str
        End date for calibration period (inclusive)
    test_start_date : str
        Start date for test period (inclusive)

    Returns
    -------
    train_df, cal_df, test_df : tuple of DataFrames
    """
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)

    # Convert dates to timestamps (UTC timezone to match data)
    train_end = pd.Timestamp(train_end_date, tz='UTC') + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    cal_end = pd.Timestamp(cal_end_date, tz='UTC') + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    test_start = pd.Timestamp(test_start_date, tz='UTC')

    # Split data
    train_df = df_sorted[df_sorted['timestamp'] <= train_end].copy()
    cal_df = df_sorted[(df_sorted['timestamp'] > train_end) &
                       (df_sorted['timestamp'] <= cal_end)].copy()
    test_df = df_sorted[df_sorted['timestamp'] >= test_start].copy()

    print(f"\nTemporal Train/Calibration/Test Split:")
    print(f"  Train period: {train_df['timestamp'].min().date()} to {train_df['timestamp'].max().date()}")
    print(f"  Train: {len(train_df)} episodes from {train_df['userId'].nunique()} users")
    print(f"  ")
    print(f"  Calibration period: {cal_df['timestamp'].min().date()} to {cal_df['timestamp'].max().date()}")
    print(f"  Calibration: {len(cal_df)} episodes from {cal_df['userId'].nunique()} users")
    print(f"  ")
    print(f"  Test period: {test_df['timestamp'].min().date()} to {test_df['timestamp'].max().date()}")
    print(f"  Test: {len(test_df)} episodes from {test_df['userId'].nunique()} users")

    # Check user overlap
    train_users = set(train_df['userId'].unique())
    cal_users = set(cal_df['userId'].unique())
    test_users = set(test_df['userId'].unique())

    overlap_train_cal = train_users & cal_users
    overlap_cal_test = cal_users & test_users
    overlap_train_test = train_users & test_users

    print(f"\nUser overlap (temporal split allows same users across periods):")
    print(f"  Train ∩ Calibration: {len(overlap_train_cal)} users")
    print(f"  Calibration ∩ Test: {len(overlap_cal_test)} users")
    print(f"  Train ∩ Test: {len(overlap_train_test)} users")

    return train_df, cal_df, test_df


def run_temporal_threshold_calibration(
    data_path: str = 'results (2).csv',
    output_dir: str = 'report_figures',
    high_intensity_threshold: int = 7,
    n_lags: int = 3,
    window_days: list = [7],
    model_name: str = 'xgboost',
    target_col: str = 'target_next_high_intensity',
    train_end_date: str = '2025-07-15',
    cal_end_date: str = '2025-07-31',
    test_start_date: str = '2025-08-01'
):
    """
    Run threshold calibration for temporal validation.

    Parameters
    ----------
    data_path : str
        Path to data CSV
    output_dir : str
        Directory to save results
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
    train_end_date : str
        End date for training period
    cal_end_date : str
        End date for calibration period
    test_start_date : str
        Start date for test period
    """
    print("="*80)
    print("TEMPORAL THRESHOLD CALIBRATION")
    print("="*80)
    print(f"\nModel: {model_name}")
    print(f"Target: {target_col}")
    print(f"High-intensity threshold: ≥{high_intensity_threshold}")
    print(f"Train period: up to {train_end_date}")
    print(f"Calibration period: {train_end_date} to {cal_end_date}")
    print(f"Test period: {test_start_date} onwards")
    print()

    # Load and prepare data
    print("Loading data...")
    df = load_and_clean_data(data_path)

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

    # Temporal split
    print("\nSplitting data temporally...")
    train_df, cal_df, test_df = temporal_train_cal_test_split(
        df.dropna(subset=[target_col]),
        train_end_date=train_end_date,
        cal_end_date=cal_end_date,
        test_start_date=test_start_date
    )

    X_train = train_df[numeric_feature_cols]
    y_train = train_df[target_col]
    X_cal = cal_df[numeric_feature_cols]
    y_cal = cal_df[target_col]
    X_test = test_df[numeric_feature_cols]
    y_test = test_df[target_col]

    print(f"\nPositive class rate (high-intensity):")
    print(f"  Train: {y_train.mean():.2%} ({y_train.sum()}/{len(y_train)})")
    print(f"  Calibration: {y_cal.mean():.2%} ({y_cal.sum()}/{len(y_cal)})")
    print(f"  Test: {y_test.mean():.2%} ({y_test.sum()}/{len(y_test)})")

    # Train model on train set only
    print(f"\nTraining {model_name} model on training period (May 29 - July 15)...")
    factory = ModelFactory()
    model = factory.get_model(model_name, task_type='classification')

    # Use best hyperparameters from hyperparameter search
    if model_name == 'xgboost':
        model.set_params(
            n_estimators=100,
            max_depth=10,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
    elif model_name == 'random_forest':
        model.set_params(
            n_estimators=100,
            max_depth=30,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=0.5,
            random_state=42
        )

    model.fit(X_train, y_train)
    print("Model trained successfully!")

    # Calibrate threshold on calibration period (July 16-31)
    print("\nCalibrating threshold on calibration period (July 16-31)...")
    calibrator = ProperThresholdCalibrator(model, X_cal, y_cal)
    best_threshold, cal_metrics, metrics_df = calibrator.find_optimal_threshold(metric='f1')

    print(f"\n{'='*60}")
    print("CALIBRATION RESULTS (on calibration period)")
    print(f"{'='*60}")
    print(f"\nOptimal Threshold: {best_threshold:.4f}")
    print(f"  Precision: {cal_metrics['precision']:.4f}")
    print(f"  Recall:    {cal_metrics['recall']:.4f}")
    print(f"  F1-Score:  {cal_metrics['f1']:.4f}")
    print(f"  Accuracy:  {cal_metrics['accuracy']:.4f}")

    # Evaluate on test period (August 1 - October 26)
    print(f"\n{'='*60}")
    print("TEST SET EVALUATION (August 1 - October 26)")
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

    if default_metrics['f1'] > 0:
        f1_improvement = ((test_metrics['f1']/default_metrics['f1'] - 1)*100)
    else:
        f1_improvement = 0

    if default_metrics['recall'] > 0:
        recall_improvement = ((test_metrics['recall']/default_metrics['recall'] - 1)*100)
    else:
        recall_improvement = 0

    print(f"\nImprovement from Calibration:")
    print(f"  F1-Score:  +{(test_metrics['f1'] - default_metrics['f1']):.4f} ({f1_improvement:.1f}%)")
    print(f"  Recall:    +{(test_metrics['recall'] - default_metrics['recall']):.4f} ({recall_improvement:.1f}%)")

    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results_path = Path(output_dir) / 'temporal_threshold_calibration_results.csv'

    results_summary = pd.DataFrame({
        'validation_type': ['temporal', 'temporal'],
        'threshold': [best_threshold, 0.5],
        'threshold_type': ['calibrated', 'default'],
        'test_precision': [test_metrics['precision'], default_metrics['precision']],
        'test_recall': [test_metrics['recall'], default_metrics['recall']],
        'test_f1': [test_metrics['f1'], default_metrics['f1']],
        'test_accuracy': [test_metrics['accuracy'], default_metrics['accuracy']],
        'cal_precision': [cal_metrics['precision'], None],
        'cal_recall': [cal_metrics['recall'], None],
        'cal_f1': [cal_metrics['f1'], None]
    })
    results_summary.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")

    # Also save the full threshold scan
    metrics_df['validation_type'] = 'temporal'
    metrics_path = Path(output_dir) / 'temporal_threshold_scan.csv'
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Threshold scan saved to: {metrics_path}")

    return best_threshold, cal_metrics, test_metrics, default_metrics, model, metrics_df


if __name__ == "__main__":
    # Run temporal threshold calibration
    best_threshold, cal_metrics, test_metrics, default_metrics, model, metrics_df = \
        run_temporal_threshold_calibration()

    print("\n" + "="*80)
    print("TEMPORAL THRESHOLD CALIBRATION COMPLETE")
    print("="*80)
    print(f"\nRecommendation for temporal validation deployment:")
    print(f"  Use threshold = {best_threshold:.4f}")
    print(f"  Expected F1-Score: {test_metrics['f1']:.4f}")
    print(f"  Expected Recall: {test_metrics['recall']:.4f} "
          f"(catches {test_metrics['recall']*100:.1f}% of high-intensity episodes)")
    print(f"  Expected Precision: {test_metrics['precision']:.4f} "
          f"({test_metrics['precision']*100:.1f}% of alerts are correct)")
