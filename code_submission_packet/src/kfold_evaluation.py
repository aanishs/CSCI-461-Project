#!/usr/bin/env python3
"""
5-Fold Cross-Validation Evaluation Module.

Implements user-grouped 5-fold CV where each user appears
in the test fold exactly once. This provides robust evaluation
and allows for per-user performance analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                            precision_score, recall_score, f1_score, accuracy_score,
                            roc_auc_score, precision_recall_curve, auc as auc_score)
from typing import List, Dict, Tuple
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_and_clean_data, filter_users_by_min_episodes
from feature_engineering import FeatureEngineer
from target_generation import TargetGenerator
from models import ModelFactory


class UserGroupedKFold:
    """
    K-Fold cross-validation where users are split into folds.
    Each user appears in exactly one test fold.
    """

    def __init__(self, n_splits=5, random_state=42):
        """
        Initialize user-grouped k-fold.

        Parameters
        ----------
        n_splits : int
            Number of folds (default 5)
        random_state : int
            Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def split(self, df, user_col='userId'):
        """
        Generate train/test splits by users.

        Parameters
        ----------
        df : pd.DataFrame
            Input data
        user_col : str
            Column containing user IDs

        Yields
        ------
        train_df, test_df : tuple of DataFrames
        """
        users = df[user_col].unique()

        for train_user_idx, test_user_idx in self.kfold.split(users):
            train_users = users[train_user_idx]
            test_users = users[test_user_idx]

            train_df = df[df[user_col].isin(train_users)].copy()
            test_df = df[df[user_col].isin(test_users)].copy()

            yield train_df, test_df, train_users, test_users


def evaluate_regression_kfold(model_name, df, feature_cols, target_col,
                              model_params=None, n_splits=5, random_state=42):
    """
    Evaluate regression model with user-grouped k-fold CV.

    Parameters
    ----------
    model_name : str
        Name of the model
    df : pd.DataFrame
        Input data
    feature_cols : list
        List of feature columns
    target_col : str
        Target column name
    model_params : dict, optional
        Model hyperparameters
    n_splits : int
        Number of folds
    random_state : int
        Random seed

    Returns
    -------
    dict
        Fold-wise and aggregated results
    """
    factory = ModelFactory()
    splitter = UserGroupedKFold(n_splits=n_splits, random_state=random_state)

    fold_results = []
    user_results = []

    print(f"\nRunning {n_splits}-Fold CV for {model_name} regression...")

    for fold_idx, (train_df, test_df, train_users, test_users) in enumerate(splitter.split(df)):
        print(f"\n  Fold {fold_idx + 1}/{n_splits}:")
        print(f"    Train: {len(train_df)} episodes from {len(train_users)} users")
        print(f"    Test: {len(test_df)} episodes from {len(test_users)} users")

        # Prepare data
        X_train = train_df[feature_cols].fillna(train_df[feature_cols].median())
        y_train = train_df[target_col].values
        X_test = test_df[feature_cols].fillna(train_df[feature_cols].median())
        y_test = test_df[target_col].values

        # Train model
        model = factory.get_model(model_name, task_type='regression', params=model_params)
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Calculate fold metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        fold_results.append({
            'fold': fold_idx + 1,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'n_train': len(X_train),
            'n_test': len(X_test)
        })

        print(f"    MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

        # Per-user metrics for this fold
        for user_id in test_users:
            user_mask = test_df['userId'] == user_id
            user_indices = np.where(user_mask)[0]

            if len(user_indices) > 0:
                user_y_test = y_test[user_indices]
                user_y_pred = y_pred[user_indices]

                user_mae = mean_absolute_error(user_y_test, user_y_pred)

                user_results.append({
                    'userId': user_id,
                    'fold': fold_idx + 1,
                    'mae': user_mae,
                    'n_episodes': len(user_indices)
                })

    # Aggregate results
    fold_df = pd.DataFrame(fold_results)
    user_df = pd.DataFrame(user_results)

    overall_results = {
        'mean_mae': fold_df['mae'].mean(),
        'std_mae': fold_df['mae'].std(),
        'mean_rmse': fold_df['rmse'].mean(),
        'std_rmse': fold_df['rmse'].std(),
        'mean_r2': fold_df['r2'].mean(),
        'std_r2': fold_df['r2'].std(),
    }

    print(f"\n  Overall Results:")
    print(f"    MAE: {overall_results['mean_mae']:.4f} ± {overall_results['std_mae']:.4f}")
    print(f"    RMSE: {overall_results['mean_rmse']:.4f} ± {overall_results['std_rmse']:.4f}")
    print(f"    R²: {overall_results['mean_r2']:.4f} ± {overall_results['std_r2']:.4f}")

    return {
        'fold_results': fold_df,
        'user_results': user_df,
        'overall': overall_results
    }


def evaluate_classification_kfold(model_name, df, feature_cols, target_col,
                                 model_params=None, n_splits=5, random_state=42,
                                 calibrated_threshold=None):
    """
    Evaluate classification model with user-grouped k-fold CV.

    Parameters
    ----------
    model_name : str
        Name of the model
    df : pd.DataFrame
        Input data
    feature_cols : list
        List of feature columns
    target_col : str
        Target column name
    model_params : dict, optional
        Model hyperparameters
    n_splits : int
        Number of folds
    random_state : int
        Random seed
    calibrated_threshold : float, optional
        Pre-calibrated threshold to use (if None, uses 0.5)

    Returns
    -------
    dict
        Fold-wise and aggregated results
    """
    factory = ModelFactory()
    splitter = UserGroupedKFold(n_splits=n_splits, random_state=random_state)

    threshold = calibrated_threshold if calibrated_threshold is not None else 0.5

    fold_results = []
    user_results = []

    print(f"\nRunning {n_splits}-Fold CV for {model_name} classification...")
    print(f"Using threshold: {threshold:.4f}")

    for fold_idx, (train_df, test_df, train_users, test_users) in enumerate(splitter.split(df)):
        print(f"\n  Fold {fold_idx + 1}/{n_splits}:")
        print(f"    Train: {len(train_df)} episodes from {len(train_users)} users")
        print(f"    Test: {len(test_df)} episodes from {len(test_users)} users")

        # Prepare data
        X_train = train_df[feature_cols].fillna(train_df[feature_cols].median())
        y_train = train_df[target_col].values
        X_test = test_df[feature_cols].fillna(train_df[feature_cols].median())
        y_test = test_df[target_col].values

        # Train model
        model = factory.get_model(model_name, task_type='classification', params=model_params)
        model.fit(X_train, y_train)

        # Predict probabilities
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.predict(X_test)

        # Apply threshold
        y_pred = (y_proba >= threshold).astype(int)

        # Calculate fold metrics
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)

        # ROC-AUC and PR-AUC
        if len(np.unique(y_test)) > 1:  # Need both classes for AUC
            roc_auc = roc_auc_score(y_test, y_proba)
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
            pr_auc = auc_score(recall_curve, precision_curve)
        else:
            roc_auc = np.nan
            pr_auc = np.nan

        fold_results.append({
            'fold': fold_idx + 1,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'n_train': len(X_train),
            'n_test': len(X_test)
        })

        print(f"    Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

        # Per-user metrics for this fold
        for user_id in test_users:
            user_mask = test_df['userId'] == user_id
            user_indices = np.where(user_mask)[0]

            if len(user_indices) > 0:
                user_y_test = y_test[user_indices]
                user_y_pred = y_pred[user_indices]

                if len(np.unique(user_y_test)) > 1:
                    user_f1 = f1_score(user_y_test, user_y_pred, zero_division=0)
                    user_precision = precision_score(user_y_test, user_y_pred, zero_division=0)
                    user_recall = recall_score(user_y_test, user_y_pred, zero_division=0)
                else:
                    user_f1 = np.nan
                    user_precision = np.nan
                    user_recall = np.nan

                user_results.append({
                    'userId': user_id,
                    'fold': fold_idx + 1,
                    'precision': user_precision,
                    'recall': user_recall,
                    'f1': user_f1,
                    'n_episodes': len(user_indices)
                })

    # Aggregate results
    fold_df = pd.DataFrame(fold_results)
    user_df = pd.DataFrame(user_results)

    overall_results = {
        'mean_precision': fold_df['precision'].mean(),
        'std_precision': fold_df['precision'].std(),
        'mean_recall': fold_df['recall'].mean(),
        'std_recall': fold_df['recall'].std(),
        'mean_f1': fold_df['f1'].mean(),
        'std_f1': fold_df['f1'].std(),
        'mean_accuracy': fold_df['accuracy'].mean(),
        'std_accuracy': fold_df['accuracy'].std(),
        'mean_roc_auc': fold_df['roc_auc'].mean(),
        'std_roc_auc': fold_df['roc_auc'].std(),
        'mean_pr_auc': fold_df['pr_auc'].mean(),
        'std_pr_auc': fold_df['pr_auc'].std(),
    }

    print(f"\n  Overall Results:")
    print(f"    Precision: {overall_results['mean_precision']:.4f} ± {overall_results['std_precision']:.4f}")
    print(f"    Recall: {overall_results['mean_recall']:.4f} ± {overall_results['std_recall']:.4f}")
    print(f"    F1: {overall_results['mean_f1']:.4f} ± {overall_results['std_f1']:.4f}")
    print(f"    ROC-AUC: {overall_results['mean_roc_auc']:.4f} ± {overall_results['std_roc_auc']:.4f}")

    return {
        'fold_results': fold_df,
        'user_results': user_df,
        'overall': overall_results
    }


def run_kfold_evaluation(
    data_path: str = 'results (2).csv',
    output_dir: str = 'report_figures',
    high_intensity_threshold: int = 7,
    n_lags: int = 3,
    window_days: list = [7],
    n_splits: int = 5
):
    """
    Run 5-fold CV evaluation for both regression and classification.

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
    n_splits : int
        Number of folds
    """
    print("="*80)
    print(f"{n_splits}-FOLD CROSS-VALIDATION EVALUATION")
    print("="*80)

    # Load and prepare data
    print("\nLoading data...")
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

    # Regression evaluation
    print("\n" + "="*80)
    print("REGRESSION: Next Intensity Prediction")
    print("="*80)

    df_reg = df.dropna(subset=['target_next_intensity']).copy()
    reg_results = evaluate_regression_kfold(
        model_name='random_forest',
        df=df_reg,
        feature_cols=numeric_feature_cols,
        target_col='target_next_intensity',
        model_params={'n_estimators': 100, 'max_depth': 5, 'random_state': 42},
        n_splits=n_splits
    )

    # Classification evaluation
    print("\n" + "="*80)
    print("CLASSIFICATION: High-Intensity Prediction")
    print("="*80)

    df_clf = df.dropna(subset=['target_next_high_intensity']).copy()
    clf_results = evaluate_classification_kfold(
        model_name='xgboost',
        df=df_clf,
        feature_cols=numeric_feature_cols,
        target_col='target_next_high_intensity',
        model_params={'n_estimators': 100, 'max_depth': 10, 'learning_rate': 0.1, 'random_state': 42},
        n_splits=n_splits,
        calibrated_threshold=0.04  # Use calibrated threshold from previous analysis
    )

    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    reg_results['fold_results'].to_csv(Path(output_dir) / 'kfold_regression_results.csv', index=False)
    reg_results['user_results'].to_csv(Path(output_dir) / 'kfold_regression_user_results.csv', index=False)

    clf_results['fold_results'].to_csv(Path(output_dir) / 'kfold_classification_results.csv', index=False)
    clf_results['user_results'].to_csv(Path(output_dir) / 'kfold_classification_user_results.csv', index=False)

    print(f"\nResults saved to {output_dir}/")

    return reg_results, clf_results


if __name__ == "__main__":
    # Run 5-fold CV evaluation
    reg_results, clf_results = run_kfold_evaluation(n_splits=5)

    print("\n" + "="*80)
    print("5-FOLD CROSS-VALIDATION COMPLETE")
    print("="*80)
