"""
Evaluation metrics for tic episode prediction.
Supports both regression and classification metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc,
    confusion_matrix, classification_report
)
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')


class MetricsCalculator:
    """
    Calculate evaluation metrics for regression and classification tasks.
    """

    @staticmethod
    def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                          y_baseline: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate regression metrics.

        Parameters
        ----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values
        y_baseline : np.ndarray, optional
            Baseline predictions for comparison

        Returns
        -------
        dict
            Dictionary of regression metrics
        """
        metrics = {}

        # Basic metrics
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)

        # Additional metrics
        metrics['mean_error'] = np.mean(y_pred - y_true)
        metrics['median_abs_error'] = np.median(np.abs(y_pred - y_true))
        metrics['max_error'] = np.max(np.abs(y_pred - y_true))

        # Percentage metrics
        metrics['mape'] = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100

        # Baseline comparison if provided
        if y_baseline is not None:
            baseline_mae = mean_absolute_error(y_true, y_baseline)
            baseline_rmse = np.sqrt(mean_squared_error(y_true, y_baseline))

            metrics['baseline_mae'] = baseline_mae
            metrics['baseline_rmse'] = baseline_rmse
            metrics['mae_improvement'] = (baseline_mae - metrics['mae']) / baseline_mae * 100
            metrics['rmse_improvement'] = (baseline_rmse - metrics['rmse']) / baseline_rmse * 100

        return metrics

    @staticmethod
    def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                              y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate classification metrics.

        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        y_pred_proba : np.ndarray, optional
            Predicted probabilities (for ROC-AUC and PR-AUC)

        Returns
        -------
        dict
            Dictionary of classification metrics
        """
        metrics = {}

        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)

        # Specificity and other metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Probability-based metrics
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            except:
                metrics['roc_auc'] = np.nan

            try:
                precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
                metrics['pr_auc'] = auc(recall_curve, precision_curve)
            except:
                metrics['pr_auc'] = np.nan

        # Baseline comparison (predict majority class)
        majority_class = int(np.bincount(y_true.astype(int)).argmax())
        baseline_acc = (y_true == majority_class).mean()
        metrics['baseline_accuracy'] = baseline_acc
        metrics['accuracy_improvement'] = (metrics['accuracy'] - baseline_acc) / baseline_acc * 100 if baseline_acc > 0 else 0

        return metrics

    @staticmethod
    def print_regression_metrics(metrics: Dict[str, float], title: str = "Regression Metrics"):
        """
        Print regression metrics in a formatted way.

        Parameters
        ----------
        metrics : dict
            Dictionary of metrics
        title : str
            Title to display
        """
        print("=" * 60)
        print(title)
        print("=" * 60)

        print(f"\nPrimary Metrics:")
        print(f"  MAE:  {metrics['mae']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  R²:   {metrics['r2']:.4f}")

        print(f"\nError Statistics:")
        print(f"  Mean Error:         {metrics['mean_error']:.4f}")
        print(f"  Median Abs Error:   {metrics['median_abs_error']:.4f}")
        print(f"  Max Error:          {metrics['max_error']:.4f}")
        print(f"  MAPE:               {metrics['mape']:.2f}%")

        if 'baseline_mae' in metrics:
            print(f"\nBaseline Comparison:")
            print(f"  Baseline MAE:       {metrics['baseline_mae']:.4f}")
            print(f"  MAE Improvement:    {metrics['mae_improvement']:.2f}%")
            print(f"  RMSE Improvement:   {metrics['rmse_improvement']:.2f}%")

    @staticmethod
    def print_classification_metrics(metrics: Dict[str, float], title: str = "Classification Metrics"):
        """
        Print classification metrics in a formatted way.

        Parameters
        ----------
        metrics : dict
            Dictionary of metrics
        title : str
            Title to display
        """
        print("=" * 60)
        print(title)
        print("=" * 60)

        print(f"\nPrimary Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-score:  {metrics['f1']:.4f}")

        print(f"\nConfusion Matrix:")
        print(f"  True Positives:  {metrics['true_positives']}")
        print(f"  True Negatives:  {metrics['true_negatives']}")
        print(f"  False Positives: {metrics['false_positives']}")
        print(f"  False Negatives: {metrics['false_negatives']}")
        print(f"  Specificity:     {metrics['specificity']:.4f}")

        if 'roc_auc' in metrics and not np.isnan(metrics['roc_auc']):
            print(f"\nProbability-Based Metrics:")
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"  PR-AUC:  {metrics['pr_auc']:.4f}")

        if 'baseline_accuracy' in metrics:
            print(f"\nBaseline Comparison:")
            print(f"  Baseline Accuracy:    {metrics['baseline_accuracy']:.4f}")
            print(f"  Accuracy Improvement: {metrics['accuracy_improvement']:.2f}%")

    @staticmethod
    def calculate_per_user_metrics(df: pd.DataFrame, y_true_col: str, y_pred_col: str,
                                   user_col: str = 'userId', task_type: str = 'regression') -> pd.DataFrame:
        """
        Calculate metrics per user.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with predictions and true values
        y_true_col : str
            Column name for true values
        y_pred_col : str
            Column name for predicted values
        user_col : str
            Column name for user IDs
        task_type : str
            'regression' or 'classification'

        Returns
        -------
        pd.DataFrame
            DataFrame with per-user metrics
        """
        user_metrics = []

        for user_id in df[user_col].unique():
            user_df = df[df[user_col] == user_id]
            y_true = user_df[y_true_col].values
            y_pred = user_df[y_pred_col].values

            metrics = {'userId': user_id, 'n_samples': len(user_df)}

            if task_type == 'regression':
                metrics['mae'] = mean_absolute_error(y_true, y_pred)
                metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
                metrics['mean_true'] = y_true.mean()
                metrics['std_true'] = y_true.std()
            else:
                metrics['accuracy'] = accuracy_score(y_true, y_pred)
                metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
                metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
                metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)

            user_metrics.append(metrics)

        return pd.DataFrame(user_metrics).sort_values('mae' if task_type == 'regression' else 'f1', ascending=False)


def evaluate_model(model, X_train, y_train, X_test, y_test,
                  task_type: str = 'regression',
                  print_results: bool = True) -> Dict[str, Any]:
    """
    Evaluate a trained model on train and test sets.

    Parameters
    ----------
    model : sklearn-like model
        Trained model
    X_train, y_train : array-like
        Training data
    X_test, y_test : array-like
        Test data
    task_type : str
        'regression' or 'classification'
    print_results : bool
        Whether to print formatted results

    Returns
    -------
    dict
        Dictionary with train and test metrics
    """
    calc = MetricsCalculator()

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Get probabilities for classification
    y_test_proba = None
    if task_type == 'classification' and hasattr(model, 'predict_proba'):
        y_test_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    if task_type == 'regression':
        # Create baseline (predict mean)
        y_baseline = np.full(len(y_test), y_train.mean())

        train_metrics = calc.regression_metrics(y_train, y_train_pred)
        test_metrics = calc.regression_metrics(y_test, y_test_pred, y_baseline=y_baseline)

        if print_results:
            calc.print_regression_metrics(train_metrics, "Training Set Metrics")
            print()
            calc.print_regression_metrics(test_metrics, "Test Set Metrics")

    else:
        train_metrics = calc.classification_metrics(y_train, y_train_pred)
        test_metrics = calc.classification_metrics(y_test, y_test_pred, y_pred_proba=y_test_proba)

        if print_results:
            calc.print_classification_metrics(train_metrics, "Training Set Metrics")
            print()
            calc.print_classification_metrics(test_metrics, "Test Set Metrics")

    return {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
    }


if __name__ == "__main__":
    # Test evaluation module
    print("Testing Evaluation Module...")

    # Create dummy data
    np.random.seed(42)

    # Regression test
    print("\n" + "="*60)
    print("REGRESSION TEST")
    print("="*60)

    y_true_reg = np.random.randint(1, 11, 100)
    y_pred_reg = y_true_reg + np.random.normal(0, 1.5, 100)
    y_pred_reg = np.clip(y_pred_reg, 1, 10)
    y_baseline_reg = np.full(100, y_true_reg.mean())

    calc = MetricsCalculator()
    metrics_reg = calc.regression_metrics(y_true_reg, y_pred_reg, y_baseline=y_baseline_reg)
    calc.print_regression_metrics(metrics_reg)

    # Classification test
    print("\n\n" + "="*60)
    print("CLASSIFICATION TEST")
    print("="*60)

    y_true_clf = np.random.binomial(1, 0.3, 100)
    y_pred_proba = y_true_clf + np.random.normal(0, 0.3, 100)
    y_pred_proba = np.clip(y_pred_proba, 0, 1)
    y_pred_clf = (y_pred_proba > 0.5).astype(int)

    metrics_clf = calc.classification_metrics(y_true_clf, y_pred_clf, y_pred_proba=y_pred_proba)
    calc.print_classification_metrics(metrics_clf)
