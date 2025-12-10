#!/usr/bin/env python3
"""
Formal Feature Selection Module.

Implements multiple feature selection methods:
1. Recursive Feature Elimination (RFE)
2. L1 Regularization (Lasso)
3. Mutual Information
4. Feature Importance from Tree Models

Compares performance with selected features vs all features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import (RFE, SelectKBest, mutual_info_regression,
                                      mutual_info_classif)
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, f1_score
from typing import List, Tuple, Dict
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_and_clean_data, filter_users_by_min_episodes
from feature_engineering import FeatureEngineer
from target_generation import TargetGenerator
from validation import UserGroupedSplit
from models import ModelFactory


class FeatureSelector:
    """
    Perform feature selection using multiple methods.
    """

    def __init__(self, X_train, y_train, X_test, y_test, feature_names, task_type='regression'):
        """
        Initialize feature selector.

        Parameters
        ----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        X_test : array-like
            Test features
        y_test : array-like
            Test target
        feature_names : list
            Feature names
        task_type : str
            'regression' or 'classification'
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
        self.task_type = task_type

    def rfe_selection(self, n_features=20, estimator=None):
        """
        Recursive Feature Elimination.

        Parameters
        ----------
        n_features : int
            Number of features to select
        estimator : sklearn estimator, optional
            Base estimator (if None, uses RandomForest)

        Returns
        -------
        list
            Selected feature names
        """
        if estimator is None:
            if self.task_type == 'regression':
                estimator = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
            else:
                estimator = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)

        rfe = RFE(estimator, n_features_to_select=n_features, step=1)
        rfe.fit(self.X_train, self.y_train)

        selected_features = [self.feature_names[i] for i in range(len(self.feature_names))
                           if rfe.support_[i]]

        return selected_features, rfe.ranking_

    def l1_selection(self, alpha=0.01, threshold=1e-5):
        """
        L1 Regularization (Lasso) for feature selection.

        Parameters
        ----------
        alpha : float
            Regularization strength
        threshold : float
            Coefficient threshold for selection

        Returns
        -------
        list
            Selected feature names
        """
        if self.task_type == 'regression':
            model = Lasso(alpha=alpha, random_state=42, max_iter=10000)
        else:
            model = LogisticRegression(penalty='l1', C=1/alpha, solver='liblinear',
                                      random_state=42, max_iter=10000)

        model.fit(self.X_train, self.y_train)

        if self.task_type == 'regression':
            coefficients = np.abs(model.coef_)
        else:
            coefficients = np.abs(model.coef_[0])

        selected_features = [self.feature_names[i] for i in range(len(self.feature_names))
                           if coefficients[i] > threshold]

        return selected_features, coefficients

    def mutual_info_selection(self, n_features=20):
        """
        Mutual Information based feature selection.

        Parameters
        ----------
        n_features : int
            Number of features to select

        Returns
        -------
        list
            Selected feature names
        """
        if self.task_type == 'regression':
            mi_scores = mutual_info_regression(self.X_train, self.y_train, random_state=42)
        else:
            mi_scores = mutual_info_classif(self.X_train, self.y_train, random_state=42)

        # Select top k features
        top_indices = np.argsort(mi_scores)[::-1][:n_features]
        selected_features = [self.feature_names[i] for i in top_indices]

        return selected_features, mi_scores

    def tree_importance_selection(self, n_features=20):
        """
        Feature importance from tree-based models.

        Parameters
        ----------
        n_features : int
            Number of features to select

        Returns
        -------
        list
            Selected feature names
        """
        if self.task_type == 'regression':
            model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

        model.fit(self.X_train, self.y_train)
        importances = model.feature_importances_

        # Select top k features
        top_indices = np.argsort(importances)[::-1][:n_features]
        selected_features = [self.feature_names[i] for i in top_indices]

        return selected_features, importances

    def evaluate_feature_set(self, selected_features, model_name='random_forest'):
        """
        Evaluate model performance with selected features.

        Parameters
        ----------
        selected_features : list
            List of selected feature names
        model_name : str
            Model to use for evaluation

        Returns
        -------
        dict
            Performance metrics
        """
        # Get indices of selected features
        selected_indices = [i for i, name in enumerate(self.feature_names)
                          if name in selected_features]

        X_train_selected = self.X_train[:, selected_indices]
        X_test_selected = self.X_test[:, selected_indices]

        # Train model
        factory = ModelFactory()
        if self.task_type == 'regression':
            model = factory.get_model(model_name, task_type='regression',
                                     params={'n_estimators': 100, 'max_depth': 5, 'random_state': 42})
            model.fit(X_train_selected, self.y_train)
            y_pred = model.predict(X_test_selected)
            mae = mean_absolute_error(self.y_test, y_pred)

            return {'mae': mae, 'n_features': len(selected_features)}
        else:
            model = factory.get_model(model_name, task_type='classification',
                                     params={'n_estimators': 100, 'max_depth': 10, 'random_state': 42})
            model.fit(X_train_selected, self.y_train)

            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test_selected)[:, 1]
            else:
                y_proba = model.predict(X_test_selected)

            y_pred = (y_proba >= 0.04).astype(int)  # Use calibrated threshold
            f1 = f1_score(self.y_test, y_pred, zero_division=0)

            return {'f1': f1, 'n_features': len(selected_features)}


def run_feature_selection_analysis(
    data_path: str = 'results (2).csv',
    output_dir: str = 'report_figures',
    high_intensity_threshold: int = 7,
    n_lags: int = 3,
    window_days: list = [7],
    n_features_to_select: int = 20
):
    """
    Run comprehensive feature selection analysis.

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
    n_features_to_select : int
        Number of features to select
    """
    print("="*80)
    print("FEATURE SELECTION ANALYSIS")
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

    print(f"\nTotal features: {len(numeric_feature_cols)}")

    # Split data
    splitter = UserGroupedSplit(test_size=0.2, random_state=42)

    # REGRESSION FEATURE SELECTION
    print("\n" + "="*80)
    print("REGRESSION FEATURE SELECTION")
    print("="*80)

    df_reg = df.dropna(subset=['target_next_intensity']).copy()
    train_df_reg, test_df_reg = splitter.split(df_reg)

    X_train_reg = train_df_reg[numeric_feature_cols].fillna(train_df_reg[numeric_feature_cols].median()).values
    y_train_reg = train_df_reg['target_next_intensity'].values
    X_test_reg = test_df_reg[numeric_feature_cols].fillna(train_df_reg[numeric_feature_cols].median()).values
    y_test_reg = test_df_reg['target_next_intensity'].values

    selector_reg = FeatureSelector(X_train_reg, y_train_reg, X_test_reg, y_test_reg,
                                   numeric_feature_cols, task_type='regression')

    reg_results = {}

    # Baseline: All features
    print("\nBaseline (All Features):")
    baseline_metrics = selector_reg.evaluate_feature_set(numeric_feature_cols)
    print(f"  MAE: {baseline_metrics['mae']:.4f} with {baseline_metrics['n_features']} features")
    reg_results['baseline'] = {'features': numeric_feature_cols, 'metrics': baseline_metrics}

    # RFE
    print(f"\nRecursive Feature Elimination (selecting {n_features_to_select} features):")
    rfe_features, rfe_ranking = selector_reg.rfe_selection(n_features=n_features_to_select)
    rfe_metrics = selector_reg.evaluate_feature_set(rfe_features)
    print(f"  MAE: {rfe_metrics['mae']:.4f} with {rfe_metrics['n_features']} features")
    print(f"  Top 10 features: {rfe_features[:10]}")
    reg_results['rfe'] = {'features': rfe_features, 'metrics': rfe_metrics, 'ranking': rfe_ranking}

    # L1 Regularization
    print(f"\nL1 Regularization (Lasso):")
    l1_features, l1_coefs = selector_reg.l1_selection(alpha=0.001)
    if len(l1_features) > 0:
        l1_metrics = selector_reg.evaluate_feature_set(l1_features)
        print(f"  MAE: {l1_metrics['mae']:.4f} with {l1_metrics['n_features']} features")
        print(f"  Top 10 features: {l1_features[:10]}")
        reg_results['l1'] = {'features': l1_features, 'metrics': l1_metrics, 'coefficients': l1_coefs}
    else:
        print("  No features selected (try lower alpha)")

    # Mutual Information
    print(f"\nMutual Information (selecting {n_features_to_select} features):")
    mi_features, mi_scores = selector_reg.mutual_info_selection(n_features=n_features_to_select)
    mi_metrics = selector_reg.evaluate_feature_set(mi_features)
    print(f"  MAE: {mi_metrics['mae']:.4f} with {mi_metrics['n_features']} features")
    print(f"  Top 10 features: {mi_features[:10]}")
    reg_results['mutual_info'] = {'features': mi_features, 'metrics': mi_metrics, 'scores': mi_scores}

    # Tree Importance
    print(f"\nTree-based Feature Importance (selecting {n_features_to_select} features):")
    tree_features, tree_importances = selector_reg.tree_importance_selection(n_features=n_features_to_select)
    tree_metrics = selector_reg.evaluate_feature_set(tree_features)
    print(f"  MAE: {tree_metrics['mae']:.4f} with {tree_metrics['n_features']} features")
    print(f"  Top 10 features: {tree_features[:10]}")
    reg_results['tree_importance'] = {'features': tree_features, 'metrics': tree_metrics, 'importances': tree_importances}

    # CLASSIFICATION FEATURE SELECTION
    print("\n" + "="*80)
    print("CLASSIFICATION FEATURE SELECTION")
    print("="*80)

    df_clf = df.dropna(subset=['target_next_high_intensity']).copy()
    train_df_clf, test_df_clf = splitter.split(df_clf)

    X_train_clf = train_df_clf[numeric_feature_cols].fillna(train_df_clf[numeric_feature_cols].median()).values
    y_train_clf = train_df_clf['target_next_high_intensity'].values
    X_test_clf = test_df_clf[numeric_feature_cols].fillna(train_df_clf[numeric_feature_cols].median()).values
    y_test_clf = test_df_clf['target_next_high_intensity'].values

    selector_clf = FeatureSelector(X_train_clf, y_train_clf, X_test_clf, y_test_clf,
                                   numeric_feature_cols, task_type='classification')

    clf_results = {}

    # Baseline: All features
    print("\nBaseline (All Features):")
    baseline_metrics = selector_clf.evaluate_feature_set(numeric_feature_cols, model_name='xgboost')
    print(f"  F1: {baseline_metrics['f1']:.4f} with {baseline_metrics['n_features']} features")
    clf_results['baseline'] = {'features': numeric_feature_cols, 'metrics': baseline_metrics}

    # RFE
    print(f"\nRecursive Feature Elimination (selecting {n_features_to_select} features):")
    rfe_features, rfe_ranking = selector_clf.rfe_selection(n_features=n_features_to_select)
    rfe_metrics = selector_clf.evaluate_feature_set(rfe_features, model_name='xgboost')
    print(f"  F1: {rfe_metrics['f1']:.4f} with {rfe_metrics['n_features']} features")
    print(f"  Top 10 features: {rfe_features[:10]}")
    clf_results['rfe'] = {'features': rfe_features, 'metrics': rfe_metrics, 'ranking': rfe_ranking}

    # L1 Regularization
    print(f"\nL1 Regularization (Logistic Regression):")
    l1_features, l1_coefs = selector_clf.l1_selection(alpha=0.001)
    if len(l1_features) > 0:
        l1_metrics = selector_clf.evaluate_feature_set(l1_features, model_name='xgboost')
        print(f"  F1: {l1_metrics['f1']:.4f} with {l1_metrics['n_features']} features")
        print(f"  Top 10 features: {l1_features[:10]}")
        clf_results['l1'] = {'features': l1_features, 'metrics': l1_metrics, 'coefficients': l1_coefs}
    else:
        print("  No features selected (try lower alpha)")

    # Mutual Information
    print(f"\nMutual Information (selecting {n_features_to_select} features):")
    mi_features, mi_scores = selector_clf.mutual_info_selection(n_features=n_features_to_select)
    mi_metrics = selector_clf.evaluate_feature_set(mi_features, model_name='xgboost')
    print(f"  F1: {mi_metrics['f1']:.4f} with {mi_metrics['n_features']} features")
    print(f"  Top 10 features: {mi_features[:10]}")
    clf_results['mutual_info'] = {'features': mi_features, 'metrics': mi_metrics, 'scores': mi_scores}

    # Tree Importance
    print(f"\nTree-based Feature Importance (selecting {n_features_to_select} features):")
    tree_features, tree_importances = selector_clf.tree_importance_selection(n_features=n_features_to_select)
    tree_metrics = selector_clf.evaluate_feature_set(tree_features, model_name='xgboost')
    print(f"  F1: {tree_metrics['f1']:.4f} with {tree_metrics['n_features']} features")
    print(f"  Top 10 features: {tree_features[:10]}")
    clf_results['tree_importance'] = {'features': tree_features, 'metrics': tree_metrics, 'importances': tree_importances}

    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save summary
    reg_summary = pd.DataFrame({
        'method': list(reg_results.keys()),
        'mae': [v['metrics']['mae'] for v in reg_results.values()],
        'n_features': [v['metrics']['n_features'] for v in reg_results.values()]
    })
    reg_summary.to_csv(Path(output_dir) / 'feature_selection_regression_summary.csv', index=False)

    clf_summary = pd.DataFrame({
        'method': list(clf_results.keys()),
        'f1': [v['metrics']['f1'] for v in clf_results.values()],
        'n_features': [v['metrics']['n_features'] for v in clf_results.values()]
    })
    clf_summary.to_csv(Path(output_dir) / 'feature_selection_classification_summary.csv', index=False)

    print(f"\nResults saved to {output_dir}/")

    return reg_results, clf_results


if __name__ == "__main__":
    # Run feature selection analysis
    reg_results, clf_results = run_feature_selection_analysis(n_features_to_select=20)

    print("\n" + "="*80)
    print("FEATURE SELECTION ANALYSIS COMPLETE")
    print("="*80)
