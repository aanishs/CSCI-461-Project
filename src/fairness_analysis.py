#!/usr/bin/env python3
"""
Fairness and Robustness Analysis Module.

Analyzes model performance across different user subgroups
to identify potential fairness issues and assess robustness.

Since demographic data is not available, we use proxy factors:
- Episode frequency (sparse, medium, high engagement)
- Average severity (low, medium, high intensity users)
- Tic type diversity (single type vs multiple types)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (mean_absolute_error, precision_score,
                            recall_score, f1_score, accuracy_score)
from typing import Dict, List, Tuple
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_and_clean_data, filter_users_by_min_episodes
from feature_engineering import FeatureEngineer
from target_generation import TargetGenerator
from validation import UserGroupedSplit
from models import ModelFactory


class SubgroupAnalyzer:
    """
    Analyze model performance across different user subgroups.
    """

    def __init__(self, df):
        """
        Initialize analyzer.

        Parameters
        ----------
        df : pd.DataFrame
            Input data with userId column
        """
        self.df = df
        self.user_characteristics = self._compute_user_characteristics()

    def _compute_user_characteristics(self):
        """Compute characteristics for each user."""
        user_stats = []

        for user_id in self.df['userId'].unique():
            user_data = self.df[self.df['userId'] == user_id]

            stats = {
                'userId': user_id,
                'n_episodes': len(user_data),
                'mean_intensity': user_data['intensity'].mean(),
                'std_intensity': user_data['intensity'].std(),
                'n_unique_types': user_data['type'].nunique(),
                'high_intensity_rate': (user_data['intensity'] >= 7).mean(),
            }

            user_stats.append(stats)

        return pd.DataFrame(user_stats)

    def create_engagement_groups(self):
        """
        Create subgroups based on episode frequency.

        Returns
        -------
        dict
            Mapping of userId to group ('sparse', 'medium', 'high')
        """
        groups = {}

        for _, row in self.user_characteristics.iterrows():
            user_id = row['userId']
            n_episodes = row['n_episodes']

            if n_episodes < 10:
                groups[user_id] = 'sparse'
            elif n_episodes < 50:
                groups[user_id] = 'medium'
            else:
                groups[user_id] = 'high'

        return groups

    def create_severity_groups(self):
        """
        Create subgroups based on average episode severity.

        Returns
        -------
        dict
            Mapping of userId to group ('low', 'medium', 'high')
        """
        groups = {}

        # Calculate percentiles
        p33 = self.user_characteristics['mean_intensity'].quantile(0.33)
        p67 = self.user_characteristics['mean_intensity'].quantile(0.67)

        for _, row in self.user_characteristics.iterrows():
            user_id = row['userId']
            mean_intensity = row['mean_intensity']

            if mean_intensity < p33:
                groups[user_id] = 'low_severity'
            elif mean_intensity < p67:
                groups[user_id] = 'medium_severity'
            else:
                groups[user_id] = 'high_severity'

        return groups

    def create_diversity_groups(self):
        """
        Create subgroups based on tic type diversity.

        Returns
        -------
        dict
            Mapping of userId to group ('single', 'few', 'many')
        """
        groups = {}

        for _, row in self.user_characteristics.iterrows():
            user_id = row['userId']
            n_types = row['n_unique_types']

            if n_types == 1:
                groups[user_id] = 'single_type'
            elif n_types <= 3:
                groups[user_id] = 'few_types'
            else:
                groups[user_id] = 'many_types'

        return groups


def evaluate_subgroup_fairness(
    df,
    feature_cols,
    target_col,
    task_type,
    model_name,
    model_params,
    subgroup_definitions: Dict[str, Dict],
    test_df=None,
    calibrated_threshold=0.5
):
    """
    Evaluate model performance across subgroups.

    Parameters
    ----------
    df : pd.DataFrame
        Training data
    feature_cols : list
        Feature columns
    target_col : str
        Target column
    task_type : str
        'regression' or 'classification'
    model_name : str
        Model name
    model_params : dict
        Model hyperparameters
    subgroup_definitions : dict
        Dict mapping subgroup name to user-to-group mapping
    test_df : pd.DataFrame, optional
        Test data (if None, uses df)
    calibrated_threshold : float
        Threshold for classification

    Returns
    -------
    dict
        Subgroup performance metrics
    """
    # Train model
    factory = ModelFactory()
    model = factory.get_model(model_name, task_type=task_type, params=model_params)

    X_train = df[feature_cols].fillna(df[feature_cols].median())
    y_train = df[target_col].values

    model.fit(X_train, y_train)

    # Use test set if provided, otherwise use train set
    eval_df = test_df if test_df is not None else df

    # Prepare test data
    X_test = eval_df[feature_cols].fillna(df[feature_cols].median())
    y_test = eval_df[target_col].values

    # Get predictions
    if task_type == 'regression':
        y_pred = model.predict(X_test)
    else:
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba >= calibrated_threshold).astype(int)

    # Evaluate each subgroup
    results = {}

    for subgroup_name, user_to_group in subgroup_definitions.items():
        # Add group labels to test df
        eval_df_copy = eval_df.copy()
        eval_df_copy['subgroup'] = eval_df_copy['userId'].map(user_to_group)

        subgroup_results = {}

        for group_label in eval_df_copy['subgroup'].unique():
            if pd.isna(group_label):
                continue

            # Get indices for this group
            group_mask = eval_df_copy['subgroup'] == group_label
            group_indices = np.where(group_mask)[0]

            if len(group_indices) == 0:
                continue

            group_y_test = y_test[group_indices]
            group_y_pred = y_pred[group_indices]

            # Calculate metrics
            if task_type == 'regression':
                mae = mean_absolute_error(group_y_test, group_y_pred)
                subgroup_results[group_label] = {
                    'mae': mae,
                    'n_episodes': len(group_indices),
                    'n_users': eval_df_copy[group_mask]['userId'].nunique()
                }
            else:
                # Need at least 2 classes for meaningful metrics
                if len(np.unique(group_y_test)) > 1:
                    precision = precision_score(group_y_test, group_y_pred, zero_division=0)
                    recall = recall_score(group_y_test, group_y_pred, zero_division=0)
                    f1 = f1_score(group_y_test, group_y_pred, zero_division=0)
                    accuracy = accuracy_score(group_y_test, group_y_pred)
                else:
                    precision = recall = f1 = accuracy = np.nan

                subgroup_results[group_label] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'accuracy': accuracy,
                    'n_episodes': len(group_indices),
                    'n_users': eval_df_copy[group_mask]['userId'].nunique()
                }

        results[subgroup_name] = subgroup_results

    return results


def run_fairness_analysis(
    data_path: str = 'results (2).csv',
    output_dir: str = 'report_figures',
    high_intensity_threshold: int = 7,
    n_lags: int = 3,
    window_days: list = [7]
):
    """
    Run comprehensive fairness analysis.

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
    """
    print("="*80)
    print("FAIRNESS AND ROBUSTNESS ANALYSIS")
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

    # Create subgroups
    print("\nCreating user subgroups...")
    analyzer = SubgroupAnalyzer(df)
    engagement_groups = analyzer.create_engagement_groups()
    severity_groups = analyzer.create_severity_groups()
    diversity_groups = analyzer.create_diversity_groups()

    print(f"  Engagement groups: {pd.Series(engagement_groups).value_counts().to_dict()}")
    print(f"  Severity groups: {pd.Series(severity_groups).value_counts().to_dict()}")
    print(f"  Diversity groups: {pd.Series(diversity_groups).value_counts().to_dict()}")

    subgroup_definitions = {
        'engagement': engagement_groups,
        'severity': severity_groups,
        'diversity': diversity_groups
    }

    # Split data
    print("\nSplitting data...")
    splitter = UserGroupedSplit(test_size=0.2, random_state=42)
    train_df, test_df = splitter.split(df)

    # Regression fairness analysis
    print("\n" + "="*80)
    print("REGRESSION FAIRNESS ANALYSIS")
    print("="*80)

    df_reg = df.dropna(subset=['target_next_intensity']).copy()
    train_df_reg = train_df[train_df.index.isin(df_reg.index)].copy()
    test_df_reg = test_df[test_df.index.isin(df_reg.index)].copy()

    reg_fairness = evaluate_subgroup_fairness(
        df=train_df_reg,
        feature_cols=numeric_feature_cols,
        target_col='target_next_intensity',
        task_type='regression',
        model_name='random_forest',
        model_params={'n_estimators': 100, 'max_depth': 5, 'random_state': 42},
        subgroup_definitions=subgroup_definitions,
        test_df=test_df_reg
    )

    print("\nRegression Performance by Subgroup:")
    for subgroup_name, results in reg_fairness.items():
        print(f"\n  {subgroup_name.upper()}:")
        for group_label, metrics in results.items():
            print(f"    {group_label}: MAE = {metrics['mae']:.4f} "
                  f"(n={metrics['n_episodes']} episodes, {metrics['n_users']} users)")

    # Classification fairness analysis
    print("\n" + "="*80)
    print("CLASSIFICATION FAIRNESS ANALYSIS")
    print("="*80)

    df_clf = df.dropna(subset=['target_next_high_intensity']).copy()
    train_df_clf = train_df[train_df.index.isin(df_clf.index)].copy()
    test_df_clf = test_df[test_df.index.isin(df_clf.index)].copy()

    clf_fairness = evaluate_subgroup_fairness(
        df=train_df_clf,
        feature_cols=numeric_feature_cols,
        target_col='target_next_high_intensity',
        task_type='classification',
        model_name='xgboost',
        model_params={'n_estimators': 100, 'max_depth': 10, 'learning_rate': 0.1, 'random_state': 42},
        subgroup_definitions=subgroup_definitions,
        test_df=test_df_clf,
        calibrated_threshold=0.04
    )

    print("\nClassification Performance by Subgroup:")
    for subgroup_name, results in clf_fairness.items():
        print(f"\n  {subgroup_name.upper()}:")
        for group_label, metrics in results.items():
            if not np.isnan(metrics['f1']):
                print(f"    {group_label}: F1 = {metrics['f1']:.4f}, "
                      f"Precision = {metrics['precision']:.4f}, "
                      f"Recall = {metrics['recall']:.4f} "
                      f"(n={metrics['n_episodes']} episodes, {metrics['n_users']} users)")
            else:
                print(f"    {group_label}: Insufficient data for metrics "
                      f"(n={metrics['n_episodes']} episodes, {metrics['n_users']} users)")

    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Convert to DataFrames and save
    reg_fairness_rows = []
    for subgroup_name, results in reg_fairness.items():
        for group_label, metrics in results.items():
            reg_fairness_rows.append({
                'subgroup_type': subgroup_name,
                'group': group_label,
                **metrics
            })

    reg_fairness_df = pd.DataFrame(reg_fairness_rows)
    reg_fairness_df.to_csv(Path(output_dir) / 'fairness_regression_results.csv', index=False)

    clf_fairness_rows = []
    for subgroup_name, results in clf_fairness.items():
        for group_label, metrics in results.items():
            clf_fairness_rows.append({
                'subgroup_type': subgroup_name,
                'group': group_label,
                **metrics
            })

    clf_fairness_df = pd.DataFrame(clf_fairness_rows)
    clf_fairness_df.to_csv(Path(output_dir) / 'fairness_classification_results.csv', index=False)

    print(f"\nResults saved to {output_dir}/")

    # Create visualization
    create_fairness_visualizations(reg_fairness_df, clf_fairness_df, output_dir)

    return reg_fairness, clf_fairness


def create_fairness_visualizations(reg_df, clf_df, output_dir):
    """Create fairness visualization plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Regression plots
    for idx, subgroup_type in enumerate(['engagement', 'severity', 'diversity']):
        ax = axes[0, idx]
        data = reg_df[reg_df['subgroup_type'] == subgroup_type]

        bars = ax.bar(range(len(data)), data['mae'], color='steelblue', edgecolor='black')
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(data['group'], rotation=45, ha='right')
        ax.set_ylabel('MAE', fontweight='bold')
        ax.set_title(f'Regression MAE by {subgroup_type.capitalize()}', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add values on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)

    # Classification plots
    for idx, subgroup_type in enumerate(['engagement', 'severity', 'diversity']):
        ax = axes[1, idx]
        data = clf_df[clf_df['subgroup_type'] == subgroup_type]

        # Filter out NaN values
        data = data[~data['f1'].isna()]

        if len(data) > 0:
            bars = ax.bar(range(len(data)), data['f1'], color='coral', edgecolor='black')
            ax.set_xticks(range(len(data)))
            ax.set_xticklabels(data['group'], rotation=45, ha='right')
            ax.set_ylabel('F1-Score', fontweight='bold')
            ax.set_title(f'Classification F1 by {subgroup_type.capitalize()}', fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

            # Add values on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'fairness_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {output_dir}/fairness_analysis.png")
    plt.close()


if __name__ == "__main__":
    # Run fairness analysis
    reg_fairness, clf_fairness = run_fairness_analysis()

    print("\n" + "="*80)
    print("FAIRNESS ANALYSIS COMPLETE")
    print("="*80)
