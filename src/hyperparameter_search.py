"""
Hyperparameter search orchestration module.
Coordinates the search across different models, targets, and hyperparameters.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from typing import Dict, Any, List, Optional, Tuple
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from data_loader import load_and_clean_data, filter_users_by_min_episodes
from feature_engineering import FeatureEngineer
from target_generation import TargetGenerator
from models import ModelFactory
from validation import UserGroupedSplit, UserTimeSeriesSplit
from evaluation import MetricsCalculator, evaluate_model
from experiment_tracker import ExperimentTracker


class HyperparameterSearch:
    """
    Orchestrate hyperparameter search across models and configurations.
    """

    def __init__(self, data_path: str, experiment_dir: str = 'experiments',
                 random_state: int = 42):
        """
        Initialize hyperparameter search.

        Parameters
        ----------
        data_path : str
            Path to the data CSV file
        experiment_dir : str
            Directory to save experiment logs
        random_state : int
            Random seed for reproducibility
        """
        self.data_path = data_path
        self.experiment_dir = experiment_dir
        self.random_state = random_state

        # Initialize components
        self.tracker = ExperimentTracker(experiment_dir=experiment_dir)
        self.feature_engineer = FeatureEngineer()
        self.model_factory = ModelFactory()

        # Load data
        print("Loading data...")
        self.df_raw = load_and_clean_data(data_path)
        self.df_raw = filter_users_by_min_episodes(self.df_raw, min_episodes=4)
        print(f"Loaded {len(self.df_raw)} episodes from {self.df_raw['userId'].nunique()} users")

    def prepare_data(self, high_intensity_threshold: int = 7,
                    n_lags: int = 3, window_days: List[int] = [3, 7, 14],
                    k_days_list: List[int] = [1, 3, 7, 14],
                    include_sequence: bool = True,
                    include_time_window: bool = True,
                    include_engineered: bool = True) -> pd.DataFrame:
        """
        Prepare data with features and targets.

        Parameters
        ----------
        high_intensity_threshold : int
            Threshold for high-intensity classification
        n_lags : int
            Number of lag features
        window_days : list
            List of window sizes for time-window features
        k_days_list : list
            List of prediction windows for future count targets
        include_sequence : bool
            Include sequence-based features
        include_time_window : bool
            Include time-window features
        include_engineered : bool
            Include engineered features

        Returns
        -------
        pd.DataFrame
            Prepared DataFrame with features and targets
        """
        print("Creating features...")
        df = self.feature_engineer.create_all_features(
            self.df_raw,
            n_lags=n_lags,
            window_days=window_days if include_time_window else [],
            fit=True
        )

        print("Creating targets...")
        target_generator = TargetGenerator(high_intensity_threshold=high_intensity_threshold)
        df = target_generator.create_all_targets(df, k_days_list=k_days_list)

        # Get feature columns
        feature_cols = self.feature_engineer.get_feature_columns(
            df,
            include_sequence=include_sequence,
            include_time_window=include_time_window,
            include_engineered=include_engineered
        )

        # Filter out rows with missing critical values
        df = df.dropna(subset=feature_cols[:5])  # Keep rows with at least some features

        print(f"Prepared {len(df)} samples with {len(feature_cols)} features")

        return df, feature_cols

    def run_single_experiment(self, df: pd.DataFrame, feature_cols: List[str],
                             model_name: str, target_col: str, task_type: str,
                             model_params: Dict[str, Any],
                             config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single experiment with specific configuration.

        Parameters
        ----------
        df : pd.DataFrame
            Prepared data
        feature_cols : list
            List of feature columns
        model_name : str
            Name of the model
        target_col : str
            Target column name
        task_type : str
            'regression' or 'classification'
        model_params : dict
            Model hyperparameters
        config : dict
            Experiment configuration

        Returns
        -------
        dict
            Experiment results
        """
        # Filter out samples with missing target
        df_model = df.dropna(subset=[target_col]).copy()

        # Split data
        splitter = UserGroupedSplit(test_size=0.2, random_state=self.random_state)
        train_df, test_df = splitter.split(df_model)

        # Prepare features and targets
        X_train = train_df[feature_cols].fillna(train_df[feature_cols].median())
        y_train = train_df[target_col].values
        X_test = test_df[feature_cols].fillna(train_df[feature_cols].median())
        y_test = test_df[target_col].values

        # Train model
        model = self.model_factory.get_model(model_name, task_type, params=model_params)

        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Evaluate
        calc = MetricsCalculator()
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        if task_type == 'regression':
            y_baseline = np.full(len(y_test), y_train.mean())
            train_metrics = calc.regression_metrics(y_train, y_train_pred)
            test_metrics = calc.regression_metrics(y_test, y_test_pred, y_baseline=y_baseline)
        else:
            y_test_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            train_metrics = calc.classification_metrics(y_train, y_train_pred)
            test_metrics = calc.classification_metrics(y_test, y_test_pred, y_pred_proba=y_test_proba)

        # Combine all config info
        full_config = {
            **config,
            **model_params,
            'model_name': model_name,
            'target_col': target_col,
            'task_type': task_type,
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'n_features': len(feature_cols),
        }

        # Combine metrics
        all_metrics = {
            **{'train_' + k: v for k, v in train_metrics.items()},
            **{'test_' + k: v for k, v in test_metrics.items()},
            'train_time_seconds': train_time,
        }

        return {
            'config': full_config,
            'metrics': all_metrics,
            'model': model,
        }

    def grid_search_sklearn(self, df: pd.DataFrame, feature_cols: List[str],
                           model_name: str, target_col: str, task_type: str,
                           param_grid: Dict[str, list],
                           config: Dict[str, Any],
                           cv_splits: int = 3,
                           n_jobs: int = -1) -> Dict[str, Any]:
        """
        Run sklearn GridSearchCV.

        Parameters
        ----------
        df : pd.DataFrame
            Prepared data
        feature_cols : list
            Feature columns
        model_name : str
            Model name
        target_col : str
            Target column
        task_type : str
            Task type
        param_grid : dict
            Parameter grid
        config : dict
            Configuration
        cv_splits : int
            Number of CV splits
        n_jobs : int
            Number of parallel jobs

        Returns
        -------
        dict
            Results with best parameters
        """
        # Filter out samples with missing target
        df_model = df.dropna(subset=[target_col]).copy()

        # Split data
        splitter = UserGroupedSplit(test_size=0.2, random_state=self.random_state)
        train_df, test_df = splitter.split(df_model)

        # Prepare features and targets
        X_train = train_df[feature_cols].fillna(train_df[feature_cols].median())
        y_train = train_df[target_col].values

        # Create base model
        base_model = self.model_factory.get_model(model_name, task_type, params={})

        # Determine scoring metric
        if task_type == 'regression':
            scoring = 'neg_mean_absolute_error'
        else:
            scoring = 'f1'

        # Run grid search
        print(f"  Running GridSearchCV with {cv_splits} folds...")
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv_splits,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=0
        )

        start_time = time.time()
        grid_search.fit(X_train, y_train)
        search_time = time.time() - start_time

        # Get best model and evaluate on test set
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        # Evaluate on test set
        result = self.run_single_experiment(
            df, feature_cols, model_name, target_col, task_type,
            best_params, config
        )

        result['config']['search_type'] = 'grid'
        result['config']['search_time_seconds'] = search_time
        result['config']['best_cv_score'] = grid_search.best_score_

        return result

    def random_search_sklearn(self, df: pd.DataFrame, feature_cols: List[str],
                             model_name: str, target_col: str, task_type: str,
                             param_distributions: Dict[str, list],
                             config: Dict[str, Any],
                             n_iter: int = 50,
                             cv_splits: int = 3,
                             n_jobs: int = -1) -> Dict[str, Any]:
        """
        Run sklearn RandomizedSearchCV.

        Parameters
        ----------
        df : pd.DataFrame
            Prepared data
        feature_cols : list
            Feature columns
        model_name : str
            Model name
        target_col : str
            Target column
        task_type : str
            Task type
        param_distributions : dict
            Parameter distributions
        config : dict
            Configuration
        n_iter : int
            Number of iterations
        cv_splits : int
            Number of CV splits
        n_jobs : int
            Number of parallel jobs

        Returns
        -------
        dict
            Results with best parameters
        """
        # Filter out samples with missing target
        df_model = df.dropna(subset=[target_col]).copy()

        # Split data
        splitter = UserGroupedSplit(test_size=0.2, random_state=self.random_state)
        train_df, test_df = splitter.split(df_model)

        # Prepare features and targets
        X_train = train_df[feature_cols].fillna(train_df[feature_cols].median())
        y_train = train_df[target_col].values

        # Create base model
        base_model = self.model_factory.get_model(model_name, task_type, params={})

        # Determine scoring metric
        if task_type == 'regression':
            scoring = 'neg_mean_absolute_error'
        else:
            scoring = 'f1'

        # Run random search
        print(f"  Running RandomizedSearchCV with {n_iter} iterations, {cv_splits} folds...")
        random_search = RandomizedSearchCV(
            base_model,
            param_distributions,
            n_iter=n_iter,
            cv=cv_splits,
            scoring=scoring,
            n_jobs=n_jobs,
            random_state=self.random_state,
            verbose=0
        )

        start_time = time.time()
        random_search.fit(X_train, y_train)
        search_time = time.time() - start_time

        # Get best model and evaluate on test set
        best_model = random_search.best_estimator_
        best_params = random_search.best_params_

        # Evaluate on test set
        result = self.run_single_experiment(
            df, feature_cols, model_name, target_col, task_type,
            best_params, config
        )

        result['config']['search_type'] = 'random'
        result['config']['search_time_seconds'] = search_time
        result['config']['best_cv_score'] = random_search.best_score_
        result['config']['n_iter'] = n_iter

        return result


if __name__ == "__main__":
    # Test hyperparameter search
    print("Testing Hyperparameter Search Module...")

    # Initialize search
    search = HyperparameterSearch(
        data_path='results (2).csv',
        experiment_dir='experiments_test'
    )

    # Prepare data (small config for testing)
    df, feature_cols = search.prepare_data(
        n_lags=3,
        window_days=[7],
        k_days_list=[7],
        include_sequence=True,
        include_time_window=True
    )

    # Run a simple experiment
    print("\nRunning simple experiment...")
    config = {
        'n_lags': 3,
        'window_days': [7],
        'feature_set': 'all',
    }

    result = search.run_single_experiment(
        df=df,
        feature_cols=feature_cols,
        model_name='random_forest',
        target_col='target_next_intensity',
        task_type='regression',
        model_params={'n_estimators': 50, 'max_depth': 5},
        config=config
    )

    print(f"\nTest MAE: {result['metrics']['test_mae']:.4f}")
    print(f"Test RMSE: {result['metrics']['test_rmse']:.4f}")

    # Log experiment
    run_id = search.tracker.log_experiment(
        config=result['config'],
        metrics=result['metrics'],
        model_name='random_forest',
        target_type='next_intensity'
    )

    print(f"\nExperiment logged: {run_id}")

    # Clean up
    import shutil
    shutil.rmtree('experiments_test')
    print("\nTest directory cleaned up.")
