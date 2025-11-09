#!/usr/bin/env python3
"""
Main script to run comprehensive hyperparameter search.
"""

import argparse
import yaml
import sys
from pathlib import Path
from itertools import product
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from hyperparameter_search import HyperparameterSearch
from models import ModelFactory


def load_config(config_path: str = 'config/hyperparameter_grids.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_comprehensive_search(data_path: str = 'results (2).csv',
                            config_path: str = 'config/hyperparameter_grids.yaml',
                            experiment_dir: str = 'experiments',
                            mode: str = 'quick',
                            models: list = None,
                            targets: list = None):
    """
    Run comprehensive hyperparameter search.

    Parameters
    ----------
    data_path : str
        Path to data CSV
    config_path : str
        Path to config YAML
    experiment_dir : str
        Directory for experiment logs
    mode : str
        'quick' (small search), 'medium' (moderate search), or 'full' (comprehensive)
    models : list, optional
        List of models to search (if None, use all)
    targets : list, optional
        List of targets to search (if None, use all)
    """
    # Load configuration
    config = load_config(config_path)

    # Initialize search
    print("=" * 80)
    print("HYPERPARAMETER SEARCH - TIC EPISODE PREDICTION")
    print("=" * 80)
    print(f"\nMode: {mode}")
    print(f"Data: {data_path}")
    print(f"Experiments: {experiment_dir}\n")

    search = HyperparameterSearch(
        data_path=data_path,
        experiment_dir=experiment_dir
    )

    # Determine search parameters based on mode
    if mode == 'quick':
        data_configs = [
            {
                'high_intensity_threshold': 7,
                'n_lags': 3,
                'window_days': [7],
                'k_days_list': [7],
                'feature_set': 'all',
            }
        ]
        model_names = ['random_forest', 'xgboost']
        target_configs = [
            ('target_next_intensity', 'regression'),
            ('target_next_high_intensity', 'classification'),
        ]
        search_type = 'random'
        n_iter = 20

    elif mode == 'medium':
        # Multiple feature configurations
        data_configs = []
        for threshold in [7]:
            for n_lags in [3]:
                for window_days in [[7], [3, 7, 14]]:
                    for feature_set in ['sequence_only', 'time_window_only', 'all']:
                        data_configs.append({
                            'high_intensity_threshold': threshold,
                            'n_lags': n_lags,
                            'window_days': window_days,
                            'k_days_list': [1, 3, 7],
                            'feature_set': feature_set,
                        })

        model_names = ['random_forest', 'xgboost', 'lightgbm']
        target_configs = [
            ('target_next_intensity', 'regression'),
            ('target_next_high_intensity', 'classification'),
            ('target_high_count_next_7d', 'regression'),
            ('target_time_to_high_days', 'regression'),
        ]
        search_type = 'random'
        n_iter = 50

    else:  # full
        # Comprehensive search over all hyperparameters
        data_configs = []
        for threshold in config['data_params']['high_intensity_threshold']:
            for n_lags in config['data_params']['n_lags']:
                for window_days in config['data_params']['feature_window_m_days']:
                    for feature_set_config in config['data_params']['feature_sets']:
                        data_configs.append({
                            'high_intensity_threshold': threshold,
                            'n_lags': n_lags,
                            'window_days': window_days,
                            'k_days_list': [1, 3, 7, 14],
                            'feature_set': feature_set_config['name'],
                            'include_sequence': feature_set_config['include_sequence'],
                            'include_time_window': feature_set_config['include_time_window'],
                            'include_engineered': feature_set_config['include_engineered'],
                        })

        model_names = ['ridge', 'decision_tree', 'random_forest', 'xgboost', 'lightgbm']
        target_configs = [
            ('target_next_intensity', 'regression'),
            ('target_next_high_intensity', 'classification'),
            ('target_high_count_next_3d', 'regression'),
            ('target_high_count_next_7d', 'regression'),
            ('target_has_high_next_7d', 'classification'),
            ('target_time_to_high_days', 'regression'),
        ]
        search_type = 'random'
        n_iter = 100

    # Filter models if specified
    if models is not None:
        model_names = [m for m in model_names if m in models]

    # Filter targets if specified
    if targets is not None:
        target_configs = [(t, tt) for t, tt in target_configs if t in targets]

    # Calculate total experiments
    total_experiments = len(data_configs) * len(model_names) * len(target_configs)

    print(f"Search Configuration:")
    print(f"  Data configurations: {len(data_configs)}")
    print(f"  Models: {len(model_names)} ({', '.join(model_names)})")
    print(f"  Targets: {len(target_configs)}")
    print(f"  Total experiments: {total_experiments}")
    print(f"  Search type: {search_type}")
    if search_type == 'random':
        print(f"  Iterations per experiment: {n_iter}")
    print()

    # Confirm before running
    if mode == 'full':
        response = input(f"This will run {total_experiments} experiments. Continue? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    # Run experiments
    factory = ModelFactory()
    experiment_count = 0

    with tqdm(total=total_experiments, desc="Overall Progress") as pbar:
        for data_config in data_configs:
            # Prepare data for this configuration
            print(f"\nPreparing data: threshold={data_config['high_intensity_threshold']}, "
                  f"n_lags={data_config['n_lags']}, "
                  f"windows={data_config['window_days']}, "
                  f"feature_set={data_config['feature_set']}")

            # Get feature set parameters
            if 'include_sequence' in data_config:
                include_sequence = data_config['include_sequence']
                include_time_window = data_config['include_time_window']
                include_engineered = data_config['include_engineered']
            else:
                # Use defaults based on feature_set name
                include_sequence = data_config['feature_set'] in ['sequence_only', 'both', 'all']
                include_time_window = data_config['feature_set'] in ['time_window_only', 'both', 'all']
                include_engineered = data_config['feature_set'] == 'all'

            df, feature_cols = search.prepare_data(
                high_intensity_threshold=data_config['high_intensity_threshold'],
                n_lags=data_config['n_lags'],
                window_days=data_config['window_days'],
                k_days_list=data_config['k_days_list'],
                include_sequence=include_sequence,
                include_time_window=include_time_window,
                include_engineered=include_engineered
            )

            for model_name in model_names:
                for target_col, task_type in target_configs:
                    # Skip if target doesn't exist in data
                    if target_col not in df.columns:
                        pbar.update(1)
                        continue

                    experiment_count += 1
                    print(f"\n[{experiment_count}/{total_experiments}] "
                          f"Model: {model_name}, Target: {target_col}")

                    try:
                        # Get parameter grid
                        if search_type == 'random':
                            param_grid = factory.get_param_grid(model_name, 'random')

                            result = search.random_search_sklearn(
                                df=df,
                                feature_cols=feature_cols,
                                model_name=model_name,
                                target_col=target_col,
                                task_type=task_type,
                                param_distributions=param_grid,
                                config=data_config,
                                n_iter=n_iter,
                                cv_splits=3,
                                n_jobs=-1
                            )
                        else:
                            param_grid = factory.get_param_grid(model_name, 'grid')

                            result = search.grid_search_sklearn(
                                df=df,
                                feature_cols=feature_cols,
                                model_name=model_name,
                                target_col=target_col,
                                task_type=task_type,
                                param_grid=param_grid,
                                config=data_config,
                                cv_splits=5,
                                n_jobs=-1
                            )

                        # Log experiment
                        run_id = search.tracker.log_experiment(
                            config=result['config'],
                            metrics=result['metrics'],
                            model_name=model_name,
                            target_type=target_col
                        )

                        # Print results
                        if task_type == 'regression':
                            print(f"  Test MAE: {result['metrics']['test_mae']:.4f}")
                        else:
                            print(f"  Test F1: {result['metrics']['test_f1']:.4f}")

                    except Exception as e:
                        print(f"  ERROR: {str(e)}")

                    pbar.update(1)

    # Print summary
    print("\n" + "=" * 80)
    print("SEARCH COMPLETE")
    print("=" * 80)
    search.tracker.print_summary()

    # Print best results for key metrics
    print("\n" + "=" * 80)
    print("TOP RESULTS")
    print("=" * 80)

    results_df = search.tracker.load_results()

    if len(results_df) > 0:
        # Best regression models
        print("\nTop 5 Regression Models (by Test MAE):")
        reg_results = results_df[results_df['target_type'].str.contains('intensity|time|count')]
        if 'metric_test_mae' in reg_results.columns:
            top_reg = reg_results.nsmallest(5, 'metric_test_mae')
            print(top_reg[['run_id', 'model_name', 'target_type', 'metric_test_mae', 'metric_test_rmse']])

        # Best classification models
        print("\nTop 5 Classification Models (by Test F1):")
        clf_results = results_df[results_df['target_type'].str.contains('high')]
        if 'metric_test_f1' in clf_results.columns:
            top_clf = clf_results.nlargest(5, 'metric_test_f1')
            print(top_clf[['run_id', 'model_name', 'target_type', 'metric_test_f1', 'metric_test_pr_auc']])

    print(f"\nResults saved to: {experiment_dir}/results.csv")
    print(f"Detailed logs in: {experiment_dir}/details/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run hyperparameter search for tic episode prediction')

    parser.add_argument('--data', type=str, default='results (2).csv',
                       help='Path to data CSV file')
    parser.add_argument('--config', type=str, default='config/hyperparameter_grids.yaml',
                       help='Path to configuration YAML file')
    parser.add_argument('--output', type=str, default='experiments',
                       help='Directory for experiment outputs')
    parser.add_argument('--mode', type=str, default='quick',
                       choices=['quick', 'medium', 'full'],
                       help='Search mode: quick, medium, or full')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                       help='Specific models to search (e.g., random_forest xgboost)')
    parser.add_argument('--targets', type=str, nargs='+', default=None,
                       help='Specific targets to search (e.g., target_next_intensity)')

    args = parser.parse_args()

    run_comprehensive_search(
        data_path=args.data,
        config_path=args.config,
        experiment_dir=args.output,
        mode=args.mode,
        models=args.models,
        targets=args.targets
    )
