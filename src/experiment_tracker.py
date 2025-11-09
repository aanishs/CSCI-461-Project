"""
Experiment tracking and logging module.
Logs hyperparameters, metrics, and results to CSV and JSON files.
"""

import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np


class ExperimentTracker:
    """
    Track experiments with hyperparameters and metrics.
    Saves results to CSV and detailed JSON logs.
    """

    def __init__(self, experiment_dir: str = 'experiments'):
        """
        Initialize experiment tracker.

        Parameters
        ----------
        experiment_dir : str
            Directory to save experiment logs
        """
        self.experiment_dir = experiment_dir
        self.results_file = os.path.join(experiment_dir, 'results.csv')
        self.details_dir = os.path.join(experiment_dir, 'details')

        # Create directories if they don't exist
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(self.details_dir, exist_ok=True)

        self.current_run_id = None

    def log_experiment(self, config: Dict[str, Any], metrics: Dict[str, Any],
                      model_name: str, target_type: str,
                      additional_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Log an experiment with its configuration and metrics.

        Parameters
        ----------
        config : dict
            Hyperparameters and configuration
        metrics : dict
            Evaluation metrics
        model_name : str
            Name of the model
        target_type : str
            Type of prediction target
        additional_info : dict, optional
            Additional information to log

        Returns
        -------
        str
            Experiment run ID
        """
        # Generate run ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_id = f"{model_name}_{target_type}_{timestamp}"

        # Flatten config and metrics for CSV
        row_data = {
            'run_id': run_id,
            'timestamp': timestamp,
            'model_name': model_name,
            'target_type': target_type,
        }

        # Add config parameters
        for key, value in config.items():
            if isinstance(value, (int, float, str, bool)):
                row_data[f'config_{key}'] = value
            else:
                row_data[f'config_{key}'] = str(value)

        # Add metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float, bool)):
                row_data[f'metric_{key}'] = value
            elif isinstance(value, np.ndarray):
                row_data[f'metric_{key}'] = value.tolist()
            else:
                row_data[f'metric_{key}'] = str(value)

        # Add additional info
        if additional_info:
            for key, value in additional_info.items():
                if isinstance(value, (int, float, str, bool)):
                    row_data[f'info_{key}'] = value
                else:
                    row_data[f'info_{key}'] = str(value)

        # Save to CSV
        df = pd.DataFrame([row_data])
        if os.path.exists(self.results_file):
            df_existing = pd.read_csv(self.results_file)
            df = pd.concat([df_existing, df], ignore_index=True)
        df.to_csv(self.results_file, index=False)

        # Save detailed JSON
        detailed_log = {
            'run_id': run_id,
            'timestamp': timestamp,
            'model_name': model_name,
            'target_type': target_type,
            'config': config,
            'metrics': metrics,
            'additional_info': additional_info or {},
        }

        json_file = os.path.join(self.details_dir, f'{run_id}.json')
        with open(json_file, 'w') as f:
            json.dump(detailed_log, f, indent=2, default=str)

        self.current_run_id = run_id
        return run_id

    def load_results(self) -> pd.DataFrame:
        """
        Load all experiment results from CSV.

        Returns
        -------
        pd.DataFrame
            DataFrame with all experiment results
        """
        if os.path.exists(self.results_file):
            return pd.read_csv(self.results_file)
        else:
            return pd.DataFrame()

    def load_experiment_details(self, run_id: str) -> Dict[str, Any]:
        """
        Load detailed experiment log from JSON.

        Parameters
        ----------
        run_id : str
            Experiment run ID

        Returns
        -------
        dict
            Detailed experiment log
        """
        json_file = os.path.join(self.details_dir, f'{run_id}.json')
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"No details found for run_id: {run_id}")

    def get_best_experiments(self, metric: str, n: int = 10, minimize: bool = True) -> pd.DataFrame:
        """
        Get top N experiments by a specific metric.

        Parameters
        ----------
        metric : str
            Metric to sort by (should include 'metric_' prefix)
        n : int
            Number of experiments to return
        minimize : bool
            True if lower is better, False if higher is better

        Returns
        -------
        pd.DataFrame
            Top N experiments
        """
        df = self.load_results()

        if len(df) == 0:
            return df

        metric_col = metric if metric.startswith('metric_') else f'metric_{metric}'

        if metric_col not in df.columns:
            raise ValueError(f"Metric {metric} not found in results")

        df_sorted = df.sort_values(metric_col, ascending=minimize)
        return df_sorted.head(n)

    def compare_experiments(self, run_ids: list) -> pd.DataFrame:
        """
        Compare multiple experiments.

        Parameters
        ----------
        run_ids : list
            List of run IDs to compare

        Returns
        -------
        pd.DataFrame
            DataFrame comparing the experiments
        """
        df = self.load_results()
        return df[df['run_id'].isin(run_ids)]

    def get_experiment_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of all experiments.

        Returns
        -------
        dict
            Summary statistics
        """
        df = self.load_results()

        if len(df) == 0:
            return {"message": "No experiments logged yet"}

        summary = {
            'total_experiments': len(df),
            'models_tested': df['model_name'].nunique(),
            'target_types_tested': df['target_type'].nunique(),
            'model_counts': df['model_name'].value_counts().to_dict(),
            'target_counts': df['target_type'].value_counts().to_dict(),
        }

        # Get metric columns
        metric_cols = [c for c in df.columns if c.startswith('metric_')]

        # Best scores for each metric
        summary['best_scores'] = {}
        for col in metric_cols:
            if df[col].dtype in [np.float64, np.int64]:
                metric_name = col.replace('metric_', '')
                # Assume lower is better for most metrics (MAE, RMSE, etc.)
                # Except for R2, accuracy, F1, etc.
                if any(m in metric_name.lower() for m in ['accuracy', 'f1', 'precision', 'recall', 'auc', 'r2']):
                    best_value = df[col].max()
                    best_run = df.loc[df[col].idxmax(), 'run_id']
                else:
                    best_value = df[col].min()
                    best_run = df.loc[df[col].idxmin(), 'run_id']

                summary['best_scores'][metric_name] = {
                    'value': best_value,
                    'run_id': best_run
                }

        return summary

    def print_summary(self):
        """Print experiment summary."""
        summary = self.get_experiment_summary()

        print("=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)

        if 'message' in summary:
            print(f"\n{summary['message']}")
            return

        print(f"\nTotal Experiments: {summary['total_experiments']}")
        print(f"Models Tested: {summary['models_tested']}")
        print(f"Target Types Tested: {summary['target_types_tested']}")

        print(f"\nModel Distribution:")
        for model, count in summary['model_counts'].items():
            print(f"  {model}: {count}")

        print(f"\nTarget Type Distribution:")
        for target, count in summary['target_counts'].items():
            print(f"  {target}: {count}")

        print(f"\nBest Scores:")
        for metric, info in summary['best_scores'].items():
            print(f"  {metric}: {info['value']:.4f} (run: {info['run_id']})")


if __name__ == "__main__":
    # Test experiment tracker
    print("Testing Experiment Tracker...")

    tracker = ExperimentTracker(experiment_dir='experiments_test')

    # Log a few test experiments
    for i in range(3):
        config = {
            'n_estimators': 100 + i * 50,
            'max_depth': 10 + i * 5,
            'learning_rate': 0.1,
            'feature_window_days': 7,
        }

        metrics = {
            'mae': 1.5 + i * 0.2,
            'rmse': 2.0 + i * 0.3,
            'r2': 0.5 - i * 0.05,
            'accuracy': 0.7 + i * 0.05,
        }

        run_id = tracker.log_experiment(
            config=config,
            metrics=metrics,
            model_name='random_forest',
            target_type='next_intensity',
            additional_info={'dataset_size': 1000}
        )

        print(f"\nLogged experiment: {run_id}")

    # Print summary
    print("\n")
    tracker.print_summary()

    # Get best experiments
    print("\n\nBest experiments by MAE:")
    best = tracker.get_best_experiments('mae', n=2, minimize=True)
    print(best[['run_id', 'model_name', 'metric_mae', 'config_n_estimators']])

    # Clean up test directory
    import shutil
    shutil.rmtree('experiments_test')
    print("\n\nTest directory cleaned up.")
