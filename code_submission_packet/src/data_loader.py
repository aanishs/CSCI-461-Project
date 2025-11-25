"""
Data loading and preprocessing module for tic episode prediction.
"""

import pandas as pd
import numpy as np
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Load tic episode data from CSV and perform initial cleaning.

    Parameters
    ----------
    filepath : str
        Path to the CSV file

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame sorted by userId and timestamp
    """
    # Load data
    df = pd.read_csv(filepath)

    # Parse timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = pd.to_datetime(df['date'])

    # Convert intensity to numeric
    df['intensity'] = pd.to_numeric(df['intensity'], errors='coerce')

    # Handle 'null' strings in categorical columns
    for col in ['mood', 'trigger', 'description']:
        if col in df.columns:
            df[col] = df[col].replace('null', np.nan)

    # Sort by user and timestamp (CRITICAL for time series)
    df = df.sort_values(['userId', 'timestamp']).reset_index(drop=True)

    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Generate summary statistics for the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned tic episode DataFrame

    Returns
    -------
    dict
        Dictionary containing summary statistics
    """
    summary = {
        'total_records': len(df),
        'unique_users': df['userId'].nunique(),
        'date_range': (df['date'].min(), df['date'].max()),
        'intensity_stats': {
            'mean': df['intensity'].mean(),
            'median': df['intensity'].median(),
            'std': df['intensity'].std(),
            'min': df['intensity'].min(),
            'max': df['intensity'].max(),
        },
        'missing_values': df.isnull().sum().to_dict(),
        'unique_types': df['type'].nunique() if 'type' in df.columns else 0,
        'unique_moods': df['mood'].nunique() if 'mood' in df.columns else 0,
        'unique_triggers': df['trigger'].nunique() if 'trigger' in df.columns else 0,
    }

    # Add high-intensity rate
    if 'intensity' in df.columns:
        summary['high_intensity_rate'] = (df['intensity'] >= 7).mean()

    # Per-user statistics
    user_counts = df.groupby('userId').size()
    summary['user_engagement'] = {
        'mean_episodes_per_user': user_counts.mean(),
        'median_episodes_per_user': user_counts.median(),
        'min_episodes': user_counts.min(),
        'max_episodes': user_counts.max(),
    }

    return summary


def filter_users_by_min_episodes(df: pd.DataFrame, min_episodes: int = 4) -> pd.DataFrame:
    """
    Filter out users with fewer than min_episodes.

    Parameters
    ----------
    df : pd.DataFrame
        Tic episode DataFrame
    min_episodes : int
        Minimum number of episodes required per user

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame
    """
    user_counts = df.groupby('userId').size()
    valid_users = user_counts[user_counts >= min_episodes].index
    return df[df['userId'].isin(valid_users)].copy()


def print_data_summary(summary: dict):
    """
    Print formatted data summary.

    Parameters
    ----------
    summary : dict
        Summary dictionary from get_data_summary()
    """
    print("=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)

    print(f"\nDataset Overview:")
    print(f"  Total records: {summary['total_records']:,}")
    print(f"  Unique users: {summary['unique_users']}")
    print(f"  Date range: {summary['date_range'][0].date()} to {summary['date_range'][1].date()}")

    print(f"\nIntensity Statistics:")
    stats = summary['intensity_stats']
    print(f"  Mean: {stats['mean']:.2f}")
    print(f"  Median: {stats['median']:.1f}")
    print(f"  Std: {stats['std']:.2f}")
    print(f"  Range: [{stats['min']:.0f}, {stats['max']:.0f}]")

    if 'high_intensity_rate' in summary:
        print(f"  High-intensity rate (≥7): {summary['high_intensity_rate']*100:.1f}%")

    print(f"\nUser Engagement:")
    eng = summary['user_engagement']
    print(f"  Mean episodes per user: {eng['mean_episodes_per_user']:.1f}")
    print(f"  Median episodes per user: {eng['median_episodes_per_user']:.0f}")
    print(f"  Range: [{eng['min_episodes']}, {eng['max_episodes']}]")

    print(f"\nCategorical Variables:")
    print(f"  Unique tic types: {summary['unique_types']}")
    print(f"  Unique moods: {summary['unique_moods']}")
    print(f"  Unique triggers: {summary['unique_triggers']}")

    print(f"\nMissing Values:")
    missing = summary['missing_values']
    for col, count in missing.items():
        if count > 0:
            pct = count / summary['total_records'] * 100
            print(f"  {col}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    # Test the module
    df = load_and_clean_data('results (2).csv')
    summary = get_data_summary(df)
    print_data_summary(summary)

    # Test filtering
    df_filtered = filter_users_by_min_episodes(df, min_episodes=4)
    print(f"\n\nAfter filtering (≥4 episodes):")
    print(f"  Records: {len(df_filtered):,}")
    print(f"  Users: {df_filtered['userId'].nunique()}")
