"""
Target generation module for different prediction tasks.
Supports 3 types of prediction targets:
1. Next single tic intensity (regression)
2. Count of high-intensity episodes in next k days (regression/classification)
3. Time to next high-intensity episode (regression)
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


class TargetGenerator:
    """
    Generate different types of prediction targets for tic episode data.
    """

    def __init__(self, high_intensity_threshold: int = 7):
        """
        Initialize target generator.

        Parameters
        ----------
        high_intensity_threshold : int
            Threshold for classifying a tic as high-intensity
        """
        self.high_intensity_threshold = high_intensity_threshold

    def create_next_intensity_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Target 1: Predict intensity of the next single tic episode.
        This is the baseline approach.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with tic episodes sorted by userId and timestamp

        Returns
        -------
        pd.DataFrame
            DataFrame with 'target_next_intensity' and 'target_next_high_intensity' columns
        """
        df = df.copy()

        # The target is simply the current intensity
        # (when we predict, we're predicting the "next" tic which is the current row)
        df['target_next_intensity'] = df['intensity']
        df['target_next_high_intensity'] = (df['intensity'] >= self.high_intensity_threshold).astype(int)

        return df

    def create_future_count_target(self, df: pd.DataFrame, k_days: int = 7) -> pd.DataFrame:
        """
        Target 2: Predict count of high-intensity episodes in the next k days.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with tic episodes sorted by userId and timestamp
        k_days : int
            Number of days to look ahead

        Returns
        -------
        pd.DataFrame
            DataFrame with target columns for counts
        """
        df = df.copy()

        count_targets = []
        high_count_targets = []

        for user_id in df['userId'].unique():
            user_df = df[df['userId'] == user_id].copy()
            user_df = user_df.sort_values('timestamp')

            user_counts = []
            user_high_counts = []

            for idx, row in user_df.iterrows():
                current_time = row['timestamp']
                window_end = current_time + pd.Timedelta(days=k_days)

                # Get future episodes in the next k days (excluding current episode)
                future_data = user_df[
                    (user_df['timestamp'] > current_time) &
                    (user_df['timestamp'] <= window_end)
                ]

                # Count episodes
                total_count = len(future_data)
                high_count = (future_data['intensity'] >= self.high_intensity_threshold).sum()

                user_counts.append(total_count)
                user_high_counts.append(high_count)

            count_df = pd.DataFrame({
                f'target_count_next_{k_days}d': user_counts,
                f'target_high_count_next_{k_days}d': user_high_counts,
                f'target_has_high_next_{k_days}d': [1 if c > 0 else 0 for c in user_high_counts],
            }, index=user_df.index)

            count_targets.append(count_df)

        # Concatenate all user targets
        all_targets = pd.concat(count_targets)

        # Merge back to original dataframe
        df = df.join(all_targets)

        return df

    def create_time_to_event_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Target 3: Predict time (in hours) to next high-intensity episode.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with tic episodes sorted by userId and timestamp

        Returns
        -------
        pd.DataFrame
            DataFrame with time-to-event targets
        """
        df = df.copy()

        time_targets = []
        event_occurred = []

        for user_id in df['userId'].unique():
            user_df = df[df['userId'] == user_id].copy()
            user_df = user_df.sort_values('timestamp')

            user_times = []
            user_events = []

            for idx, row in user_df.iterrows():
                current_time = row['timestamp']

                # Find next high-intensity episode
                future_high = user_df[
                    (user_df['timestamp'] > current_time) &
                    (user_df['intensity'] >= self.high_intensity_threshold)
                ]

                if len(future_high) > 0:
                    # Time to next high-intensity episode
                    next_high_time = future_high.iloc[0]['timestamp']
                    time_diff_hours = (next_high_time - current_time).total_seconds() / 3600
                    user_times.append(time_diff_hours)
                    user_events.append(1)  # Event occurred
                else:
                    # No future high-intensity episode (censored)
                    # Use maximum observation time
                    max_time = user_df['timestamp'].max()
                    time_diff_hours = (max_time - current_time).total_seconds() / 3600
                    user_times.append(time_diff_hours)
                    user_events.append(0)  # Censored

            time_df = pd.DataFrame({
                'target_time_to_high_hours': user_times,
                'target_time_to_high_days': [t / 24 for t in user_times],
                'target_event_occurred': user_events,
            }, index=user_df.index)

            time_targets.append(time_df)

        # Concatenate all user targets
        all_targets = pd.concat(time_targets)

        # Merge back to original dataframe
        df = df.join(all_targets)

        return df

    def create_all_targets(self, df: pd.DataFrame, k_days_list: Optional[list] = None) -> pd.DataFrame:
        """
        Create all target types.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with tic episodes
        k_days_list : list, optional
            List of prediction windows for future count targets

        Returns
        -------
        pd.DataFrame
            DataFrame with all target columns
        """
        if k_days_list is None:
            k_days_list = [1, 3, 7, 14]

        # Create target 1: next intensity
        df = self.create_next_intensity_target(df)

        # Create target 2: future counts for different windows
        for k_days in k_days_list:
            df = self.create_future_count_target(df, k_days=k_days)

        # Create target 3: time to event
        df = self.create_time_to_event_target(df)

        return df

    def get_target_columns(self, target_type: str, k_days: Optional[int] = None) -> list:
        """
        Get target column names for a given target type.

        Parameters
        ----------
        target_type : str
            One of: 'next_intensity', 'future_count', 'time_to_event'
        k_days : int, optional
            Required for 'future_count' target type

        Returns
        -------
        list
            List of target column names
        """
        if target_type == 'next_intensity':
            return ['target_next_intensity', 'target_next_high_intensity']
        elif target_type == 'future_count':
            if k_days is None:
                raise ValueError("k_days must be specified for future_count target")
            return [
                f'target_count_next_{k_days}d',
                f'target_high_count_next_{k_days}d',
                f'target_has_high_next_{k_days}d'
            ]
        elif target_type == 'time_to_event':
            return ['target_time_to_high_hours', 'target_time_to_high_days', 'target_event_occurred']
        else:
            raise ValueError(f"Unknown target type: {target_type}")

    def get_target_info(self) -> dict:
        """
        Get information about all available targets.

        Returns
        -------
        dict
            Dictionary describing each target type
        """
        return {
            'next_intensity': {
                'description': 'Predict intensity of the next single tic episode',
                'task_types': ['regression', 'binary_classification'],
                'targets': ['target_next_intensity', 'target_next_high_intensity'],
            },
            'future_count': {
                'description': 'Predict count of high-intensity episodes in next k days',
                'task_types': ['regression', 'binary_classification'],
                'targets': ['target_count_next_Kd', 'target_high_count_next_Kd', 'target_has_high_next_Kd'],
            },
            'time_to_event': {
                'description': 'Predict time (hours/days) until next high-intensity episode',
                'task_types': ['regression', 'survival_analysis'],
                'targets': ['target_time_to_high_hours', 'target_time_to_high_days', 'target_event_occurred'],
            },
        }


if __name__ == "__main__":
    # Test the target generation module
    from data_loader import load_and_clean_data

    print("Testing Target Generation Module...")

    df = load_and_clean_data('results (2).csv')

    tg = TargetGenerator(high_intensity_threshold=7)
    df_targets = tg.create_all_targets(df, k_days_list=[3, 7])

    print(f"\nOriginal shape: {df.shape}")
    print(f"With targets shape: {df_targets.shape}")

    print("\nTarget columns created:")
    target_cols = [c for c in df_targets.columns if c.startswith('target_')]
    for col in target_cols:
        print(f"  - {col}")

    print("\nTarget statistics (7-day window):")
    print(f"  Mean count next 7 days: {df_targets['target_count_next_7d'].mean():.2f}")
    print(f"  Mean high-intensity count next 7 days: {df_targets['target_high_count_next_7d'].mean():.2f}")
    print(f"  Rate of having high-intensity in next 7 days: {df_targets['target_has_high_next_7d'].mean()*100:.1f}%")
    print(f"  Mean time to high-intensity: {df_targets['target_time_to_high_days'].mean():.2f} days")
    print(f"  Event occurrence rate: {df_targets['target_event_occurred'].mean()*100:.1f}%")
