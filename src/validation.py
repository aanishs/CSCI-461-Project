"""
Cross-validation strategies for tic episode prediction.
Includes TimeSeriesSplit and Leave-One-User-Out validation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Iterator


class UserGroupedSplit:
    """
    Split data by user groups to prevent data leakage.
    Ensures that a user's data appears in either train or test, but not both.
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize user-grouped splitter.

        Parameters
        ----------
        test_size : float
            Proportion of users to include in test set
        random_state : int
            Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state

    def split(self, df: pd.DataFrame, user_col: str = 'userId') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test by user.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        user_col : str
            Column name containing user IDs

        Returns
        -------
        tuple of pd.DataFrame
            (train_df, test_df)
        """
        users = df[user_col].unique()
        train_users, test_users = train_test_split(
            users,
            test_size=self.test_size,
            random_state=self.random_state
        )

        train_df = df[df[user_col].isin(train_users)]
        test_df = df[df[user_col].isin(test_users)]

        return train_df, test_df


class TemporalSplit:
    """
    Split data temporally within each user.
    For each user, use first N% of their tics for training, rest for testing.
    """

    def __init__(self, train_frac: float = 0.8):
        """
        Initialize temporal splitter.

        Parameters
        ----------
        train_frac : float
            Fraction of each user's tics to use for training
        """
        self.train_frac = train_frac

    def split(self, df: pd.DataFrame, user_col: str = 'userId',
             time_col: str = 'timestamp') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data temporally within each user.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        user_col : str
            Column name containing user IDs
        time_col : str
            Column name containing timestamps

        Returns
        -------
        tuple of pd.DataFrame
            (train_df, test_df)
        """
        def temporal_split_user(group):
            n_train = int(len(group) * self.train_frac)
            group['temporal_split'] = ['train'] * n_train + ['test'] * (len(group) - n_train)
            return group

        df_split = df.sort_values([user_col, time_col]).copy()
        df_split = df_split.groupby(user_col, group_keys=False).apply(temporal_split_user)

        train_df = df_split[df_split['temporal_split'] == 'train'].drop(columns=['temporal_split'])
        test_df = df_split[df_split['temporal_split'] == 'test'].drop(columns=['temporal_split'])

        return train_df, test_df


class UserTimeSeriesSplit:
    """
    Time series cross-validation within each user.
    For each user, create multiple train/test splits with increasing training windows.
    """

    def __init__(self, n_splits: int = 5, test_size: int = None):
        """
        Initialize user-level time series splitter.

        Parameters
        ----------
        n_splits : int
            Number of splits to create
        test_size : int, optional
            Number of samples in each test fold (if None, determined automatically)
        """
        self.n_splits = n_splits
        self.test_size = test_size

    def split(self, df: pd.DataFrame, user_col: str = 'userId',
             time_col: str = 'timestamp') -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices for time series cross-validation.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame sorted by user and time
        user_col : str
            Column name containing user IDs
        time_col : str
            Column name containing timestamps

        Yields
        ------
        tuple of np.ndarray
            (train_indices, test_indices) for each fold
        """
        df = df.sort_values([user_col, time_col]).reset_index(drop=True)

        all_train_indices = []
        all_test_indices = []

        for user_id in df[user_col].unique():
            user_indices = df[df[user_col] == user_id].index.values
            n_samples = len(user_indices)

            if n_samples < self.n_splits + 1:
                # Skip users with too few samples
                continue

            if self.test_size is None:
                # Determine test size automatically
                test_size = max(1, n_samples // (self.n_splits + 1))
            else:
                test_size = self.test_size

            # Create splits for this user
            for i in range(self.n_splits):
                test_start = n_samples - (self.n_splits - i) * test_size
                test_end = test_start + test_size

                if test_start <= 0 or test_end > n_samples:
                    continue

                train_idx = user_indices[:test_start]
                test_idx = user_indices[test_start:test_end]

                if len(all_train_indices) <= i:
                    all_train_indices.append([])
                    all_test_indices.append([])

                all_train_indices[i].extend(train_idx)
                all_test_indices[i].extend(test_idx)

        # Yield splits
        for i in range(len(all_train_indices)):
            yield np.array(all_train_indices[i]), np.array(all_test_indices[i])

    def get_n_splits(self):
        """Get number of splits."""
        return self.n_splits


class LeaveOneUserOut:
    """
    Leave-one-user-out cross-validation.
    For each user, train on all other users and test on that user.
    """

    def __init__(self, min_test_samples: int = 5):
        """
        Initialize LOUO splitter.

        Parameters
        ----------
        min_test_samples : int
            Minimum number of samples required for a user to be used as test set
        """
        self.min_test_samples = min_test_samples

    def split(self, df: pd.DataFrame, user_col: str = 'userId') -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices for leave-one-user-out cross-validation.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        user_col : str
            Column name containing user IDs

        Yields
        ------
        tuple of np.ndarray
            (train_indices, test_indices) for each fold
        """
        df = df.reset_index(drop=True)
        user_counts = df[user_col].value_counts()

        # Only consider users with enough samples
        valid_users = user_counts[user_counts >= self.min_test_samples].index

        for user_id in valid_users:
            train_idx = df[df[user_col] != user_id].index.values
            test_idx = df[df[user_col] == user_id].index.values

            yield train_idx, test_idx

    def get_n_splits(self, df: pd.DataFrame, user_col: str = 'userId'):
        """
        Get number of splits.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        user_col : str
            Column name containing user IDs

        Returns
        -------
        int
            Number of splits (number of valid users)
        """
        user_counts = df[user_col].value_counts()
        valid_users = user_counts[user_counts >= self.min_test_samples].index
        return len(valid_users)


def create_validation_split(df: pd.DataFrame, split_type: str = 'user_grouped',
                           user_col: str = 'userId', time_col: str = 'timestamp',
                           **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create train/test split using specified strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    split_type : str
        One of: 'user_grouped', 'temporal', 'random'
    user_col : str
        Column name containing user IDs
    time_col : str
        Column name containing timestamps
    **kwargs
        Additional arguments passed to splitter

    Returns
    -------
    tuple of pd.DataFrame
        (train_df, test_df)
    """
    if split_type == 'user_grouped':
        splitter = UserGroupedSplit(**kwargs)
        return splitter.split(df, user_col=user_col)

    elif split_type == 'temporal':
        splitter = TemporalSplit(**kwargs)
        return splitter.split(df, user_col=user_col, time_col=time_col)

    elif split_type == 'random':
        test_size = kwargs.get('test_size', 0.2)
        random_state = kwargs.get('random_state', 42)
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        return train_df, test_df

    else:
        raise ValueError(f"Unknown split type: {split_type}")


if __name__ == "__main__":
    # Test validation module
    from data_loader import load_and_clean_data

    print("Testing Validation Module...")

    df = load_and_clean_data('results (2).csv')
    df = df[df.groupby('userId')['userId'].transform('count') >= 10]  # Filter users with ≥10 samples

    print(f"\nDataset: {len(df)} samples, {df['userId'].nunique()} users")

    # Test user-grouped split
    print("\n1. User-Grouped Split:")
    splitter = UserGroupedSplit(test_size=0.2, random_state=42)
    train_df, test_df = splitter.split(df)
    print(f"   Train: {len(train_df)} samples, {train_df['userId'].nunique()} users")
    print(f"   Test:  {len(test_df)} samples, {test_df['userId'].nunique()} users")

    # Test temporal split
    print("\n2. Temporal Split:")
    splitter = TemporalSplit(train_frac=0.8)
    train_df, test_df = splitter.split(df)
    print(f"   Train: {len(train_df)} samples")
    print(f"   Test:  {len(test_df)} samples")

    # Test time series CV
    print("\n3. User Time Series Split:")
    splitter = UserTimeSeriesSplit(n_splits=3)
    n_folds = 0
    for train_idx, test_idx in splitter.split(df):
        n_folds += 1
        print(f"   Fold {n_folds}: Train={len(train_idx)}, Test={len(test_idx)}")

    # Test LOUO
    print("\n4. Leave-One-User-Out (first 3 folds):")
    splitter = LeaveOneUserOut(min_test_samples=5)
    n_splits = splitter.get_n_splits(df)
    print(f"   Total folds: {n_splits}")

    for i, (train_idx, test_idx) in enumerate(splitter.split(df)):
        if i >= 3:
            break
        print(f"   Fold {i+1}: Train={len(train_idx)}, Test={len(test_idx)}")
