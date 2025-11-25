"""
Feature engineering module for tic episode prediction.
Supports both sequence-based and time-window based features.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Feature engineering for tic episode prediction.
    Supports both sequence-based (last n ticks) and time-window (past m days) features.
    """

    def __init__(self):
        self.label_encoders = {}
        self.feature_names = []

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create basic temporal features from timestamp.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'timestamp' column

        Returns
        -------
        pd.DataFrame
            DataFrame with additional temporal features
        """
        df = df.copy()

        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month

        return df

    def create_sequence_features(self, df: pd.DataFrame, n_lags: int = 3) -> pd.DataFrame:
        """
        Create sequence-based features (last n tic episodes).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame sorted by userId and timestamp
        n_lags : int
            Number of previous episodes to use as features

        Returns
        -------
        pd.DataFrame
            DataFrame with sequence features
        """
        df = df.copy()

        # Time since previous tic (in hours)
        df['time_since_prev_hours'] = df.groupby('userId')['timestamp'].diff().dt.total_seconds() / 3600

        # Lag features for intensity
        for i in range(1, n_lags + 1):
            df[f'prev_intensity_{i}'] = df.groupby('userId')['intensity'].shift(i)

        # Lag features for type
        for i in range(1, min(n_lags, 2) + 1):  # Only first 1-2 types
            df[f'prev_type_{i}'] = df.groupby('userId')['type'].shift(i)

        # Lag features for time of day
        df['prev_timeOfDay_1'] = df.groupby('userId')['timeOfDay'].shift(1)

        return df

    def create_time_window_features(self, df: pd.DataFrame, window_days: int = 7) -> pd.DataFrame:
        """
        Create time-window based features (aggregations over past m days).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame sorted by userId and timestamp
        window_days : int
            Number of days to look back for window features

        Returns
        -------
        pd.DataFrame
            DataFrame with time-window features
        """
        df = df.copy()

        # For each row, calculate features based on episodes in the past window_days
        # We need to ensure no data leakage - only look at PAST episodes

        window_features = []

        for user_id in df['userId'].unique():
            user_df = df[df['userId'] == user_id].copy()
            user_df = user_df.sort_values('timestamp')

            features_list = []

            for idx, row in user_df.iterrows():
                current_time = row['timestamp']
                window_start = current_time - pd.Timedelta(days=window_days)

                # Get episodes in the window (excluding current episode)
                window_data = user_df[
                    (user_df['timestamp'] < current_time) &
                    (user_df['timestamp'] >= window_start)
                ]

                # Aggregate features
                features = {
                    f'window_{window_days}d_count': len(window_data),
                    f'window_{window_days}d_mean_intensity': window_data['intensity'].mean() if len(window_data) > 0 else np.nan,
                    f'window_{window_days}d_max_intensity': window_data['intensity'].max() if len(window_data) > 0 else np.nan,
                    f'window_{window_days}d_min_intensity': window_data['intensity'].min() if len(window_data) > 0 else np.nan,
                    f'window_{window_days}d_std_intensity': window_data['intensity'].std() if len(window_data) > 0 else np.nan,
                    f'window_{window_days}d_high_intensity_count': (window_data['intensity'] >= 7).sum() if len(window_data) > 0 else 0,
                    f'window_{window_days}d_high_intensity_rate': (window_data['intensity'] >= 7).mean() if len(window_data) > 0 else np.nan,
                }

                # Time-based patterns
                if len(window_data) > 0:
                    features[f'window_{window_days}d_weekend_rate'] = window_data['is_weekend'].mean() if 'is_weekend' in window_data.columns else np.nan
                    features[f'window_{window_days}d_mean_hour'] = window_data['hour'].mean() if 'hour' in window_data.columns else np.nan
                else:
                    features[f'window_{window_days}d_weekend_rate'] = np.nan
                    features[f'window_{window_days}d_mean_hour'] = np.nan

                features_list.append(features)

            user_features = pd.DataFrame(features_list, index=user_df.index)
            window_features.append(user_features)

        # Concatenate all user features
        all_features = pd.concat(window_features)

        # Merge back to original dataframe
        df = df.join(all_features)

        return df

    def create_user_level_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create user-level aggregated features (expanding window to prevent leakage).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame sorted by userId and timestamp

        Returns
        -------
        pd.DataFrame
            DataFrame with user-level features
        """
        df = df.copy()

        # User's historical statistics (expanding mean/std)
        df['user_mean_intensity'] = df.groupby('userId')['intensity'].expanding().mean().shift(1).reset_index(level=0, drop=True)
        df['user_std_intensity'] = df.groupby('userId')['intensity'].expanding().std().shift(1).reset_index(level=0, drop=True)
        df['user_max_intensity'] = df.groupby('userId')['intensity'].expanding().max().shift(1).reset_index(level=0, drop=True)
        df['user_min_intensity'] = df.groupby('userId')['intensity'].expanding().min().shift(1).reset_index(level=0, drop=True)

        # Number of tics so far for this user
        df['user_tic_count'] = df.groupby('userId').cumcount()

        return df

    def create_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered/interaction features.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with base features

        Returns
        -------
        pd.DataFrame
            DataFrame with engineered features
        """
        df = df.copy()

        # Existing interaction features
        if 'user_mean_intensity' in df.columns and 'user_tic_count' in df.columns:
            df['intensity_x_count'] = df['user_mean_intensity'] * df['user_tic_count']

        # Intensity trend (is user's intensity increasing or decreasing?)
        if 'prev_intensity_1' in df.columns and 'prev_intensity_2' in df.columns:
            df['intensity_trend'] = df['prev_intensity_1'] - df['prev_intensity_2']

        # Volatility (recent intensity variance)
        if all(f'prev_intensity_{i}' in df.columns for i in range(1, 4)):
            df['recent_intensity_volatility'] = df[[f'prev_intensity_{i}' for i in range(1, 4)]].std(axis=1)

        # NEW INTERACTION FEATURES (from prelim promises)

        # Interaction 1: mood × timeOfDay
        if 'mood_encoded' in df.columns and 'timeOfDay_encoded' in df.columns:
            df['mood_x_timeOfDay'] = df['mood_encoded'] * df['timeOfDay_encoded']

        # Interaction 2: trigger × type
        if 'trigger_encoded' in df.columns and 'type_encoded' in df.columns:
            df['trigger_x_type'] = df['trigger_encoded'] * df['type_encoded']

        # Interaction 3: mood × recent intensity
        if 'mood_encoded' in df.columns and 'prev_intensity_1' in df.columns:
            df['mood_x_prev_intensity'] = df['mood_encoded'] * df['prev_intensity_1']

        # Interaction 4: timeOfDay × hour (categorical × continuous)
        if 'timeOfDay_encoded' in df.columns and 'hour' in df.columns:
            df['timeOfDay_x_hour'] = df['timeOfDay_encoded'] * df['hour']

        # Interaction 5: type × hour
        if 'type_encoded' in df.columns and 'hour' in df.columns:
            df['type_x_hour'] = df['type_encoded'] * df['hour']

        # Interaction 6: is_weekend × hour
        if 'is_weekend' in df.columns and 'hour' in df.columns:
            df['weekend_x_hour'] = df['is_weekend'] * df['hour']

        return df

    def encode_categorical_features(self, df: pd.DataFrame,
                                   categorical_cols: Optional[List[str]] = None,
                                   fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features using LabelEncoder.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with categorical columns
        categorical_cols : List[str], optional
            List of categorical columns to encode. If None, uses default list.
        fit : bool
            Whether to fit new encoders (True for training, False for test)

        Returns
        -------
        pd.DataFrame
            DataFrame with encoded categorical features
        """
        df = df.copy()

        if categorical_cols is None:
            categorical_cols = ['type', 'timeOfDay', 'mood', 'trigger', 'prev_type_1', 'prev_timeOfDay_1']

        for col in categorical_cols:
            if col not in df.columns:
                continue

            encoded_col = f'{col}_encoded'

            if fit:
                # Fit new encoder
                le = LabelEncoder()
                df[encoded_col] = le.fit_transform(df[col].fillna('missing'))
                self.label_encoders[col] = le
            else:
                # Use existing encoder
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    # Handle unseen labels
                    df[col] = df[col].fillna('missing')
                    df[encoded_col] = df[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
                else:
                    raise ValueError(f"No encoder found for column {col}. Make sure to fit first.")

        # Add binary flags for optional fields
        if 'mood' in df.columns:
            df['has_mood'] = (~df['mood'].isna()).astype(int)
        if 'trigger' in df.columns:
            df['has_trigger'] = (~df['trigger'].isna()).astype(int)

        return df

    def create_all_features(self, df: pd.DataFrame,
                          n_lags: int = 3,
                          window_days: Optional[List[int]] = None,
                          fit: bool = True) -> pd.DataFrame:
        """
        Create all features: temporal, sequence, time-window, user-level, and engineered.

        Parameters
        ----------
        df : pd.DataFrame
            Raw DataFrame with userId, timestamp, intensity, type, etc.
        n_lags : int
            Number of lag features to create
        window_days : List[int], optional
            List of window sizes (in days) for time-window features.
            If None, defaults to [3, 7, 14].
        fit : bool
            Whether to fit label encoders (True for training, False for test)

        Returns
        -------
        pd.DataFrame
            DataFrame with all features
        """
        if window_days is None:
            window_days = [3, 7, 14]

        # Create features step by step
        df = self.create_temporal_features(df)
        df = self.create_sequence_features(df, n_lags=n_lags)
        df = self.create_user_level_features(df)

        # Create time-window features for each window size
        for window in window_days:
            df = self.create_time_window_features(df, window_days=window)

        df = self.create_engineered_features(df)
        df = self.encode_categorical_features(df, fit=fit)

        return df

    def get_feature_columns(self, df: pd.DataFrame,
                           include_sequence: bool = True,
                           include_time_window: bool = True,
                           include_engineered: bool = True) -> List[str]:
        """
        Get list of feature columns based on what's included.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with features
        include_sequence : bool
            Include sequence-based features
        include_time_window : bool
            Include time-window features
        include_engineered : bool
            Include engineered/interaction features

        Returns
        -------
        List[str]
            List of feature column names
        """
        feature_cols = []

        # Always include temporal features
        temporal = ['hour', 'day_of_week', 'is_weekend', 'day_of_month', 'month']
        feature_cols.extend([c for c in temporal if c in df.columns])

        # Encoded categorical features
        categorical_encoded = ['type_encoded', 'timeOfDay_encoded', 'mood_encoded',
                              'trigger_encoded', 'has_mood', 'has_trigger',
                              'prev_type_1_encoded', 'prev_timeOfDay_1_encoded']
        feature_cols.extend([c for c in categorical_encoded if c in df.columns])

        # User-level features
        user_level = ['user_mean_intensity', 'user_std_intensity', 'user_max_intensity',
                     'user_min_intensity', 'user_tic_count']
        feature_cols.extend([c for c in user_level if c in df.columns])

        # Sequence features
        if include_sequence:
            sequence = [c for c in df.columns if c.startswith('prev_intensity_') or c == 'time_since_prev_hours']
            feature_cols.extend(sequence)

        # Time-window features
        if include_time_window:
            time_window = [c for c in df.columns if c.startswith('window_')]
            feature_cols.extend(time_window)

        # Engineered features
        if include_engineered:
            engineered = [
                'intensity_x_count',
                'intensity_trend',
                'recent_intensity_volatility',
                'mood_x_timeOfDay',
                'trigger_x_type',
                'mood_x_prev_intensity',
                'timeOfDay_x_hour',
                'type_x_hour',
                'weekend_x_hour'
            ]
            feature_cols.extend([c for c in engineered if c in df.columns])

        return feature_cols


if __name__ == "__main__":
    # Test the feature engineering module
    from data_loader import load_and_clean_data

    print("Testing Feature Engineering Module...")

    df = load_and_clean_data('results (2).csv')

    fe = FeatureEngineer()
    df_features = fe.create_all_features(df, n_lags=3, window_days=[3, 7])

    print(f"\nOriginal shape: {df.shape}")
    print(f"With features shape: {df_features.shape}")

    feature_cols = fe.get_feature_columns(df_features)
    print(f"\nTotal features: {len(feature_cols)}")
    print(f"\nFeature columns:")
    for col in feature_cols:
        print(f"  - {col}")
