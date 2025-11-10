#!/usr/bin/env python3
"""
Generate feature correlation heatmap for preliminary report.
This exploratory analysis helps understand feature relationships and multicollinearity.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')
from data_loader import load_and_clean_data, filter_users_by_min_episodes
from feature_engineering import FeatureEngineer
from target_generation import TargetGenerator

# Set style
sns.set_style('white')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 9


def generate_feature_correlation_heatmap(
    data_path='results (2).csv',
    output_path='report_figures/fig0_feature_correlation.png'
):
    """
    Generate feature correlation heatmap.

    Parameters
    ----------
    data_path : str
        Path to data CSV
    output_path : str
        Path to save figure
    """
    print("=" * 70)
    print("GENERATING FEATURE CORRELATION HEATMAP")
    print("=" * 70)

    # Load and prepare data
    print("\n1. Loading data...")
    df_raw = load_and_clean_data(data_path)
    df_raw = filter_users_by_min_episodes(df_raw, min_episodes=4)
    print(f"   Loaded {len(df_raw)} episodes from {df_raw['userId'].nunique()} users")

    # Create features
    print("\n2. Generating features...")
    feature_engineer = FeatureEngineer()
    df = feature_engineer.create_all_features(
        df_raw,
        n_lags=3,
        window_days=[7],
        fit=True
    )

    # Get feature columns (using 'all' feature set)
    feature_cols = feature_engineer.get_feature_columns(
        df,
        include_sequence=True,
        include_time_window=True,
        include_engineered=True
    )

    print(f"   Generated {len(feature_cols)} features")

    # Extract feature data (remove rows with missing values)
    print("\n3. Computing correlation matrix...")
    df_features = df[feature_cols].copy()

    # Select only numeric columns for correlation
    numeric_features = df_features.select_dtypes(include=[np.number]).columns.tolist()
    df_numeric = df_features[numeric_features].dropna()

    print(f"   Using {len(numeric_features)} numeric features")
    print(f"   Using {len(df_numeric)} complete samples")

    # Compute correlation matrix
    corr_matrix = df_numeric.corr()

    # Identify top positive and negative correlations
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_pairs.append({
                'feature1': corr_matrix.columns[i],
                'feature2': corr_matrix.columns[j],
                'correlation': corr_matrix.iloc[i, j]
            })

    corr_df = pd.DataFrame(corr_pairs).sort_values('correlation', ascending=False)

    print(f"\n   Top 3 Positive Correlations:")
    for idx, row in corr_df.head(3).iterrows():
        print(f"   - {row['feature1']} <-> {row['feature2']}: {row['correlation']:.3f}")

    print(f"\n   Top 3 Negative Correlations:")
    for idx, row in corr_df.tail(3).iterrows():
        print(f"   - {row['feature1']} <-> {row['feature2']}: {row['correlation']:.3f}")

    # Plot heatmap
    print("\n4. Generating heatmap...")
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create heatmap
    sns.heatmap(
        corr_matrix,
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8},
        ax=ax,
        annot=False  # Too many features for annotations
    )

    ax.set_title('Feature Correlation Matrix\n(34 Features, All Feature Set)',
                 fontsize=14, fontweight='bold', pad=20)

    # Rotate labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right', fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)

    plt.tight_layout()

    # Save figure
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close()

    print("\n" + "=" * 70)
    print("CORRELATION HEATMAP GENERATION COMPLETE")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate feature correlation heatmap for report'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='results (2).csv',
        help='Path to data CSV file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='report_figures/fig0_feature_correlation.png',
        help='Output path for heatmap'
    )

    args = parser.parse_args()

    generate_feature_correlation_heatmap(args.data, args.output)
