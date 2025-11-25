"""
Detailed error analysis stratified by user engagement, intensity range, tic type, and time of day.
Addresses prelim Section 6.5 and pending todo.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.metrics import mean_absolute_error, f1_score

from data_loader import load_and_clean_data
from feature_engineering import FeatureEngineer
from target_generation import TargetGenerator
from validation import UserGroupedSplit

def stratified_error_analysis(df, predictions_reg, predictions_clf):
    """
    Analyze prediction errors stratified by multiple dimensions.
    """
    print("\n" + "="*80)
    print("STRATIFIED ERROR ANALYSIS")
    print("="*80)

    target_reg = 'target_next_intensity'
    target_clf = 'target_next_high_intensity'

    # Add predictions to dataframe
    df_analysis = df.copy()
    df_analysis['pred_intensity'] = predictions_reg
    df_analysis['pred_high_intensity'] = predictions_clf
    df_analysis['abs_error'] = np.abs(df_analysis[target_reg] - df_analysis['pred_intensity'])
    df_analysis['clf_correct'] = (df_analysis[target_clf] == df_analysis['pred_high_intensity']).astype(int)

    # ----- STRATIFICATION 1: User Engagement -----
    print("\n--- Error by User Engagement Level ---")

    # Calculate episodes per user
    user_episode_counts = df_analysis.groupby('userId').size().to_dict()
    df_analysis['user_episode_count'] = df_analysis['userId'].map(user_episode_counts)

    # Define tiers
    df_analysis['engagement_tier'] = pd.cut(
        df_analysis['user_episode_count'],
        bins=[0, 9, 49, 1000],
        labels=['Sparse (1-9)', 'Medium (10-49)', 'High (50+)']
    )

    engagement_errors = df_analysis.groupby('engagement_tier').agg({
        'abs_error': ['mean', 'std', 'count'],
        'clf_correct': 'mean'
    })

    print(engagement_errors)

    # ----- STRATIFICATION 2: Intensity Range -----
    print("\n--- Error by Intensity Range ---")

    df_analysis['intensity_range'] = pd.cut(
        df_analysis[target_reg],
        bins=[0, 3, 6, 10],
        labels=['Low (1-3)', 'Medium (4-6)', 'High (7-10)']
    )

    intensity_errors = df_analysis.groupby('intensity_range').agg({
        'abs_error': ['mean', 'std', 'count'],
        'clf_correct': 'mean'
    })

    print(intensity_errors)

    # ----- STRATIFICATION 3: Tic Type (Common vs Rare) -----
    print("\n--- Error by Tic Type Frequency ---")

    # Count occurrences of each type
    type_counts = df_analysis['type'].value_counts()
    common_types = type_counts[type_counts >= 20].index.tolist()

    df_analysis['type_category'] = df_analysis['type'].apply(
        lambda x: 'Common (≥20 occurrences)' if x in common_types else 'Rare (<20 occurrences)'
    )

    type_errors = df_analysis.groupby('type_category').agg({
        'abs_error': ['mean', 'std', 'count'],
        'clf_correct': 'mean'
    })

    print(type_errors)

    # ----- STRATIFICATION 4: Time of Day -----
    print("\n--- Error by Time of Day ---")

    time_errors = df_analysis.groupby('timeOfDay').agg({
        'abs_error': ['mean', 'std', 'count'],
        'clf_correct': 'mean'
    })

    print(time_errors)

    # Visualizations
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Plot 1: MAE by engagement tier
    ax1 = fig.add_subplot(gs[0, 0])
    engagement_mae = df_analysis.groupby('engagement_tier')['abs_error'].mean().sort_values()
    engagement_mae.plot(kind='bar', ax=ax1, color='#3498db', edgecolor='black')
    ax1.set_ylabel('Mean Absolute Error', fontsize=11, fontweight='bold')
    ax1.set_xlabel('User Engagement Tier', fontsize=11, fontweight='bold')
    ax1.set_title('Regression Error by Engagement', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

    # Plot 2: Accuracy by engagement tier
    ax2 = fig.add_subplot(gs[0, 1])
    engagement_acc = df_analysis.groupby('engagement_tier')['clf_correct'].mean() * 100
    engagement_acc.plot(kind='bar', ax=ax2, color='#e74c3c', edgecolor='black')
    ax2.set_ylabel('Classification Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('User Engagement Tier', fontsize=11, fontweight='bold')
    ax2.set_title('Classification Accuracy by Engagement', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

    # Plot 3: MAE by intensity range
    ax3 = fig.add_subplot(gs[0, 2])
    intensity_mae = df_analysis.groupby('intensity_range')['abs_error'].mean()
    intensity_mae.plot(kind='bar', ax=ax3, color='#2ecc71', edgecolor='black')
    ax3.set_ylabel('Mean Absolute Error', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Actual Intensity Range', fontsize=11, fontweight='bold')
    ax3.set_title('Regression Error by Intensity', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')

    # Plot 4: Error distribution by engagement
    ax4 = fig.add_subplot(gs[1, :])
    df_analysis.boxplot(column='abs_error', by='engagement_tier', ax=ax4)
    ax4.set_ylabel('Absolute Error', fontsize=11, fontweight='bold')
    ax4.set_xlabel('User Engagement Tier', fontsize=11, fontweight='bold')
    ax4.set_title('Error Distribution by Engagement Tier', fontsize=12, fontweight='bold')
    ax4.get_figure().suptitle('')
    ax4.grid(True, alpha=0.3, axis='y')

    # Plot 5: Error by time of day
    ax5 = fig.add_subplot(gs[2, 0])
    time_order = ['Morning', 'Afternoon', 'Evening', 'Night']
    time_mae = df_analysis.groupby('timeOfDay')['abs_error'].mean().reindex(time_order)
    time_mae.plot(kind='bar', ax=ax5, color='#9b59b6', edgecolor='black')
    ax5.set_ylabel('Mean Absolute Error', fontsize=11, fontweight='bold')
    ax5.set_xlabel('Time of Day', fontsize=11, fontweight='bold')
    ax5.set_title('Regression Error by Time of Day', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45, ha='right')

    # Plot 6: Error by tic type category
    ax6 = fig.add_subplot(gs[2, 1])
    type_mae = df_analysis.groupby('type_category')['abs_error'].mean()
    type_mae.plot(kind='bar', ax=ax6, color='#f39c12', edgecolor='black')
    ax6.set_ylabel('Mean Absolute Error', fontsize=11, fontweight='bold')
    ax6.set_xlabel('Tic Type Category', fontsize=11, fontweight='bold')
    ax6.set_title('Regression Error by Tic Type Frequency', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45, ha='right')

    # Plot 7: Prediction vs actual scatter colored by error magnitude
    ax7 = fig.add_subplot(gs[2, 2])
    scatter = ax7.scatter(df_analysis[target_reg], df_analysis['pred_intensity'],
                         c=df_analysis['abs_error'], cmap='YlOrRd', alpha=0.6, s=20)
    ax7.plot([0, 10], [0, 10], 'k--', lw=2)
    ax7.set_xlabel('Actual Intensity', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Predicted Intensity', fontsize=11, fontweight='bold')
    ax7.set_title('Predictions Colored by Error Magnitude', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax7, label='Absolute Error')

    plt.savefig('report_figures/fig36_error_analysis.png', dpi=300, bbox_inches='tight')
    print("\nSaved: report_figures/fig36_error_analysis.png")

    return {
        'engagement': engagement_errors,
        'intensity': intensity_errors,
        'type': type_errors,
        'time': time_errors
    }

def main():
    print("="*80)
    print("DETAILED ERROR ANALYSIS")
    print("="*80)

    # Load data
    df = load_and_clean_data('results (2).csv')

    # Generate features
    fe = FeatureEngineer()
    df = fe.create_all_features(df, n_lags=3, window_days=[7], fit=True)

    # Generate targets
    tg = TargetGenerator(high_intensity_threshold=7)
    df = tg.create_next_intensity_target(df)

    # Get feature columns
    feature_cols = fe.get_feature_columns(df,
                                         include_sequence=True,
                                         include_time_window=True,
                                         include_engineered=True)

    # Remove NaN
    target_reg = 'target_next_intensity'
    target_clf = 'target_next_high_intensity'
    df_clean = df.dropna(subset=[target_reg, target_clf] + feature_cols)

    # Train-test split using UserGroupedSplit
    splitter = UserGroupedSplit(test_size=0.2, random_state=42)
    df_train, df_test = splitter.split(df_clean)

    X_train = df_train[feature_cols]
    X_test = df_test[feature_cols]
    y_reg_train = df_train[target_reg]
    y_reg_test = df_test[target_reg]
    y_clf_train = df_train[target_clf]
    y_clf_test = df_test[target_clf]

    # Train models
    print("\nTraining models...")
    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf_reg.fit(X_train, y_reg_train)

    xgb_clf = XGBClassifier(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42)
    xgb_clf.fit(X_train, y_clf_train)

    # Make predictions on test set
    pred_reg = rf_reg.predict(X_test)
    pred_clf = xgb_clf.predict(X_test)

    print(f"\nTest set: {len(df_test)} episodes")
    print(f"Regression MAE: {mean_absolute_error(y_reg_test, pred_reg):.4f}")
    print(f"Classification F1: {f1_score(y_clf_test, pred_clf):.4f}")

    # Run error analysis
    error_results = stratified_error_analysis(df_test, pred_reg, pred_clf)

    print("\n" + "="*80)
    print("ERROR ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
