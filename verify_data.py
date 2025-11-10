#!/usr/bin/env python3
"""
Quick data verification script.
Checks actual data statistics vs. what's reported in PreliminaryReport_TicPrediction.md
"""

try:
    import pandas as pd
    import numpy as np
    from datetime import datetime

    print("="*80)
    print("DATA VERIFICATION REPORT")
    print("="*80)

    # Load data
    df = pd.read_csv('results (2).csv')

    print(f"\n{'ACTUAL DATA STATISTICS':-^80}")

    # Basic counts
    total_records = len(df)
    unique_users = df['userId'].nunique()

    print(f"\n1. DATASET SIZE:")
    print(f"   Total tic episodes: {total_records}")
    print(f"   Unique users: {unique_users}")

    # Date range
    df['date'] = pd.to_datetime(df['date'])
    min_date = df['date'].min()
    max_date = df['date'].max()
    days_spanned = (max_date - min_date).days + 1
    unique_dates = df['date'].nunique()

    print(f"\n2. TIME SPAN:")
    print(f"   Start date: {min_date.date()}")
    print(f"   End date: {max_date.date()}")
    print(f"   Days spanned: {days_spanned} days")
    print(f"   Unique logging days: {unique_dates} days")

    # Intensity statistics
    intensity_mean = df['intensity'].mean()
    intensity_std = df['intensity'].std()
    intensity_median = df['intensity'].median()
    intensity_min = df['intensity'].min()
    intensity_max = df['intensity'].max()

    print(f"\n3. INTENSITY STATISTICS:")
    print(f"   Mean: {intensity_mean:.2f}")
    print(f"   Std Dev: {intensity_std:.2f}")
    print(f"   Median: {intensity_median:.1f}")
    print(f"   Range: {intensity_min}-{intensity_max}")

    # High intensity
    high_intensity_7 = len(df[df['intensity'] >= 7])
    high_intensity_pct = (high_intensity_7 / total_records) * 100

    print(f"\n4. HIGH-INTENSITY EPISODES (≥7):")
    print(f"   Count: {high_intensity_7}")
    print(f"   Percentage: {high_intensity_pct:.1f}%")

    # User engagement
    user_counts = df.groupby('userId').size()
    high_users = len(user_counts[user_counts > 50])
    medium_users = len(user_counts[(user_counts >= 10) & (user_counts <= 50)])
    minimal_users = len(user_counts[user_counts < 10])

    print(f"\n5. USER ENGAGEMENT:")
    print(f"   High (>50 events): {high_users} users ({high_users/unique_users*100:.1f}%)")
    print(f"   Medium (10-50 events): {medium_users} users ({medium_users/unique_users*100:.1f}%)")
    print(f"   Minimal (<10 events): {minimal_users} users ({minimal_users/unique_users*100:.1f}%)")

    # Tic types
    unique_types = df['type'].nunique()
    top_types = df['type'].value_counts().head(10)

    print(f"\n6. TIC TYPES:")
    print(f"   Unique types: {unique_types}")
    print(f"   Top 10 types:")
    for i, (tic_type, count) in enumerate(top_types.items(), 1):
        pct = (count / total_records) * 100
        print(f"      {i}. {tic_type}: {count} ({pct:.1f}%)")

    # Missing data
    print(f"\n7. MISSING DATA:")
    for col in ['mood', 'trigger', 'description']:
        if col in df.columns:
            missing_count = df[col].isna().sum() + (df[col] == 'null').sum()
            missing_pct = (missing_count / total_records) * 100
            coverage_pct = 100 - missing_pct
            print(f"   {col.capitalize()}: {missing_count} missing ({missing_pct:.1f}% missing, {coverage_pct:.1f}% coverage)")

    # Time of day
    if 'timeOfDay' in df.columns:
        print(f"\n8. TIME OF DAY DISTRIBUTION:")
        time_dist = df['timeOfDay'].value_counts()
        for time_cat, count in time_dist.items():
            pct = (count / total_records) * 100
            print(f"   {time_cat}: {count} ({pct:.1f}%)")

    # Day of week
    df['day_of_week'] = df['date'].dt.day_name()
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    weekends = ['Saturday', 'Sunday']

    weekday_count = df[df['day_of_week'].isin(weekdays)].shape[0]
    weekend_count = df[df['day_of_week'].isin(weekends)].shape[0]
    weekday_pct = (weekday_count / total_records) * 100
    weekend_pct = (weekend_count / total_records) * 100

    print(f"\n9. DAY OF WEEK PATTERNS:")
    print(f"   Weekday: {weekday_count} ({weekday_pct:.1f}%)")
    print(f"   Weekend: {weekend_count} ({weekend_pct:.1f}%)")

    # Age if available
    if 'age' in df.columns:
        print(f"\n10. AGE STATISTICS:")
        print(f"   Range: {df['age'].min()}-{df['age'].max()}")
        print(f"   Mean: {df['age'].mean():.1f}")

    print("\n" + "="*80)
    print("COMPARISON WITH PRELIMINARY REPORT")
    print("="*80)

    # Compare with report values
    report_stats = {
        'Total episodes': (1367, total_records),
        'Unique users': (70, unique_users),
        'Days spanned': (103, days_spanned),
        'Mean intensity': (4.5, intensity_mean),
        'Std Dev': (2.8, intensity_std),
        'High-intensity %': (22.0, high_intensity_pct),
    }

    print("\n   Metric                Report Value    Actual Value    Match?")
    print("   " + "-"*70)
    for metric, (report_val, actual_val) in report_stats.items():
        if isinstance(report_val, int):
            match = "✓" if report_val == actual_val else "✗"
            print(f"   {metric:25} {report_val:15} {actual_val:15} {match:>7}")
        else:
            # For floats, allow small difference
            match = "✓" if abs(report_val - actual_val) < 0.2 else "✗"
            print(f"   {metric:25} {report_val:15.2f} {actual_val:15.2f} {match:>7}")

    print("\n" + "="*80)

    # Check if major discrepancies
    if total_records != 1367 or unique_users != 70:
        print("\n⚠️  WARNING: Significant differences found!")
        print("   The Preliminary Report needs to be updated with correct statistics.")
        print(f"\n   Update these values in PreliminaryReport_TicPrediction.md:")
        print(f"   - Total episodes: 1,367 → {total_records:,}")
        print(f"   - Unique users: 70 → {unique_users}")
        print(f"   - Days spanned: 103 → {days_spanned}")
    else:
        print("\n✓ All major statistics match the preliminary report!")

    print("\n" + "="*80)

except ImportError:
    print("\nERROR: pandas not installed")
    print("Please run: pip install pandas numpy")
    print("\nOr use conda/virtualenv with required dependencies")
except FileNotFoundError:
    print("\nERROR: Data file 'results (2).csv' not found")
    print("Please ensure the data file is in the current directory")
except Exception as e:
    print(f"\nERROR: {str(e)}")
    import traceback
    traceback.print_exc()
