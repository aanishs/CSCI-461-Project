# Data Verification Results

## Summary

**⚠️ DISCREPANCIES FOUND between reported and actual data statistics**

---

## Actual Data (from baseline_timeseries_model.ipynb)

| Metric | Actual Value |
|--------|--------------|
| **Total tic episodes** | **1,533** |
| **Unique users** | **89** |
| **Usable sequences** | 1,316 (after filtering ≥4 events) |
| **Unique tic types** | 82 |
| **Train/test split** | 80/20 by user |
| **Date range** | April 26, 2025 - October 25, 2025 (~182 days) |

---

## Currently Reported (in Plan.md and PreliminaryReport)

| Metric | Reported Value | Status |
|--------|----------------|--------|
| **Total tic episodes** | **1,367** | ❌ INCORRECT |
| **Unique users** | **70** | ❌ INCORRECT |
| **Days spanned** | **103** | ❌ INCORRECT (actual ~182 days) |
| **Unique tic types** | 77 | ❌ INCORRECT (actual 82) |

---

## Required Updates

### Files to Update:

1. **Plan.md** - Line 70:
   - ❌ OLD: "Size: 1,367 tic occurrence records from 70 unique users across 103 days"
   - ✅ NEW: "Size: 1,533 tic occurrence records from 89 unique users across 182 days"

2. **PreliminaryReport_TicPrediction.md** - Multiple locations:

   **Abstract (line ~10):**
   - ❌ OLD: "longitudinal dataset of 1,367 tic occurrence records from 70 unique users collected over 103 days"
   - ✅ NEW: "longitudinal dataset of 1,533 tic occurrence records from 89 unique users collected over 182 days"

   **Section 3.1 Data Overview (line ~360):**
   - ❌ OLD: "Total tic episodes: 1,367 recorded events"
   - ✅ NEW: "Total tic episodes: 1,533 recorded events"
   - ❌ OLD: "Unique users: 70 individuals"
   - ✅ NEW: "Unique users: 89 individuals"
   - ❌ OLD: "Time span: 103 days"
   - ✅ NEW: "Time span: 182 days (April 26 - October 25, 2025)"

   **Section 3.3.2 User Engagement (line ~420):**
   - Percentages will change slightly with 89 users vs 70 users

   **Section 3.4 Pre-processing (line ~540):**
   - ❌ OLD: "1,316 sequence instances (after filtering)"
   - ✅ This is actually CORRECT!

   **Appendix B.4 Tic Type Distribution:**
   - Update to show actual top 15 from 82 unique types (not 77)

---

## Correct Statistics to Use

### Dataset Size:
- **Total records**: 1,533 tic events
- **Unique users**: 89
- **Date range**: April 26 - October 25, 2025 (182 days)
- **Usable for modeling**: 1,316 sequences (after filtering for users with ≥4 events)

### User Engagement (89 total users):
- Need to recalculate percentages:
  - High (>50 events): ? users (?%)
  - Medium (10-50 events): ? users (?%)
  - Minimal (<10 events): ? users (?%)

### Performance Metrics (from baseline model - VERIFIED):
- Test MAE: 1.778
- Test RMSE: 2.194
- F1-score (classification): 0.265
- These are CORRECT ✓

---

## Verification Status

✅ **Baseline model README** - Has correct statistics (1,533 events, 89 users)
❌ **Plan.md** - Needs update (says 1,367 events, 70 users)
❌ **PreliminaryReport_TicPrediction.md** - Needs update (says 1,367 events, 70 users)
✅ **Baseline performance metrics** - All correct
✅ **Feature counts** - Correct (17 features in baseline, 34 in hyperparameter search)

---

## Action Required

**Before running experiments, update:**

1. Plan.md (1 location)
2. PreliminaryReport_TicPrediction.md (5-6 locations)

**Then verify** by searching for:
- "1,367" → should be "1,533"
- "70 unique users" → should be "89 unique users"
- "103 days" → should be "182 days"
- "77 unique types" → should be "82 unique types"

---

## Note

The discrepancy likely occurred because:
1. Initial project planning used preliminary data
2. More users joined the mobile app between planning and execution
3. Data collection period extended from 103 days to 182 days
4. More tic types were recorded as dataset grew

This is NORMAL and GOOD - you have MORE data than expected, which will improve model performance!

---

**Status**: Ready to update files with correct statistics.
