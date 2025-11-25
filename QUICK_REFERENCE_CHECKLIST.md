# Quick Reference Checklist: Complete the Project

**Status**: 85% → Target: 95%+
**Estimated Time**: 18-20 hours
**Last Updated**: November 24, 2025

---

## Critical Gaps Summary

### What's Missing from Prelim Promises
1. ❌ Medium search incomplete (36% done)
2. ❌ Interaction features not implemented
3. ❌ Targets 2-3 never evaluated (code exists!)
4. ❌ No temporal validation
5. ❌ No SHAP values
6. ❌ No feature ablation
7. ❌ No error analysis

---

## Quick Start: Run These Commands

### 1. Complete Medium Search (2-3 hours, run overnight)
```bash
cd /Users/aanishsachdev/Desktop/CSCI-461-Project
python run_hyperparameter_search.py --mode medium --output experiments_medium
```

### 2. Evaluate Future Targets (1 hour after creating script)
```bash
# First create src/evaluate_future_targets.py from IMPLEMENTATION_GUIDE.md
python src/evaluate_future_targets.py
```

### 3. Temporal Validation (1 hour after creating script)
```bash
# First create src/temporal_validation.py from IMPLEMENTATION_GUIDE.md
python src/temporal_validation.py
```

### 4. SHAP Analysis (2 hours)
```bash
pip install shap
# First create src/shap_analysis.py from IMPLEMENTATION_GUIDE.md
python src/shap_analysis.py
```

### 5. Feature Ablation (2 hours)
```bash
# First create src/feature_ablation.py from IMPLEMENTATION_GUIDE.md
python src/feature_ablation.py
```

### 6. Error Analysis (2 hours)
```bash
# First create src/error_analysis.py from IMPLEMENTATION_GUIDE.md
python src/error_analysis.py
```

### 7. Survival Analysis - OPTIONAL (3 hours)
```bash
pip install lifelines
# First create src/survival_analysis.py from IMPLEMENTATION_GUIDE.md
python src/survival_analysis.py
```

---

## Code Files to Create

All code is in `IMPLEMENTATION_GUIDE.md`. Copy and create these files:

- [ ] `src/evaluate_future_targets.py` - Targets 2-3 evaluation
- [ ] `src/temporal_validation.py` - Temporal split validation
- [ ] `src/shap_analysis.py` - SHAP explainability
- [ ] `src/feature_ablation.py` - Feature ablation study
- [ ] `src/error_analysis.py` - Stratified error analysis
- [ ] `src/survival_analysis.py` - Cox PH and Kaplan-Meier (optional)

---

## Code Files to Modify

### 1. `src/feature_engineering.py`

**Location**: Lines 180-208 in `create_engineered_features()` method

**Add these lines** after existing features:

```python
# NEW INTERACTION FEATURES
if 'mood_encoded' in df.columns and 'timeOfDay_encoded' in df.columns:
    df['mood_x_timeOfDay'] = df['mood_encoded'] * df['timeOfDay_encoded']

if 'trigger_encoded' in df.columns and 'type_encoded' in df.columns:
    df['trigger_x_type'] = df['trigger_encoded'] * df['type_encoded']

if 'mood_encoded' in df.columns and 'prev_intensity_1' in df.columns:
    df['mood_x_prev_intensity'] = df['mood_encoded'] * df['prev_intensity_1']

if 'timeOfDay_encoded' in df.columns and 'hour' in df.columns:
    df['timeOfDay_x_hour'] = df['timeOfDay_encoded'] * df['hour']

if 'type_encoded' in df.columns and 'hour' in df.columns:
    df['type_x_hour'] = df['type_encoded'] * df['hour']

if 'is_weekend' in df.columns and 'hour' in df.columns:
    df['weekend_x_hour'] = df['is_weekend'] * df['hour']
```

**Location**: Lines 358-361 in `get_feature_columns()` method

**Replace** the engineered features section:

```python
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
```

### 2. `src/validation.py` (for error analysis)

**Modify** `create_user_grouped_split()` to optionally return indices:

```python
def create_user_grouped_split(X, y, user_ids, test_size=0.2, random_state=42, return_indices=False):
    # ... existing code ...
    # At the end, before return:

    if return_indices:
        return X_train, X_test, y_train, y_test, train_idx, test_idx
    else:
        return X_train, X_test, y_train, y_test
```

---

## Report Sections to Add

### Add to FINAL_REPORT.md

**After Section 3.3 (Feature Engineering)**:
```markdown
**Interaction Features.** Six interaction features capture...
[Copy from IMPLEMENTATION_GUIDE.md Task 2.4]
```

**After Section 4.4 (create new Section 4.5)**:
```markdown
### 4.5 Temporal Validation
[Copy from IMPLEMENTATION_GUIDE.md Task 4.3]
```

**After Section 5.2 (create new Section 5.3)**:
```markdown
### 5.3 Extended Prediction Targets
[Copy from IMPLEMENTATION_GUIDE.md Task 3.3]
```

**After current Section 5 (create new Section 5.5)**:
```markdown
### 5.5 Extended Hyperparameter Search Results
[Copy from IMPLEMENTATION_GUIDE.md Task 1.3]
```

**In Section 6.2 (add subsections 6.2.2 and 6.2.3)**:
```markdown
### 6.2.2 SHAP Value Analysis
[Copy from IMPLEMENTATION_GUIDE.md Task 5.4]

### 6.2.3 Feature Ablation Study
[Copy from IMPLEMENTATION_GUIDE.md Task 6.3]
```

**After Section 6.3 (create new Section 6.4)**:
```markdown
### 6.4 Systematic Error Analysis
[Copy from IMPLEMENTATION_GUIDE.md Task 7.3]
```

**Optional - After Section 5.3 (create Section 5.4)**:
```markdown
### 5.4 Survival Analysis
[Copy from IMPLEMENTATION_GUIDE.md Task 8.4]
```

---

## Expected New Figures

After running all scripts, you should have:

- ✅ fig27_target2_future_count.png
- ✅ fig28_target3_time_to_event.png
- ✅ fig29_temporal_validation.png
- ✅ fig30_shap_regression_bar.png
- ✅ fig31_shap_regression_beeswarm.png
- ✅ fig32_shap_force_low.png
- ✅ fig32_shap_force_medium.png
- ✅ fig32_shap_force_high.png
- ✅ fig33_shap_classification_bar.png
- ✅ fig34_shap_classification_beeswarm.png
- ✅ fig35_feature_ablation.png
- ✅ fig36_error_analysis.png
- ✅ fig37_cox_hazard_ratios.png (if survival analysis)
- ✅ fig38_kaplan_meier.png (if survival analysis)

**Total**: 12-14 new figures (26 → 38-40 total)

---

## Execution Plan

### Day 1 (4-5 hours)
- [ ] Start medium mode search (overnight)
- [ ] Create `evaluate_future_targets.py` and run
- [ ] Create `shap_analysis.py`, install shap, run
- [ ] Update report with Targets 2-3 and SHAP sections

### Day 2 (4-5 hours)
- [ ] Check medium search completed
- [ ] Create `temporal_validation.py` and run
- [ ] Create `feature_ablation.py` and run
- [ ] Update report with validation and ablation sections

### Day 3 (4-5 hours)
- [ ] Implement interaction features in `feature_engineering.py`
- [ ] Re-run quick experiments with new features
- [ ] Create `error_analysis.py` and run
- [ ] Update report with interaction features and error analysis

### Day 4 (3-4 hours)
- [ ] (Optional) Create `survival_analysis.py`, install lifelines, run
- [ ] Complete all reference citations
- [ ] Final report review and formatting
- [ ] Generate PDF using pandoc

---

## Pre-Flight Checklist

Before starting, verify:

- [ ] Python environment active
- [ ] All required packages installed (sklearn, xgboost, lightgbm, pandas, numpy, matplotlib, seaborn)
- [ ] `results (2).csv` data file present
- [ ] ~15 GB disk space available (for medium search results)
- [ ] 16+ GB RAM (for medium search)

---

## Final Submission Checklist

Before submitting, verify:

- [ ] Medium search: 72/72 experiments complete
- [ ] All new figures generated and saved in `report_figures/`
- [ ] All "[INSERT]" placeholders filled
- [ ] All reference placeholders ([40], [41], etc.) completed
- [ ] No "TODO" comments remaining
- [ ] Figure numbers sequential (1-38+)
- [ ] Table numbers sequential
- [ ] All figures referenced in text
- [ ] PDF renders all figures correctly
- [ ] Table of contents accurate
- [ ] Report length: 60-80 pages
- [ ] Code runs without errors
- [ ] Repository clean (no .DS_Store, __pycache__, etc.)

---

## Troubleshooting

### Medium search crashes
```bash
# Resume with same command - will skip completed experiments
python run_hyperparameter_search.py --mode medium --output experiments_medium
```

### SHAP too slow
- Sample size already set to 500 in script
- Reduce to 200 if still slow: `sample_size = min(200, len(X_test))`

### Out of memory
- Close other applications
- Reduce n_estimators in models temporarily
- Run analyses one at a time, not in parallel

### Figures don't render in PDF
```bash
# Use pandoc with xelatex
pandoc FINAL_REPORT.md -o FINAL_REPORT.pdf --pdf-engine=xelatex --toc -N

# Or use other PDF engine
pandoc FINAL_REPORT.md -o FINAL_REPORT.pdf --pdf-engine=pdflatex --toc -N
```

---

## Time Estimates by Priority

### HIGH Priority (Must Do) - 10-12 hours
1. ✅ Complete medium search (2-3 hrs passive)
2. ✅ Implement interaction features (3 hrs)
3. ✅ Evaluate Targets 2-3 (2 hrs)
4. ✅ SHAP analysis (2 hrs)
5. ✅ Complete references + PDF (1.5 hrs)

### MEDIUM Priority (Should Do) - 6 hours
6. ✅ Temporal validation (2 hrs)
7. ✅ Feature ablation (2 hrs)
8. ✅ Error analysis (2 hrs)

### LOW Priority (Nice to Have) - 3 hours
9. ✅ Survival analysis (3 hrs)

---

## Contact & Resources

- **Full Details**: See `IMPLEMENTATION_GUIDE.md` (complete code and instructions)
- **Current Report**: `FINAL_REPORT.md` (860 lines, 20,216 words)
- **Prelim Report**: `PRELIMINARY_REPORT.md` (reference for promises)
- **Sample Report**: `Fall2022_AUD_ProjectFinalReport.pdf` (quality standard)

---

**Remember**: The goal is 95%+ completion by fulfilling all preliminary report promises. Focus on HIGH priority items first, then MEDIUM, then LOW if time permits.

Good luck! 🚀
