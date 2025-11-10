#!/bin/bash
#
# Master script to run all experiments and generate visualizations.
# This script orchestrates the entire experimental workflow.
#

set -e  # Exit on error

echo "======================================================================"
echo "  Tic Episode Prediction - Complete Experiment Workflow"
echo "======================================================================"
echo ""

# Check if data file exists
DATA_FILE="results (2).csv"
if [ ! -f "$DATA_FILE" ]; then
    echo "ERROR: Data file '$DATA_FILE' not found!"
    echo "Please ensure the data file is in the project directory."
    exit 1
fi

echo "Data file found: $DATA_FILE"
echo ""

# ============================================================================
# PHASE 1: Quick Test (Optional - Skip if already done)
# ============================================================================
read -p "Run quick test first? (y/n, default=n): " run_quick
run_quick=${run_quick:-n}

if [ "$run_quick" = "y" ]; then
    echo ""
    echo "======================================================================"
    echo "  PHASE 1: Quick Mode Test (~30 seconds)"
    echo "======================================================================"
    echo ""
    echo "This will run 4 experiments to verify the framework works..."
    python run_hyperparameter_search.py --mode quick
    echo ""
    echo "Quick test complete! Check experiments/results.csv"
    echo ""
fi

# ============================================================================
# PHASE 2: Medium Hyperparameter Search
# ============================================================================
echo ""
echo "======================================================================"
echo "  PHASE 2: Medium Hyperparameter Search (~1-2 hours)"
echo "======================================================================"
echo ""
echo "This will run ~200-300 experiments with:"
echo "  - Models: Random Forest, XGBoost, LightGBM"
echo "  - Targets: All 3 prediction targets"
echo "  - Feature configs: Multiple combinations"
echo "  - Iterations: 50 per configuration"
echo ""
read -p "Start medium search? (y/n, default=y): " run_medium
run_medium=${run_medium:-y}

if [ "$run_medium" = "y" ]; then
    echo ""
    echo "Starting medium hyperparameter search..."
    echo "This may take 1-2 hours. You can monitor progress in experiments/results.csv"
    echo ""

    python run_hyperparameter_search.py --mode medium

    echo ""
    echo "Medium search complete!"
    echo "Total experiments run: $(wc -l < experiments/results.csv)"
    echo ""
else
    echo "Skipping medium search."
fi

# ============================================================================
# PHASE 3: Generate Analysis Plots
# ============================================================================
echo ""
echo "======================================================================"
echo "  PHASE 3: Generate Analysis Plots (~30 seconds)"
echo "======================================================================"
echo ""
echo "Generating 8 analysis plots from experiment results..."
echo ""

python src/generate_analysis_plots.py

echo ""
echo "Analysis plots complete! Check figures/03_results_analysis/"
echo ""

# ============================================================================
# PHASE 4: Generate Best Model Plots
# ============================================================================
echo ""
echo "======================================================================"
echo "  PHASE 4: Generate Best Model Deep-Dive Plots (~30 seconds)"
echo "======================================================================"
echo ""
echo "Generating detailed plots for top 3 models..."
echo ""

python src/generate_best_model_plots.py

echo ""
echo "Best model plots complete! Check figures/04_best_models/"
echo ""

# ============================================================================
# PHASE 5: Results Summary
# ============================================================================
echo ""
echo "======================================================================"
echo "  Experiment Workflow Complete!"
echo "======================================================================"
echo ""
echo "Summary:"
echo "  - Total experiments: $(wc -l < experiments/results.csv)"
echo "  - Analysis plots: figures/03_results_analysis/ (8 plots)"
echo "  - Best model plots: figures/04_best_models/ (varies)"
echo ""
echo "Next steps:"
echo "  1. Review plots in figures/ directory"
echo "  2. Run baseline notebook to generate EDA plots:"
echo "     jupyter notebook baseline_timeseries_model.ipynb"
echo "  3. Update PreliminaryReport_TicPrediction.md with figure references"
echo "  4. (Optional) Run full search for final optimization:"
echo "     python run_hyperparameter_search.py --mode full"
echo ""
echo "======================================================================"
