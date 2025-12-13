#!/bin/bash
# Quick Start Script for Flag Pattern Classification Pipeline
# This script runs the complete ML pipeline from data preprocessing to evaluation

set -e  # Exit on error

echo "========================================================================"
echo "FLAG PATTERN CLASSIFICATION - COMPLETE PIPELINE"
echo "========================================================================"
echo ""

# Check if we're in the right directory
if [ ! -d "src" ] || [ ! -f "src/config.py" ]; then
    echo "ERROR: Please run this script from the project root directory"
    exit 1
fi

# Change to src directory
cd src

# Create log directory
mkdir -p ../log

# Clear previous log
> ../log/run.log

echo "Step 1/4: Data Preprocessing"
echo "------------------------------------------------------------------------"
echo "Extracting segments from Label Studio JSON..."
echo "Engineering features and creating train/test split..."
echo ""
python -u 01-data-preprocessing.py 2>&1 | tee -a ../log/run.log
echo ""
echo "✓ Preprocessing complete"
echo "  Output: data/export/segments_preproc_24.csv (train)"
echo "          data/export/segments_preproc_24_test.csv (test)"
echo ""

echo "Step 2/4: Model Training"
echo "------------------------------------------------------------------------"
echo "Training CNN model with PyTorch Lightning..."
echo "This may take 10-30 minutes depending on data size and hardware..."
echo ""
python -u train_model.py 2>&1 | tee -a ../log/run.log
echo ""
echo "✓ Training complete"
echo "  Output: Best model checkpoint saved"
echo "          Training metadata saved"
echo ""

echo "Step 3/4: Model Evaluation"
echo "------------------------------------------------------------------------"
echo "Evaluating on test set and comparing with baseline..."
echo "Generating visualizations..."
echo ""
python -u 03-evaluation.py 2>&1 | tee -a ../log/run.log
echo ""
echo "✓ Evaluation complete"
echo "  Output: 03_evaluation/evaluation_metrics.txt"
echo "          03_evaluation/figures/*.png"
echo ""

echo "Step 4/4: Inference Example"
echo "------------------------------------------------------------------------"
echo "Running inference on test set as demonstration..."
echo ""
python -u 04-inference.py \
    --input ../data/export/segments_preproc_24_test.csv \
    --output ../test_predictions.csv 2>&1 | tee -a ../log/run.log
echo ""
echo "✓ Inference complete"
echo "  Output: test_predictions.csv"
echo ""

echo "========================================================================"
echo "PIPELINE COMPLETED SUCCESSFULLY"
echo "========================================================================"
echo ""
echo "Results Summary:"
echo "  • Training logs: log/run.log"
echo "  • Best model: data/export/checkpoints_v2/best_model_*.ckpt"
echo "  • Evaluation metrics: src/03_evaluation/evaluation_metrics.txt"
echo "  • Visualizations: src/03_evaluation/figures/"
echo "  • Example predictions: test_predictions.csv"
echo ""
echo "To view evaluation results:"
echo "  cat src/03_evaluation/evaluation_metrics.txt"
echo ""
echo "To view logs:"
echo "  tail -n 100 log/run.log"
echo ""
echo "To view logs:"
echo "  tail -n 100 log/run.log"
echo ""
echo "For more information, see:"
echo "  • src/README.md - Module documentation"
echo "  • TRANSFORMATION_SUMMARY.md - Detailed transformation notes"
echo ""
