#!/bin/bash

#########################################
# MACHINE LEARNING TURNOVER PIPELINE
# Wrapper script for:
#   1. turnover_pipeline_run.R
#   2. turnover_evaluation_suite.R
#
# Usage:
#   ./run_machine_learning_pipeline.sh input.csv [output_folder] [model] [--skip-shap] [--skip-pdp]
#
# model = "default" (original) or "alt" (drops movement vars)
#########################################


# --- ARGUMENT CHECK ---
if [ -z "$1" ]; then
    echo "Error: No input dataset provided."
    echo "Usage: ./run_machine_learning_pipeline.sh input.csv [output_folder] [model] [--skip-shap] [--skip-pdp]"
    exit 1
fi

INPUT_DATASET="$1"
shift

# Optional: output directory
if [ -n "$1" ] && [[ "$1" != "--skip-shap" && "$1" != "--skip-pdp" ]]; then
    OUTPUT_DIR="$1"
    shift
else
    OUTPUT_DIR="turnover_pipeline_output"
fi

# Optional: model type
if [ -n "$1" ] && [[ "$1" != "--skip-shap" && "$1" != "--skip-pdp" ]]; then
    MODEL_TYPE="$1"
    shift
else
    MODEL_TYPE="default"
fi

# Flags for skipping
SKIP_SHAP=false
SKIP_PDP=false

for arg in "$@"; do
    case $arg in
        --skip-shap)
            SKIP_SHAP=true
            ;;
        --skip-pdp)
            SKIP_PDP=true
            ;;
    esac
done

# Validate model type
if [[ "$MODEL_TYPE" != "default" && "$MODEL_TYPE" != "alt" ]]; then
    echo "Error: Model type must be 'default' or 'alt'"
    exit 1
fi

# Create output directory if needed
mkdir -p "$OUTPUT_DIR"


#########################################
# 1. RUN TURNOVER PIPELINE
#########################################

echo "--------------------------------------"
echo "Running turnover pipeline..."
echo "Input dataset : $INPUT_DATASET"
echo "Output folder : $OUTPUT_DIR"
echo "Model type    : $MODEL_TYPE"
echo "--------------------------------------"

Rscript --vanilla turnover_pipeline_run.R \
    --input "$INPUT_DATASET" \
    --output "$OUTPUT_DIR" \
    --model "$MODEL_TYPE"

PIPELINE_STATUS=$?

if [ $PIPELINE_STATUS -ne 0 ]; then
    echo "ERROR: turnover_pipeline_run.R failed! Exiting."
    exit 1
fi


#########################################
# 2. RUN EVALUATION SUITE
#########################################

echo "--------------------------------------"
echo "Generating evaluation plots..."
echo "--------------------------------------"

# Correct filename depending on model type
if [ "$MODEL_TYPE" == "alt" ]; then
    RESULTS_FILE="$OUTPUT_DIR/turnover_results_alt.rds"
else
    RESULTS_FILE="$OUTPUT_DIR/turnover_results.rds"
fi

EVAL_CMD="Rscript --vanilla turnover_evaluation_suite.R \
    --input \"$RESULTS_FILE\" \
    --output \"$OUTPUT_DIR\""

if [ "$SKIP_SHAP" = true ]; then
    EVAL_CMD+=" --skip-shap"
fi

if [ "$SKIP_PDP" = true ]; then
    EVAL_CMD+=" --skip-pdp"
fi

# Run the evaluation suite
eval $EVAL_CMD
EVAL_STATUS=$?

if [ $EVAL_STATUS -ne 0 ]; then
    echo "ERROR: turnover_evaluation_suite.R failed! Exiting."
    exit 1
fi


#########################################
# DONE
#########################################

echo ""
echo "======================================="
echo " TURNOVER PIPELINE SUCCESSFULLY FINISHED"
echo "======================================="
echo "Model      : $MODEL_TYPE"
echo "Output dir : $OUTPUT_DIR"
if [ "$SKIP_SHAP" = true ]; then
    echo "SHAP       : Skipped"
else
    echo "SHAP       : Computed"
fi
if [ "$SKIP_PDP" = true ]; then
    echo "PDP        : Skipped"
else
    echo "PDP        : Computed"
fi
echo ""
