#!/bin/bash
# Run experiment with full logging

# Create timestamp for log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="experiment_output_${TIMESTAMP}.log"

echo "Starting experiment at $(date)"
echo "Logging to: $LOG_FILE"
echo "Program logs also saved to: experiment_results/experiment.log"
echo "========================================"

# Run the experiment and capture all output
python run_experiment.py "$@" 2>&1 | tee "$LOG_FILE"

echo "========================================"
echo "Experiment finished at $(date)"
echo "Output saved to: $LOG_FILE"

