#!/bin/bash

# Script to run label propagation with optional validation
# Usage: ./run_labelpropagation.sh [--skip-validation]

SKIP_VALIDATION=false

# Parse arguments
for arg in "$@"
do
    case $arg in
        --skip-validation)
        SKIP_VALIDATION=true
        shift
        ;;
    esac
done

# Run the label propagation burst
PYTHONPATH=. uv run labelpropagation.py \
  --ow-host localhost \
  --ow-port 31001 \
  --lp-endpoint http://minio-service.default:9000 \
  --partitions 8 \
  --num-nodes 10000 \
  --bucket test-bucket \
  --key graphs/cluster-test \
  --granularity 1 \
  --backend redis-list \
  --chunk-size 1024 \
  --max-iterations 10 \
  --convergence-threshold 0 \
  --runtime-memory 2048

LP_EXIT_CODE=$?

# Run validation if not skipped and LP succeeded
if [ "$SKIP_VALIDATION" = false ] && [ $LP_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Running correctness validation..."
    echo "=========================================="
    
    python3 validate_results.py \
        --num-communities 4 \
        --nodes-per-community 100 \
        --num-workers 8 \
        --output validation_report.html
    
    VALIDATION_EXIT_CODE=$?
    
    if [ $VALIDATION_EXIT_CODE -ne 0 ]; then
        echo "ERROR: Validation failed!"
        exit $VALIDATION_EXIT_CODE
    fi
    
    echo "Validation passed successfully"
fi

exit $LP_EXIT_CODE

