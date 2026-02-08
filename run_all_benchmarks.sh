#!/bin/bash
# run_all_benchmarks.sh

NODES_LIST=(1000000 2000000 5000000 8000000 10000000 12500000 15000000)
LOG_FILE="final_benchmark_results.log"
CSV_FILE="final_benchmark_results.csv"

echo "Nodes,Standalone_ms,Burst_ms,Speedup" > $CSV_FILE
echo "Starting final benchmarks..." > $LOG_FILE

for NODES in "${NODES_LIST[@]}"
do
    echo "------------------------------------------" | tee -a $LOG_FILE
    echo "Benchmarking $NODES nodes..." | tee -a $LOG_FILE
    
    # Flush Redis to ensure clean state
    kubectl exec pod/dragonfly -- redis-cli FLUSHALL > /dev/null 2>&1
    
    # Run benchmark
    # Use 4 partitions as we did before for large graphs
    # Iter 10 to match previous runs
    # Memory 2048 for large ones
    OUTPUT=$(uv run python benchmark_lp.py --nodes $NODES --partitions 4 --granularity 1 --iter 10 --memory 2048)
    
    echo "$OUTPUT" >> $LOG_FILE
    
    # Extract times
    LPST_TIME=$(echo "$OUTPUT" | grep "LPST Time:" | awk '{print $3}')
    BURST_TIME=$(echo "$OUTPUT" | grep "Burst Time:" | awk '{print $3}')
    SPEEDUP=$(echo "$OUTPUT" | grep "Speedup:" | awk '{print $2}' | sed 's/x//')
    
    if [ -z "$LPST_TIME" ]; then LPST_TIME="0"; fi
    if [ -z "$BURST_TIME" ]; then BURST_TIME="0"; fi
    if [ -z "$SPEEDUP" ]; then SPEEDUP="0"; fi
    
    echo "$NODES,$LPST_TIME,$BURST_TIME,$SPEEDUP" >> $CSV_FILE
    
    echo "Done: LPST=$LPST_TIME ms, Burst=$BURST_TIME ms, Speedup=${SPEEDUP}x" | tee -a $LOG_FILE
done

echo "Benchmarks completed. Results in $CSV_FILE"
