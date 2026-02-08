#!/bin/bash
# Script to find crossover point between Burst and Standalone

echo "================================================================================"
echo "FINDING CROSSOVER POINT: Burst vs Standalone"
echo "================================================================================"
echo ""

# Configuration
PARTITIONS=4
GRANULARITY=1
MEMORY=1024
ITERATIONS=10

# Test sizes
SIZES=(100000 200000 500000 1000000 2000000)

echo "Configuration:"
echo "  Partitions: $PARTITIONS"
echo "  Granularity: $GRANULARITY"
echo "  Memory: ${MEMORY}MB"
echo "  Iterations: $ITERATIONS"
echo ""

# Results file
RESULTS_FILE="crossover_results.csv"
echo "nodes,standalone_ms,burst_ms,speedup" > $RESULTS_FILE

for NODES in "${SIZES[@]}"; do
    echo "================================================================================"
    echo "Testing: $(printf "%'d" $NODES) nodes"
    echo "================================================================================"
    echo ""
    
    # Generate graph
    echo "Generating graph..."
    PYTHONPATH=. python setup_large_lp_data.py \
        --nodes $NODES \
        --partitions $PARTITIONS \
        --endpoint localhost:9000 \
        --density 20
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to generate graph for $NODES nodes"
        break
    fi
    
    # Run benchmark
    echo ""
    echo "Running benchmark..."
    OUTPUT=$(PYTHONPATH=. python benchmark_lp.py \
        --nodes $NODES \
        --partitions $PARTITIONS \
        --granularity $GRANULARITY \
        --iter $ITERATIONS \
        --memory $MEMORY \
        --ow-host localhost \
        --ow-port 31001 2>&1)
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Benchmark failed for $NODES nodes"
        echo "$OUTPUT"
        break
    fi
    
    # Parse results
    STANDALONE=$(echo "$OUTPUT" | grep "Standalone Time:" | awk '{print $3}')
    BURST=$(echo "$OUTPUT" | grep "Burst Time:" | awk '{print $3}')
    SPEEDUP=$(echo "$OUTPUT" | grep "Speedup:" | awk '{print $2}' | tr -d 'x')
    
    # Save to CSV
    echo "$NODES,$STANDALONE,$BURST,$SPEEDUP" >> $RESULTS_FILE
    
    echo ""
    echo "Results:"
    echo "  Standalone: $STANDALONE ms"
    echo "  Burst:      $BURST ms"
    echo "  Speedup:    ${SPEEDUP}x"
    echo ""
    
    # Check if we found crossover
    if [ $(echo "$SPEEDUP >= 1.0" | bc -l) -eq 1 ]; then
        echo "🎯 CROSSOVER FOUND! Burst is faster at $NODES nodes (${SPEEDUP}x)"
        break
    fi
    
    # Check if speedup is getting worse (resource limits)
    if [ $(echo "$SPEEDUP < 0.01" | bc -l) -eq 1 ]; then
        echo "⚠ Speedup very low - may be hitting resource limits"
        echo "Stopping search"
        break
    fi
done

echo ""
echo "================================================================================"
echo "FINAL RESULTS"
echo "================================================================================"
echo ""
cat $RESULTS_FILE | column -t -s ','
echo ""
echo "Results saved to: $RESULTS_FILE"
