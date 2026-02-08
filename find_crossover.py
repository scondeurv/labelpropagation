#!/usr/bin/env python3
"""
Script to find the crossover point where Burst becomes faster than standalone.
Tests increasing graph sizes until speedup > 1.0
"""

import subprocess
import sys
import time
import os
from typing import Tuple, Optional

def run_benchmark(nodes: int, partitions: int, granularity: int, memory: int) -> Optional[Tuple[float, float, float]]:
    """
    Run benchmark for given configuration.
    Returns (standalone_time, burst_time, speedup) or None if failed.
    """
    print(f"\n{'='*80}")
    print(f"Testing: {nodes:,} nodes, {partitions} partitions, {granularity} granularity, {memory}MB memory")
    print(f"{'='*80}\n")
    
    # Generate graph data
    print(f"Generating graph with {nodes:,} nodes...")
    gen_cmd = [
        sys.executable, "setup_large_lp_data.py",
        "--nodes", str(nodes),
        "--partitions", str(partitions),
        "--endpoint", "localhost:9000",
        "--density", "20"
    ]
    
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    result = subprocess.run(gen_cmd, env=env, capture_output=True, text=True, shell=False)
    if result.returncode != 0:
        print(f"Failed to generate graph: {result.stderr}")
        return None
    
    # Run benchmark
    print(f"Running benchmark...")
    bench_cmd = [
        sys.executable, "benchmark_lp.py",
        "--nodes", str(nodes),
        "--partitions", str(partitions),
        "--granularity", str(granularity),
        "--iter", "10",
        "--memory", str(memory),
        "--ow-host", "localhost",
        "--ow-port", "31001"
    ]
    
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    result = subprocess.run(bench_cmd, env=env, capture_output=True, text=True, shell=False)
    if result.returncode != 0:
        print(f"Benchmark failed: {result.stderr}")
        return None
    
    # Parse output
    output = result.stdout
    standalone_time = None
    burst_time = None
    speedup = None
    
    for line in output.split('\n'):
        if "Standalone Time:" in line:
            try:
                standalone_time = float(line.split(':')[1].strip().split()[0])
            except:
                pass
        elif "Burst Time:" in line:
            try:
                burst_time = float(line.split(':')[1].strip())
            except:
                pass
        elif "Speedup:" in line:
            try:
                speedup_str = line.split(':')[1].strip().replace('x', '')
                speedup = float(speedup_str)
            except:
                pass
    
    if standalone_time and burst_time and speedup is not None:
        print(f"\n✓ Results:")
        print(f"  Standalone: {standalone_time:.0f} ms")
        print(f"  Burst:      {burst_time:.0f} ms")
        print(f"  Speedup:    {speedup:.2f}x")
        return (standalone_time, burst_time, speedup)
    
    return None


def find_crossover():
    """
    Find the crossover point using binary search approach.
    """
    print("="*80)
    print("FINDING CROSSOVER POINT: Burst vs Standalone")
    print("="*80)
    
    # Configuration
    partitions = 4
    granularity = 1
    memory = 1024
    
    # Test sizes (nodes)
    test_sizes = [
        100_000,    # Already tested: 0.01x
        200_000,    # 2x
        500_000,    # 5x
        1_000_000,  # 10x
        2_000_000,  # 20x
        5_000_000,  # 50x
    ]
    
    results = []
    
    for nodes in test_sizes:
        result = run_benchmark(nodes, partitions, granularity, memory)
        
        if result is None:
            print(f"\n⚠ Failed at {nodes:,} nodes - stopping search")
            break
        
        standalone_time, burst_time, speedup = result
        results.append({
            'nodes': nodes,
            'standalone_ms': standalone_time,
            'burst_ms': burst_time,
            'speedup': speedup
        })
        
        # Check if we found crossover
        if speedup >= 1.0:
            print(f"\n{'='*80}")
            print(f"🎯 CROSSOVER FOUND!")
            print(f"{'='*80}")
            print(f"Burst becomes faster at approximately {nodes:,} nodes")
            print(f"Speedup: {speedup:.2f}x")
            
            # If we have previous result, interpolate
            if len(results) > 1:
                prev = results[-2]
                print(f"\nCrossover is between {prev['nodes']:,} and {nodes:,} nodes")
                print(f"  {prev['nodes']:,} nodes: {prev['speedup']:.2f}x")
                print(f"  {nodes:,} nodes: {speedup:.2f}x")
            
            break
        
        # If speedup is getting worse, stop
        if len(results) >= 2 and speedup < results[-2]['speedup']:
            print(f"\n⚠ Speedup decreasing - may have resource limits")
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"{'Nodes':>12} | {'Standalone (ms)':>16} | {'Burst (ms)':>12} | {'Speedup':>10} | {'Winner':>10}")
    print("-" * 80)
    
    for r in results:
        winner = "✓ Burst" if r['speedup'] >= 1.0 else "Standalone"
        print(f"{r['nodes']:>12,} | {r['standalone_ms']:>16,.0f} | {r['burst_ms']:>12,.0f} | {r['speedup']:>9.2f}x | {winner:>10}")
    
    # Final recommendation
    if results:
        max_speedup = max(r['speedup'] for r in results)
        if max_speedup >= 1.0:
            best = max(results, key=lambda r: r['speedup'])
            print(f"\n✓ Best configuration found: {best['nodes']:,} nodes with {best['speedup']:.2f}x speedup")
        else:
            print(f"\n✗ Burst never faster than standalone in tested range")
            print(f"   Maximum speedup achieved: {max_speedup:.2f}x at {max(results, key=lambda r: r['speedup'])['nodes']:,} nodes")
            print(f"   Try larger graphs or more workers for better parallelism")


if __name__ == "__main__":
    try:
        find_crossover()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
