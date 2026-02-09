#!/usr/bin/env python3
"""
Validate crossover point with targeted benchmarks around 4.37M nodes
"""
import subprocess
import json
import sys
import time
from datetime import datetime

# Strategic test points around estimated crossover (4.37M)
TEST_POINTS = [
    3000000,   # 3M - Before crossover (Standalone should win)
    4000000,   # 4M - Near crossover
    4500000,   # 4.5M - Very close to crossover
    5000000,   # 5M - Just after crossover (Burst should win)
    6000000,   # 6M - Confirmation (Burst clearly faster)
]

PARTITIONS = 4
MAX_ITER = 10
MEMORY = 4096
S3_ENDPOINT = "http://minio-service.default.svc.cluster.local:9000"
BUCKET = "test-bucket"

def log(message):
    """Print with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def generate_graph(nodes):
    """Generate graph data if needed"""
    log(f"Generating {nodes/1e6:.1f}M node graph...")
    
    result = subprocess.run([
        ".venv/bin/python", "setup_large_lp_data.py",
        "--nodes", str(nodes),
        "--partitions", str(PARTITIONS),
        "--bucket", BUCKET,
        "--endpoint", "http://localhost:9000",  # From host
        "--density", "20"
    ], capture_output=True, text=True, timeout=600)
    
    if result.returncode != 0:
        log(f"❌ Failed to generate graph: {result.stderr}")
        return False
    
    log(f"✅ Graph generated successfully")
    return True

def run_benchmark(nodes):
    """Run benchmark for given size"""
    log(f"\n{'='*80}")
    log(f"BENCHMARKING: {nodes/1e6:.1f}M nodes")
    log(f"{'='*80}")
    
    # Generate graph if needed
    if not generate_graph(nodes):
        return None
    
    # Run benchmark (both modes by default)
    log(f"Running benchmark (Standalone + Burst)...")
    
    result = subprocess.run([
        ".venv/bin/python", "benchmark_lp.py",
        "--nodes", str(nodes),
        "--partitions", str(PARTITIONS),
        "--iter", str(MAX_ITER),
        "--memory", str(MEMORY),
        "--s3-endpoint", S3_ENDPOINT,
        "--bucket", BUCKET
    ], capture_output=True, text=True, timeout=1200)
    
    if result.returncode != 0:
        log(f"❌ Benchmark failed: {result.stderr}")
        return None
    
    # Parse output to extract times
    output = result.stdout
    print(output)  # Show full output
    
    standalone_time = None
    burst_time = None
    
    for line in output.split('\n'):
        if 'Standalone Time:' in line:
            try:
                standalone_time = float(line.split(':')[1].strip().split()[0])
            except:
                pass
        if 'Burst Time:' in line:
            try:
                burst_time = float(line.split(':')[1].strip().split()[0])
            except:
                pass
    
    if standalone_time and burst_time:
        speedup = standalone_time / burst_time
        log(f"\n📊 Results for {nodes/1e6:.1f}M nodes:")
        log(f"   Standalone: {standalone_time:.2f} ms")
        log(f"   Burst:      {burst_time:.2f} ms")
        log(f"   Speedup:    {speedup:.2f}x")
        
        winner = "Burst" if speedup > 1.0 else "Standalone"
        log(f"   Winner:     {winner} ✅")
        
        return {
            'nodes': nodes,
            'standalone_ms': standalone_time,
            'burst_ms': burst_time,
            'speedup': speedup,
            'winner': winner
        }
    
    log(f"⚠️  Could not parse results")
    return None

def main():
    """Run crossover validation"""
    log("="*80)
    log("CROSSOVER VALIDATION BENCHMARK")
    log("="*80)
    log(f"Testing {len(TEST_POINTS)} points around estimated crossover (4.37M)")
    log(f"Test points: {[f'{n/1e6:.1f}M' for n in TEST_POINTS]}")
    log("="*80)
    
    results = []
    
    for nodes in TEST_POINTS:
        result = run_benchmark(nodes)
        if result:
            results.append(result)
        else:
            log(f"⚠️  Skipping {nodes/1e6:.1f}M due to errors")
        
        # Brief pause between benchmarks
        time.sleep(5)
    
    # Summary
    log("\n" + "="*80)
    log("CROSSOVER VALIDATION SUMMARY")
    log("="*80)
    log(f"{'Nodes':>12} {'Standalone':>12} {'Burst':>12} {'Speedup':>10} {'Winner':>12}")
    log("-"*80)
    
    crossover_found = False
    crossover_point = None
    
    for i, r in enumerate(results):
        log(f"{r['nodes']/1e6:>10.1f}M {r['standalone_ms']:>10.0f}ms {r['burst_ms']:>10.0f}ms {r['speedup']:>9.2f}x {r['winner']:>12}")
        
        # Detect crossover
        if i > 0 and not crossover_found:
            prev = results[i-1]
            if prev['speedup'] < 1.0 and r['speedup'] >= 1.0:
                crossover_found = True
                # Linear interpolation
                m = (r['speedup'] - prev['speedup']) / (r['nodes'] - prev['nodes'])
                b = prev['speedup'] - m * prev['nodes']
                crossover_point = (1.0 - b) / m
                log("-"*80)
                log(f"📍 CROSSOVER DETECTED between {prev['nodes']/1e6:.1f}M and {r['nodes']/1e6:.1f}M")
                log(f"📍 Refined estimate: {crossover_point/1e6:.2f}M nodes")
                log("-"*80)
    
    log("="*80)
    
    # Save results
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'test_points': TEST_POINTS,
        'results': results,
        'crossover_estimate': crossover_point,
        'configuration': {
            'partitions': PARTITIONS,
            'max_iter': MAX_ITER,
            'memory_mb': MEMORY
        }
    }
    
    with open('crossover_validation_results.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    log(f"💾 Results saved: crossover_validation_results.json")
    log("✅ CROSSOVER VALIDATION COMPLETE")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        log(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
