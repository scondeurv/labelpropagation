#!/usr/bin/env python3
"""
Benchmark Label Propagation: Standalone vs Burst versions
"""
import argparse
import json
import subprocess
import sys
import os
from ow_client.openwhisk_executor import OpenwhiskExecutor
from ow_client.time_helper import get_millis
from labelpropagation_utils import generate_payload

def benchmark_standalone(graph_file, num_nodes, max_iter):
    """Run standalone Label Propagation and return execution time in ms"""
    binary_path = "lpst/target/release/label-propagation"
    
    if not os.path.exists(binary_path):
        print(f"Error: Binary not found at {binary_path}", file=sys.stderr)
        print("Run: cd lpst && cargo build --release", file=sys.stderr)
        return None
    
    if not os.path.exists(graph_file):
        print(f"Error: Graph file not found: {graph_file}", file=sys.stderr)
        return None
    
    try:
        result = subprocess.run(
            [binary_path, graph_file, str(num_nodes), str(max_iter)],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode != 0:
            print(f"Error running standalone: {result.stderr}", file=sys.stderr)
            return None
        
        # Parse JSON output
        output = json.loads(result.stdout.strip())
        return output.get("total_time_ms", 0)
    except subprocess.TimeoutExpired:
        print("Error: Standalone version timed out", file=sys.stderr)
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing standalone output: {e}", file=sys.stderr)
        print(f"Output was: {result.stdout}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return None

def run_validation(graph_file, num_nodes, bucket, key, endpoint):
    """Run validation comparing standalone vs burst results"""
    import subprocess
    
    standalone_output = "lpst_output.json"
    
    # Run standalone and save full output
    binary_path = "lpst/target/release/label-propagation"
    try:
        result = subprocess.run(
            [binary_path, graph_file, str(num_nodes), "50"],
            capture_output=True,
            text=True,
            timeout=600
        )
        if result.returncode != 0:
            print(f"Standalone failed: {result.stderr}", file=sys.stderr)
            return False
        
        with open(standalone_output, 'w') as f:
            f.write(result.stdout)
    except Exception as e:
        print(f"Error running standalone for validation: {e}", file=sys.stderr)
        return False
    
    # Run validation script
    val_result = subprocess.run([
        "python3", "validate_results.py",
        "--standalone", standalone_output,
        "--graph", graph_file,
        "--bucket", bucket,
        "--key", key,
        "--endpoint", endpoint,
        "--num-nodes", str(num_nodes)
    ], capture_output=True, text=True)
    
    print(val_result.stdout)
    if val_result.returncode != 0:
        print("VALIDATION FAILED!", file=sys.stderr)
        print(val_result.stderr, file=sys.stderr)
        return False
    
    return True

def benchmark_burst(num_nodes, num_partitions, max_iter, memory_mb, granularity=1, ow_host="localhost", ow_port=31001, s3_endpoint="http://minio-service.default.svc.cluster.local:9000"):
    """Run burst Label Propagation and return execution time in ms"""
    s3_prefix = f"graphs/large-{num_nodes}"
    
    params = generate_payload(
        endpoint=s3_endpoint,
        partitions=num_partitions,
        num_nodes=num_nodes,
        bucket="test-bucket",
        key=s3_prefix,
        convergence_threshold=0,
        max_iterations=max_iter,
        granularity=granularity
    )
    
    executor = OpenwhiskExecutor(ow_host, ow_port, debug=True)
    
    try:
        host_submit = get_millis()
        dt = executor.burst(
            "labelpropagation",
            params,
            file="./labelpropagation.zip",
            memory=memory_mb,
            custom_image="burstcomputing/runtime-rust-burst:latest",
            debug_mode=True,
            burst_size=1,
            join=False,
            backend="redis-list",
            chunk_size=1024,
            is_zip=True,
            timeout=900000
        )
        finished = get_millis()
        
        # Get results to ensure completion
        results = dt.get_results()
        if not results:
            print("Error: No results from burst execution", file=sys.stderr)
            return None
        
        return finished - host_submit
    except Exception as e:
        print(f"Error running burst: {e}", file=sys.stderr)
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark LP: Standalone vs Burst")
    parser.add_argument("--nodes", type=int, required=True, help="Number of nodes")
    parser.add_argument("--partitions", type=int, default=8, help="Number of partitions for burst")
    parser.add_argument("--granularity", type=int, default=1, help="Granularity for burst")
    parser.add_argument("--iter", type=int, default=10, help="Max iterations")
    parser.add_argument("--memory", type=int, default=512, help="Memory per worker (MB)")
    parser.add_argument("--ow-host", type=str, default="localhost", help="OpenWhisk host")
    parser.add_argument("--ow-port", type=int, default=31001, help="OpenWhisk port")
    parser.add_argument("--skip-standalone", action="store_true", help="Skip standalone benchmark")
    parser.add_argument("--skip-burst", action="store_true", help="Skip burst benchmark")
    parser.add_argument("--validate", action="store_true", help="Validate burst results against standalone")
    parser.add_argument("--s3-endpoint", default="http://minio-service.default.svc.cluster.local:9000", help="S3 endpoint for workers inside cluster")
    parser.add_argument("--bucket", default="test-bucket", help="S3 bucket name")
    parser.add_argument("--key-prefix", default="graphs", help="S3 key prefix")
    
    args = parser.parse_args()
    
    graph_file = f"large_{args.nodes}.txt"
    
    # Benchmark standalone
    lpst_time = None
    if not args.skip_standalone:
        print(f"Running standalone version...")
        lpst_time = benchmark_standalone(graph_file, args.nodes, args.iter)
        if lpst_time is not None:
            print(f"LPST Time: {lpst_time}")
        else:
            print("LPST Time: FAILED")
    
    # Benchmark burst
    burst_time = None
    if not args.skip_burst:
        print(f"Running burst version...")
        burst_time = benchmark_burst(
            args.nodes, 
            args.partitions, 
            args.iter, 
            args.memory,
            args.granularity,
            args.ow_host,
            args.ow_port,
            args.s3_endpoint
        )
        if burst_time is not None:
            print(f"Burst Time: {burst_time}")
        else:
            print("Burst Time: FAILED")
    
    # Calculate speedup
    if lpst_time and burst_time:
        speedup = lpst_time / burst_time
        print(f"Speedup: {speedup:.2f}x")
        if speedup > 1.0:
            print("✓ Burst is faster!")
        else:
            print("✗ Standalone is faster")
    
    # Validation
    if args.validate:
        # Skip validation for very large graphs (too slow for standalone)
        if args.nodes > 5000:
            print(f"\n⚠ Skipping validation for {args.nodes} nodes (too large for standalone reference)")
        else:
            print("\n=== Running Validation ===")
            key_prefix = f"{args.key_prefix}/large-{args.nodes}"
            if not run_validation(graph_file, args.nodes, args.bucket, key_prefix, args.s3_endpoint):
                print("\n✗ VALIDATION FAILED - Results do not match!")
                sys.exit(1)
            print("\n✓ VALIDATION PASSED - Results match!")
