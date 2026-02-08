#!/usr/bin/env python3
"""
Comparative Benchmark: Standalone vs Burst Implementation
Compares performance from 1M to 10M nodes
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from ow_client.openwhisk_executor import OpenwhiskExecutor
from ow_client.time_helper import get_millis
from labelpropagation_utils import generate_payload

def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def benchmark_standalone(graph_file, num_nodes, max_iter=10):
    """Run standalone benchmark and return time in ms"""
    binary = "lpst/target/release/label-propagation"
    
    print(f"🔄 Running standalone ({num_nodes:,} nodes)...")
    
    try:
        result = subprocess.run(
            [binary, graph_file, str(num_nodes), str(max_iter)],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode != 0:
            print(f"❌ Failed: {result.stderr}")
            return None
        
        output = json.loads(result.stdout.strip())
        time_ms = output.get("total_time_ms", 0)
        
        print(f"✓ Completed in {time_ms/1000:.2f}s")
        return time_ms
        
    except subprocess.TimeoutExpired:
        print("❌ Timed out")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def benchmark_burst(num_nodes, num_partitions, max_iter, memory_mb, granularity, 
                   ow_host, ow_port, s3_endpoint):
    """Run burst benchmark and return (total_time, algo_time) in ms"""
    s3_prefix = f"graphs/large-{num_nodes}"
    
    print(f"🔄 Running burst ({num_nodes:,} nodes, {num_partitions} partitions)...")
    
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
    
    executor = OpenwhiskExecutor(ow_host, ow_port, debug=False)
    
    try:
        host_submit = get_millis()
        dt = executor.burst(
            "labelpropagation",
            params,
            file="./labelpropagation.zip",
            memory=memory_mb,
            custom_image="burstcomputing/runtime-rust-burst:latest",
            debug_mode=False,
            burst_size=1,
            join=False,
            backend="redis-list",
            chunk_size=1024,
            is_zip=True,
            timeout=900000
        )
        finished = get_millis()
        
        results = dt.get_results()
        if not results:
            print("❌ No results returned")
            return None, None
        
        # Extract timing information
        worker_starts = []
        worker_ends = []
        iterations = None
        converged = None
        
        for r in results:
            worker_data = r
            if isinstance(r, list) and len(r) > 0:
                worker_data = r[0]
            
            if isinstance(worker_data, dict):
                if "timestamps" in worker_data:
                    ts_map = {ts["key"]: int(ts["value"]) for ts in worker_data["timestamps"]}
                    if "worker_start" in ts_map:
                        worker_starts.append(ts_map["worker_start"])
                    if "worker_end" in ts_map:
                        worker_ends.append(ts_map["worker_end"])
                
                # Extract convergence info from results string
                if "results" in worker_data and worker_data["results"]:
                    results_str = worker_data["results"]
                    if "iterations:" in results_str:
                        try:
                            iterations = int(results_str.split("iterations:")[1].split()[0])
                        except:
                            pass
                    if "converged:" in results_str:
                        converged = "true" in results_str.lower()
        
        algo_time = None
        if worker_starts and worker_ends:
            algo_time = max(worker_ends) - min(worker_starts)
        
        total_time = finished - host_submit
        
        print(f"✓ Completed in {total_time/1000:.2f}s (algo: {algo_time/1000:.2f}s)" if algo_time 
              else f"✓ Completed in {total_time/1000:.2f}s")
        
        if iterations is not None:
            print(f"  Iterations: {iterations}, Converged: {converged}")
        
        return total_time, algo_time
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def run_comparison(nodes_list, args):
    """Run comparison for multiple graph sizes"""
    results = []
    
    for num_nodes in nodes_list:
        print_section(f"Benchmarking {num_nodes:,} nodes")
        
        graph_file = f"large_{num_nodes}.txt"
        
        # Standalone
        standalone_time = benchmark_standalone(graph_file, num_nodes, args.iter)
        
        # Burst
        burst_total, burst_algo = benchmark_burst(
            num_nodes,
            args.partitions,
            args.iter,
            args.memory,
            args.granularity,
            args.ow_host,
            args.ow_port,
            args.s3_endpoint
        )
        
        # Calculate metrics
        result = {
            'nodes': num_nodes,
            'standalone_ms': standalone_time,
            'burst_total_ms': burst_total,
            'burst_algo_ms': burst_algo,
        }
        
        if standalone_time and burst_total:
            result['speedup_total'] = standalone_time / burst_total
            result['overhead_ms'] = burst_total - burst_algo if burst_algo else None
            result['overhead_pct'] = ((burst_total - burst_algo) / burst_total * 100) if burst_algo else None
            
            if burst_algo:
                result['speedup_algo'] = standalone_time / burst_algo
        
        results.append(result)
        
        # Print summary for this size
        print("\n📊 Summary:")
        print(f"  Standalone:    {standalone_time/1000:.2f}s" if standalone_time else "  Standalone:    FAILED")
        print(f"  Burst (total): {burst_total/1000:.2f}s" if burst_total else "  Burst (total): FAILED")
        if burst_algo:
            print(f"  Burst (algo):  {burst_algo/1000:.2f}s")
        if 'speedup_total' in result:
            print(f"  Speedup:       {result['speedup_total']:.2f}x")
        if 'speedup_algo' in result:
            print(f"  Algo Speedup:  {result['speedup_algo']:.2f}x")
        if 'overhead_pct' in result and result['overhead_pct']:
            print(f"  OW Overhead:   {result['overhead_pct']:.1f}%")
    
    return results

def generate_report(results, args):
    """Generate comprehensive comparison report"""
    print_section("COMPARATIVE BENCHMARK REPORT")
    
    # Table header
    print("\n{:<12} {:<15} {:<15} {:<15} {:<12} {:<12}".format(
        "Nodes", "Standalone", "Burst Total", "Burst Algo", "Speedup", "OW Overhead"
    ))
    print("-" * 80)
    
    # Table rows
    for r in results:
        nodes = f"{r['nodes']:,}"
        standalone = f"{r['standalone_ms']/1000:.2f}s" if r['standalone_ms'] else "FAILED"
        burst_total = f"{r['burst_total_ms']/1000:.2f}s" if r['burst_total_ms'] else "FAILED"
        burst_algo = f"{r['burst_algo_ms']/1000:.2f}s" if r.get('burst_algo_ms') else "N/A"
        speedup = f"{r['speedup_total']:.2f}x" if r.get('speedup_total') else "N/A"
        overhead = f"{r['overhead_pct']:.1f}%" if r.get('overhead_pct') else "N/A"
        
        print("{:<12} {:<15} {:<15} {:<15} {:<12} {:<12}".format(
            nodes, standalone, burst_total, burst_algo, speedup, overhead
        ))
    
    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    valid_results = [r for r in results if r.get('speedup_total')]
    
    if valid_results:
        avg_speedup = sum(r['speedup_total'] for r in valid_results) / len(valid_results)
        max_speedup = max(r['speedup_total'] for r in valid_results)
        min_speedup = min(r['speedup_total'] for r in valid_results)
        
        print(f"\n🚀 Speedup Statistics:")
        print(f"   Average: {avg_speedup:.2f}x")
        print(f"   Maximum: {max_speedup:.2f}x")
        print(f"   Minimum: {min_speedup:.2f}x")
        
        algo_results = [r for r in valid_results if r.get('speedup_algo')]
        if algo_results:
            avg_algo_speedup = sum(r['speedup_algo'] for r in algo_results) / len(algo_results)
            print(f"\n⚡ Algorithmic Speedup: {avg_algo_speedup:.2f}x")
            
            avg_overhead = sum(r['overhead_pct'] for r in algo_results) / len(algo_results)
            print(f"📊 Average OpenWhisk Overhead: {avg_overhead:.1f}%")
        
        if avg_speedup > 1.0:
            print(f"\n✅ Burst is {avg_speedup:.2f}x faster on average!")
        else:
            print(f"\n⚠️  Standalone is {1/avg_speedup:.2f}x faster on average")
            if algo_results and avg_algo_speedup > 1.0:
                print(f"   But algorithmically, burst is {avg_algo_speedup:.2f}x faster")
                print(f"   OpenWhisk overhead reduces overall performance")
    
    # Save detailed results
    output_file = f"comparison_results_{int(time.time())}.json"
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'partitions': args.partitions,
            'granularity': args.granularity,
            'iterations': args.iter,
            'memory_mb': args.memory,
        },
        'results': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Compare Standalone vs Burst LP")
    parser.add_argument("--nodes", type=str, default="1000000,10000000",
                       help="Comma-separated list of node counts (e.g., '1000000,5000000,10000000')")
    parser.add_argument("--partitions", type=int, default=4, help="Number of partitions")
    parser.add_argument("--granularity", type=int, default=1, help="Granularity")
    parser.add_argument("--iter", type=int, default=10, help="Max iterations")
    parser.add_argument("--memory", type=int, default=2048, help="Memory per worker (MB)")
    parser.add_argument("--ow-host", default="localhost", help="OpenWhisk host")
    parser.add_argument("--ow-port", type=int, default=31001, help="OpenWhisk port")
    parser.add_argument("--s3-endpoint", default="http://minio-service.default.svc.cluster.local:9000",
                       help="S3 endpoint")
    
    args = parser.parse_args()
    
    # Parse node list
    nodes_list = [int(n.strip()) for n in args.nodes.split(',')]
    
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "COMPARATIVE BENCHMARK SUITE" + " "*31 + "║")
    print("║" + " "*24 + "Standalone vs Burst" + " "*35 + "║")
    print("╚" + "="*78 + "╝")
    
    print(f"\nConfiguration:")
    print(f"  Graph sizes: {', '.join(f'{n:,}' for n in nodes_list)} nodes")
    print(f"  Partitions: {args.partitions}")
    print(f"  Granularity: {args.granularity}")
    print(f"  Max iterations: {args.iter}")
    print(f"  Worker memory: {args.memory} MB")
    
    start_time = time.time()
    
    try:
        results = run_comparison(nodes_list, args)
        output_file = generate_report(results, args)
        
        elapsed = time.time() - start_time
        print(f"\n⏱️  Total benchmark time: {elapsed:.1f}s")
        
        print("\n" + "="*80)
        print("✅ COMPARISON COMPLETED")
        print("="*80)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
