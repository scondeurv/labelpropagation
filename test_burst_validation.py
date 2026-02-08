#!/usr/bin/env python3
"""
Test script to validate that the burst implementation generates correct results.

This script:
1. Generates a small test graph with clear community structure
2. Runs the standalone (lpst) version
3. Runs the burst distributed version
4. Compares results to verify correctness
"""
import argparse
import json
import os
import subprocess
import sys
import time
import boto3
from botocore.exceptions import ClientError

# Test configuration  
TEST_NUM_NODES = 100
TEST_NUM_COMMUNITIES = 2
TEST_PARTITIONS = 4
TEST_SEED_RATIO = 0.1
TEST_BUCKET = "lp-test"
TEST_KEY = "test-graph"
TEST_MAX_ITER = 50
TEST_CONVERGENCE = 0

# File paths
TEST_GRAPH_JSON = "test_graph.json"
TEST_GRAPH_TSV = "test_graph.tsv"
TEST_STANDALONE_OUTPUT = "test_standalone_results.json"
TEST_VALIDATION_REPORT = "test_validation_report.json"


def print_header(message):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {message}")
    print("=" * 70)


def generate_test_graph():
    """Generate a small graph with 2 clear communities for testing"""
    print_header("STEP 1: Generating Test Graph")
    
    import random
    random.seed(42)  # Reproducible results
    
    # Create two communities
    community_size = TEST_NUM_NODES // TEST_NUM_COMMUNITIES
    edges = []
    labeled_nodes = {}
    
    # Generate edges within each community (high connectivity)
    for community in range(TEST_NUM_COMMUNITIES):
        start = community * community_size
        end = start + community_size
        
        # Create dense connections within community
        for i in range(start, end):
            for j in range(i + 1, end):
                if random.random() < 0.4:  # 40% edge probability within community
                    edges.append([i, j])
        
        # Label 10% of nodes in each community
        num_seeds = int(community_size * TEST_SEED_RATIO)
        for _ in range(num_seeds):
            node = random.randint(start, end - 1)
            labeled_nodes[str(node)] = community * 100  # Label 0 or 100
    
    # Add sparse connections between communities
    for i in range(0, community_size):
        for j in range(community_size, TEST_NUM_NODES):
            if random.random() < 0.02:  # 2% edge probability between communities
                edges.append([i, j])
    
    # Save as JSON for standalone version
    graph_data = {
        "edges": edges,
        "labeled_nodes": {int(k): v for k, v in labeled_nodes.items()},
        "num_nodes": TEST_NUM_NODES
    }
    
    with open(TEST_GRAPH_JSON, 'w') as f:
        json.dump(graph_data, f, indent=2)
    
    print(f"✓ Generated graph with {len(edges)} edges and {len(labeled_nodes)} labeled nodes")
    print(f"  Communities: {TEST_NUM_COMMUNITIES}")
    print(f"  Nodes per community: {community_size}")
    print(f"  Saved to: {TEST_GRAPH_JSON}")
    
    # Also save in TSV format for burst version
    with open(TEST_GRAPH_TSV, 'w') as f:
        for src, dst in edges:
            # Check if source or destination has initial label
            if str(src) in labeled_nodes:
                f.write(f"{src}\t{dst}\t{labeled_nodes[str(src)]}\n")
            elif str(dst) in labeled_nodes:
                f.write(f"{src}\t{dst}\t{labeled_nodes[str(dst)]}\n")
            else:
                f.write(f"{src}\t{dst}\n")
    
    print(f"✓ Saved TSV format to: {TEST_GRAPH_TSV}")
    
    return graph_data


def compile_standalone():
    """Compile the standalone Rust implementation"""
    print_header("STEP 2: Compiling Standalone Version")
    
    lpst_dir = "lpst"
    print(f"Compiling {lpst_dir}...")
    
    result = subprocess.run(
        ["cargo", "build", "--release"],
        cwd=lpst_dir,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print("✗ Compilation failed:")
        print(result.stderr)
        return False
    
    print("✓ Standalone version compiled successfully")
    return True


def run_standalone():
    """Run the standalone version"""
    print_header("STEP 3: Running Standalone Version")
    
    binary = "lpst/target/release/run_label_propagation"
    
    if not os.path.exists(binary):
        print(f"✗ Binary not found: {binary}")
        return False
    
    print(f"Running: {binary} {TEST_GRAPH_JSON} {TEST_MAX_ITER} 0.0 {TEST_STANDALONE_OUTPUT}")
    
    result = subprocess.run(
        [binary, TEST_GRAPH_JSON, str(TEST_MAX_ITER), "0.0", TEST_STANDALONE_OUTPUT],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print("✗ Standalone execution failed:")
        print(result.stderr)
        return False
    
    print(result.stdout)
    print(f"✓ Standalone results saved to: {TEST_STANDALONE_OUTPUT}")
    
    return True


def setup_s3(endpoint):
    """Setup S3 client and create bucket if needed"""
    print_header("STEP 4: Setting up S3")
    
    s3_client = boto3.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID', 'minioadmin'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY', 'minioadmin'),
        region_name='us-east-1'
    )
    
    # Create bucket if it doesn't exist
    try:
        s3_client.head_bucket(Bucket=TEST_BUCKET)
        print(f"✓ Bucket '{TEST_BUCKET}' already exists")
    except ClientError:
        try:
            s3_client.create_bucket(Bucket=TEST_BUCKET)
            print(f"✓ Created bucket '{TEST_BUCKET}'")
        except ClientError as e:
            print(f"✗ Failed to create bucket: {e}")
            return None
    
    return s3_client


def upload_graph_to_s3(s3_client, endpoint):
    """Upload test graph partitions to S3"""
    print("Uploading graph partitions to S3...")
    
    # Read TSV graph
    with open(TEST_GRAPH_TSV, 'r') as f:
        lines = f.readlines()
    
    # Partition by source node (modulo distribution)
    partitions = {i: [] for i in range(TEST_PARTITIONS)}
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            src = int(parts[0])
            part_idx = src % TEST_PARTITIONS
            partitions[part_idx].append(line)
    
    # Upload each partition
    for part_idx, part_lines in partitions.items():
        if not part_lines:
            continue
        
        data = ''.join(part_lines)
        key = f"{TEST_KEY}/part-{str(part_idx).zfill(5)}"
        
        s3_client.put_object(
            Bucket=TEST_BUCKET,
            Key=key,
            Body=data.encode('utf-8'),
            ContentType='text/plain'
        )
        print(f"  ✓ Uploaded partition {part_idx} to s3://{TEST_BUCKET}/{key} ({len(part_lines)} lines)")
    
    print(f"✓ All partitions uploaded to S3")


def compile_burst():
    """Compile the burst implementation"""
    print_header("STEP 5: Compiling Burst Version")
    
    ow_lp_dir = "ow-lp"
    print(f"Compiling {ow_lp_dir}...")
    
    result = subprocess.run(
        ["cargo", "build", "--release"],
        cwd=ow_lp_dir,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print("✗ Compilation failed:")
        print(result.stderr)
        return False
    
    # Copy binary to bin/exec
    import shutil
    src = f"{ow_lp_dir}/target/release/labelpropagation"
    dst = f"{ow_lp_dir}/bin/exec"
    os.makedirs(f"{ow_lp_dir}/bin", exist_ok=True)
    shutil.copy(src, dst)
    
    print("✓ Burst version compiled successfully")
    return True


def run_burst(ow_host, ow_port, lp_endpoint, backend):
    """Run the burst distributed version"""
    print_header("STEP 6: Running Burst Version")
    
    # Check if action needs to be deployed
    print("Deploying burst action...")
    
    cmd = [
        "python3", "labelpropagation.py",
        "--ow-host", ow_host,
        "--ow-port", str(ow_port),
        "--lp-endpoint", lp_endpoint,
        "--partitions", str(TEST_PARTITIONS),
        "--num-nodes", str(TEST_NUM_NODES),
        "--bucket", TEST_BUCKET,
        "--key", TEST_KEY,
        "--granularity", str(TEST_PARTITIONS),
        "--backend", backend,
        "--max-iterations", str(TEST_MAX_ITER),
        "--convergence-threshold", str(TEST_CONVERGENCE),
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(result.stdout)
    if result.returncode != 0:
        print("✗ Burst execution failed:")
        print(result.stderr)
        return False
    
    print("✓ Burst execution completed")
    return True


def validate_results(s3_endpoint):
    """Validate that burst and standalone produce the same results"""
    print_header("STEP 7: Validating Results")
    
    cmd = [
        "python3", "validate_results.py",
        "--standalone", TEST_STANDALONE_OUTPUT,
        "--graph", TEST_GRAPH_TSV,
        "--bucket", TEST_BUCKET,
        "--key", TEST_KEY,
        "--endpoint", s3_endpoint,
        "--num-nodes", str(TEST_NUM_NODES),
        "--report", TEST_VALIDATION_REPORT
    ]
    
    print(f"Running validation...")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(result.stdout)
    
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Test burst validation")
    parser.add_argument("--ow-host", default="localhost", help="OpenWhisk host")
    parser.add_argument("--ow-port", type=int, default=31001, help="OpenWhisk port")
    parser.add_argument("--s3-endpoint", default="http://localhost:9000", help="S3 endpoint")
    parser.add_argument("--backend", default="redis", help="Burst backend (redis, rabbitmq)")
    parser.add_argument("--skip-compile", action="store_true", help="Skip compilation")
    parser.add_argument("--skip-standalone", action="store_true", help="Skip standalone execution")
    parser.add_argument("--skip-burst", action="store_true", help="Skip burst execution")
    args = parser.parse_args()
    
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "BURST VALIDATION TEST SUITE" + " " * 26 + "║")
    print("╚" + "═" * 68 + "╝")
    
    start_time = time.time()
    
    try:
        # Generate test graph
        graph_data = generate_test_graph()
        
        # Compile and run standalone
        if not args.skip_compile:
            if not compile_standalone():
                return 1
        
        if not args.skip_standalone:
            if not run_standalone():
                return 1
        
        # Setup S3 and upload graph
        s3_client = setup_s3(args.s3_endpoint)
        if not s3_client:
            return 1
        
        upload_graph_to_s3(s3_client, args.s3_endpoint)
        
        # Compile and run burst
        if not args.skip_compile:
            if not compile_burst():
                return 1
        
        if not args.skip_burst:
            if not run_burst(args.ow_host, args.ow_port, args.s3_endpoint, args.backend):
                return 1
        
        # Wait a bit for results to be written to S3
        print("\nWaiting 5 seconds for results to be written to S3...")
        time.sleep(5)
        
        # Validate results
        if validate_results(args.s3_endpoint):
            elapsed = time.time() - start_time
            print_header("✓ ALL TESTS PASSED!")
            print(f"Total time: {elapsed:.2f}s")
            return 0
        else:
            print_header("✗ VALIDATION FAILED")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n✗ Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
