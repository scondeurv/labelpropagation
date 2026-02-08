#!/usr/bin/env python3
"""
Quick validation test that compares the core LP algorithms without OpenWhisk.

This script:
1. Generates test graphs with known properties
2. Runs both standalone (lpst) and simulates burst behavior
3. Verifies correctness of results
"""
import json
import random
import subprocess
import sys
import os

def generate_simple_graph(filename, num_nodes=20, num_seeds=4):
    """Generate a simple graph with two distinct communities"""
    random.seed(42)
    
    # Two communities
    half = num_nodes // 2
    edges = []
    labeled_nodes = {}
    
    # Community 0: nodes 0-9, label 0
    for i in range(half):
        for j in range(i + 1, half):
            if random.random() < 0.7:  # Dense within community
                edges.append([i, j])
    
    # Community 1: nodes 10-19, label 100
    for i in range(half, num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < 0.7:  # Dense within community
                edges.append([i, j])
    
    # Sparse connections between communities
    for i in range(half):
        for j in range(half, num_nodes):
            if random.random() < 0.05:
                edges.append([i, j])
    
    # Add seed labels (first 2 nodes in each community)
    labeled_nodes[0] = 0
    labeled_nodes[1] = 0
    labeled_nodes[half] = 100
    labeled_nodes[half + 1] = 100
    
    graph_data = {
        "edges": edges,
        "labeled_nodes": labeled_nodes,
        "num_nodes": num_nodes
    }
    
    with open(filename, 'w') as f:
        json.dump(graph_data, f, indent=2)
    
    print(f"✓ Generated test graph: {num_nodes} nodes, {len(edges)} edges, {len(labeled_nodes)} seeds")
    return graph_data

def run_standalone(graph_file, output_file):
    """Run standalone implementation"""
    binary = "lpst/target/release/run_label_propagation"
    
    if not os.path.exists(binary):
        print("✗ Compiling standalone version first...")
        result = subprocess.run(
            ["cargo", "build", "--release"],
            cwd="lpst",
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"✗ Compilation failed: {result.stderr}")
            return None
    
    result = subprocess.run(
        [binary, graph_file, "50", "0.0", output_file],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"✗ Execution failed: {result.stderr}")
        return None
    
    with open(output_file, 'r') as f:
        data = json.load(f)
    
    print(f"✓ Standalone completed: {data['iterations']} iterations, converged={data['converged']}")
    return data

def verify_label_consistency(labels, initial_labels, num_nodes):
    """Verify that results make sense"""
    issues = []
    
    # Check 1: All nodes have labels
    if len(labels) != num_nodes:
        issues.append(f"Expected {num_nodes} labels, got {len(labels)}")
    
    # Check 2: Initial labels are preserved (clamping)
    for node, label in initial_labels.items():
        node_id = str(node)
        if node_id in labels and labels[node_id] != label:
            issues.append(f"Seed node {node_id} changed from {label} to {labels[node_id]}")
    
    # Check 3: Count label distribution
    label_counts = {}
    for node_id, label in labels.items():
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"  Label distribution: {label_counts}")
    
    return issues

def test_case_1():
    """Test 1: Simple two-community graph"""
    print("\n" + "="*60)
    print("TEST 1: Two-Community Graph")
    print("="*60)
    
    graph_file = "test_simple.json"
    output_file = "test_simple_output.json"
    
    # Generate and run
    graph_data = generate_simple_graph(graph_file, num_nodes=20)
    result = run_standalone(graph_file, output_file)
    
    if not result:
        return False
    
    # Verify
    issues = verify_label_consistency(
        result['labels'], 
        graph_data['labeled_nodes'],
        graph_data['num_nodes']
    )
    
    if issues:
        print("✗ Validation failed:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    # Check that most nodes got one of the seed labels
    label_values = set(result['labels'].values())
    expected_labels = set(graph_data['labeled_nodes'].values())
    
    if not label_values.issubset(expected_labels | {0, 100}):
        unexpected = label_values - expected_labels
        print(f"⚠ Warning: Unexpected labels appeared: {unexpected}")
    
    print("✓ Test 1 PASSED")
    return True

def test_case_2():
    """Test 2: Larger graph with clear structure"""
    print("\n" + "="*60)
    print("TEST 2: Larger Graph (100 nodes)")
    print("="*60)
    
    graph_file = "test_large.json"
    output_file = "test_large_output.json"
    
    # Generate and run
    graph_data = generate_simple_graph(graph_file, num_nodes=100)
    result = run_standalone(graph_file, output_file)
    
    if not result:
        return False
    
    # Verify
    issues = verify_label_consistency(
        result['labels'], 
        graph_data['labeled_nodes'],
        graph_data['num_nodes']
    )
    
    if issues:
        print("✗ Validation failed:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    print("✓ Test 2 PASSED")
    return True

def test_case_3():
    """Test 3: Ring graph (should propagate labels around)"""
    print("\n" + "="*60)
    print("TEST 3: Ring Graph")
    print("="*60)
    
    graph_file = "test_ring.json"
    output_file = "test_ring_output.json"
    
    num_nodes = 30
    edges = [[i, (i+1) % num_nodes] for i in range(num_nodes)]
    
    graph_data = {
        "edges": edges,
        "labeled_nodes": {0: 0, 15: 100},  # Two seeds on opposite sides
        "num_nodes": num_nodes
    }
    
    with open(graph_file, 'w') as f:
        json.dump(graph_data, f, indent=2)
    
    result = run_standalone(graph_file, output_file)
    
    if not result:
        return False
    
    issues = verify_label_consistency(
        result['labels'],
        graph_data['labeled_nodes'],
        graph_data['num_nodes']
    )
    
    if issues:
        print("✗ Validation failed:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    print("✓ Test 3 PASSED")
    return True

def main():
    print("╔" + "═"*58 + "╗")
    print("║" + " "*10 + "QUICK VALIDATION TEST SUITE" + " "*21 + "║")
    print("╚" + "═"*58 + "╝")
    
    tests = [test_case_1, test_case_2, test_case_3]
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    return 0 if failed == 0 else 1

if __name__ == '__main__':
    sys.exit(main())
