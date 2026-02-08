#!/usr/bin/env python3
"""
Validate Label Propagation results by comparing burst vs standalone outputs
"""
import argparse
import json
import sys
import boto3
from collections import Counter

def download_burst_labels(bucket, key, endpoint):
    """Download burst results from S3"""
    s3_client = boto3.client('s3', endpoint_url=endpoint)
    output_key = f"{key}/output/labels_final.json"
    
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=output_key)
        data = json.loads(obj['Body'].read().decode('utf-8'))
        return {int(k): v for k, v in data['labels'].items()}
    except Exception as e:
        print(f"Error downloading burst results: {e}", file=sys.stderr)
        return None

def load_standalone_labels(output_file, graph_file, num_nodes):
    """Extract labels from standalone JSON output or graph file with initial labels"""
    try:
        with open(output_file, 'r') as f:
            data = json.load(f)
            if 'labels' in data:
                # lpst output format
                labels_list = data['labels']
                return {i: labels_list[i] for i in range(len(labels_list))}
    except Exception as e:
        print(f"Error loading standalone results: {e}", file=sys.stderr)
        return None
    
    # Fallback: load from graph file with initial labels
    labels = {}
    try:
        with open(graph_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    node = int(parts[0])
                    label = int(parts[2])
                    labels[node] = label
        return labels
    except:
        return None

def compare_labels(burst_labels, standalone_labels, initial_seeds=None):
    """Compare burst and standalone label assignments"""
    if not burst_labels or not standalone_labels:
        return None
    
    num_nodes = max(max(burst_labels.keys()), max(standalone_labels.keys())) + 1
    
    matching = 0
    different = []
    seed_violations = []
    
    for i in range(num_nodes):
        burst_l = burst_labels.get(i, None)
        standalone_l = standalone_labels.get(i, None)
        
        if burst_l is None or standalone_l is None:
            continue
            
        if burst_l == standalone_l:
            matching += 1
        else:
            different.append((i, burst_l, standalone_l))
        
        # Check seed stability
        if initial_seeds and i in initial_seeds:
            if burst_l != initial_seeds[i]:
                seed_violations.append((i, initial_seeds[i], burst_l))
    
    accuracy = (matching / num_nodes) * 100 if num_nodes > 0 else 0
    
    return {
        'num_nodes': num_nodes,
        'matching': matching,
        'different': len(different),
        'accuracy': accuracy,
        'different_nodes': different[:20],  # First 20 for brevity
        'seed_violations': seed_violations,
        'burst_distribution': Counter(burst_labels.values()),
        'standalone_distribution': Counter(standalone_labels.values())
    }

def main():
    parser = argparse.ArgumentParser(description="Validate LP burst results")
    parser.add_argument('--standalone', required=True, help='Standalone output JSON file')
    parser.add_argument('--graph', required=True, help='Graph file for seed labels')
    parser.add_argument('--bucket', required=True, help='S3 bucket for burst results')
    parser.add_argument('--key', required=True, help='S3 key prefix for burst results')
    parser.add_argument('--endpoint', default='http://localhost:9000', help='S3 endpoint')
    parser.add_argument('--num-nodes', type=int, required=True, help='Number of nodes')
    parser.add_argument('--report', default='validation_report.json', help='Output report file')
    
    args = parser.parse_args()
    
    print("=== Label Propagation Validation ===")
    print(f"Standalone: {args.standalone}")
    print(f"Burst S3: s3://{args.bucket}/{args.key}/output/labels_final.json")
    print()
    
    # Load results
    print("Loading burst results from S3...")
    burst_labels = download_burst_labels(args.bucket, args.key, args.endpoint)
    if not burst_labels:
        print("✗ Failed to load burst results")
        return 1
    print(f"✓ Loaded {len(burst_labels)} burst labels")
    
    print("Loading standalone results...")
    standalone_labels = load_standalone_labels(args.standalone, args.graph, args.num_nodes)
    if not standalone_labels:
        print("✗ Failed to load standalone results")
        return 1
    print(f"✓ Loaded {len(standalone_labels)} standalone labels")
    
    # Load initial seeds for stability check
    initial_seeds = {}
    with open(args.graph, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                node = int(parts[0])
                label = int(parts[2])
                initial_seeds[node] = label
    
    # Compare
    print("\nComparing results...")
    report = compare_labels(burst_labels, standalone_labels, initial_seeds)
    
    if not report:
        print("✗ Comparison failed")
        return 1
    
    # Display results
    print(f"\n=== Validation Report ===")
    print(f"Total nodes: {report['num_nodes']}")
    print(f"Matching labels: {report['matching']} ({report['accuracy']:.2f}%)")
    print(f"Different labels: {report['different']}")
    
    if report['seed_violations']:
        print(f"\n⚠ SEED VIOLATIONS: {len(report['seed_violations'])}")
        for node, expected, got in report['seed_violations'][:10]:
            print(f"  Node {node}: expected {expected}, got {got}")
    
    if report['different'] > 0:
        print(f"\n⚠ LABEL MISMATCHES (showing first 20):")
        for node, burst_l, standalone_l in report['different_nodes']:
            print(f"  Node {node}: burst={burst_l}, standalone={standalone_l}")
    
    print(f"\nBurst label distribution: {dict(report['burst_distribution'].most_common(10))}")
    print(f"Standalone label distribution: {dict(report['standalone_distribution'].most_common(10))}")
    
    # Save report
    with open(args.report, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {args.report}")
    
    # Exit code
    if report['accuracy'] == 100.0 and not report['seed_violations']:
        print("\n✓ VALIDATION PASSED: Results match exactly!")
        return 0
    else:
        print(f"\n✗ VALIDATION FAILED: {report['different']} mismatches")
        return 1

if __name__ == '__main__':
    sys.exit(main())
