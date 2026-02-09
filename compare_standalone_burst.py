#!/usr/bin/env python3
"""
Compare results between standalone (CSV output) and burst (validation pickle) modes
"""

import subprocess
import pickle
from pathlib import Path
from collections import Counter

def parse_standalone_output(nodes: int):
    """Run standalone binary and parse CSV output"""
    graph_file = f"large_{nodes}.txt"
    
    if not Path(graph_file).exists():
        print(f"❌ Graph file {graph_file} not found")
        return None
    
    print(f"\n{'='*60}")
    print(f"🔍 Analyzing Standalone: {nodes:,} nodes")
    print(f"{'='*60}")
    
    # Run standalone binary
    cmd = ["lpst/target/release/label-propagation", graph_file, str(nodes), "10"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Standalone failed: {result.stderr}")
        return None
    
    try:
        output_str = result.stdout.strip()
        
        # Find where JSON metadata starts (last '{')
        json_start = output_str.rfind('{')
        
        if json_start == -1:
            print(f"⚠️  Warning: No JSON metadata found")
            labels_csv = output_str
        else:
            # Labels are CSV before JSON
            labels_csv = output_str[:json_start].rstrip(',')
        
        # Parse labels from CSV
        if labels_csv:
            labels = [int(x) for x in labels_csv.split(',') if x.strip()]
        else:
            labels = []
        
        # Count distribution
        label_counts = Counter(labels)
        
        print(f"\n📊 Standalone Label Distribution:")
        print(f"   Total nodes: {len(labels):,}")
        
        for label in sorted(label_counts.keys()):
            count = label_counts[label]
            pct = (count / len(labels)) * 100 if labels else 0
            print(f"   Label {label:>3}: {count:>10,} nodes ({pct:>5.2f}%)")
        
        # Check for unexpected labels
        expected_labels = {0, 100, 200, 300}
        unexpected = set(label_counts.keys()) - expected_labels
        if unexpected:
            print(f"\n⚠️  WARNING: Unexpected labels found: {unexpected}")
        
        return label_counts
        
    except Exception as e:
        print(f"❌ Error parsing output: {e}")
        return None


def parse_burst_validation(nodes: int, partitions: int = 4):
    """Parse burst validation results from pickle"""
    validation_file = f"validation_{nodes}_{partitions}p.pkl"
    
    if not Path(validation_file).exists():
        print(f"❌ Validation file {validation_file} not found")
        return None
    
    print(f"\n{'='*60}")
    print(f"🔍 Analyzing Burst: {nodes:,} nodes")
    print(f"{'='*60}")
    
    try:
        with open(validation_file, 'rb') as f:
            data = pickle.load(f)
        
        # Extract burst results
        if 'burst_time' not in data:
            print(f"❌ No burst results in validation file")
            return None
        
        # The burst results should have label distribution
        # Check what's in the data structure
        print(f"\n📊 Burst results keys: {data.keys()}")
        
        # Try to find label distribution
        # Sometimes it's in 'label_distribution', sometimes in activations
        if 'label_distribution' in data:
            label_dist = data['label_distribution']
            print(f"\n📊 Burst Label Distribution:")
            for label in sorted(label_dist.keys()):
                count = label_dist[label]
                pct = (count / nodes) * 100
                print(f"   Label {label:>3}: {count:>10,} nodes ({pct:>5.2f}%)")
            return label_dist
        else:
            print(f"\n⚠️  Looking for label distribution in data structure...")
            # Print sample to help debug
            for key, value in list(data.items())[:5]:
                print(f"   {key}: {type(value)}")
            return None
            
    except Exception as e:
        print(f"❌ Error reading validation file: {e}")
        return None


def compare_results(nodes: int):
    """Compare standalone and burst results"""
    
    # Get standalone results
    standalone_labels = parse_standalone_output(nodes)
    
    # Get burst results from pickle
    burst_labels = parse_burst_validation(nodes)
    
    # Compare
    if standalone_labels and burst_labels:
        print(f"\n{'='*60}")
        print(f"📊 Comparison:")
        print(f"{'='*60}")
        
        all_labels = sorted(set(standalone_labels.keys()) | set(burst_labels.keys()))
        
        print(f"\n{'Label':<10} {'Standalone':<15} {'Burst':<15} {'Difference':<15}")
        print(f"{'-'*60}")
        
        for label in all_labels:
            s_count = standalone_labels.get(label, 0)
            b_count = burst_labels.get(label, 0)
            diff = s_count - b_count
            diff_pct = (diff / nodes) * 100 if nodes > 0 else 0
            
            print(f"{label:<10} {s_count:<15,} {b_count:<15,} {diff:>+10,} ({diff_pct:>+5.2f}%)")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        nodes = int(sys.argv[1])
    else:
        # Test all available sizes
        nodes = 3000000
    
    compare_results(nodes)
