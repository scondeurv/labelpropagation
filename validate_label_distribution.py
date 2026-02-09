#!/usr/bin/env python3
"""
Validate label distribution and propagation in detail
"""
import subprocess
import json
import sys

def test_label_distribution(nodes, output_file="large_{}.txt"):
    """Run Label Propagation and analyze detailed label distribution"""
    
    graph_file = output_file.format(nodes)
    binary = "lpst/target/release/label-propagation"
    
    print(f"\n{'='*80}")
    print(f"TESTING LABEL DISTRIBUTION: {nodes/1e6:.1f}M nodes")
    print(f"{'='*80}")
    
    # Run label propagation
    result = subprocess.run(
        [binary, graph_file, str(nodes), "10"],
        capture_output=True,
        text=True,
        timeout=120
    )
    
    if result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        return None
    
    # Parse output
    try:
        output = json.loads(result.stdout.strip())
    except:
        print(f"❌ Failed to parse JSON output")
        return None
    
    # Analyze label distribution
    label_counts = output.get("label_counts", {})
    final_labels = output.get("final_labels", [])
    
    print(f"\n📊 Global Label Distribution:")
    total_labeled = 0
    for label, count in sorted(label_counts.items(), key=lambda x: int(x[0])):
        pct = (count / nodes) * 100
        total_labeled += count
        print(f"   Label {label:>3}: {count:>10} nodes ({pct:>5.2f}%)")
    
    print(f"\n   Total labeled: {total_labeled:>10} / {nodes} nodes")
    
    if total_labeled != nodes:
        print(f"   ⚠️  WARNING: {nodes - total_labeled} unlabeled nodes!")
    
    # Sample distribution across different regions
    if final_labels and len(final_labels) >= nodes:
        print(f"\n🔍 Label Distribution by Region:")
        regions = [
            (0, 100, "Start (0-99)"),
            (nodes//4 - 50, nodes//4 + 50, "Quarter boundary"),
            (nodes//2 - 50, nodes//2 + 50, "Midpoint"),
            (3*nodes//4 - 50, 3*nodes//4 + 50, "3/4 boundary"),
            (nodes - 100, nodes, "End (last 100)")
        ]
        
        for start, end, name in regions:
            start = max(0, int(start))
            end = min(nodes, int(end))
            
            region_labels = {}
            for i in range(start, end):
                label = final_labels[i]
                region_labels[label] = region_labels.get(label, 0) + 1
            
            print(f"\n   {name} (nodes {start}-{end}):")
            for label, count in sorted(region_labels.items()):
                pct = (count / (end - start)) * 100
                print(f"      Label {label:>3}: {count:>4} ({pct:>5.1f}%)")
    
    # Check for label mixing (how well propagated)
    print(f"\n🔬 Label Mixing Analysis:")
    mixing_samples = [0, nodes//8, nodes//4, 3*nodes//8, nodes//2, 5*nodes//8, 3*nodes//4, 7*nodes//8]
    
    print(f"   Sample nodes across graph:")
    for node_id in mixing_samples:
        node_id = int(node_id)
        if node_id < len(final_labels):
            label = final_labels[node_id]
            expected_label = (node_id // (nodes // 4)) * 100
            match = "✅" if label == expected_label else "❌"
            print(f"      Node {node_id:>10}: Label {label:>3} (expected {expected_label:>3}) {match}")
    
    return {
        'nodes': nodes,
        'label_counts': label_counts,
        'total_labeled': total_labeled,
        'all_labeled': total_labeled == nodes
    }

def main():
    """Test label distribution for all sizes"""
    test_sizes = [3000000, 4000000, 4500000, 5000000, 6000000]
    
    results = []
    for size in test_sizes:
        result = test_label_distribution(size)
        if result:
            results.append(result)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"LABEL DISTRIBUTION SUMMARY")
    print(f"{'='*80}")
    print(f"{'Nodes':>12} {'All Labeled':>15} {'Label 0':>12} {'Label 100':>12} {'Label 200':>12} {'Label 300':>12}")
    print(f"{'-'*80}")
    
    for r in results:
        nodes = r['nodes']
        all_ok = "✅" if r['all_labeled'] else "❌"
        counts = r['label_counts']
        l0 = counts.get('0', 0)
        l100 = counts.get('100', 0)
        l200 = counts.get('200', 0)
        l300 = counts.get('300', 0)
        
        print(f"{nodes/1e6:>10.1f}M {all_ok:>15} {l0:>12} {l100:>12} {l200:>12} {l300:>12}")
    
    print(f"{'='*80}")
    
    # Check balance
    print(f"\n✅ VALIDATION CHECKS:")
    all_pass = True
    for r in results:
        nodes = r['nodes']
        counts = r['label_counts']
        expected = nodes / 4
        
        for label in ['0', '100', '200', '300']:
            count = counts.get(label, 0)
            deviation = abs(count - expected) / expected * 100
            
            if deviation > 1.0:  # More than 1% deviation
                print(f"   ⚠️  {nodes/1e6:.1f}M: Label {label} has {deviation:.2f}% deviation from expected")
                all_pass = False
    
    if all_pass:
        print(f"   ✅ All label distributions are balanced (within 1% of expected 25%)")
    
    # Check for propagation issues
    print(f"\n🔍 PROPAGATION CHECK:")
    print(f"   Expected: Labels should be distributed based on graph structure")
    print(f"   Initial seeds: 10% of nodes (every 10th node)")
    print(f"   After 10 iterations: All nodes should have propagated labels")
    print(f"   Distribution: Should reflect connectivity structure (ring topology)")
    

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
