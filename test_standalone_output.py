#!/usr/bin/env python3
"""Quick test of standalone binary output"""
import subprocess
import json

# Run the binary with 3M nodes  
cmd = ["lpst/target/release/label-propagation", "large_3000000.txt", "3000000", "10"]
result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

if result.returncode != 0:
    print(f"Error: {result.stderr}")
    exit(1)

output_str = result.stdout

# Try to parse as JSON
try:
    data = json.loads(output_str)
    print(" JSON parsing successful!")
    print(f"\nKeys in output: {list(data.keys())}")
    
    if 'labels' in data:
        labels_array = data['labels']
        print(f"Labels array length: {len(labels_array)}")
        print(f"First 20 labels: {labels_array[:20]}")
        print(f"Last 20 labels: {labels_array[-20:]}")
        
        # Count distribution
        from collections import Counter
        counts = Counter(labels_array)
        print(f"\nLabel distribution:")
        for label in sorted(counts):
            print(f"  Label {label}: {counts[label]:,} nodes ({100*counts[label]/len(labels_array):.2f}%)")
    
    if 'execution_time_ms' in data:
        print(f"\nExecution time: {data['execution_time_ms']} ms")
    
except json.JSONDecodeError as e:
    print(f"❌ JSON parsing failed: {e}")
    print(f"\nOutput type: {type(output_str)}")
    print(f"Output length: {len(output_str)} bytes")
    print(f"First 500 chars:\n{output_str[:500]}")
    print(f"\n...\n")
    print(f"Last 500 chars:\n{output_str[-500:]}")
