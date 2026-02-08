#!/usr/bin/env python3
"""
Detailed crossover analysis with multiple data points
Runs benchmarks from 1M to 15M in 0.5M increments
"""
import subprocess
import sys
import json
import time
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt
import numpy as np

def run_benchmark_point(nodes: int, partitions: int = 4) -> Optional[Tuple[float, float, float]]:
    """Run benchmark for a specific node count"""
    print(f"\n{'='*80}")
    print(f"  BENCHMARKING {nodes:,} nodes")
    print(f"{'='*80}")
    
    # Check if graph file exists
    graph_file = f"large_{nodes}.txt"
    import os
    if not os.path.exists(graph_file):
        print(f"⚠️  Graph file not found: {graph_file}")
        print(f"   Generating graph...")
        result = subprocess.run([
            "python3", "setup_large_lp_data.py",
            "--nodes", str(nodes),
            "--partitions", str(partitions),
            "--density", "20",
            "--communities", "4"
        ], capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            print(f"✗ Graph generation failed")
            return None
        print(f"✓ Graph generated")
    else:
        print(f"✓ Using existing graph file")
    
    # Run comparison benchmark
    print(f"Running comparison...")
    result = subprocess.run([
        "python3", "compare_implementations.py",
        "--nodes", str(nodes),
        "--partitions", str(partitions),
        "--granularity", "1",
        "--iter", "10",
        "--memory", "2048"
    ], capture_output=True, text=True, timeout=1200)
    
    if result.returncode != 0:
        print(f"✗ Benchmark failed")
        return None
    
    # Parse output
    output = result.stdout
    standalone_ms = None
    burst_ms = None
    
    for line in output.split('\n'):
        if "Standalone:" in line and "LPST" not in line:
            try:
                time_str = line.split(':')[1].strip()
                if 's' in time_str:
                    standalone_ms = float(time_str.replace('s', '')) * 1000
            except:
                pass
        elif "Burst (total):" in line:
            try:
                time_str = line.split(':')[1].strip()
                if 's' in time_str:
                    burst_ms = float(time_str.replace('s', '')) * 1000
            except:
                pass
    
    if standalone_ms and burst_ms:
        speedup = standalone_ms / burst_ms
        print(f"\n✓ Results:")
        print(f"  Standalone: {standalone_ms/1000:.2f}s")
        print(f"  Burst:      {burst_ms/1000:.2f}s")
        print(f"  Speedup:    {speedup:.2f}x")
        
        return (standalone_ms, burst_ms, speedup)
    
    print("✗ Could not parse results")
    return None

def run_detailed_analysis(start_nodes: int, end_nodes: int, increment: int) -> List[Tuple[int, float, float, float]]:
    """Run benchmarks across the specified range"""
    results = []
    node_counts = list(range(start_nodes, end_nodes + 1, increment))
    
    print(f"\n{'#'*80}")
    print(f"# Running {len(node_counts)} benchmarks from {start_nodes/1e6:.1f}M to {end_nodes/1e6:.1f}M")
    print(f"# Increment: {increment/1e6:.1f}M nodes")
    print(f"{'#'*80}")
    
    for i, nodes in enumerate(node_counts, 1):
        print(f"\n{'█'*80}")
        print(f"█ Progress: {i}/{len(node_counts)} - Testing {nodes/1e6:.1f}M nodes")
        print(f"{'█'*80}")
        
        result = run_benchmark_point(nodes)
        if result:
            standalone_ms, burst_ms, speedup = result
            results.append((nodes, standalone_ms, burst_ms, speedup))
            
            # Show running comparison
            if len(results) > 1:
                prev = results[-2]
                print(f"\n  Trend: Speedup changed from {prev[3]:.2f}x → {speedup:.2f}x")
        else:
            print(f"⚠️  Skipping {nodes:,} nodes due to benchmark failure")
    
    return results

def find_crossover(results: List[Tuple[int, float, float, float]]) -> Optional[int]:
    """Find crossover point from results"""
    results_sorted = sorted(results, key=lambda x: x[0])
    
    for i in range(len(results_sorted) - 1):
        nodes1, s1, b1, sp1 = results_sorted[i]
        nodes2, s2, b2, sp2 = results_sorted[i + 1]
        
        if sp1 < 1.0 and sp2 >= 1.0:
            # Linear interpolation
            fraction = (1.0 - sp1) / (sp2 - sp1)
            crossover = int(nodes1 + fraction * (nodes2 - nodes1))
            
            print(f"\n{'='*80}")
            print(f"📍 CROSSOVER FOUND!")
            print(f"{'='*80}")
            print(f"Between: {nodes1/1e6:.1f}M nodes (speedup {sp1:.2f}x)")
            print(f"   and:  {nodes2/1e6:.1f}M nodes (speedup {sp2:.2f}x)")
            print(f"Crossover point: ~{crossover:,} nodes ({crossover/1e6:.2f}M)")
            
            return crossover
    
    return None

def plot_detailed_results(results: List[Tuple[int, float, float, float]], crossover: Optional[int]):
    """Generate detailed visualization"""
    results_sorted = sorted(results, key=lambda x: x[0])
    
    nodes = np.array([r[0] / 1e6 for r in results_sorted])
    standalone = np.array([r[1] / 1000 for r in results_sorted])
    burst = np.array([r[2] / 1000 for r in results_sorted])
    speedup = np.array([r[3] for r in results_sorted])
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Execution Times (Large)
    ax1 = fig.add_subplot(gs[0:2, 0])
    ax1.plot(nodes, standalone, 'o-', linewidth=2.5, markersize=8, label='Standalone', color='#3498db')
    ax1.plot(nodes, burst, 's-', linewidth=2.5, markersize=8, label='Burst', color='#e74c3c')
    
    if crossover:
        ax1.axvline(x=crossover/1e6, color='green', linestyle='--', linewidth=3, 
                   label=f'Crossover: {crossover/1e6:.2f}M', alpha=0.8)
        ax1.fill_betweenx([0, max(standalone.max(), burst.max())], 0, crossover/1e6, 
                         alpha=0.1, color='blue', label='Standalone faster')
        ax1.fill_betweenx([0, max(standalone.max(), burst.max())], crossover/1e6, nodes.max(), 
                         alpha=0.1, color='red', label='Burst faster')
    
    ax1.set_xlabel('Graph Size (millions of nodes)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Execution Time (seconds)', fontsize=13, fontweight='bold')
    ax1.set_title('Standalone vs Burst: Detailed Performance Comparison', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Speedup (Large)
    ax2 = fig.add_subplot(gs[0:2, 1])
    colors = ['#e74c3c' if s < 1.0 else '#27ae60' for s in speedup]
    bars = ax2.bar(nodes, speedup, width=0.4, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=2.5, label='Break-even (1x)', zorder=10)
    
    if crossover:
        ax2.axvline(x=crossover/1e6, color='green', linestyle='--', linewidth=3, alpha=0.8)
    
    ax2.set_xlabel('Graph Size (millions of nodes)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Speedup (Standalone / Burst)', fontsize=13, fontweight='bold')
    ax2.set_title('Detailed Speedup Analysis', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val, n in zip(bars, speedup, nodes):
        if val > ax2.get_ylim()[1] * 0.9:
            va = 'top'
            offset = -5
        else:
            va = 'bottom'
            offset = 5
        ax2.text(bar.get_x() + bar.get_width()/2., val, f'{val:.2f}x',
                ha='center', va=va, fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Plot 3: Time Difference
    ax3 = fig.add_subplot(gs[2, 0])
    time_diff = standalone - burst
    colors_diff = ['green' if d > 0 else 'red' for d in time_diff]
    ax3.bar(nodes, time_diff, width=0.4, color=colors_diff, alpha=0.7, edgecolor='black')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=2)
    ax3.set_xlabel('Graph Size (millions of nodes)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Time Difference (s)\n(Standalone - Burst)', fontsize=12, fontweight='bold')
    ax3.set_title('Absolute Time Savings', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Data Table
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis('tight')
    ax4.axis('off')
    
    table_data = [['Nodes (M)', 'Standalone', 'Burst', 'Speedup']]
    for n, s, b, sp in zip(nodes, standalone, burst, speedup):
        color = '🟢' if sp >= 1.0 else '🔴'
        table_data.append([f'{n:.1f}', f'{s:.1f}s', f'{b:.1f}s', f'{color} {sp:.2f}x'])
    
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.2, 0.25, 0.25, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('Detailed Results Summary', fontsize=13, fontweight='bold', pad=20)
    
    plt.savefig('detailed_crossover_analysis.png', dpi=150, bbox_inches='tight')
    print("\n📊 Detailed graph saved: detailed_crossover_analysis.png")

def save_results(results: List[Tuple[int, float, float, float]], crossover: Optional[int]):
    """Save results to JSON"""
    data = {
        'crossover_nodes': crossover,
        'crossover_millions': round(crossover / 1e6, 2) if crossover else None,
        'num_data_points': len(results),
        'data_points': [
            {
                'nodes': r[0],
                'nodes_millions': round(r[0] / 1e6, 2),
                'standalone_ms': r[1],
                'burst_ms': r[2],
                'speedup': r[3],
                'time_saved_ms': r[1] - r[2],
                'winner': 'burst' if r[3] >= 1.0 else 'standalone'
            } for r in sorted(results, key=lambda x: x[0])
        ]
    }
    
    with open('detailed_crossover_results.json', 'w') as f:
        json.dump(data, f, indent=2)
    print("💾 Detailed results saved: detailed_crossover_results.json")

def print_summary(results: List[Tuple[int, float, float, float]], crossover: Optional[int]):
    """Print summary table"""
    print("\n" + "="*90)
    print("DETAILED CROSSOVER ANALYSIS SUMMARY")
    print("="*90)
    print()
    print(f"{'Nodes':<12} {'Millions':<10} {'Standalone':<12} {'Burst':<12} {'Speedup':<12} {'Winner':<12}")
    print("-"*90)
    
    for nodes, standalone, burst, speedup in sorted(results, key=lambda x: x[0]):
        winner = "🚀 Burst" if speedup >= 1.0 else "💻 Standalone"
        millions = f"{nodes/1e6:.1f}M"
        print(f"{nodes:>10,}  {millions:<10} {standalone/1000:>10.2f}s  {burst/1000:>10.2f}s  "
              f"{speedup:>10.2f}x  {winner}")
    
    print()
    if crossover:
        print(f"📍 CROSSOVER POINT: {crossover:,} nodes ({crossover/1e6:.2f}M)")
        print(f"   • Below {crossover/1e6:.2f}M: Standalone is faster")
        print(f"   • Above {crossover/1e6:.2f}M: Burst is faster")
    
    # Statistics
    burst_wins = sum(1 for r in results if r[3] >= 1.0)
    standalone_wins = len(results) - burst_wins
    max_speedup = max(r[3] for r in results)
    max_speedup_nodes = [r[0] for r in results if r[3] == max_speedup][0]
    
    print(f"\n📊 STATISTICS:")
    print(f"   • Total data points: {len(results)}")
    print(f"   • Burst wins: {burst_wins} ({burst_wins/len(results)*100:.1f}%)")
    print(f"   • Standalone wins: {standalone_wins} ({standalone_wins/len(results)*100:.1f}%)")
    print(f"   • Maximum speedup: {max_speedup:.2f}x at {max_speedup_nodes/1e6:.1f}M nodes")

def main():
    print("╔" + "="*88 + "╗")
    print("║" + " "*25 + "DETAILED CROSSOVER ANALYSIS" + " "*36 + "║")
    print("║" + " "*20 + "1M to 15M nodes in 0.5M increments" + " "*33 + "║")
    print("╚" + "="*88 + "╝")
    
    start_time = time.time()
    
    try:
        # Run analysis from 1M to 15M in 0.5M increments
        results = run_detailed_analysis(
            start_nodes=1000000,
            end_nodes=15000000,
            increment=500000
        )
        
        if not results:
            print("\n❌ No successful benchmarks")
            return 1
        
        # Find crossover
        crossover = find_crossover(results)
        
        # Print summary
        print_summary(results, crossover)
        
        # Save results
        save_results(results, crossover)
        
        # Generate plots
        print("\n" + "="*90)
        print("GENERATING VISUALIZATIONS")
        print("="*90)
        plot_detailed_results(results, crossover)
        
        elapsed = time.time() - start_time
        print(f"\n⏱️  Total analysis time: {elapsed/60:.1f} minutes ({elapsed:.0f}s)")
        print("\n" + "="*90)
        print("✅ DETAILED CROSSOVER ANALYSIS COMPLETE")
        print("="*90)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
