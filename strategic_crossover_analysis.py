#!/usr/bin/env python3
"""
Strategic crossover analysis - focus on key points around crossover
"""
import subprocess
import sys
import json
import time
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt
import numpy as np

# Known data points
KNOWN_DATA = [
    (1000000, 2900, 15610),
    (10000000, 35180, 14910),
]

# Strategic key points only - focus on crossover region and upper range
STRATEGIC_POINTS = [
    2000000,   # 2M
    4000000,   # 4M (near estimated crossover)
    5000000,   # 5M (past crossover)
    7000000,   # 7M (confirm burst advantage)
    12000000,  # 12M (upper limit)
]

def generate_graph_fast(nodes: int, partitions: int = 4):
    """Generate graph with optimized settings"""
    print(f"  Generating {nodes/1e6:.1f}M node graph...")
    
    result = subprocess.run([
        "python3", "setup_large_lp_data.py",
        "--nodes", str(nodes),
        "--partitions", str(partitions),
        "--density", "20",
        "--communities", "4"
    ], capture_output=True, text=True, timeout=600)
    
    return result.returncode == 0

def run_benchmark_point(nodes: int) -> Optional[Tuple[float, float]]:
    """Run benchmark and return (standalone_ms, burst_ms)"""
    import os
    graph_file = f"large_{nodes}.txt"
    
    # Generate if needed
    if not os.path.exists(graph_file):
        if not generate_graph_fast(nodes):
            return None
    
    print(f"  Running benchmarks...")
    result = subprocess.run([
        "python3", "compare_implementations.py",
        "--nodes", str(nodes),
        "--partitions", "4",
        "--granularity", "1",
        "--iter", "10",
        "--memory", "2048"
    ], capture_output=True, text=True, timeout=1200)
    
    if result.returncode != 0:
        return None
    
    # Parse
    standalone_ms = None
    burst_ms = None
    
    for line in result.stdout.split('\n'):
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
        print(f"  ✓ Standalone: {standalone_ms/1000:.2f}s | Burst: {burst_ms/1000:.2f}s | Speedup: {speedup:.2f}x")
        return (standalone_ms, burst_ms)
    
    return None

def run_strategic_analysis():
    """Run benchmarks on strategic points"""
    results = []
    
    # Add known data
    for nodes, standalone, burst in KNOWN_DATA:
        speedup = standalone / burst
        results.append((nodes, standalone, burst, speedup))
        print(f"\n✓ Using known: {nodes/1e6:.1f}M nodes -> {speedup:.2f}x speedup")
    
    # Test strategic points
    print(f"\n{'='*80}")
    print(f"Testing {len(STRATEGIC_POINTS)} strategic points around crossover")
    print(f"{'='*80}")
    
    for i, nodes in enumerate(STRATEGIC_POINTS, 1):
        print(f"\n[{i}/{len(STRATEGIC_POINTS)}] Testing {nodes/1e6:.1f}M nodes:")
        
        result = run_benchmark_point(nodes)
        if result:
            standalone_ms, burst_ms = result
            speedup = standalone_ms / burst_ms
            results.append((nodes, standalone_ms, burst_ms, speedup))
        else:
            print(f"  ✗ Benchmark failed")
    
    return sorted(results, key=lambda x: x[0])

def find_crossover(results):
    """Find crossover point"""
    for i in range(len(results) - 1):
        nodes1, s1, b1, sp1 = results[i]
        nodes2, s2, b2, sp2 = results[i + 1]
        
        if sp1 < 1.0 and sp2 >= 1.0:
            # Linear interpolation
            fraction = (1.0 - sp1) / (sp2 - sp1)
            crossover = int(nodes1 + fraction * (nodes2 - nodes1))
            
            print(f"\n{'='*80}")
            print(f"📍 PRECISE CROSSOVER FOUND!")
            print(f"{'='*80}")
            print(f"Between: {nodes1/1e6:.2f}M nodes (speedup {sp1:.2f}x)")
            print(f"   and:  {nodes2/1e6:.2f}M nodes (speedup {sp2:.2f}x)")
            print(f"\nCrossover point: {crossover:,} nodes ({crossover/1e6:.2f}M)")
            
            return crossover
    
    return None

def plot_results(results, crossover):
    """Generate comprehensive visualization"""
    nodes = np.array([r[0] / 1e6 for r in results])
    standalone = np.array([r[1] / 1000 for r in results])
    burst = np.array([r[2] / 1000 for r in results])
    speedup = np.array([r[3] for r in results])
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Execution Times
    ax1.plot(nodes, standalone, 'o-', linewidth=2.5, markersize=10, label='Standalone', color='#3498db')
    ax1.plot(nodes, burst, 's-', linewidth=2.5, markersize=10, label='Burst', color='#e74c3c')
    
    if crossover:
        ax1.axvline(x=crossover/1e6, color='green', linestyle='--', linewidth=3, 
                   label=f'Crossover: {crossover/1e6:.2f}M', alpha=0.8)
        ax1.fill_betweenx([0, max(standalone.max(), burst.max())], 0, crossover/1e6, 
                         alpha=0.15, color='blue')
        ax1.fill_betweenx([0, max(standalone.max(), burst.max())], crossover/1e6, nodes.max(), 
                         alpha=0.15, color='red')
    
    ax1.set_xlabel('Graph Size (millions of nodes)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Execution Time (seconds)', fontsize=13, fontweight='bold')
    ax1.set_title('Standalone vs Burst: Performance Comparison', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Speedup
    colors = ['#e74c3c' if s < 1.0 else '#27ae60' for s in speedup]
    bars = ax2.bar(nodes, speedup, width=0.3, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=2.5, label='Break-even (1x)')
    
    if crossover:
        ax2.axvline(x=crossover/1e6, color='green', linestyle='--', linewidth=3, alpha=0.8)
    
    ax2.set_xlabel('Graph Size (millions of nodes)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Speedup (Standalone / Burst)', fontsize=13, fontweight='bold')
    ax2.set_title('Speedup Analysis', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, speedup):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height, f'{val:.2f}x',
                ha='center', va='bottom' if val < 3 else 'top', fontsize=9, fontweight='bold')
    
    # Plot 3: Time Savings
    time_diff = standalone - burst
    colors_diff = ['green' if d > 0 else 'red' for d in time_diff]
    ax3.bar(nodes, time_diff, width=0.3, color=colors_diff, alpha=0.7, edgecolor='black', linewidth=2)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=2)
    if crossover:
        ax3.axvline(x=crossover/1e6, color='green', linestyle='--', linewidth=2, alpha=0.5)
    ax3.set_xlabel('Graph Size (millions of nodes)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Time Savings (seconds)\n(Standalone - Burst)', fontsize=12, fontweight='bold')
    ax3.set_title('Absolute Time Savings with Burst', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Scaling Efficiency
    # Normalize times to 1M baseline
    baseline_idx = 0  # 1M nodes
    standalone_scaled = standalone / standalone[baseline_idx]
    burst_scaled = burst / burst[baseline_idx]
    nodes_scaled = nodes / nodes[baseline_idx]
    
    ax4.plot(nodes, standalone_scaled / nodes_scaled, 'o-', linewidth=2.5, markersize=8, 
            label='Standalone Efficiency', color='#3498db')
    ax4.plot(nodes, burst_scaled / nodes_scaled, 's-', linewidth=2.5, markersize=8, 
            label='Burst Efficiency', color='#e74c3c')
    ax4.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label='Linear scaling')
    
    ax4.set_xlabel('Graph Size (millions of nodes)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Scaling Efficiency\n(normalized to 1M baseline)', fontsize=12, fontweight='bold')
    ax4.set_title('Scaling Efficiency Comparison', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('final_crossover_analysis.png', dpi=150, bbox_inches='tight')
    print("\n📊 Final analysis graph saved: final_crossover_analysis.png")

def save_results(results, crossover):
    """Save comprehensive results"""
    data = {
        'crossover_nodes': crossover,
        'crossover_millions': round(crossover / 1e6, 2) if crossover else None,
        'total_points': len(results),
        'data_points': [
            {
                'nodes': r[0],
                'nodes_M': round(r[0] / 1e6, 2),
                'standalone_ms': r[1],
                'standalone_s': round(r[1] / 1000, 2),
                'burst_ms': r[2],
                'burst_s': round(r[2] / 1000, 2),
                'speedup': round(r[3], 2),
                'time_saved_s': round((r[1] - r[2]) / 1000, 2),
                'winner': 'burst' if r[3] >= 1.0 else 'standalone'
            } for r in results
        ]
    }
    
    with open('final_crossover_results.json', 'w') as f:
        json.dump(data, f, indent=2)
    print("💾 Results saved: final_crossover_results.json")

def print_summary(results, crossover):
    """Print final summary"""
    print("\n" + "="*90)
    print("FINAL CROSSOVER ANALYSIS")
    print("="*90)
    print(f"\n{'Nodes':<12} {'M':<8} {'Standalone':<14} {'Burst':<14} {'Speedup':<12} {'Winner':<12}")
    print("-"*90)
    
    for nodes, standalone, burst, speedup in results:
        winner = "🚀 Burst" if speedup >= 1.0 else "💻 Standalone"
        print(f"{nodes:>10,}  {nodes/1e6:>6.1f}M  {standalone/1000:>12.2f}s  {burst/1000:>12.2f}s  "
              f"{speedup:>10.2f}x  {winner}")
    
    if crossover:
        print(f"\n📍 CROSSOVER: {crossover:,} nodes ({crossover/1e6:.2f}M)")
        print(f"   • Use Standalone for < {crossover/1e6:.2f}M nodes")
        print(f"   • Use Burst for > {crossover/1e6:.2f}M nodes")
    
    # Stats
    burst_wins = sum(1 for r in results if r[3] >= 1.0)
    max_speedup = max(r[3] for r in results)
    max_speedup_nodes = [r[0] for r in results if r[3] == max_speedup][0]
    max_savings = max((r[1] - r[2])/1000 for r in results)
    
    print(f"\n📊 KEY METRICS:")
    print(f"   • Data points: {len(results)}")
    print(f"   • Burst wins: {burst_wins}/{len(results)} ({burst_wins/len(results)*100:.0f}%)")
    print(f"   • Max speedup: {max_speedup:.2f}x at {max_speedup_nodes/1e6:.1f}M nodes")
    print(f"   • Max time savings: {max_savings:.1f}s")

def main():
    print("╔" + "="*88 + "╗")
    print("║" + " "*25 + "STRATEGIC CROSSOVER ANALYSIS" + " "*35 + "║")
    print("╚" + "="*88 + "╝")
    
    start_time = time.time()
    
    try:
        results = run_strategic_analysis()
        
        if len(results) < 2:
            print("\n❌ Insufficient data points")
            return 1
        
        crossover = find_crossover(results)
        print_summary(results, crossover)
        save_results(results, crossover)
        
        print("\n" + "="*90)
        print("GENERATING VISUALIZATIONS")
        print("="*90)
        plot_results(results, crossover)
        
        elapsed = time.time() - start_time
        print(f"\n⏱️  Total time: {elapsed/60:.1f} minutes")
        print("\n✅ ANALYSIS COMPLETE\n")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
        return 1
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
