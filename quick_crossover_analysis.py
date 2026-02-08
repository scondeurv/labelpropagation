#!/usr/bin/env python3
"""
Quick crossover analysis using existing data and interpolation
"""
import json
import matplotlib.pyplot as plt
import numpy as np

# Known data points from benchmarks
DATA_POINTS = [
    # (nodes, standalone_ms, burst_total_ms, burst_algo_ms)
    (1000000, 2900, 15610, 6740),
    (10000000, 35180, 14910, 5410),
]

def calculate_speedup(standalone, burst):
    """Calculate speedup"""
    return standalone / burst if burst > 0 else 0

def interpolate_crossover():
    """Find crossover point using linear interpolation"""
    nodes1, s1, b1, _ = DATA_POINTS[0]
    nodes2, s2, b2, _ = DATA_POINTS[1]
    
    speedup1 = calculate_speedup(s1, b1)
    speedup2 = calculate_speedup(s2, b2)
    
    print(f"Point 1: {nodes1:,} nodes -> speedup {speedup1:.2f}x")
    print(f"Point 2: {nodes2:,} nodes -> speedup {speedup2:.2f}x")
    
    # Linear interpolation to find where speedup = 1.0
    # speedup = m * nodes + b
    # We have two points (nodes1, speedup1) and (nodes2, speedup2)
    
    m = (speedup2 - speedup1) / (nodes2 - nodes1)
    b = speedup1 - m * nodes1
    
    # Solve for speedup = 1.0
    # 1.0 = m * crossover + b
    # crossover = (1.0 - b) / m
    
    crossover = (1.0 - b) / m
    
    print(f"\n📍 Estimated crossover point: {int(crossover):,} nodes")
    print(f"   (approximately {crossover/1e6:.2f} million nodes)")
    
    return int(crossover)

def create_interpolated_curve(crossover):
    """Create smooth curves showing the trend"""
    # Generate more points using exponential/polynomial fit
    nodes_range = np.linspace(1e6, 10e6, 50)
    
    # Fit exponential curves to the data
    nodes_data = np.array([p[0] for p in DATA_POINTS])
    standalone_data = np.array([p[1] for p in DATA_POINTS])
    burst_data = np.array([p[2] for p in DATA_POINTS])
    
    # Log-linear fit (works well for algorithmic complexity)
    # time = a * nodes + b
    standalone_fit = np.polyfit(nodes_data, standalone_data, 1)
    burst_fit = np.polyfit(nodes_data, burst_data, 1)
    
    standalone_curve = np.polyval(standalone_fit, nodes_range)
    burst_curve = np.polyval(burst_fit, nodes_range)
    
    return nodes_range, standalone_curve, burst_curve

def plot_crossover_analysis(crossover):
    """Generate comprehensive crossover visualization"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Get interpolated curves
    nodes_curve, standalone_curve, burst_curve = create_interpolated_curve(crossover)
    nodes_curve_M = nodes_curve / 1e6
    
    # Extract data points
    nodes_points = [p[0] / 1e6 for p in DATA_POINTS]
    standalone_points = [p[1] / 1000 for p in DATA_POINTS]
    burst_total_points = [p[2] / 1000 for p in DATA_POINTS]
    burst_algo_points = [p[3] / 1000 for p in DATA_POINTS]
    
    # Plot 1: Execution Time Comparison
    ax1.plot(nodes_curve_M, standalone_curve/1000, '-', linewidth=2, label='Standalone (interpolated)', color='#3498db', alpha=0.6)
    ax1.plot(nodes_curve_M, burst_curve/1000, '-', linewidth=2, label='Burst (interpolated)', color='#e74c3c', alpha=0.6)
    ax1.plot(nodes_points, standalone_points, 'o', markersize=12, label='Standalone (measured)', color='#2980b9', zorder=5)
    ax1.plot(nodes_points, burst_total_points, 's', markersize=12, label='Burst Total (measured)', color='#c0392b', zorder=5)
    ax1.plot(nodes_points, burst_algo_points, '^', markersize=10, label='Burst Algo (measured)', color='#e67e22', zorder=5)
    
    ax1.axvline(x=crossover/1e6, color='green', linestyle='--', linewidth=2.5, label=f'Crossover: {crossover/1e6:.1f}M', alpha=0.8)
    ax1.fill_between(nodes_curve_M, 0, standalone_curve/1000, where=(nodes_curve < crossover), 
                     alpha=0.1, color='blue', label='Standalone faster')
    ax1.fill_between(nodes_curve_M, 0, burst_curve/1000, where=(nodes_curve > crossover), 
                     alpha=0.1, color='red', label='Burst faster')
    
    ax1.set_xlabel('Graph Size (millions of nodes)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Standalone vs Burst: Performance Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.5, 10.5)
    
    # Plot 2: Speedup Over Graph Size
    speedup_curve = standalone_curve / burst_curve
    speedup_points = [s/b for s, b in zip([p[1] for p in DATA_POINTS], [p[2] for p in DATA_POINTS])]
    
    ax2.plot(nodes_curve_M, speedup_curve, '-', linewidth=3, color='purple', label='Speedup (interpolated)')
    ax2.plot(nodes_points, speedup_points, 'o', markersize=12, color='darkviolet', label='Speedup (measured)', zorder=5)
    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Break-even (1x)')
    ax2.axvline(x=crossover/1e6, color='green', linestyle='--', linewidth=2.5, alpha=0.8)
    ax2.fill_between(nodes_curve_M, 0, 1, where=(speedup_curve < 1), alpha=0.2, color='red', label='Standalone wins')
    ax2.fill_between(nodes_curve_M, 1, speedup_curve, where=(speedup_curve > 1), alpha=0.2, color='green', label='Burst wins')
    
    # Add annotations
    for n, s in zip(nodes_points, speedup_points):
        ax2.annotate(f'{s:.2f}x', xy=(n, s), xytext=(5, 5), textcoords='offset points', 
                    fontsize=10, fontweight='bold')
    
    ax2.set_xlabel('Graph Size (millions of nodes)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Speedup (Standalone / Burst)', fontsize=12, fontweight='bold')
    ax2.set_title('Burst Speedup vs Graph Size', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.5, 10.5)
    ax2.set_ylim(0, 3)
    
    # Plot 3: algorithm Time Only (excluding OpenWhisk overhead)
    algo_speedup_points = [s/a for s, a in zip([p[1] for p in DATA_POINTS], [p[3] for p in DATA_POINTS])]
    
    bars = ax3.bar(nodes_points, algo_speedup_points, width=0.8, color=['red' if s < 1 else 'green' for s in algo_speedup_points], 
                   alpha=0.7, edgecolor='black', linewidth=2)
    ax3.axhline(y=1.0, color='black', linestyle='--', linewidth=2)
    ax3.set_xlabel('Graph Size (millions of nodes)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Algorithmic Speedup', fontsize=12, fontweight='bold')
    ax3.set_title('Pure Algorithm Performance (no OpenWhisk overhead)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, algo_speedup_points):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height, f'{val:.2f}x',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 4: OpenWhisk Overhead Analysis
    overheads = [(b - a) / b * 100 for b, a in zip([p[2] for p in DATA_POINTS], [p[3] for p in DATA_POINTS])]
    
    bars = ax4.bar(nodes_points, overheads, width=0.8, color='orange', alpha=0.7, edgecolor='black', linewidth=2)
    ax4.set_xlabel('Graph Size (millions of nodes)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('OpenWhisk Overhead (%)', fontsize=12, fontweight='bold')
    ax4.set_title('OpenWhisk Infrastructure Overhead', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(0, 100)
    
    for bar, val in zip(bars, overheads):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height, f'{val:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('crossover_analysis.png', dpi=150, bbox_inches='tight')
    print("\n📊 Comprehensive analysis saved: crossover_analysis.png")

def generate_report(crossover):
    """Generate text report"""
    print("\n" + "="*80)
    print("CROSSOVER ANALYSIS REPORT")
    print("="*80)
    
    print("\n📊 DATA POINTS:")
    print(f"{'Nodes':<15} {'Standalone':<15} {'Burst Total':<15} {'Burst Algo':<15} {'Speedup':<15}")
    print("-"*80)
    
    for nodes, standalone, burst_total, burst_algo in DATA_POINTS:
        speedup = standalone / burst_total
        print(f"{nodes:>13,}  {standalone/1000:>13.2f}s  {burst_total/1000:>13.2f}s  "
              f"{burst_algo/1000:>13.2f}s  {speedup:>13.2f}x")
    
    print(f"\n📍 CROSSOVER POINT: ~{crossover:,} nodes ({crossover/1e6:.2f}M)")
    print(f"\n💡 INSIGHTS:")
    print(f"   • Below {crossover/1e6:.1f}M nodes: Standalone is faster (overhead dominates)")
    print(f"   • Above {crossover/1e6:.1f}M nodes: Burst is faster (parallelism wins)")
    print(f"   • Algorithmic speedup: {DATA_POINTS[1][1]/DATA_POINTS[1][3]:.2f}x at 10M nodes")
    print(f"   • OpenWhisk overhead: ~60% of total execution time")
    print(f"   • Burst scales better: {DATA_POINTS[1][1]/DATA_POINTS[0][1]:.1f}x time for 10x data (standalone)")
    print(f"   •                     vs {DATA_POINTS[1][2]/DATA_POINTS[0][2]:.1f}x time for 10x data (burst)")
    
    # Save JSON
    data = {
        'crossover_nodes': crossover,
        'crossover_millions': round(crossover / 1e6, 2),
        'data_points': [
            {
                'nodes': p[0],
                'standalone_ms': p[1],
                'burst_total_ms': p[2],
                'burst_algo_ms': p[3],
                'speedup_total': p[1] / p[2],
                'speedup_algo': p[1] / p[3],
                'overhead_pct': (p[2] - p[3]) / p[2] * 100
            } for p in DATA_POINTS
        ]
    }
    
    with open('crossover_data.json', 'w') as f:
        json.dump(data, f, indent=2)
    print("\n💾 Data saved: crossover_data.json")

def main():
    print("╔" + "="*78 + "╗")
    print("║" + " "*25 + "CROSSOVER ANALYSIS" + " "*34 + "║")
    print("╚" + "="*78 + "╝")
    print()
    
    # Calculate crossover
    crossover = interpolate_crossover()
    
    # Generate visualization
    print("\nGenerating comprehensive visualization...")
    plot_crossover_analysis(crossover)
    
    # Generate report
    generate_report(crossover)
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  📊 crossover_analysis.png - Comprehensive 4-panel visualization")
    print("  💾 crossover_data.json - Raw data and metrics")

if __name__ == '__main__':
    main()
