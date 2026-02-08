#!/usr/bin/env python3
"""
Plot crossover results: Standalone vs Burst Label Propagation performance
"""
import matplotlib.pyplot as plt
import argparse
import sys

def parse_results(results_file):
    """Parse results file with format: nodes lpst_time burst_time speedup"""
    nodes = []
    lpst_times = []
    burst_times = []
    speedups = []
    
    try:
        with open(results_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    nodes.append(int(parts[0]))
                    lpst_times.append(float(parts[1]))
                    burst_times.append(float(parts[2]))
                    speedups.append(float(parts[3]))
    except FileNotFoundError:
        print(f"Error: Results file not found: {results_file}", file=sys.stderr)
        return None, None, None, None
    except Exception as e:
        print(f"Error parsing results: {e}", file=sys.stderr)
        return None, None, None, None
    
    return nodes, lpst_times, burst_times, speedups

def plot_crossover(nodes, lpst_times, burst_times, speedups, output_file="crossover_plot.png"):
    """Generate crossover plot"""
    if not nodes:
        print("No data to plot", file=sys.stderr)
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Execution time comparison
    ax1.plot(nodes, lpst_times, 'o-', label='Standalone (LPST)', linewidth=2, markersize=8)
    ax1.plot(nodes, burst_times, 's-', label='Burst (Distributed)', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Nodes', fontsize=12)
    ax1.set_ylabel('Execution Time (ms)', fontsize=12)
    ax1.set_title('Label Propagation: Standalone vs Burst', fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    # Find and mark crossover point
    for i in range(len(speedups)):
        if speedups[i] > 1.0:
            ax1.axvline(x=nodes[i], color='green', linestyle='--', alpha=0.5, linewidth=2)
            ax1.text(nodes[i], max(max(lpst_times), max(burst_times)) * 0.5, 
                    f'Crossover\n{nodes[i]} nodes', 
                    rotation=0, ha='center', fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
            break
    
    # Plot 2: Speedup
    ax2.plot(nodes, speedups, 'D-', color='purple', linewidth=2, markersize=8)
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Break-even (1.0x)')
    ax2.set_xlabel('Number of Nodes', fontsize=12)
    ax2.set_ylabel('Speedup (Standalone / Burst)', fontsize=12)
    ax2.set_title('Burst Speedup Factor', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.fill_between(nodes, 1.0, speedups, where=[s > 1.0 for s in speedups], 
                     alpha=0.3, color='green', label='Burst faster')
    ax2.fill_between(nodes, speedups, 1.0, where=[s < 1.0 for s in speedups], 
                     alpha=0.3, color='red', label='Standalone faster')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for i, n in enumerate(nodes):
        status = "✓ BURST FASTER" if speedups[i] > 1.0 else "✗ Standalone faster"
        print(f"{n:>10} nodes: LPST={lpst_times[i]:>8.0f}ms, Burst={burst_times[i]:>8.0f}ms, "
              f"Speedup={speedups[i]:>5.2f}x {status}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot LP crossover results")
    parser.add_argument("--input", type=str, default="/tmp/crossover_results.txt", 
                       help="Results file")
    parser.add_argument("--output", type=str, default="crossover_plot.png", 
                       help="Output plot file")
    
    args = parser.parse_args()
    
    nodes, lpst_times, burst_times, speedups = parse_results(args.input)
    
    if nodes:
        plot_crossover(nodes, lpst_times, burst_times, speedups, args.output)
    else:
        print("Failed to generate plot", file=sys.stderr)
        sys.exit(1)
