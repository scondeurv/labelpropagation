#!/usr/bin/env python3
"""
Generate comprehensive plots from the crossover validation results
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Data extracted from validation run (in seconds)
data = {
    'nodes': [3.0, 4.0, 4.5, 5.0, 6.0],  # Millions
    'standalone': [16.787, 23.079, 26.104, 28.998, 35.125],  # seconds
    'burst_algo': [6.355, 9.567, 9.166, 9.557, 10.870],  # seconds (algorithmic only)
    'burst_total': [19.497, 25.587, 23.431, 25.526, 27.585],  # seconds (includes overhead)
    'overhead': [13.142, 16.020, 14.265, 15.969, 16.715],  # seconds
    'speedup_algo': [2.64, 2.41, 2.85, 3.03, 3.23],  # algorithmic speedup
    'speedup_total': [0.86, 0.90, 1.11, 1.14, 1.27],  # total speedup
}

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 1. Execution Time Comparison
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(data['nodes'], data['standalone'], 'o-', linewidth=2, markersize=8, 
         label='Standalone', color='#2E86AB')
ax1.plot(data['nodes'], data['burst_algo'], 's-', linewidth=2, markersize=8,
         label='Burst (Algorithmic)', color='#A23B72')
ax1.plot(data['nodes'], data['burst_total'], '^-', linewidth=2, markersize=8,
         label='Burst (Total)', color='#F18F01', alpha=0.6)
ax1.set_xlabel('Graph Size (Million Nodes)', fontsize=12)
ax1.set_ylabel('Execution Time (seconds)', fontsize=12)
ax1.set_title('Execution Time vs Graph Size', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 2. Speedup Comparison
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(data['nodes'], data['speedup_algo'], 'o-', linewidth=2, markersize=8,
         label='Algorithmic Speedup', color='#06A77D')
ax2.plot(data['nodes'], data['speedup_total'], 's-', linewidth=2, markersize=8,
         label='Total Speedup', color='#F18F01', alpha=0.7)
ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='No Speedup (1.0x)')
ax2.set_xlabel('Graph Size (Million Nodes)', fontsize=12)
ax2.set_ylabel('Speedup Factor', fontsize=12)
ax2.set_title('Burst vs Standalone Speedup', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# 3. OpenWhisk Overhead
ax3 = fig.add_subplot(gs[1, 0])
overhead_pct = [oh / total * 100 for oh, total in zip(data['overhead'], data['burst_total'])]
bars = ax3.bar(data['nodes'], data['overhead'], width=0.3, color='#D62828', alpha=0.7)
ax3.set_xlabel('Graph Size (Million Nodes)', fontsize=12)
ax3.set_ylabel('Overhead Time (seconds)', fontsize=12)
ax3.set_title('OpenWhisk Infrastructure Overhead', fontsize=14, fontweight='bold')
# Add percentage labels on bars
for i, (bar, pct) in enumerate(zip(bars, overhead_pct)):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
ax3.grid(True, alpha=0.3, axis='y')

# 4. Throughput (nodes/second)
ax4 = fig.add_subplot(gs[1, 1])
nodes_absolute = [n * 1e6 for n in data['nodes']]
throughput_standalone = [n / t for n, t in zip(nodes_absolute, data['standalone'])]
throughput_burst = [n / t for n, t in zip(nodes_absolute, data['burst_algo'])]
ax4.plot(data['nodes'], [t/1000 for t in throughput_standalone], 'o-', linewidth=2, 
         markersize=8, label='Standalone', color='#2E86AB')
ax4.plot(data['nodes'], [t/1000 for t in throughput_burst], 's-', linewidth=2,
         markersize=8, label='Burst', color='#A23B72')
ax4.set_xlabel('Graph Size (Million Nodes)', fontsize=12)
ax4.set_ylabel('Throughput (K nodes/sec)', fontsize=12)
ax4.set_title('Processing Throughput', fontsize=14, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

# 5. Linear Regression Analysis for Standalone
ax5 = fig.add_subplot(gs[2, 0])
# Fit linear model for standalone
z = np.polyfit(data['nodes'], data['standalone'], 1)
p = np.poly1d(z)
r_squared = 1 - (sum((np.array(data['standalone']) - p(np.array(data['nodes'])))**2) / 
                 sum((np.array(data['standalone']) - np.mean(data['standalone']))**2))

ax5.scatter(data['nodes'], data['standalone'], s=100, alpha=0.6, color='#2E86AB')
x_fit = np.linspace(min(data['nodes']), max(data['nodes']), 100)
ax5.plot(x_fit, p(x_fit), 'r--', linewidth=2, 
         label=f'y = {z[0]:.2f}x + {z[1]:.2f}\nR² = {r_squared:.4f}')
ax5.set_xlabel('Graph Size (Million Nodes)', fontsize=12)
ax5.set_ylabel('Standalone Time (seconds)', fontsize=12)
ax5.set_title('Standalone Linear Scaling', fontsize=14, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)

# 6. Summary Statistics Table
ax6 = fig.add_subplot(gs[2, 1])
ax6.axis('off')

# Calculate statistics
avg_speedup_algo = np.mean(data['speedup_algo'])
avg_speedup_total = np.mean(data['speedup_total'])
avg_overhead_pct = np.mean(overhead_pct)
burst_time_variance = np.std(data['burst_algo'])

summary_text = f"""
SUMMARY STATISTICS
{'='*50}

Algorithmic Speedup:
  • Average:    {avg_speedup_algo:.2f}x
  • Range:      {min(data['speedup_algo']):.2f}x - {max(data['speedup_algo']):.2f}x
  • Burst ALWAYS faster algorithmically

Total Speedup (with overhead):
  • Average:    {avg_speedup_total:.2f}x  
  • Range:      {min(data['speedup_total']):.2f}x - {max(data['speedup_total']):.2f}x
  • Crossover at ~4.5M nodes

Infrastructure Overhead:
  • Average:    {avg_overhead_pct:.1f}% of total time
  • Range:      {min(overhead_pct):.1f}% - {max(overhead_pct):.1f}%

Burst Algorithmic Time:
  • Average:    {np.mean(data['burst_algo']):.2f}s
  • Std Dev:    {burst_time_variance:.2f}s
  • Relatively constant (parallel efficiency)

Standalone Scaling:
  • Linear rate: {z[0]:.2f} sec/M nodes
  • R² = {r_squared:.4f} (excellent fit)
"""

ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
         fontsize=10, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Overall title
fig.suptitle('Label Propagation: Burst vs Standalone Performance Analysis', 
             fontsize=16, fontweight='bold', y=0.98)

# Save figure
plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Saved: comprehensive_analysis.png")

# Create a second figure focused on crossover analysis
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Crossover plot - Total time
ax1.plot(data['nodes'], data['standalone'], 'o-', linewidth=2.5, markersize=10,
         label='Standalone', color='#2E86AB')
ax1.plot(data['nodes'], data['burst_total'], 's-', linewidth=2.5, markersize=10,
         label='Burst (Total)', color='#F18F01')
ax1.axvline(x=4.5, color='green', linestyle='--', linewidth=2, alpha=0.5, 
            label='Crossover Point (~4.5M)')
ax1.set_xlabel('Graph Size (Million Nodes)', fontsize=13)
ax1.set_ylabel('Total Execution Time (seconds)', fontsize=13)
ax1.set_title('Crossover Point Analysis\n(Including Infrastructure Overhead)', 
              fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.fill_between([3, 4.5], 0, 40, alpha=0.15, color='orange', 
                  label='Standalone wins')
ax1.fill_between([4.5, 6], 0, 40, alpha=0.15, color='green',
                  label='Burst wins')

# Speedup over scale
ax2.plot(data['nodes'], data['speedup_algo'], 'o-', linewidth=2.5, markersize=10,
         label='Algorithmic Speedup', color='#06A77D')
ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax2.fill_between(data['nodes'], 1, data['speedup_algo'], alpha=0.2, color='green')
ax2.set_xlabel('Graph Size (Million Nodes)', fontsize=13)
ax2.set_ylabel('Speedup Factor (vs Standalone)', fontsize=13)
ax2.set_title('Algorithmic Speedup Trend\n(Pure Execution Time)', 
              fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.text(4.5, 2.5, f'Avg: {avg_speedup_algo:.2f}x', fontsize=12, 
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

plt.tight_layout()
plt.savefig('crossover_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Saved: crossover_analysis.png")

# Print summary to console
print("\n" + "="*70)
print("CROSSOVER VALIDATION RESULTS")
print("="*70)
print(f"\n{'Nodes (M)':<12} {'Standalone':<15} {'Burst (Algo)':<15} {'Speedup':<10}")
print("-"*70)
for i in range(len(data['nodes'])):
    print(f"{data['nodes'][i]:<12.1f} {data['standalone'][i]:<15.2f} "
          f"{data['burst_algo'][i]:<15.2f} {data['speedup_algo'][i]:<10.2f}x")

print("\n" + "="*70)
print("KEY FINDINGS:")
print("="*70)
print(f"✓ No algorithmic crossover - Burst is ALWAYS faster")
print(f"✓ Average algorithmic speedup: {avg_speedup_algo:.2f}x")
print(f"✓ Speedup improves with scale: {min(data['speedup_algo']):.2f}x → {max(data['speedup_algo']):.2f}x")
print(f"✓ Total time crossover at ~4.5M nodes (with {avg_overhead_pct:.0f}% overhead)")
print(f"✓ Burst algorithmic time remains constant: ~{np.mean(data['burst_algo']):.1f}s ± {burst_time_variance:.1f}s")
print("="*70 + "\n")

plt.show()
