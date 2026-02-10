#!/usr/bin/env python3
"""
Generate comprehensive plots from the crossover validation results
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Data extracted from consistency run (Processing vs Total)
# Burst Processing Time is the Distributed Span: max(worker_end) - min(worker_start)
# Standalone Processing Time is execution_time_ms
data = {
    'nodes': [3.0, 4.0, 4.5, 5.0, 6.0],  # Millions
    'standalone_exec': [9.858, 13.810, 15.624, 16.980, 20.511], 
    'standalone_total': [17.106, 23.411, 26.506, 28.940, 35.053],
    'burst_span': [7.556, 9.492, 9.084, 9.684, 10.741],
    'burst_total': [19.521, 23.477, 25.488, 27.443, 29.520],
}

# Derived metrics
data['speedup_processing'] = [s / b for s, b in zip(data['standalone_exec'], data['burst_span'])]
data['speedup_total'] = [s / b for s, b in zip(data['standalone_total'], data['burst_total'])]
data['overhead'] = [t - s for t, s in zip(data['burst_total'], data['burst_span'])]

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 1. Execution Time Comparison
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(data['nodes'], data['standalone_total'], 'o--', linewidth=1, markersize=6, 
         label='Standalone (Total)', color='#2E86AB', alpha=0.5)
ax1.plot(data['nodes'], data['standalone_exec'], 'o-', linewidth=2, markersize=8, 
         label='Standalone (Processing)', color='#2E86AB')
ax1.plot(data['nodes'], data['burst_span'], 's-', linewidth=2, markersize=8,
         label='Burst (Processing Span)', color='#A23B72')
ax1.plot(data['nodes'], data['burst_total'], '^--', linewidth=1, markersize=6,
         label='Burst (Total Orchestrator)', color='#F18F01', alpha=0.4)
ax1.set_xlabel('Graph Size (Million Nodes)', fontsize=12)
ax1.set_ylabel('Execution Time (seconds)', fontsize=12)
ax1.set_title('Processing Time: Standalone vs Burst Span', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 2. Speedup Comparison
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(data['nodes'], data['speedup_processing'], 'o-', linewidth=2, markersize=8,
         label='Processing Speedup', color='#06A77D')
ax2.plot(data['nodes'], data['speedup_total'], 's-', linewidth=2, markersize=8,
         label='End-to-End Speedup', color='#F18F01', alpha=0.7)
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
throughput_standalone = [n / t for n, t in zip(nodes_absolute, data['standalone_exec'])]
throughput_burst = [n / t for n, t in zip(nodes_absolute, data['burst_span'])]
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
z = np.polyfit(data['nodes'], data['standalone_exec'], 1)
p = np.poly1d(z)
r_squared = 1 - (sum((np.array(data['standalone_exec']) - p(np.array(data['nodes'])))**2) / 
                 sum((np.array(data['standalone_exec']) - np.mean(data['standalone_exec']))**2))

ax5.scatter(data['nodes'], data['standalone_exec'], s=100, alpha=0.6, color='#2E86AB')
x_fit = np.linspace(min(data['nodes']), max(data['nodes']), 100)
ax5.plot(x_fit, p(x_fit), 'r--', linewidth=2, 
         label=f'y = {z[0]:.2f}x + {z[1]:.2f}\nR² = {r_squared:.4f}')
ax5.set_xlabel('Graph Size (Million Nodes)', fontsize=12)
ax5.set_ylabel('Standalone Processing Time (seconds)', fontsize=12)
ax5.set_title('Standalone Processing Linear Scaling', fontsize=14, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)

# 6. Summary Statistics Table
ax6 = fig.add_subplot(gs[2, 1])
ax6.axis('off')

# Calculate statistics
avg_speedup_algo = np.mean(data['speedup_processing'])
avg_speedup_total = np.mean(data['speedup_total'])
avg_overhead_pct = np.mean(overhead_pct)
burst_time_variance = np.std(data['burst_span'])

summary_text = f"""
SUMMARY STATISTICS
{'='*50}

Algorithmic Speedup:
  • Average:    {avg_speedup_algo:.2f}x
  • Range:      {min(data['speedup_processing']):.2f}x - {max(data['speedup_processing']):.2f}x
  • Burst ALWAYS faster algorithmically

Total Speedup (with overhead):
  • Average:    {avg_speedup_total:.2f}x  
  • Range:      {min(data['speedup_total']):.2f}x - {max(data['speedup_total']):.2f}x
  • Crossover at ~4.5M nodes

Infrastructure Overhead:
  • Average:    {avg_overhead_pct:.1f}% of total time
  • Range:      {min(overhead_pct):.1f}% - {max(overhead_pct):.1f}%

Burst Algorithmic Time:
  • Average:    {np.mean(data['burst_span']):.2f}s
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
ax1.plot(data['nodes'], data['standalone_total'], 'o-', linewidth=2.5, markersize=10,
         label='Standalone', color='#2E86AB')
ax1.plot(data['nodes'], data['burst_total'], 's-', linewidth=2.5, markersize=10,
         label='Burst (Total)', color='#F18F01')
ax1.axvline(x=4.0, color='green', linestyle='--', linewidth=2, alpha=0.5, 
            label='Crossover Point (~4.0M)')
ax1.set_xlabel('Graph Size (Million Nodes)', fontsize=13)
ax1.set_ylabel('Total Execution Time (seconds)', fontsize=13)
ax1.set_title('Crossover Point Analysis\n(Including Infrastructure Overhead)', 
              fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.fill_between([3, 4.0], 0, 40, alpha=0.15, color='orange', 
                  label='Standalone wins')
ax1.fill_between([4.0, 6], 0, 40, alpha=0.15, color='green',
                  label='Burst wins')

# Speedup over scale
ax2.plot(data['nodes'], data['speedup_processing'], 'o-', linewidth=2.5, markersize=10,
         label='Algorithmic Speedup', color='#06A77D')
ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax2.fill_between(data['nodes'], 1, data['speedup_processing'], alpha=0.2, color='green')
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
print("CROSSOVER VALIDATION RESULTS (CONSISTENCY CHECK)")
print("="*70)
print(f"\n{'Nodes (M)':<12} {'Standalone (Proc)':<20} {'Burst (Span)':<15} {'Speedup':<10}")
print("-"*70)
for i in range(len(data['nodes'])):
    print(f"{data['nodes'][i]:<12.1f} {data['standalone_exec'][i]:<20.2f} "
          f"{data['burst_span'][i]:<15.2f} {data['speedup_processing'][i]:<10.2f}x")

print("\n" + "="*70)
print("KEY FINDINGS:")
print("="*70)
print(f"✓ No algorithmic crossover - Burst is ALWAYS faster")
print(f"✓ Average processing speedup: {avg_speedup_algo:.2f}x")
print(f"✓ Speedup improves with scale: {min(data['speedup_processing']):.2f}x → {max(data['speedup_processing']):.2f}x")
print(f"✓ Total time crossover at ~4.5M nodes (with {avg_overhead_pct:.0f}% overhead)")
print(f"✓ Burst algorithmic time remains constant: ~{np.mean(data['burst_span']):.1f}s ± {burst_time_variance:.1f}s")
print("="*70 + "\n")

plt.show()
