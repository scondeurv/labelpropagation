#!/usr/bin/env python3
"""
Generate crossover visualization from validated experimental data
"""
import matplotlib.pyplot as plt
import numpy as np

# Validated experimental data
nodes = np.array([3, 4, 4.5, 5, 6])  # In millions
standalone = np.array([16.92, 23.26, 26.39, 29.42, 34.79])  # seconds
burst_total = np.array([23.00, 24.04, 25.59, 23.61, 29.58])  # seconds
burst_algo = np.array([8.28, 9.54, 9.25, 8.96, 9.90])  # seconds
speedup = standalone / burst_total

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 10))

# Main layout: 2x2 grid
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Plot 1: Execution Time Comparison (Top Left)
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(nodes, standalone, 'o-', linewidth=3, markersize=10, 
         label='Standalone', color='#3498db', alpha=0.8)
ax1.plot(nodes, burst_total, 's-', linewidth=3, markersize=10, 
         label='Burst (Total)', color='#e74c3c', alpha=0.8)
ax1.plot(nodes, burst_algo, '^-', linewidth=2, markersize=8, 
         label='Burst (Algorithmic)', color='#2ecc71', alpha=0.7, linestyle='--')

# Mark crossover
crossover_idx = 2  # 4.5M
ax1.axvline(x=nodes[crossover_idx], color='gold', linestyle='--', 
            linewidth=2.5, alpha=0.7, label='Crossover (~4.5M)')
ax1.fill_between(nodes[:crossover_idx+1], 0, 40, alpha=0.1, color='blue')
ax1.fill_between(nodes[crossover_idx:], 0, 40, alpha=0.1, color='red')

ax1.set_xlabel('Graph Size (million nodes)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Execution Time (seconds)', fontsize=13, fontweight='bold')
ax1.set_title('Standalone vs Burst: Performance Comparison', fontsize=15, fontweight='bold')
ax1.legend(fontsize=11, loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 40])

# Plot 2: Speedup Factor (Top Right)
ax2 = fig.add_subplot(gs[0, 1])
colors = ['#e74c3c' if s < 1.0 else '#2ecc71' for s in speedup]
bars = ax2.bar(nodes, speedup, width=0.3, color=colors, alpha=0.7, edgecolor='black')
ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Break-even')

# Label bars with values
for i, (n, s) in enumerate(zip(nodes, speedup)):
    color = 'red' if s < 1.0 else 'green'
    ax2.text(n, s + 0.05, f'{s:.2f}x', ha='center', fontsize=11, 
             fontweight='bold', color=color)

ax2.set_xlabel('Graph Size (million nodes)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Speedup (Standalone / Burst)', fontsize=13, fontweight='bold')
ax2.set_title('Burst Speedup Factor', fontsize=15, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim([0, 1.5])

# Plot 3: Overhead Analysis (Bottom Left)
ax3 = fig.add_subplot(gs[1, 0])
overhead = burst_total - burst_algo
overhead_pct = (overhead / burst_total) * 100

x = np.arange(len(nodes))
width = 0.35

bars1 = ax3.bar(x - width/2, burst_algo, width, label='Algorithm', 
                color='#2ecc71', alpha=0.8)
bars2 = ax3.bar(x + width/2, overhead, width, label='OpenWhisk Overhead', 
                color='#e67e22', alpha=0.8)

ax3.set_xlabel('Graph Size (million nodes)', fontsize=13, fontweight='bold')
ax3.set_ylabel('Time (seconds)', fontsize=13, fontweight='bold')
ax3.set_title('Burst: Algorithm vs Overhead Breakdown', fontsize=15, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels([f'{n:.1f}M' for n in nodes])
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3, axis='y')

# Add percentage labels
for i, (a, o, p) in enumerate(zip(burst_algo, overhead, overhead_pct)):
    ax3.text(i, a + o + 0.5, f'{p:.0f}%\noverhead', ha='center', 
             fontsize=9, fontweight='bold', color='#e67e22')

# Plot 4: Algorithmic Speedup (Bottom Right)
ax4 = fig.add_subplot(gs[1, 1])
algo_speedup = standalone / burst_algo

ax4.plot(nodes, algo_speedup, 'D-', linewidth=3, markersize=10, 
         color='#9b59b6', alpha=0.8, label='Algorithmic Speedup')
ax4.axhline(y=1.0, color='black', linestyle='--', linewidth=2, alpha=0.5)

# Highlight the trend
for i, (n, s) in enumerate(zip(nodes, algo_speedup)):
    ax4.text(n, s + 0.1, f'{s:.2f}x', ha='center', fontsize=11, 
             fontweight='bold', color='#9b59b6')

ax4.set_xlabel('Graph Size (million nodes)', fontsize=13, fontweight='bold')
ax4.set_ylabel('Speedup (Standalone / Burst Algo)', fontsize=13, fontweight='bold')
ax4.set_title('Algorithmic Speedup (Without Overhead)', fontsize=15, fontweight='bold')
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)
ax4.set_ylim([0, 4.5])

# Add text annotation
ax4.text(0.5, 0.95, 'Burst algorithm is ALWAYS faster\nOverhead only matters for small graphs', 
         transform=ax4.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Overall title
fig.suptitle('Label Propagation: Crossover Analysis (Validated Experimental Data)', 
             fontsize=17, fontweight='bold', y=0.98)

plt.savefig('validated_crossover_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Visualization saved: validated_crossover_analysis.png")

# Print summary table
print("\n" + "="*80)
print("VALIDATED CROSSOVER RESULTS")
print("="*80)
print(f"{'Nodes':>10} {'Standalone':>12} {'Burst Total':>12} {'Burst Algo':>11} {'Speedup':>8} {'Winner':>12}")
print("-"*80)

for n, s, bt, ba, sp in zip(nodes, standalone, burst_total, burst_algo, speedup):
    winner = "Burst" if sp > 1.0 else "Standalone"
    marker = "🟢" if sp > 1.0 else "⚪"
    print(f"{n:>8.1f}M {s:>10.2f}s {bt:>10.2f}s {ba:>10.2f}s {sp:>7.2f}x {marker} {winner:>10}")

print("="*80)
print(f"\n📍 CROSSOVER POINT: Between 4.0M and 4.5M nodes")
print(f"   At 4.0M: Speedup = 0.97x (Standalone wins by 3%)")
print(f"   At 4.5M: Speedup = 1.03x (Burst wins by 3%)")
print(f"\n💡 RECOMMENDATION:")
print(f"   < 4M nodes:  Use Standalone (simpler, faster)")
print(f"   > 5M nodes:  Use Burst (significantly faster, better scaling)")
print(f"   4-5M nodes:  Either works (crossover zone)")
print("="*80)
