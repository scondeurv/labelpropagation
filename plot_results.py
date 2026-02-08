#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import sys

def main():
    csv_file = "final_benchmark_results.csv"
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: {csv_file} not found.")
        return
    
    if df.empty:
        print("CSV is empty.")
        return

    # Filter out rows with zero or missing values
    df = df[df['Standalone_ms'] > 0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Execution Time
    ax1.plot(df['Nodes'], df['Standalone_ms'], 'o-', label='Standalone (Rust)', linewidth=2)
    ax1.plot(df['Nodes'], df['Burst_ms'], 's-', label='Burst (Distributed)', linewidth=2)
    ax1.set_xlabel('Number of Nodes')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Execution Time: Standalone vs Burst')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Plot 2: Speedup
    ax2.plot(df['Nodes'], df['Speedup'], 'D-', color='green', linewidth=2)
    ax2.axhline(y=1.0, color='red', linestyle='--', label='Crossover (1x)')
    ax2.set_xlabel('Number of Nodes')
    ax2.set_ylabel('Speedup (x)')
    ax2.set_title('Burst Speedup over Standalone')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('final_crossover_results.png')
    print("Graph saved as final_crossover_results.png")

if __name__ == "__main__":
    main()
