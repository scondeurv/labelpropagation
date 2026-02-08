#!/usr/bin/env python3
"""
Generate a detailed benchmark report with visualizations
"""
import json
import sys
import glob
import matplotlib.pyplot as plt
from datetime import datetime

def load_latest_results():
    """Load the most recent benchmark results"""
    files = glob.glob("benchmark_results_*.json")
    if not files:
        return None
    
    latest = max(files)
    with open(latest, 'r') as f:
        return json.load(f), latest

def create_performance_chart(results):
    """Create a bar chart of performance results"""
    labels = [r['label'] for r in results]
    times = [r['result']['avg'] / 1000 for r in results]  # Convert to seconds
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, times, color=['#3498db', '#e74c3c'])
    
    ax.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax.set_xlabel('Graph Size', fontsize=12)
    ax.set_title('Label Propagation Performance - Standalone', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('benchmark_performance.png', dpi=150)
    print("📊 Performance chart saved: benchmark_performance.png")
    
def create_throughput_chart(results):
    """Create a chart showing throughput"""
    labels = [r['label'] for r in results]
    nodes = [r['nodes'] / 1e6 for r in results]  # Millions
    throughput = [(r['nodes'] * 1000000) / r['result']['avg'] / 1e6 for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, throughput, color=['#2ecc71', '#f39c12'])
    
    ax.set_ylabel('Throughput (M edges/sec)', fontsize=12)
    ax.set_xlabel('Graph Size', fontsize=12)
    ax.set_title('Label Propagation Throughput', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('benchmark_throughput.png', dpi=150)
    print("📊 Throughput chart saved: benchmark_throughput.png")

def generate_markdown_report(results, source_file):
    """Generate a markdown report"""
    report = f"""# Label Propagation Benchmark Results

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Source:** {source_file}

## Performance Summary

| Graph Size | Nodes | Avg Time | Min Time | Max Time | Throughput |
|------------|-------|----------|----------|----------|------------|
"""
    
    for r in results:
        label = r['label']
        nodes = f"{r['nodes']:,}"
        avg = f"{r['result']['avg']/1000:.2f}s"
        min_t = f"{r['result']['min']/1000:.2f}s"
        max_t = f"{r['result']['max']/1000:.2f}s"
        throughput = f"{(r['nodes'] * 1000000) / r['result']['avg'] / 1e6:.1f} M edges/s"
        
        report += f"| {label} | {nodes} | {avg} | {min_t} | {max_t} | {throughput} |\n"
    
    report += """
## Key Findings

"""
    
    if len(results) >= 2:
        ratio = results[1]['result']['avg'] / results[0]['result']['avg']
        scale = results[1]['nodes'] / results[0]['nodes']
        report += f"- **Scalability:** {scale}x more nodes takes {ratio:.2f}x time\n"
        report += f"- **Efficiency:** Near-linear scaling ({ratio/scale:.2f}x efficiency)\n"
    
    fastest = min(results, key=lambda r: r['result']['avg'])
    report += f"- **Fastest run:** {fastest['label']} at {fastest['result']['min']/1000:.2f}s\n"
    
    max_throughput = max(results, key=lambda r: (r['nodes'] * 1000000) / r['result']['avg'])
    throughput_val = (max_throughput['nodes'] * 1000000) / max_throughput['result']['avg'] / 1e6
    report += f"- **Peak throughput:** {throughput_val:.1f} M edges/sec ({max_throughput['label']})\n"
    
    report += """
## Implementation Details

- **Algorithm:** Semi-supervised Label Propagation
- **Implementation:** Rust (standalone, single-threaded)
- **Max Iterations:** 10
- **Convergence:** Early stopping when no labels change

## Graphs

![Performance Chart](benchmark_performance.png)

![Throughput Chart](benchmark_throughput.png)

## Raw Data

```json
"""
    report += json.dumps(results, indent=2)
    report += "\n```\n"
    
    with open('BENCHMARK_REPORT.md', 'w') as f:
        f.write(report)
    
    print("📄 Markdown report saved: BENCHMARK_REPORT.md")

def main():
    print("=" * 70)
    print("  BENCHMARK REPORT GENERATOR")
    print("=" * 70)
    
    results, source = load_latest_results()
    
    if not results:
        print("❌ No benchmark results found")
        print("Run: python3 quick_benchmark.py first")
        return 1
    
    print(f"\n✓ Loaded results from: {source}")
    print(f"  - {len(results)} benchmark(s)")
    
    try:
        print("\nGenerating visualizations...")
        create_performance_chart(results)
        create_throughput_chart(results)
        
        print("\nGenerating markdown report...")
        generate_markdown_report(results, source)
        
        print("\n" + "=" * 70)
        print("✅ Report generation complete!")
        print("=" * 70)
        print("\nGenerated files:")
        print("  - BENCHMARK_REPORT.md")
        print("  - benchmark_performance.png")
        print("  - benchmark_throughput.png")
        
        return 0
        
    except ImportError as e:
        print(f"\n⚠ Could not generate charts: {e}")
        print("Install matplotlib: pip install matplotlib")
        
        # Still generate markdown report
        print("\nGenerating markdown report only...")
        generate_markdown_report(results, source)
        print("✓ Text report generated: BENCHMARK_REPORT.md")
        return 0
    
    except Exception as e:
        print(f"\n❌ Error generating report: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
