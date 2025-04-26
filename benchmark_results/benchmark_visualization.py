#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QTrust Benchmark Visualization Tool
Generate visualizations from benchmark results
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any

# Benchmark result files
BENCHMARK_FILES = [
    "throughput_test.json",
    "latency_test.json", 
    "cross_shard_test.json",
    "byzantine_test.json",
    "large_scale_test.json",
    "extreme_scale_test.json",
    "paper_config_test.json",
    "final_paper_benchmark.json"
]

def load_benchmark_data(dir_path='./'):
    """Load benchmark data from JSON files"""
    results = {}
    
    for filename in BENCHMARK_FILES:
        filepath = os.path.join(dir_path, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                try:
                    data = json.load(f)
                    # Use filename (without extension) as key
                    key = os.path.splitext(filename)[0]
                    results[key] = data
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from {filepath}")
    
    return results

def plot_throughput_comparison(data):
    """Generate throughput comparison chart across configurations"""
    # Prepare data
    configs = []
    tps_values = []
    latency_values = []
    
    for key, benchmark in data.items():
        if "results" in benchmark and "throughput" in benchmark["results"]:
            # Extract configuration information
            shards = benchmark.get("configuration", {}).get("shards", 0)
            validators = benchmark.get("configuration", {}).get("validators", 0)
            config_name = f"{shards} shards, {validators} validators"
            
            # Extract results
            tps = benchmark["results"]["throughput"].get("tps", 0)
            latency = benchmark["results"]["throughput"].get("latency_ms", 0)
            
            configs.append(config_name)
            tps_values.append(tps)
            latency_values.append(latency)
    
    # Create visualization
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Plot bars for TPS
    x = np.arange(len(configs))
    width = 0.35
    rects1 = ax1.bar(x - width/2, tps_values, width, label='TPS', color='blue', alpha=0.7)
    ax1.set_ylabel('Transactions Per Second (TPS)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Add secondary axis for latency
    ax2 = ax1.twinx()
    ax2.plot(x, latency_values, 'ro-', linewidth=2, label='Latency (ms)')
    ax2.set_ylabel('Latency (ms)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Configure common elements
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=45, ha='right')
    ax1.set_title('Throughput and Latency Comparison Across Different Configurations')
    
    # Add value labels
    for i, v in enumerate(tps_values):
        ax1.text(i - width/2, v + 100, f"{v}", ha='center')
    
    for i, v in enumerate(latency_values):
        ax2.text(i, v + 0.1, f"{v:.2f}", ha='center')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    fig.tight_layout()
    plt.savefig('throughput_comparison.png')
    
def plot_comparison_with_competitors():
    """Generate comparative analysis chart versus competing blockchain solutions"""
    # Comparative data from research paper
    solutions = ['QTrust', 'Ethereum 2.0', 'Polkadot', 'Harmony', 'Zilliqa']
    tps = [12400, 8900, 11000, 8500, 7600]
    latency = [1.2, 5.0, 4.0, 3.5, 4.2]
    
    # Create visualization
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot bars for TPS
    x = np.arange(len(solutions))
    width = 0.35
    rects1 = ax1.bar(x - width/2, tps, width, label='TPS', color='green', alpha=0.7)
    ax1.set_ylabel('Transactions Per Second (TPS)', color='green')
    ax1.tick_params(axis='y', labelcolor='green')
    
    # Add secondary axis for latency
    ax2 = ax1.twinx()
    ax2.bar(x + width/2, latency, width, label='Latency (ms)', color='orange', alpha=0.7)
    ax2.set_ylabel('Latency (ms)', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    
    # Configure common elements
    ax1.set_xticks(x)
    ax1.set_xticklabels(solutions)
    ax1.set_title('Performance Comparison with Competing Solutions')
    
    # Add value labels
    for i, v in enumerate(tps):
        ax1.text(i - width/2, v + 100, f"{v}", ha='center')
    
    for i, v in enumerate(latency):
        ax2.text(i + width/2, v + 0.1, f"{v:.1f}", ha='center')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    fig.tight_layout()
    plt.savefig('competitor_comparison.png')

def plot_byzantine_detection():
    """Visualize Byzantine attack detection performance metrics"""
    # Sample data
    configs = ['8 shards', '16 shards', '32 shards', '64 shards']
    detection_rates = [0.997, 0.999, 0.999, 1.0]
    false_positive_rates = [0.027, 0.018, 0.012, 0.001]
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot line chart for detection rates
    ax.plot(configs, detection_rates, 'go-', linewidth=2, label='Detection Rate')
    
    # Plot line chart for false positive rates
    ax.plot(configs, false_positive_rates, 'ro-', linewidth=2, label='False Positive Rate')
    
    # Set y-axis limits from 0 to 1.1
    ax.set_ylim(0, 1.1)
    
    # Add labels and title
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Rate')
    ax.set_title('Byzantine Attack Detection Performance')
    
    # Add value labels
    for i, v in enumerate(detection_rates):
        ax.text(i, v + 0.02, f"{v:.3f}", ha='center')
    
    for i, v in enumerate(false_positive_rates):
        ax.text(i, v - 0.05, f"{v:.3f}", ha='center')
    
    # Add legend
    ax.legend()
    
    fig.tight_layout()
    plt.savefig('byzantine_detection.png')

def main():
    """Main function to generate all visualization charts"""
    # Create directory for charts if it doesn't exist
    os.makedirs('charts', exist_ok=True)
    
    # Path to directory containing benchmark result files
    benchmark_dir = './'
    
    # Load data
    data = load_benchmark_data(benchmark_dir)
    
    # Generate visualizations
    plot_throughput_comparison(data)
    plot_comparison_with_competitors()
    plot_byzantine_detection()
    
    print("Charts have been successfully generated in the 'charts' directory")

if __name__ == "__main__":
    main() 