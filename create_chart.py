import matplotlib.pyplot as plt
import json
import numpy as np
import os

# Import data from JSON files
max_data = json.load(open('benchmark_results/max_benchmark.json'))
ultra_data = json.load(open('benchmark_results/ultra_performance.json'))

# Prepare data for visualization
labels = ['Max (768 validators)', 'Ultra (1024 validators)']
tps_values = [max_data['results']['throughput']['tps'], ultra_data['results']['throughput']['tps']]

# Generate visualization
plt.figure(figsize=(10, 6))
plt.bar(labels, tps_values, color=['blue', 'orange'])
plt.title('QTrust Performance Comparison')
plt.ylabel('Transactions Per Second (TPS)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels
for i, v in enumerate(tps_values):
    plt.text(i, v + 1000, f'{v:,.0f}', ha='center', fontweight='bold')

# Save visualization
plt.tight_layout()
plt.savefig('benchmark_results/performance_comparison.png')

print('Chart has been successfully saved to benchmark_results/performance_comparison.png') 