#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QTrust Benchmark Runner Script
This script runs both large-scale and small-scale benchmarks and compares the results.
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run QTrust benchmarks')
    parser.add_argument('--output-dir', type=str, default='benchmark_results', help='Output directory (default: benchmark_results)')
    parser.add_argument('--skip-large', action='store_true', help='Skip large-scale benchmark')
    parser.add_argument('--skip-small', action='store_true', help='Skip small-scale benchmark')
    parser.add_argument('--compare', action='store_true', help='Compare benchmark results')
    return parser.parse_args()

def run_large_scale_benchmark(args):
    """Run large-scale benchmark."""
    print("\n=== Running Large-Scale Benchmark ===\n")
    
    # Build command
    cmd = [
        'py', '-3.10', 'benchmark/large_scale_benchmark.py',
        '--nodes', '768',
        '--shards', '64',
        '--target-tps', '12000',
        '--duration', '300',
        '--output-dir', args.output_dir
    ]
    
    # Run benchmark
    try:
        subprocess.run(cmd, check=True)
        print("\n=== Large-Scale Benchmark Completed ===\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nError running large-scale benchmark: {e}")
        return False

def run_small_scale_benchmark(args):
    """Run small-scale benchmark."""
    print("\n=== Running Small-Scale Benchmark ===\n")
    
    # Build command
    cmd = [
        'py', '-3.10', 'benchmark/small_scale_benchmark.py',
        '--nodes', '64',
        '--shards', '16',
        '--target-tps', '2000',
        '--duration', '180',
        '--output-dir', args.output_dir
    ]
    
    # Run benchmark
    try:
        subprocess.run(cmd, check=True)
        print("\n=== Small-Scale Benchmark Completed ===\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nError running small-scale benchmark: {e}")
        return False

def compare_benchmark_results(args):
    """Compare benchmark results."""
    print("\n=== Comparing Benchmark Results ===\n")
    
    # Load large-scale results
    large_scale_file = os.path.join(args.output_dir, 'large_scale', 'benchmark_results.json')
    if not os.path.exists(large_scale_file):
        print(f"Large-scale benchmark results not found: {large_scale_file}")
        return False
    
    with open(large_scale_file, 'r') as f:
        large_scale = json.load(f)
    
    # Load small-scale results
    small_scale_file = os.path.join(args.output_dir, 'small_scale', 'benchmark_results.json')
    if not os.path.exists(small_scale_file):
        print(f"Small-scale benchmark results not found: {small_scale_file}")
        return False
    
    with open(small_scale_file, 'r') as f:
        small_scale = json.load(f)
    
    # Create comparison
    comparison = {
        'configuration': {
            'large_scale': {
                'nodes': large_scale['nodes'],
                'shards': large_scale['shards'],
                'target_tps': large_scale['target_tps']
            },
            'small_scale': {
                'nodes': small_scale['nodes'],
                'shards': small_scale['shards'],
                'target_tps': small_scale['target_tps']
            }
        },
        'throughput': {
            'large_scale': {
                'average': large_scale['throughput']['average'],
                'peak': large_scale['throughput']['peak']
            },
            'small_scale': {
                'average': small_scale['throughput']['average'],
                'peak': small_scale['throughput']['peak']
            },
            'ratio': large_scale['throughput']['average'] / small_scale['throughput']['average']
        },
        'latency': {
            'large_scale': {
                'average': large_scale['latency']['average'],
                'p95': large_scale['latency']['p95']
            },
            'small_scale': {
                'average': small_scale['latency']['average'],
                'p95': small_scale['latency']['p95']
            },
            'ratio': large_scale['latency']['average'] / small_scale['latency']['average']
        },
        'cross_shard': {
            'large_scale': {
                'cost_multiplier': large_scale['cross_shard']['cost_multiplier'],
                'average_latency': large_scale['cross_shard']['average_latency']
            },
            'small_scale': {
                'cost_multiplier': small_scale['cross_shard']['cost_multiplier'],
                'average_latency': small_scale['cross_shard']['average_latency']
            }
        },
        'trust': {
            'large_scale': {
                'convergence_time': large_scale['trust']['convergence_time'],
                'average_score': large_scale['trust']['average_score']
            },
            'small_scale': {
                'convergence_time': small_scale['trust']['convergence_time'],
                'average_score': small_scale['trust']['average_score']
            },
            'convergence_ratio': large_scale['trust']['convergence_time'] / small_scale['trust']['convergence_time']
        },
        'byzantine': {
            'large_scale': {
                'detection_rate': large_scale['byzantine']['detection_rate'],
                'false_positive_rate': large_scale['byzantine']['false_positive_rate']
            },
            'small_scale': {
                'detection_rate': small_scale['byzantine']['detection_rate'],
                'false_positive_rate': small_scale['byzantine']['false_positive_rate']
            }
        },
        'resources': {
            'large_scale': {
                'cpu_usage': large_scale['resources']['cpu_usage'],
                'ram_usage': large_scale['resources']['ram_usage'],
                'network_bandwidth': large_scale['resources']['network_bandwidth']
            },
            'small_scale': {
                'cpu_usage': small_scale['resources']['cpu_usage'],
                'ram_usage': small_scale['resources']['ram_usage'],
                'network_bandwidth': small_scale['resources']['network_bandwidth']
            },
            'cpu_ratio': large_scale['resources']['cpu_usage'] / small_scale['resources']['cpu_usage'],
            'ram_ratio': large_scale['resources']['ram_usage'] / small_scale['resources']['ram_usage'],
            'bandwidth_ratio': large_scale['resources']['network_bandwidth'] / small_scale['resources']['network_bandwidth']
        },
        'scaling_efficiency': {
            'node_ratio': large_scale['nodes'] / small_scale['nodes'],
            'shard_ratio': large_scale['shards'] / small_scale['shards'],
            'throughput_ratio': large_scale['throughput']['average'] / small_scale['throughput']['average'],
            'efficiency': (large_scale['throughput']['average'] / small_scale['throughput']['average']) / 
                         (large_scale['nodes'] / small_scale['nodes'])
        }
    }
    
    # Save comparison to file
    comparison_file = os.path.join(args.output_dir, 'benchmark_comparison.json')
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Save summary to text file
    summary_file = os.path.join(args.output_dir, 'benchmark_comparison.txt')
    with open(summary_file, 'w') as f:
        f.write(f"QTrust Benchmark Comparison\n")
        f.write(f"==========================\n\n")
        
        f.write(f"Configuration:\n")
        f.write(f"- Large-Scale: {comparison['configuration']['large_scale']['nodes']} nodes, ")
        f.write(f"{comparison['configuration']['large_scale']['shards']} shards, ")
        f.write(f"{comparison['configuration']['large_scale']['target_tps']} target TPS\n")
        
        f.write(f"- Small-Scale: {comparison['configuration']['small_scale']['nodes']} nodes, ")
        f.write(f"{comparison['configuration']['small_scale']['shards']} shards, ")
        f.write(f"{comparison['configuration']['small_scale']['target_tps']} target TPS\n\n")
        
        f.write(f"Throughput:\n")
        f.write(f"- Large-Scale: {comparison['throughput']['large_scale']['average']:.2f} TPS average, ")
        f.write(f"{comparison['throughput']['large_scale']['peak']:.2f} TPS peak\n")
        
        f.write(f"- Small-Scale: {comparison['throughput']['small_scale']['average']:.2f} TPS average, ")
        f.write(f"{comparison['throughput']['small_scale']['peak']:.2f} TPS peak\n")
        
        f.write(f"- Ratio: {comparison['throughput']['ratio']:.2f}x\n\n")
        
        f.write(f"Latency:\n")
        f.write(f"- Large-Scale: {comparison['latency']['large_scale']['average']:.2f} ms average, ")
        f.write(f"{comparison['latency']['large_scale']['p95']:.2f} ms P95\n")
        
        f.write(f"- Small-Scale: {comparison['latency']['small_scale']['average']:.2f} ms average, ")
        f.write(f"{comparison['latency']['small_scale']['p95']:.2f} ms P95\n")
        
        f.write(f"- Ratio: {comparison['latency']['ratio']:.2f}x\n\n")
        
        f.write(f"Cross-Shard Transactions:\n")
        f.write(f"- Large-Scale: {comparison['cross_shard']['large_scale']['cost_multiplier']:.2f}x cost multiplier, ")
        f.write(f"{comparison['cross_shard']['large_scale']['average_latency']:.2f} ms average latency\n")
        
        f.write(f"- Small-Scale: {comparison['cross_shard']['small_scale']['cost_multiplier']:.2f}x cost multiplier, ")
        f.write(f"{comparison['cross_shard']['small_scale']['average_latency']:.2f} ms average latency\n\n")
        
        f.write(f"Trust Propagation:\n")
        f.write(f"- Large-Scale: {comparison['trust']['large_scale']['convergence_time']:.2f} ms convergence time, ")
        f.write(f"{comparison['trust']['large_scale']['average_score']:.2f} average score\n")
        
        f.write(f"- Small-Scale: {comparison['trust']['small_scale']['convergence_time']:.2f} ms convergence time, ")
        f.write(f"{comparison['trust']['small_scale']['average_score']:.2f} average score\n")
        
        f.write(f"- Convergence Ratio: {comparison['trust']['convergence_ratio']:.2f}x\n\n")
        
        f.write(f"Byzantine Detection:\n")
        f.write(f"- Large-Scale: {comparison['byzantine']['large_scale']['detection_rate']:.2f} detection rate, ")
        f.write(f"{comparison['byzantine']['large_scale']['false_positive_rate']:.2f} false positive rate\n")
        
        f.write(f"- Small-Scale: {comparison['byzantine']['small_scale']['detection_rate']:.2f} detection rate, ")
        f.write(f"{comparison['byzantine']['small_scale']['false_positive_rate']:.2f} false positive rate\n\n")
        
        f.write(f"Resource Usage:\n")
        f.write(f"- Large-Scale: {comparison['resources']['large_scale']['cpu_usage']:.2f}% CPU, ")
        f.write(f"{comparison['resources']['large_scale']['ram_usage']:.2f}% RAM, ")
        f.write(f"{comparison['resources']['large_scale']['network_bandwidth']:.2f} Mbps bandwidth\n")
        
        f.write(f"- Small-Scale: {comparison['resources']['small_scale']['cpu_usage']:.2f}% CPU, ")
        f.write(f"{comparison['resources']['small_scale']['ram_usage']:.2f}% RAM, ")
        f.write(f"{comparison['resources']['small_scale']['network_bandwidth']:.2f} Mbps bandwidth\n")
        
        f.write(f"- Ratios: {comparison['resources']['cpu_ratio']:.2f}x CPU, ")
        f.write(f"{comparison['resources']['ram_ratio']:.2f}x RAM, ")
        f.write(f"{comparison['resources']['bandwidth_ratio']:.2f}x bandwidth\n\n")
        
        f.write(f"Scaling Efficiency:\n")
        f.write(f"- Node Ratio: {comparison['scaling_efficiency']['node_ratio']:.2f}x\n")
        f.write(f"- Shard Ratio: {comparison['scaling_efficiency']['shard_ratio']:.2f}x\n")
        f.write(f"- Throughput Ratio: {comparison['scaling_efficiency']['throughput_ratio']:.2f}x\n")
        f.write(f"- Efficiency: {comparison['scaling_efficiency']['efficiency']:.2f}\n")
        f.write(f"  (1.0 = linear scaling, >1.0 = super-linear, <1.0 = sub-linear)\n")
    
    print(f"Benchmark comparison saved to:")
    print(f"- {comparison_file}")
    print(f"- {summary_file}")
    
    # Print summary
    print("\nScaling Summary:")
    print(f"- Node Ratio: {comparison['scaling_efficiency']['node_ratio']:.2f}x")
    print(f"- Shard Ratio: {comparison['scaling_efficiency']['shard_ratio']:.2f}x")
    print(f"- Throughput Ratio: {comparison['scaling_efficiency']['throughput_ratio']:.2f}x")
    print(f"- Scaling Efficiency: {comparison['scaling_efficiency']['efficiency']:.2f}")
    print(f"  (1.0 = linear scaling, >1.0 = super-linear, <1.0 = sub-linear)")
    
    return True

def main():
    """Main function."""
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run benchmarks
    large_scale_success = True
    small_scale_success = True
    
    if not args.skip_large:
        large_scale_success = run_large_scale_benchmark(args)
    else:
        print("Skipping large-scale benchmark")
    
    if not args.skip_small:
        small_scale_success = run_small_scale_benchmark(args)
    else:
        print("Skipping small-scale benchmark")
    
    # Compare results
    if args.compare and large_scale_success and small_scale_success:
        compare_benchmark_results(args)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
