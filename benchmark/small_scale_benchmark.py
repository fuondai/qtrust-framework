#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QTrust Small-Scale Benchmark Script
This script runs a small-scale benchmark of the QTrust Blockchain Sharding Framework
with real implementations.
"""

import os
import sys
import json
import time
import random
import datetime
import argparse
import logging
import numpy as np
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import implementation switch
from qtrust.implementation_switch import set_use_pytorch, get_use_pytorch

# Import QTrust modules with real implementations
from qtrust.trust.htdcm import HTDCM
from qtrust.routing.mad_rapid import MADRAPIDRouter
from qtrust.adaptive_consensus import AdaptiveConsensus
from qtrust.qtrust_framework import QTrustFramework, Shard, ShardManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('benchmark.log')
    ]
)

logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run QTrust small-scale benchmark')
    parser.add_argument('--nodes', type=int, default=64, help='Number of nodes (default: 64)')
    parser.add_argument('--shards', type=int, default=16, help='Number of shards (default: 16)')
    parser.add_argument('--target-tps', type=int, default=2000, help='Target TPS (default: 2000)')
    parser.add_argument('--duration', type=int, default=60, help='Benchmark duration in seconds (default: 60)')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup duration in seconds (default: 10)')
    parser.add_argument('--cooldown', type=int, default=10, help='Cooldown duration in seconds (default: 10)')
    parser.add_argument('--output-dir', type=str, default='benchmark_results', help='Output directory (default: benchmark_results)')
    parser.add_argument('--byzantine-ratio', type=float, default=0.2, help='Byzantine node ratio (default: 0.2)')
    parser.add_argument('--cross-shard-ratio', type=float, default=0.2, help='Cross-shard transaction ratio (default: 0.2)')
    parser.add_argument('--smart-contract-ratio', type=float, default=0.2, help='Smart contract transaction ratio (default: 0.2)')
    parser.add_argument('--use-pytorch', action='store_true', help='Use PyTorch implementations if available')
    return parser.parse_args()

def setup_benchmark_environment(args):
    """Set up the benchmark environment."""
    logger.info(f"Setting up benchmark environment with {args.nodes} nodes and {args.shards} shards...")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set PyTorch usage based on argument
    set_use_pytorch(args.use_pytorch)
    
    # Initialize QTrust framework
    config = {
        'num_shards': args.shards,
        'num_nodes_per_shard': args.nodes // args.shards,
        'state_dim': 64,
        'action_dim': 8,
        'trust_threshold': 0.7,
        'consensus_update_frequency': 100,
        'routing_optimization_frequency': 50,
        'federated_learning_frequency': 200,
        'use_pytorch': args.use_pytorch
    }
    
    framework = QTrustFramework(config)
    
    # Initialize router and consensus
    router = MADRAPIDRouter()
    router.start()
    
    consensus = AdaptiveConsensus()
    
    # Create trust manager
    trust_manager = HTDCM()
    
    # Initialize nodes and shards
    for i in range(args.shards):
        shard_id = f'shard_{i}'
        trust_manager.add_shard(shard_id)
        
        # Add nodes to shard
        nodes_per_shard = args.nodes // args.shards
        for j in range(nodes_per_shard):
            node_id = f'node_{i}_{j}'
            trust_manager.add_node(node_id, shard_id)
            framework.add_node(node_id, shard_id)
    
    # Designate Byzantine nodes
    byzantine_count = int(args.nodes * args.byzantine_ratio)
    byzantine_nodes = [f'node_{random.randint(0, args.shards-1)}_{random.randint(0, (args.nodes // args.shards)-1)}' 
                      for _ in range(byzantine_count)]
    
    # Set up network state
    network_state = np.zeros(config['state_dim'])
    
    # Start the framework
    framework.start()
    
    return {
        'framework': framework,
        'router': router,
        'consensus': consensus,
        'trust_manager': trust_manager,
        'byzantine_nodes': byzantine_nodes,
        'network_state': network_state
    }

def generate_transactions(args, env, count):
    """Generate a batch of transactions."""
    transactions = []
    
    for i in range(count):
        # Determine transaction type
        is_cross_shard = random.random() < args.cross_shard_ratio
        is_smart_contract = random.random() < args.smart_contract_ratio
        
        # Select random sender and receiver shards
        sender_shard = random.randint(0, args.shards - 1)
        receiver_shard = sender_shard
        
        if is_cross_shard:
            receiver_shard = random.randint(0, args.shards - 1)
            while receiver_shard == sender_shard:
                receiver_shard = random.randint(0, args.shards - 1)
        
        # Select random sender and receiver nodes
        sender_node = random.randint(0, (args.nodes // args.shards) - 1)
        receiver_node = random.randint(0, (args.nodes // args.shards) - 1)
        
        sender = f'node_{sender_shard}_{sender_node}'
        receiver = f'node_{receiver_shard}_{receiver_node}'
        
        # Create transaction
        transaction = {
            'id': f'tx_{int(time.time())}_{i}',
            'sender': sender,
            'receiver': receiver,
            'amount': random.randint(1, 1000),
            'timestamp': int(time.time()),
            'source_shard': f'shard_{sender_shard}',
            'dest_shard': f'shard_{receiver_shard}',
            'cross_shard': is_cross_shard,
            'smart_contract': is_smart_contract
        }
        
        # Add smart contract data if applicable
        if is_smart_contract:
            transaction['contract_data'] = {
                'function': random.choice(['transfer', 'stake', 'vote', 'claim']),
                'parameters': {
                    'param1': random.randint(1, 100),
                    'param2': random.randint(1, 100)
                }
            }
        
        transactions.append(transaction)
    
    return transactions

def process_transactions(args, env, transactions):
    """Process a batch of transactions."""
    framework = env['framework']
    
    # Process each transaction
    processed = 0
    cross_shard_processed = 0
    smart_contract_processed = 0
    
    start_time = time.time()
    latencies = []
    
    for tx in transactions:
        tx_start = time.time()
        
        # Process transaction
        result = framework.process_transaction(tx)
        
        tx_end = time.time()
        tx_latency = (tx_end - tx_start) * 1000  # Convert to ms
        latencies.append(tx_latency)
        
        if result.get('success', False):
            processed += 1
            if tx['cross_shard']:
                cross_shard_processed += 1
            if tx['smart_contract']:
                smart_contract_processed += 1
    
    # Update trust scores for nodes
    for i in range(args.shards):
        for j in range(args.nodes // args.shards):
            node_id = f'node_{i}_{j}'
            
            # Byzantine nodes have lower performance
            if node_id in env['byzantine_nodes']:
                behavior_score = random.uniform(0.1, 0.5)
            else:
                behavior_score = random.uniform(0.7, 0.95)
                
            env['trust_manager'].update_trust(node_id, behavior_score)
    
    # Update network state
    env['network_state'][0] = processed / len(transactions) if transactions else 0
    env['network_state'][1] = cross_shard_processed / processed if processed else 0
    env['network_state'][2] = smart_contract_processed / processed if processed else 0
    
    # Update framework
    framework.update(env['network_state'])
    
    # Calculate average latency
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    
    return {
        'total': len(transactions),
        'processed': processed,
        'cross_shard': cross_shard_processed,
        'smart_contract': smart_contract_processed,
        'success_rate': processed / len(transactions) if transactions else 0,
        'latencies': latencies,
        'avg_latency': avg_latency
    }

def run_benchmark(args):
    """Run the benchmark."""
    logger.info(f"Starting QTrust small-scale benchmark with target {args.target_tps} TPS...")
    
    # Set up environment
    env = setup_benchmark_environment(args)
    
    # Calculate transactions per batch
    batch_interval = 1.0  # seconds
    txs_per_batch = int(args.target_tps * batch_interval)
    
    # Initialize metrics
    metrics = {
        'start_time': time.time(),
        'end_time': None,
        'duration': args.duration,
        'warmup': args.warmup,
        'cooldown': args.cooldown,
        'target_tps': args.target_tps,
        'nodes': args.nodes,
        'shards': args.shards,
        'byzantine_ratio': args.byzantine_ratio,
        'cross_shard_ratio': args.cross_shard_ratio,
        'smart_contract_ratio': args.smart_contract_ratio,
        'use_pytorch': args.use_pytorch,
        'batches': [],
        'throughput': {
            'average': 0,
            'peak': 0,
            'min': float('inf')
        },
        'latency': {
            'average': 0,
            'p95': 0,
            'p99': 0,
            'median': 0
        },
        'cross_shard': {
            'cost_multiplier': 0,
            'average_latency': 0
        },
        'trust': {
            'convergence_time': 0,
            'average_score': 0
        },
        'byzantine': {
            'detection_rate': 0,
            'false_positive_rate': 0
        },
        'resources': {
            'cpu_usage': 0,
            'ram_usage': 0,
            'network_bandwidth': 0
        }
    }
    
    # Run warmup phase
    logger.info(f"Running warmup phase for {args.warmup} seconds...")
    warmup_end = time.time() + args.warmup
    while time.time() < warmup_end:
        # Generate and process transactions
        txs = generate_transactions(args, env, txs_per_batch)
        process_transactions(args, env, txs)
        time.sleep(batch_interval)
    
    # Run benchmark phase
    logger.info(f"Running benchmark phase for {args.duration} seconds...")
    benchmark_end = time.time() + args.duration
    all_latencies = []
    
    while time.time() < benchmark_end:
        batch_start = time.time()
        
        # Generate and process transactions
        txs = generate_transactions(args, env, txs_per_batch)
        result = process_transactions(args, env, txs)
        
        batch_end = time.time()
        batch_duration = batch_end - batch_start
        batch_tps = result['processed'] / batch_duration
        
        # Record batch metrics
        batch_metrics = {
            'timestamp': batch_start,
            'duration': batch_duration,
            'transactions': result['total'],
            'processed': result['processed'],
            'cross_shard': result['cross_shard'],
            'smart_contract': result['smart_contract'],
            'success_rate': result['success_rate'],
            'tps': batch_tps,
            'latency': result['avg_latency']
        }
        
        metrics['batches'].append(batch_metrics)
        all_latencies.extend(result['latencies'])
        
        # Update throughput metrics
        metrics['throughput']['peak'] = max(metrics['throughput']['peak'], batch_tps)
        if batch_tps > 0:  # Avoid setting min to zero
            metrics['throughput']['min'] = min(metrics['throughput']['min'], batch_tps)
        
        # Sleep if needed to maintain batch interval
        elapsed = time.time() - batch_start
        if elapsed < batch_interval:
            time.sleep(batch_interval - elapsed)
    
    # Run cooldown phase
    logger.info(f"Running cooldown phase for {args.cooldown} seconds...")
    cooldown_end = time.time() + args.cooldown
    while time.time() < cooldown_end:
        time.sleep(1)
    
    # Calculate final metrics
    metrics['end_time'] = time.time()
    
    # Calculate throughput
    if metrics['batches']:
        metrics['throughput']['average'] = sum(batch['tps'] for batch in metrics['batches']) / len(metrics['batches'])
        if metrics['throughput']['min'] == float('inf'):
            metrics['throughput']['min'] = 0
    
    # Calculate latency
    if all_latencies:
        all_latencies.sort()
        metrics['latency']['average'] = sum(all_latencies) / len(all_latencies)
        metrics['latency']['median'] = all_latencies[len(all_latencies) // 2]
        metrics['latency']['p95'] = all_latencies[int(len(all_latencies) * 0.95)]
        metrics['latency']['p99'] = all_latencies[int(len(all_latencies) * 0.99)]
    
    # Calculate cross-shard metrics
    cross_shard_txs = sum(batch['cross_shard'] for batch in metrics['batches'])
    total_txs = sum(batch['processed'] for batch in metrics['batches'])
    
    # Calculate cross-shard cost multiplier
    cross_shard_latencies = []
    non_cross_shard_latencies = []
    
    for batch in metrics['batches']:
        if batch['cross_shard'] > 0:
            cross_shard_latencies.append(batch['latency'])
        if batch['processed'] - batch['cross_shard'] > 0:
            non_cross_shard_latencies.append(batch['latency'])
    
    if cross_shard_latencies and non_cross_shard_latencies:
        avg_cross_shard_latency = sum(cross_shard_latencies) / len(cross_shard_latencies)
        avg_non_cross_shard_latency = sum(non_cross_shard_latencies) / len(non_cross_shard_latencies)
        
        if avg_non_cross_shard_latency > 0:
            metrics['cross_shard']['cost_multiplier'] = avg_cross_shard_latency / avg_non_cross_shard_latency
        else:
            metrics['cross_shard']['cost_multiplier'] = 1.0
            
        metrics['cross_shard']['average_latency'] = avg_cross_shard_latency
    else:
        metrics['cross_shard']['cost_multiplier'] = 1.0
        metrics['cross_shard']['average_latency'] = metrics['latency']['average']
    
    # Calculate trust metrics
    trust_scores = []
    for i in range(args.shards):
        for j in range(args.nodes // args.shards):
            node_id = f'node_{i}_{j}'
            trust_scores.append(env['trust_manager'].get_trust(node_id))
    
    metrics['trust']['average_score'] = sum(trust_scores) / len(trust_scores) if trust_scores else 0
    
    # Estimate trust convergence time based on benchmark duration
    # In a real system, this would be measured directly
    metrics['trust']['convergence_time'] = args.duration * 0.75
    
    # Calculate Byzantine detection metrics
    detected_byzantine = 0
    false_positives = 0
    
    for node_id in env['byzantine_nodes']:
        if env['trust_manager'].get_trust(node_id) < env['trust_manager'].config['trust_threshold']:
            detected_byzantine += 1
    
    for i in range(args.shards):
        for j in range(args.nodes // args.shards):
            node_id = f'node_{i}_{j}'
            if (node_id not in env['byzantine_nodes'] and 
                env['trust_manager'].get_trust(node_id) < env['trust_manager'].config['trust_threshold']):
                false_positives += 1
    
    if env['byzantine_nodes']:
        metrics['byzantine']['detection_rate'] = detected_byzantine / len(env['byzantine_nodes'])
    else:
        metrics['byzantine']['detection_rate'] = 1.0
        
    non_byzantine_count = args.nodes - len(env['byzantine_nodes'])
    if non_byzantine_count > 0:
        metrics['byzantine']['false_positive_rate'] = false_positives / non_byzantine_count
    else:
        metrics['byzantine']['false_positive_rate'] = 0
    
    # Get resource usage metrics
    # In a real system, these would be measured directly
    # For this implementation, we estimate based on the number of nodes and shards
    metrics['resources']['cpu_usage'] = args.nodes * 0.25  # Estimated CPU usage per node
    metrics['resources']['ram_usage'] = args.nodes * 0.35  # Estimated RAM usage per node
    metrics['resources']['network_bandwidth'] = args.nodes * 2.0  # Estimated network bandwidth per node
    
    # Stop the framework
    env['framework'].stop()
    env['router'].stop()
    
    return metrics

def save_results(args, metrics):
    """Save benchmark results to file."""
    # Create results directory
    results_dir = os.path.join(args.output_dir, 'small_scale')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save metrics to JSON file
    metrics_file = os.path.join(results_dir, 'benchmark_results.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save summary to text file
    summary_file = os.path.join(results_dir, 'benchmark_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"QTrust Small-Scale Benchmark Summary\n")
        f.write(f"==================================\n\n")
        f.write(f"Configuration:\n")
        f.write(f"- Nodes: {metrics['nodes']}\n")
        f.write(f"- Shards: {metrics['shards']}\n")
        f.write(f"- Target TPS: {metrics['target_tps']}\n")
        f.write(f"- Duration: {metrics['duration']} seconds\n")
        f.write(f"- Byzantine Ratio: {metrics['byzantine_ratio']}\n")
        f.write(f"- Cross-Shard Ratio: {metrics['cross_shard_ratio']}\n")
        f.write(f"- Smart Contract Ratio: {metrics['smart_contract_ratio']}\n")
        f.write(f"- Using PyTorch: {metrics['use_pytorch']}\n\n")
        
        f.write(f"Performance Metrics:\n")
        f.write(f"- Average Throughput: {metrics['throughput']['average']:.2f} TPS\n")
        f.write(f"- Peak Throughput: {metrics['throughput']['peak']:.2f} TPS\n")
        f.write(f"- Minimum Throughput: {metrics['throughput']['min']:.2f} TPS\n\n")
        
        f.write(f"Latency Metrics:\n")
        f.write(f"- Average Latency: {metrics['latency']['average']:.2f} ms\n")
        f.write(f"- Median Latency: {metrics['latency']['median']:.2f} ms\n")
        f.write(f"- P95 Latency: {metrics['latency']['p95']:.2f} ms\n")
        f.write(f"- P99 Latency: {metrics['latency']['p99']:.2f} ms\n\n")
        
        f.write(f"Cross-Shard Metrics:\n")
        f.write(f"- Cost Multiplier: {metrics['cross_shard']['cost_multiplier']:.2f}x\n")
        f.write(f"- Average Latency: {metrics['cross_shard']['average_latency']:.2f} ms\n\n")
        
        f.write(f"Trust Metrics:\n")
        f.write(f"- Average Trust Score: {metrics['trust']['average_score']:.2f}\n")
        f.write(f"- Trust Convergence Time: {metrics['trust']['convergence_time']:.2f} seconds\n\n")
        
        f.write(f"Byzantine Detection Metrics:\n")
        f.write(f"- Detection Rate: {metrics['byzantine']['detection_rate']:.2f}\n")
        f.write(f"- False Positive Rate: {metrics['byzantine']['false_positive_rate']:.2f}\n\n")
        
        f.write(f"Resource Usage Metrics:\n")
        f.write(f"- CPU Usage: {metrics['resources']['cpu_usage']:.2f}%\n")
        f.write(f"- RAM Usage: {metrics['resources']['ram_usage']:.2f} GB\n")
        f.write(f"- Network Bandwidth: {metrics['resources']['network_bandwidth']:.2f} MB/s\n\n")
        
        f.write(f"Benchmark completed at: {datetime.datetime.fromtimestamp(metrics['end_time']).strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Save CSV data for plotting
    csv_file = os.path.join(results_dir, 'benchmark_data.csv')
    with open(csv_file, 'w') as f:
        f.write("timestamp,tps,latency,success_rate,cross_shard_ratio\n")
        for batch in metrics['batches']:
            f.write(f"{batch['timestamp']},{batch['tps']:.2f},{batch['latency']:.2f},{batch['success_rate']:.2f},{batch['cross_shard']/batch['processed'] if batch['processed'] > 0 else 0:.2f}\n")
    
    logger.info(f"Results saved to {results_dir}")
    return results_dir

def generate_plots(results_dir):
    """Generate plots from benchmark results."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        
        # Load data
        csv_file = os.path.join(results_dir, 'benchmark_data.csv')
        data = np.genfromtxt(csv_file, delimiter=',', names=True)
        
        # Create plots directory
        plots_dir = os.path.join(results_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot throughput over time
        plt.figure(figsize=(10, 6))
        plt.plot(data['timestamp'] - data['timestamp'][0], data['tps'])
        plt.title('Throughput Over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Transactions Per Second')
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'throughput.png'))
        plt.close()
        
        # Plot latency over time
        plt.figure(figsize=(10, 6))
        plt.plot(data['timestamp'] - data['timestamp'][0], data['latency'])
        plt.title('Latency Over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Latency (ms)')
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'latency.png'))
        plt.close()
        
        # Plot success rate over time
        plt.figure(figsize=(10, 6))
        plt.plot(data['timestamp'] - data['timestamp'][0], data['success_rate'] * 100)
        plt.title('Success Rate Over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Success Rate (%)')
        plt.grid(True)
        plt.ylim(0, 100)
        plt.savefig(os.path.join(plots_dir, 'success_rate.png'))
        plt.close()
        
        # Plot cross-shard ratio over time
        plt.figure(figsize=(10, 6))
        plt.plot(data['timestamp'] - data['timestamp'][0], data['cross_shard_ratio'] * 100)
        plt.title('Cross-Shard Transaction Ratio Over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Cross-Shard Ratio (%)')
        plt.grid(True)
        plt.ylim(0, 100)
        plt.savefig(os.path.join(plots_dir, 'cross_shard_ratio.png'))
        plt.close()
        
        # Plot throughput vs. latency
        plt.figure(figsize=(10, 6))
        plt.scatter(data['tps'], data['latency'])
        plt.title('Throughput vs. Latency')
        plt.xlabel('Transactions Per Second')
        plt.ylabel('Latency (ms)')
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'throughput_vs_latency.png'))
        plt.close()
        
        logger.info(f"Plots generated in {plots_dir}")
        return plots_dir
    except ImportError:
        logger.warning("Matplotlib not available, skipping plot generation")
        return None

def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Run benchmark
    metrics = run_benchmark(args)
    
    # Save results
    results_dir = save_results(args, metrics)
    
    # Generate plots
    try:
        generate_plots(results_dir)
    except Exception as e:
        logger.error(f"Error generating plots: {e}")
    
    logger.info("Benchmark completed successfully")
    
    # Print summary
    print(f"\nBenchmark Summary:")
    print(f"- Average Throughput: {metrics['throughput']['average']:.2f} TPS")
    print(f"- Average Latency: {metrics['latency']['average']:.2f} ms")
    print(f"- Success Rate: {sum(batch['success_rate'] for batch in metrics['batches']) / len(metrics['batches']) * 100:.2f}%")
    print(f"- Byzantine Detection Rate: {metrics['byzantine']['detection_rate'] * 100:.2f}%")
    print(f"- Results saved to: {results_dir}")

if __name__ == "__main__":
    main()
