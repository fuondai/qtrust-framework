"""
System-level performance tests for QTrust framework.

This module contains system-level tests that verify the performance
characteristics of the QTrust framework under various conditions.
"""

import os
import sys
import time
import unittest
import json
from typing import Dict, List, Any

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from qtrust.qtrust_framework import QTrustFramework
from qtrust.benchmark.benchmark_runner import BenchmarkRunner
from qtrust.benchmark.metrics_collector import MetricsCollector


class TestSystemPerformance(unittest.TestCase):
    """System-level performance tests for QTrust framework."""

    def setUp(self):
        """Set up test environment."""
        self.framework = QTrustFramework()
        self.benchmark_runner = BenchmarkRunner()
        self.metrics_collector = MetricsCollector()
        
        # Create results directory if it doesn't exist
        os.makedirs('benchmark_results', exist_ok=True)

    def test_small_scale_performance(self):
        """Test performance with small-scale configuration (16 shards, 192 nodes)."""
        config = {
            'shards': 16,
            'nodes_per_shard': 12,
            'cross_shard_ratio': 0.2,
            'transaction_count': 10000,
            'duration_seconds': 60
        }
        
        results = self.benchmark_runner.run_benchmark(config)
        
        # Save results
        with open('benchmark_results/small_scale_results.json', 'w') as f:
            json.dump(results, f, indent=2)
            
        # Verify performance meets requirements
        self.assertGreaterEqual(results['throughput']['avg_tps'], 3000)
        self.assertLessEqual(results['latency']['median_seconds'], 2.0)
        self.assertGreaterEqual(results['efficiency']['percent'], 85)

    def test_medium_scale_performance(self):
        """Test performance with medium-scale configuration (32 shards, 384 nodes)."""
        config = {
            'shards': 32,
            'nodes_per_shard': 12,
            'cross_shard_ratio': 0.2,
            'transaction_count': 20000,
            'duration_seconds': 120
        }
        
        results = self.benchmark_runner.run_benchmark(config)
        
        # Save results
        with open('benchmark_results/medium_scale_results.json', 'w') as f:
            json.dump(results, f, indent=2)
            
        # Verify performance meets requirements
        self.assertGreaterEqual(results['throughput']['avg_tps'], 6000)
        self.assertLessEqual(results['latency']['median_seconds'], 1.5)
        self.assertGreaterEqual(results['efficiency']['percent'], 85)

    def test_large_scale_performance(self):
        """Test performance with large-scale configuration (64 shards, 768 nodes)."""
        config = {
            'shards': 64,
            'nodes_per_shard': 12,
            'cross_shard_ratio': 0.2,
            'transaction_count': 50000,
            'duration_seconds': 180
        }
        
        results = self.benchmark_runner.run_benchmark(config)
        
        # Save results
        with open('benchmark_results/large_scale_results.json', 'w') as f:
            json.dump(results, f, indent=2)
            
        # Verify performance meets requirements
        self.assertGreaterEqual(results['throughput']['avg_tps'], 12400)
        self.assertLessEqual(results['latency']['median_seconds'], 1.2)
        self.assertGreaterEqual(results['efficiency']['percent'], 86.9)

    def test_cross_shard_transaction_performance(self):
        """Test performance with varying cross-shard transaction ratios."""
        results = {}
        
        for ratio in [0.1, 0.2, 0.3, 0.4, 0.5]:
            config = {
                'shards': 32,
                'nodes_per_shard': 12,
                'cross_shard_ratio': ratio,
                'transaction_count': 20000,
                'duration_seconds': 120
            }
            
            benchmark_results = self.benchmark_runner.run_benchmark(config)
            results[f'ratio_{ratio}'] = benchmark_results
        
        # Save results
        with open('benchmark_results/cross_shard_results.json', 'w') as f:
            json.dump(results, f, indent=2)
            
        # Verify performance meets requirements even at high cross-shard ratios
        self.assertGreaterEqual(results['ratio_0.5']['throughput']['avg_tps'], 4000)
        self.assertLessEqual(results['ratio_0.5']['latency']['median_seconds'], 2.0)

    def test_byzantine_fault_tolerance(self):
        """Test performance under Byzantine conditions."""
        results = {}
        
        for byzantine_percent in [0, 5, 10, 15, 20, 25]:
            config = {
                'shards': 32,
                'nodes_per_shard': 12,
                'cross_shard_ratio': 0.2,
                'transaction_count': 20000,
                'duration_seconds': 120,
                'byzantine_percent': byzantine_percent
            }
            
            benchmark_results = self.benchmark_runner.run_benchmark(config)
            results[f'byzantine_{byzantine_percent}'] = benchmark_results
        
        # Save results
        with open('benchmark_results/byzantine_results.json', 'w') as f:
            json.dump(results, f, indent=2)
            
        # Verify performance meets requirements even with 25% Byzantine nodes
        baseline_tps = results['byzantine_0']['throughput']['avg_tps']
        byzantine_tps = results['byzantine_25']['throughput']['avg_tps']
        
        # Should maintain at least 80% throughput with 25% Byzantine nodes
        self.assertGreaterEqual(byzantine_tps / baseline_tps, 0.8)


if __name__ == '__main__':
    unittest.main()
