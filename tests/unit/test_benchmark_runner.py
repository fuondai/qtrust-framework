#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit tests for the benchmark_runner module.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add parent directory to path to import QTrust modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from qtrust.benchmark.benchmark_runner import BenchmarkRunner
from qtrust.benchmark.transaction_generator import TransactionGenerator
from qtrust.benchmark.metrics_collector import MetricsCollector
from qtrust.simulation.network_simulation import NetworkSimulation


class TestBenchmarkRunner(unittest.TestCase):
    """Test cases for the benchmark_runner module."""

    def setUp(self):
        """Set up test environment before each test."""
        # Mock dependencies
        self.mock_simulation = MagicMock(spec=NetworkSimulation)
        self.mock_tx_generator = MagicMock(spec=TransactionGenerator)
        self.mock_metrics_collector = MagicMock(spec=MetricsCollector)
        
        # Patch dependencies
        self.simulation_patcher = patch('qtrust.simulation.network_simulation.NetworkSimulation')
        self.tx_generator_patcher = patch('qtrust.benchmark.transaction_generator.TransactionGenerator')
        self.metrics_collector_patcher = patch('qtrust.benchmark.metrics_collector.MetricsCollector')
        
        # Get patched objects
        self.mock_simulation_class = self.simulation_patcher.start()
        self.mock_tx_generator_class = self.tx_generator_patcher.start()
        self.mock_metrics_collector_class = self.metrics_collector_patcher.start()
        
        # Set return values
        self.mock_simulation_class.return_value = self.mock_simulation
        self.mock_tx_generator_class.return_value = self.mock_tx_generator
        self.mock_metrics_collector_class.return_value = self.mock_metrics_collector
        
        # Set mock behaviors
        self.mock_simulation.initialize.return_value = True
        self.mock_simulation.start_simulation.return_value = True
        self.mock_simulation.stop_simulation.return_value = True
        
        self.mock_tx_generator.start_generation.return_value = True
        self.mock_tx_generator.running = False
        self.mock_tx_generator.total_generated = 100
        self.mock_tx_generator.get_tps.return_value = 10.0
        
        self.mock_metrics_collector.start_collection.return_value = True
        self.mock_metrics_collector.stop_collection.return_value = True
        self.mock_metrics_collector.save_metrics.return_value = "metrics.json"

    def tearDown(self):
        """Clean up after each test."""
        # Stop all patchers
        self.simulation_patcher.stop()
        self.tx_generator_patcher.stop()
        self.metrics_collector_patcher.stop()

    def test_initialization(self):
        """Test initialization of benchmark runner."""
        runner = BenchmarkRunner()
        
        # Check attributes
        self.assertIsNotNone(runner.output_dir)
        self.assertIsNotNone(runner.warmup_duration)
        self.assertIsNotNone(runner.benchmark_duration)
        self.assertIsNotNone(runner.cooldown_duration)
        self.assertIsNotNone(runner.simulation_config)
        self.assertIsNotNone(runner.transaction_config)
        self.assertIsNotNone(runner.metrics_config)

    def test_setup(self):
        """Test setup method."""
        runner = BenchmarkRunner()
        
        # Run setup
        success = runner.setup()
        
        # Check result
        self.assertTrue(success)
        self.mock_simulation_class.assert_called_once()
        self.mock_tx_generator_class.assert_called_once()
        self.mock_metrics_collector_class.assert_called_once()
        self.mock_simulation.initialize.assert_called_once()

    def test_run_benchmark(self):
        """Test run method."""
        # Create runner with mocks
        runner = BenchmarkRunner()
        runner.simulation = self.mock_simulation
        runner.tx_generator = self.mock_tx_generator
        runner.metrics_collector = self.mock_metrics_collector
        runner.output_dir = "test_output"
        
        # Mock time.sleep to avoid actual waiting
        with patch('time.sleep'):
            # Run benchmark
            success = runner.run()
            
            # Check result
            self.assertTrue(success)
            self.mock_simulation.start_simulation.assert_called_once()
            self.mock_metrics_collector.start_collection.assert_called_once()
            self.mock_tx_generator.start_generation.assert_called_once()
            self.mock_metrics_collector.stop_collection.assert_called_once()
            self.mock_simulation.stop_simulation.assert_called_once()


if __name__ == '__main__':
    unittest.main()
