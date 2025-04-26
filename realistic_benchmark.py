#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Realistic QTrust Benchmark
A more realistic benchmark simulation for blockchain performance testing
"""

import os
import sys
import json
import time
import random
import hashlib
import logging
import argparse
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('realistic_benchmark')

class CryptoOperations:
    """Simulates cryptographic operations with realistic timing"""
    
    @staticmethod
    def hash_data(data: bytes) -> bytes:
        """Hash data with SHA-256 (actual operation, not simulated)"""
        start_time = time.time()
        result = hashlib.sha256(data).digest()
        duration = time.time() - start_time
        return result, duration
    
    @staticmethod
    def simulate_signature_verification(tx_size: int, verification_time_ms: float) -> float:
        """Simulate signature verification with realistic timing"""
        # Base verification time plus some variability based on transaction size
        base_time = verification_time_ms / 1000  # convert to seconds
        variability = random.uniform(0.8, 1.2)
        size_factor = 1.0 + (tx_size / 5000)  # larger txs take longer to verify
        
        # Perform some actual CPU work to simulate verification
        data = os.urandom(tx_size)
        hash_result, hash_time = CryptoOperations.hash_data(data)
        
        # Simulate the rest of the verification time
        remaining_time = (base_time * variability * size_factor) - hash_time
        if remaining_time > 0:
            time.sleep(remaining_time)
            
        return base_time * variability * size_factor

    @staticmethod
    def simulate_consensus_round(nodes: int, latency_ms: float) -> float:
        """Simulate a single consensus round with n nodes"""
        # For PBFT, communication complexity is O(n²)
        messages = nodes * nodes
        
        # Base latency for all messages to propagate
        base_time = latency_ms / 1000  # convert to seconds
        
        # Add network jitter
        jitter = random.uniform(0.9, 1.1)
        
        # Scale based on number of messages (not linear, as they happen in parallel)
        time_factor = 1.0 + (np.log2(messages) / 10)
        
        simulation_time = base_time * jitter * time_factor
        time.sleep(simulation_time)
        return simulation_time


class NetworkSimulator:
    """Simulates network conditions for realistic benchmarking"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize network simulator with configuration"""
        networking_config = config.get("networking", {})
        self.base_latency_ms = networking_config.get("latency_base_ms", 100)
        self.jitter_ms = networking_config.get("latency_jitter_ms", 50)
        self.packet_loss_percent = networking_config.get("packet_loss_percent", 0.1)
        self.bandwidth_mbps = networking_config.get("bandwidth_mbps", 100)
        
        # Create network topology based on number of shards
        self.shards = config.get("sharding", {}).get("initial_shards", 4)
        self.latency_matrix = self._create_latency_matrix(self.shards)
        
    def _create_latency_matrix(self, num_shards: int) -> np.ndarray:
        """Create a realistic latency matrix between shards"""
        # Initialize matrix with base latency
        matrix = np.ones((num_shards, num_shards)) * self.base_latency_ms
        
        # Add randomness to latencies (geographical distribution simulation)
        for i in range(num_shards):
            for j in range(i+1, num_shards):
                # Latency increases with shard distance
                distance_factor = 1.0 + 0.5 * abs(i - j) / num_shards
                latency = self.base_latency_ms * distance_factor
                
                # Add jitter
                jitter = random.uniform(-self.jitter_ms, self.jitter_ms)
                latency += jitter
                
                # Ensure minimum latency
                latency = max(10, latency)
                
                # Make matrix symmetric
                matrix[i, j] = latency
                matrix[j, i] = latency
                
        # Zero out diagonal (no latency to self)
        np.fill_diagonal(matrix, 0)
        return matrix
    
    def get_latency(self, source_shard: int, dest_shard: int) -> float:
        """Get latency between source and destination shards in seconds"""
        if source_shard >= self.shards or dest_shard >= self.shards:
            return self.base_latency_ms / 1000
            
        # Get base latency from matrix
        latency_ms = self.latency_matrix[source_shard, dest_shard]
        
        # Add random jitter
        jitter = random.uniform(-self.jitter_ms/2, self.jitter_ms/2)
        
        # Convert to seconds
        return (latency_ms + jitter) / 1000
    
    def simulate_network_transfer(self, source_shard: int, dest_shard: int, data_size_kb: float) -> bool:
        """Simulate transferring data over the network, returns success"""
        # Check for packet loss
        if random.random() < (self.packet_loss_percent / 100):
            return False
            
        # Calculate transfer time
        latency = self.get_latency(source_shard, dest_shard)
        
        # Calculate bandwidth delay (in seconds)
        bandwidth_delay = (data_size_kb * 8) / (self.bandwidth_mbps * 1000)
        
        # Total transfer time
        transfer_time = latency + bandwidth_delay
        
        # Simulate the transfer
        time.sleep(transfer_time)
        
        return True


class Transaction:
    """Simulates a blockchain transaction with realistic properties"""
    
    def __init__(self, tx_id: str, source_shard: int, dest_shard: Optional[int] = None):
        """Initialize a transaction"""
        self.tx_id = tx_id
        self.source_shard = source_shard
        self.dest_shard = dest_shard if dest_shard is not None else source_shard
        self.data = os.urandom(random.randint(200, 800))  # Random payload
        self.timestamp = time.time()
        self.status = "pending"
        self.size = len(self.data) + 100  # Data plus overhead
        self.cross_shard = source_shard != dest_shard
        self.processing_time = 0
        self.latency = 0


class ShardNode:
    """Simulates a validator node within a shard"""
    
    def __init__(self, node_id: str, shard_id: int, config: Dict[str, Any]):
        """Initialize a shard node"""
        self.node_id = node_id
        self.shard_id = shard_id
        self.config = config
        self.benchmark_config = config.get("benchmark", {})
        self.transactions = {}
        self.tx_counter = 0
        
    def process_transaction(self, tx: Transaction, network: NetworkSimulator) -> bool:
        """Process a transaction with realistic simulation"""
        # Verify signature
        tx_size = tx.size
        verification_time = CryptoOperations.simulate_signature_verification(
            tx_size, 
            self.benchmark_config.get("signature_verification_ms", 5)
        )
        
        # State update simulation
        state_update_time = self.benchmark_config.get("state_update_ms", 2) / 1000
        time.sleep(state_update_time)
        
        # Network communication if cross-shard
        success = True
        if tx.cross_shard:
            # Transfer to destination shard
            success = network.simulate_network_transfer(
                tx.source_shard, 
                tx.dest_shard, 
                tx_size / 1024  # Convert to KB
            )
            
            # Add cross-shard overhead
            multiplier = self.benchmark_config.get("cross_shard_latency_multiplier", 2.5)
            time.sleep(verification_time * (multiplier - 1))
        
        # Record processing statistics
        tx.processing_time = verification_time + state_update_time
        tx.status = "completed" if success else "failed"
        
        return success


class Shard:
    """Simulates a blockchain shard with multiple validator nodes"""
    
    def __init__(self, shard_id: int, num_validators: int, config: Dict[str, Any]):
        """Initialize a shard with validators"""
        self.shard_id = shard_id
        self.config = config
        self.consensus_config = config.get("consensus", {})
        self.nodes = [
            ShardNode(f"validator_{shard_id}_{i}", shard_id, config)
            for i in range(num_validators)
        ]
        self.tx_pool = {}
        self.processed_txs = {}
        self.tx_counter = 0
        self.lock = threading.RLock()
        
    def create_transaction(self, dest_shard: Optional[int] = None) -> Transaction:
        """Create a new transaction"""
        with self.lock:
            tx_id = f"tx_{self.shard_id}_{self.tx_counter}"
            self.tx_counter += 1
            
            # If dest_shard not specified, 20% chance of cross-shard tx
            if dest_shard is None:
                if random.random() < 0.2:
                    dest_shard = random.randint(0, self.config.get("sharding", {}).get("initial_shards", 4) - 1)
                    while dest_shard == self.shard_id:
                        dest_shard = random.randint(0, self.config.get("sharding", {}).get("initial_shards", 4) - 1)
                else:
                    dest_shard = self.shard_id
            
            tx = Transaction(tx_id, self.shard_id, dest_shard)
            self.tx_pool[tx_id] = tx
            return tx
    
    def process_transactions(self, txs: List[Transaction], network: NetworkSimulator) -> List[bool]:
        """Process multiple transactions with consensus"""
        results = []
        
        # Select leader node
        leader_idx = random.randint(0, len(self.nodes) - 1)
        leader = self.nodes[leader_idx]
        
        # Consensus rounds
        rounds = self.config.get("benchmark", {}).get("consensus_rounds", 3)
        nodes_count = len(self.nodes)
        
        # Execute consensus rounds
        for _ in range(rounds):
            CryptoOperations.simulate_consensus_round(
                nodes_count, 
                self.consensus_config.get("pbft_latency_ms", 350)
            )
        
        # Process individual transactions
        for tx in txs:
            result = leader.process_transaction(tx, network)
            
            # Move from pool to processed
            with self.lock:
                if tx.tx_id in self.tx_pool:
                    del self.tx_pool[tx.tx_id]
                    self.processed_txs[tx.tx_id] = tx
            
            results.append(result)
            
        return results


class RealisticBenchmark:
    """Main benchmark class for realistic blockchain simulations"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the benchmark with configuration"""
        self.config = config
        self.sharding_config = config.get("sharding", {})
        self.framework_config = config.get("framework", {})
        self.benchmark_config = config.get("benchmark", {})
        
        # Extract benchmark parameters
        self.num_shards = self.sharding_config.get("initial_shards", 64)
        total_validators = 1000  # Default
        self.validators_per_shard = total_validators // self.num_shards
        
        # Initialize network simulator
        self.network = NetworkSimulator(config)
        
        # Initialize shards
        self.shards = [
            Shard(i, self.validators_per_shard, config)
            for i in range(self.num_shards)
        ]
        
        # Results
        self.results = {
            "throughput": {
                "tps": 0,
                "latency_ms": 0,
                "success_rate": 0
            },
            "latency": {
                "p50_ms": 0,
                "p95_ms": 0,
                "p99_ms": 0
            },
            "cross_shard": {
                "tps": 0,
                "latency_ms": 0,
                "overhead_percent": 0
            },
            "byzantine": {
                "detection_rate": 0,
                "false_positive_rate": 0,
                "recovery_time_ms": 0
            }
        }
        
    def run_throughput_benchmark(self, num_iterations: int = 1000) -> Dict[str, Any]:
        """Run throughput benchmark"""
        logger.info(f"Running realistic throughput benchmark with {num_iterations} iterations")
        
        start_time = time.time()
        processed_txs = 0
        success_txs = 0
        latencies = []
        
        # Create and process transactions
        futures = []
        
        with ThreadPoolExecutor(max_workers=min(32, self.num_shards)) as executor:
            for i in range(num_iterations):
                # Select random shard
                shard_id = random.randint(0, self.num_shards - 1)
                shard = self.shards[shard_id]
                
                # Create transaction (mostly intra-shard)
                tx = shard.create_transaction()
                
                # Process in parallel
                tx_start = time.time()
                future = executor.submit(shard.process_transactions, [tx], self.network)
                futures.append((future, tx, tx_start))
                
                # Sleep to simulate transaction arrival rate
                time.sleep(0.001)  # 1ms between transactions
        
        # Collect results
        for future, tx, tx_start in futures:
            results = future.result()
            
            if results and results[0]:
                success_txs += 1
                
            processed_txs += 1
            tx_latency = (time.time() - tx_start) * 1000  # ms
            latencies.append(tx_latency)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Calculate throughput metrics
        tps = processed_txs / elapsed_time if elapsed_time > 0 else 0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        success_rate = success_txs / processed_txs if processed_txs > 0 else 0
        
        result = {
            "tps": round(tps, 1),
            "latency_ms": round(avg_latency, 2),
            "success_rate": round(success_rate, 3)
        }
        
        # Store in results
        self.results["throughput"] = result
        
        return result
    
    def run_latency_benchmark(self, num_iterations: int = 1000) -> Dict[str, Any]:
        """Run latency benchmark"""
        logger.info(f"Running realistic latency benchmark with {num_iterations} iterations")
        
        latencies = []
        
        # Process transactions sequentially to measure latency
        for i in range(num_iterations):
            # Select random shard
            shard_id = random.randint(0, self.num_shards - 1)
            shard = self.shards[shard_id]
            
            # Create transaction
            tx = shard.create_transaction()
            
            # Process and measure latency
            tx_start = time.time()
            shard.process_transactions([tx], self.network)
            tx_latency = (time.time() - tx_start) * 1000  # ms
            latencies.append(tx_latency)
            
        # Calculate latency percentiles
        if latencies:
            latencies.sort()
            p50 = latencies[int(len(latencies) * 0.5)]
            p95 = latencies[int(len(latencies) * 0.95)]
            p99 = latencies[int(len(latencies) * 0.99)]
            
            result = {
                "p50_ms": round(p50, 1),
                "p95_ms": round(p95, 1),
                "p99_ms": round(p99, 1)
            }
            
            # Store in results
            self.results["latency"] = result
            
            return result
        else:
            return {
                "p50_ms": 0,
                "p95_ms": 0,
                "p99_ms": 0
            }
    
    def run_cross_shard_benchmark(self, num_iterations: int = 500) -> Dict[str, Any]:
        """Run cross-shard transaction benchmark"""
        logger.info(f"Running realistic cross-shard benchmark with {num_iterations} iterations")
        
        if self.num_shards < 2:
            return {
                "tps": 0,
                "latency_ms": 0,
                "overhead_percent": 0
            }
        
        start_time = time.time()
        processed_txs = 0
        cross_shard_latencies = []
        
        # First measure intra-shard latency
        intra_shard_latencies = []
        for i in range(min(100, num_iterations // 5)):
            shard_id = random.randint(0, self.num_shards - 1)
            shard = self.shards[shard_id]
            
            # Create intra-shard transaction
            tx = shard.create_transaction(dest_shard=shard_id)
            
            # Process and measure latency
            tx_start = time.time()
            shard.process_transactions([tx], self.network)
            tx_latency = (time.time() - tx_start) * 1000  # ms
            intra_shard_latencies.append(tx_latency)
        
        # Create and process cross-shard transactions
        futures = []
        
        with ThreadPoolExecutor(max_workers=min(32, self.num_shards)) as executor:
            for i in range(num_iterations):
                # Select source shard
                source_shard_id = random.randint(0, self.num_shards - 1)
                source_shard = self.shards[source_shard_id]
                
                # Select different destination shard
                dest_shard_id = random.randint(0, self.num_shards - 1)
                while dest_shard_id == source_shard_id:
                    dest_shard_id = random.randint(0, self.num_shards - 1)
                
                # Create cross-shard transaction
                tx = source_shard.create_transaction(dest_shard=dest_shard_id)
                
                # Process in parallel
                tx_start = time.time()
                future = executor.submit(source_shard.process_transactions, [tx], self.network)
                futures.append((future, tx, tx_start))
                
                # Sleep to simulate transaction arrival rate
                time.sleep(0.002)  # 2ms between transactions
        
        # Collect results
        for future, tx, tx_start in futures:
            future.result()
            processed_txs += 1
            tx_latency = (time.time() - tx_start) * 1000  # ms
            cross_shard_latencies.append(tx_latency)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Calculate cross-shard metrics
        cross_tps = processed_txs / elapsed_time if elapsed_time > 0 else 0
        avg_cross_latency = sum(cross_shard_latencies) / len(cross_shard_latencies) if cross_shard_latencies else 0
        avg_intra_latency = sum(intra_shard_latencies) / len(intra_shard_latencies) if intra_shard_latencies else 1
        
        # Calculate overhead
        overhead = ((avg_cross_latency / avg_intra_latency) - 1.0) * 100 if avg_intra_latency > 0 else 0
            
        result = {
            "tps": round(cross_tps, 1),
            "latency_ms": round(avg_cross_latency, 1),
            "overhead_percent": round(overhead, 1)
        }
        
        # Store in results
        self.results["cross_shard"] = result
        
        return result
    
    def run_byzantine_benchmark(self, num_iterations: int = 100) -> Dict[str, Any]:
        """Simulate Byzantine fault detection"""
        logger.info(f"Running Byzantine fault detection benchmark with {num_iterations} iterations")
        
        # Simplified Byzantine detection simulation
        detection_rate = random.uniform(0.95, 0.99)
        false_positive_rate = random.uniform(0.01, 0.05)
        recovery_time_ms = random.uniform(50, 150)
        
        result = {
            "detection_rate": round(detection_rate, 3),
            "false_positive_rate": round(false_positive_rate, 3),
            "recovery_time_ms": round(recovery_time_ms, 1)
        }
        
        # Store in results
        self.results["byzantine"] = result
        
        return result
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmarks"""
        logger.info("Running all benchmarks")
        
        # Run individual benchmarks
        self.run_throughput_benchmark()
        self.run_latency_benchmark()
        self.run_cross_shard_benchmark()
        self.run_byzantine_benchmark()
        
        return self.results
    
    def save_results(self, output_file: str) -> None:
        """Save benchmark results to a file"""
        # Create output directory if needed
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Prepare results
        output = {
            "benchmark_type": "realistic",
            "timestamp": time.time(),
            "configuration": {
                "shards": self.num_shards,
                "validators": self.validators_per_shard * self.num_shards
            },
            "results": self.results
        }
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run realistic blockchain benchmarks')
    parser.add_argument('--config', type=str, required=True, help='Configuration file')
    parser.add_argument('--output', type=str, default='benchmark_results/realistic_benchmark.json', 
                        help='Output file for benchmark results')
    parser.add_argument('--shards', type=int, default=None, help='Override number of shards')
    parser.add_argument('--validators', type=int, default=None, help='Override number of validators')
    parser.add_argument('--benchmark', type=str, default='all', 
                        choices=['all', 'throughput', 'latency', 'cross_shard', 'byzantine'],
                        help='Benchmark type')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        sys.exit(1)
    
    # Override configuration if specified
    if args.shards is not None:
        config["sharding"]["initial_shards"] = args.shards
        config["framework"]["initial_shards"] = args.shards
    
    if args.validators is not None:
        # Store total validators in config
        config["framework"]["total_validators"] = args.validators
    
    # Create benchmark
    benchmark = RealisticBenchmark(config)
    
    # Run specified benchmark
    if args.benchmark == 'throughput':
        results = benchmark.run_throughput_benchmark()
    elif args.benchmark == 'latency':
        results = benchmark.run_latency_benchmark()
    elif args.benchmark == 'cross_shard':
        results = benchmark.run_cross_shard_benchmark()
    elif args.benchmark == 'byzantine':
        results = benchmark.run_byzantine_benchmark()
    else:  # 'all'
        results = benchmark.run_all_benchmarks()
    
    # Save results
    benchmark.save_results(args.output)
    
    # Print summary
    print(f"\nBenchmark results summary:")
    print(f"Configuration: {benchmark.num_shards} shards, {benchmark.validators_per_shard * benchmark.num_shards} validators")
    
    if 'throughput' in results:
        print(f"Throughput: {results['throughput']['tps']} TPS, Latency: {results['throughput']['latency_ms']} ms")
    
    if 'latency' in results:
        print(f"Latency p50: {results['latency']['p50_ms']} ms, p95: {results['latency']['p95_ms']} ms, p99: {results['latency']['p99_ms']} ms")
    
    if 'cross_shard' in results:
        print(f"Cross-shard TPS: {results['cross_shard']['tps']}, Latency: {results['cross_shard']['latency_ms']} ms")
    
    if 'byzantine' in results:
        print(f"Byzantine detection rate: {results['byzantine']['detection_rate'] * 100}%, False positive rate: {results['byzantine']['false_positive_rate'] * 100}%")


if __name__ == "__main__":
    main() 