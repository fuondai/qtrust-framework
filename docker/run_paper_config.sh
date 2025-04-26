#!/bin/bash

# QTrust Paper Configuration Runner
# This script runs the QTrust framework with the paper configuration
# 64 shards, 1000 nodes, 5 regions with 200 nodes each

# Set up environment
set -e
cd "$(dirname "$0")/.."
mkdir -p benchmark_results

echo "=== Building QTrust Docker Images ==="
docker build -t qtrust .

echo "=== Starting QTrust with Paper Configuration ==="
echo "Configuration: 64 shards, 1000 nodes, 5 regions with 200 nodes each"

# Run with docker-compose
docker-compose -f docker/docker-compose-paper.yml up -d

echo "=== QTrust Started with Paper Configuration ==="
echo "Wait for all regions to initialize and connect (this might take a few minutes)..."
sleep 60

echo "=== Running Benchmark ==="
# Execute benchmark on the control node
docker exec qtrust-control python qtrust_main.py --benchmark all --config /app/config/paper_large_scale.json --output /app/benchmark_results/paper_benchmark_results.json

# Copy results to host
echo "=== Copying Benchmark Results ==="
docker cp qtrust-control:/app/benchmark_results/paper_benchmark_results.json ./benchmark_results/

echo "=== Benchmark Complete ==="
echo "Results saved to: ./benchmark_results/paper_benchmark_results.json"

# Option to shutdown after benchmark
read -p "Shutdown the QTrust cluster? (y/n): " shutdown
if [[ $shutdown == "y" ]]; then
    echo "=== Shutting Down QTrust Cluster ==="
    docker-compose -f docker/docker-compose-paper.yml down
    echo "=== QTrust Cluster Shutdown Complete ==="
else
    echo "=== QTrust Cluster Left Running ==="
    echo "You can manually shut it down with: docker-compose -f docker/docker-compose-paper.yml down"
fi 