#!/bin/bash

# Script to deploy QTrust across multiple regions
# This script orchestrates the deployment of the entire QTrust network

# Create networks first
echo "Creating Docker networks..."
docker network create qtrust-network

for region in us-east us-west eu-central asia-east australia; do
  docker network create qtrust-${region}-network
done

# Deploy seed nodes first
echo "Deploying seed nodes for inter-region communication..."
docker-compose -f docker-compose-seeds.yml up -d

# Wait for seed nodes to initialize
echo "Waiting for seed nodes to initialize..."
sleep 30

# Deploy each region
echo "Deploying regional nodes..."
for region in us-east us-west eu-central asia-east australia; do
  echo "Deploying region: $region"
  docker-compose -f regions/docker-compose-${region}.yml up -d
  
  # Wait between region deployments to prevent resource contention
  sleep 10
done

echo "QTrust network deployment complete!"
echo "Deployed 5 regions with 200 nodes per region (1000 nodes total)"
echo "To monitor the network, use: docker-compose -f docker-compose.yml logs -f"
