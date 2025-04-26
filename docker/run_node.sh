#!/bin/bash

# Script to run QTrust in a Docker container
# This script simplifies the process of running a single node for development and testing

# Default values
REGION="us-east"
SHARD_ID=0
NODE_ID="dev-node"
ROLE="validator"
PORT=9000

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --region)
      REGION="$2"
      shift
      shift
      ;;
    --shard)
      SHARD_ID="$2"
      shift
      shift
      ;;
    --node)
      NODE_ID="$2"
      shift
      shift
      ;;
    --role)
      ROLE="$2"
      shift
      shift
      ;;
    --port)
      PORT="$2"
      shift
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Build the Docker image
echo "Building QTrust Docker image..."
docker build -t qtrust:latest -f Dockerfile ..

# Run the container
echo "Starting QTrust node with the following configuration:"
echo "  Region: $REGION"
echo "  Shard ID: $SHARD_ID"
echo "  Node ID: $NODE_ID"
echo "  Role: $ROLE"
echo "  Port: $PORT"

docker run -it --rm \
  --name qtrust-$NODE_ID \
  -p $PORT:$PORT \
  -e NODE_ID=$NODE_ID \
  -e REGION=$REGION \
  -e SHARD_ID=$SHARD_ID \
  -e ROLE=$ROLE \
  -e P2P_PORT=$PORT \
  -e LOG_LEVEL=INFO \
  -v qtrust-data:/app/data \
  qtrust:latest
