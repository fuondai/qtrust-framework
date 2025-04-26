# Docker Deployment

This document details the Docker deployment configuration for the QTrust blockchain sharding framework.

## Overview

The QTrust framework includes a comprehensive Docker deployment architecture that supports large-scale testing and production environments with 5 regions and 200 nodes per region (1000 nodes total).

## Architecture

The deployment architecture consists of:

- 5 geographic regions
- 10 shards per region (20 nodes per shard)
- Coordinator nodes for each shard
- Cross-region communication channels
- Resource-limited containers for realistic testing

## Base Dockerfile

```dockerfile
FROM python:3.8-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    netcat \
    curl \
    iputils-ping \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY qtrust /app/qtrust
COPY config /app/config

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose ports
EXPOSE 8000 8001 8002

# Default command
CMD ["python", "-m", "qtrust.qtrust_main"]
```

## Docker Compose Template

The base Docker Compose template defines the core services:

```yaml
version: '3.8'

services:
  coordinator:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    image: qtrust/node:latest
    command: python -m qtrust.qtrust_main --role coordinator
    environment:
      - NODE_ID=coordinator
      - REGION=${REGION}
      - SHARD_ID=${SHARD_ID}
      - COORDINATOR_HOST=coordinator
      - COORDINATOR_PORT=8000
    ports:
      - "${COORDINATOR_PORT}:8000"
    volumes:
      - coordinator_data:/app/data
    networks:
      - qtrust_network
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G

  node:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    image: qtrust/node:latest
    command: python -m qtrust.qtrust_main --role validator
    environment:
      - REGION=${REGION}
      - SHARD_ID=${SHARD_ID}
      - COORDINATOR_HOST=coordinator
      - COORDINATOR_PORT=8000
    depends_on:
      - coordinator
    volumes:
      - node_data:/app/data
    networks:
      - qtrust_network
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 2G

networks:
  qtrust_network:
    driver: bridge

volumes:
  coordinator_data:
  node_data:
```

## Configuration Generation

The framework includes a script to generate Docker Compose files for all regions and shards:

```bash
#!/bin/bash
# generate_docker_config.sh

# Configuration
NUM_REGIONS=5
SHARDS_PER_REGION=10
NODES_PER_SHARD=20
BASE_COORDINATOR_PORT=8000

# Create output directory
mkdir -p docker-compose-configs

# Generate region configs
for region_id in $(seq 1 $NUM_REGIONS); do
  region_name="region-${region_id}"
  region_dir="docker-compose-configs/${region_name}"
  mkdir -p $region_dir
  
  # Create region docker-compose file
  cat > $region_dir/docker-compose.yml <<EOF
version: '3.8'

services:
  region-coordinator:
    build:
      context: ../..
      dockerfile: docker/Dockerfile
    image: qtrust/node:latest
    command: python -m qtrust.qtrust_main --role region_coordinator
    environment:
      - NODE_ID=${region_name}-coordinator
      - REGION=${region_id}
    ports:
      - "9${region_id}00:9000"
    volumes:
      - region_coordinator_data:/app/data
    networks:
      - region_network
      - global_network

networks:
  region_network:
    driver: bridge
  global_network:
    external: true

volumes:
  region_coordinator_data:
EOF

  # Generate shard configs
  for shard_id in $(seq 1 $SHARDS_PER_REGION); do
    shard_name="shard-${shard_id}"
    shard_dir="${region_dir}/${shard_name}"
    mkdir -p $shard_dir
    
    coordinator_port=$((BASE_COORDINATOR_PORT + (region_id-1)*100 + shard_id))
    
    # Create shard docker-compose file
    cat > $shard_dir/docker-compose.yml <<EOF
version: '3.8'

services:
  coordinator:
    build:
      context: ../../..
      dockerfile: docker/Dockerfile
    image: qtrust/node:latest
    command: python -m qtrust.qtrust_main --role coordinator
    environment:
      - NODE_ID=${region_name}-${shard_name}-coordinator
      - REGION=${region_id}
      - SHARD_ID=${shard_id}
      - COORDINATOR_HOST=coordinator
      - COORDINATOR_PORT=8000
      - REGION_COORDINATOR_HOST=region-${region_id}-coordinator
      - REGION_COORDINATOR_PORT=9000
    ports:
      - "${coordinator_port}:8000"
    volumes:
      - coordinator_data:/app/data
    networks:
      - shard_network
      - region_network
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
EOF

    # Add node services
    for node_id in $(seq 1 $NODES_PER_SHARD); do
      node_name="node-${node_id}"
      
      cat >> $shard_dir/docker-compose.yml <<EOF
  ${node_name}:
    build:
      context: ../../..
      dockerfile: docker/Dockerfile
    image: qtrust/node:latest
    command: python -m qtrust.qtrust_main --role validator
    environment:
      - NODE_ID=${region_name}-${shard_name}-${node_name}
      - REGION=${region_id}
      - SHARD_ID=${shard_id}
      - COORDINATOR_HOST=coordinator
      - COORDINATOR_PORT=8000
    depends_on:
      - coordinator
    volumes:
      - ${node_name}_data:/app/data
    networks:
      - shard_network
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 2G
EOF
    done
    
    # Add networks and volumes
    cat >> $shard_dir/docker-compose.yml <<EOF

networks:
  shard_network:
    driver: bridge
  region_network:
    external: true

volumes:
  coordinator_data:
EOF

    # Add node volumes
    for node_id in $(seq 1 $NODES_PER_SHARD); do
      node_name="node-${node_id}"
      echo "  ${node_name}_data:" >> $shard_dir/docker-compose.yml
    done
  done
done

# Create global docker-compose file
cat > docker-compose-configs/docker-compose.yml <<EOF
version: '3.8'

networks:
  global_network:
    driver: bridge
EOF

echo "Configuration generated successfully!"
```

## Deployment Script

The framework includes a script to deploy the entire network:

```bash
#!/bin/bash
# deploy.sh

# Create global network
docker network create global_network

# Deploy regions
for region_dir in docker-compose-configs/region-*; do
  region_name=$(basename $region_dir)
  echo "Deploying $region_name..."
  
  # Deploy region coordinator
  docker-compose -f $region_dir/docker-compose.yml up -d
  
  # Deploy shards
  for shard_dir in $region_dir/shard-*; do
    shard_name=$(basename $shard_dir)
    echo "Deploying $region_name/$shard_name..."
    docker-compose -f $shard_dir/docker-compose.yml up -d
  done
done

echo "Deployment complete!"
```

## Node Run Script

For development and testing, the framework includes a script to run individual nodes:

```bash
#!/bin/bash
# run_node.sh

# Default values
ROLE="validator"
REGION="1"
SHARD="1"
NODE_ID=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --role)
      ROLE="$2"
      shift
      shift
      ;;
    --region)
      REGION="$2"
      shift
      shift
      ;;
    --shard)
      SHARD="$2"
      shift
      shift
      ;;
    --node-id)
      NODE_ID="$2"
      shift
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Generate node ID if not provided
if [ -z "$NODE_ID" ]; then
  NODE_ID="region-${REGION}-shard-${SHARD}-node-$(date +%s)"
fi

# Run node
docker run -it --rm \
  --name $NODE_ID \
  -e NODE_ID=$NODE_ID \
  -e REGION=$REGION \
  -e SHARD_ID=$SHARD \
  -e COORDINATOR_HOST=coordinator \
  -e COORDINATOR_PORT=8000 \
  -v ${NODE_ID}_data:/app/data \
  --network qtrust_network \
  qtrust/node:latest \
  python -m qtrust.qtrust_main --role $ROLE
```

## Scaling

The deployment architecture is designed for horizontal scaling:

1. **Region Scaling**: Add more regions by updating the `NUM_REGIONS` parameter
2. **Shard Scaling**: Adjust `SHARDS_PER_REGION` to change the number of shards per region
3. **Node Scaling**: Modify `NODES_PER_SHARD` to change the number of nodes per shard

## Resource Management

The deployment includes resource limits for realistic testing:

```yaml
deploy:
  resources:
    limits:
      cpus: '1'
      memory: 2G
```

These limits can be adjusted based on the available hardware and testing requirements.

## Network Configuration

The deployment uses a hierarchical network structure:

1. **Global Network**: Connects all regions
2. **Region Network**: Connects all shards within a region
3. **Shard Network**: Connects all nodes within a shard

This structure mimics real-world network topology and allows for realistic testing of cross-region and cross-shard transactions.

## Monitoring

The deployment includes ports for monitoring:

- Coordinator nodes expose port 8000 for API access
- Region coordinators expose port 9000 for cross-region communication
- Each node exposes metrics on port 8001 for monitoring

## Data Persistence

The deployment uses Docker volumes for data persistence:

```yaml
volumes:
  coordinator_data:
  node_data:
```

This ensures that blockchain data is preserved across container restarts.
