#!/bin/bash

# Script to generate docker-compose files for 5 regions with 200 nodes per region
# This script creates the necessary configuration for a large-scale deployment

# Define constants
REGIONS=("us-east" "us-west" "eu-central" "asia-east" "australia")
NODES_PER_REGION=200
SHARDS_PER_REGION=10  # 20 nodes per shard
BASE_PORT=9000
SEED_PORT=8000

# Create directory for region-specific docker-compose files
mkdir -p ../docker/regions

# Generate seed node configuration for inter-region communication
cat > ../docker/docker-compose-seeds.yml << EOF
version: '3.8'

services:
EOF

# Generate seed nodes (one per region)
for i in "${!REGIONS[@]}"; do
  REGION=${REGIONS[$i]}
  SEED_NODE_PORT=$((SEED_PORT + i))
  
  cat >> ../docker/docker-compose-seeds.yml << EOF
  qtrust-seed-${REGION}:
    extends:
      file: docker-compose-base.yml
      service: qtrust-node-base
    container_name: qtrust-seed-${REGION}
    hostname: qtrust-seed-${REGION}
    environment:
      - NODE_ID=seed-${REGION}
      - REGION=${REGION}
      - ROLE=seed
      - P2P_PORT=${SEED_NODE_PORT}
    ports:
      - "${SEED_NODE_PORT}:${SEED_NODE_PORT}"
    networks:
      - qtrust-network
      - qtrust-${REGION}-network

networks:
EOF

  # Add networks for each region
  for r in "${REGIONS[@]}"; do
    cat >> ../docker/docker-compose-seeds.yml << EOF
  qtrust-${r}-network:
    external: true
EOF
  done

# Generate region-specific docker-compose files
for i in "${!REGIONS[@]}"; do
  REGION=${REGIONS[$i]}
  REGION_FILE="../docker/regions/docker-compose-${REGION}.yml"
  
  # Create the docker-compose file header
  cat > $REGION_FILE << EOF
version: '3.8'

services:
EOF

  # Add seed node reference
  cat >> $REGION_FILE << EOF
  qtrust-seed-${REGION}:
    extends:
      file: ../docker-compose-seeds.yml
      service: qtrust-seed-${REGION}
    networks:
      - qtrust-network
      - qtrust-${REGION}-network
EOF

  # Generate nodes for each shard in the region
  for ((shard=0; shard<SHARDS_PER_REGION; shard++)); do
    # Nodes per shard
    NODES_PER_SHARD=$((NODES_PER_REGION / SHARDS_PER_REGION))
    
    for ((node=0; node<NODES_PER_SHARD; node++)); do
      NODE_ID="${REGION}-s${shard}-n${node}"
      PORT=$((BASE_PORT + i*1000 + shard*100 + node))
      
      # Determine node role (first node in shard is coordinator)
      if [ $node -eq 0 ]; then
        ROLE="coordinator"
      else
        ROLE="validator"
      fi
      
      cat >> $REGION_FILE << EOF
  qtrust-node-${NODE_ID}:
    extends:
      file: ../docker-compose-base.yml
      service: qtrust-node-base
    container_name: qtrust-node-${NODE_ID}
    hostname: qtrust-node-${NODE_ID}
    environment:
      - NODE_ID=${NODE_ID}
      - REGION=${REGION}
      - SHARD_ID=${shard}
      - ROLE=${ROLE}
      - P2P_PORT=${PORT}
      - SEED_NODE=qtrust-seed-${REGION}:$((SEED_PORT + i))
    networks:
      - qtrust-${REGION}-network
EOF
    done
  done

  # Add networks section
  cat >> $REGION_FILE << EOF

networks:
  qtrust-network:
    external: true
  qtrust-${REGION}-network:
    driver: bridge
EOF
done

# Create a master docker-compose file that includes all regions
cat > ../docker/docker-compose.yml << EOF
version: '3.8'

services:
  # This is a placeholder service that ensures networks are created
  # The actual services are defined in the region-specific files
  qtrust-master:
    image: alpine
    command: sh -c "echo 'QTrust network initialized' && sleep 5"
    networks:
      - qtrust-network

networks:
  qtrust-network:
    driver: bridge
EOF

for REGION in "${REGIONS[@]}"; do
  cat >> ../docker/docker-compose.yml << EOF
  qtrust-${REGION}-network:
    driver: bridge
EOF
done

echo "Docker Compose configuration generated for 5 regions with 200 nodes per region"
