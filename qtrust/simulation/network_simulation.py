#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QTrust Blockchain Sharding Framework - Network Simulation
This module implements realistic WAN environment simulation using tc and netem.
"""

import os
import subprocess
import time
import json
import random
from typing import Dict, List, Any, Optional, Tuple


class NetworkSimulator:
    """
    Simulates realistic WAN environment using tc and netem.
    Configures per-region latency, packet loss, and jitter.
    """

    def __init__(self, config_file: str = None):
        """
        Initialize the network simulator.

        Args:
            config_file: Path to configuration file
        """
        self.config = {}
        if config_file and os.path.exists(config_file):
            with open(config_file, "r") as f:
                self.config = json.load(f)

        # Default configuration
        self.regions = self.config.get(
            "regions",
            {
                "us-east": {"latency": 10, "jitter": 2, "loss": 0.1},
                "us-west": {"latency": 80, "jitter": 5, "loss": 0.2},
                "eu-central": {"latency": 100, "jitter": 10, "loss": 0.5},
                "asia-east": {"latency": 200, "jitter": 20, "loss": 1.0},
            },
        )

        # Region latency matrix (ms)
        self.latency_matrix = self.config.get(
            "latency_matrix",
            {
                "us-east": {
                    "us-east": 10,
                    "us-west": 80,
                    "eu-central": 100,
                    "asia-east": 200,
                },
                "us-west": {
                    "us-east": 80,
                    "us-west": 10,
                    "eu-central": 150,
                    "asia-east": 150,
                },
                "eu-central": {
                    "us-east": 100,
                    "us-west": 150,
                    "eu-central": 10,
                    "asia-east": 250,
                },
                "asia-east": {
                    "us-east": 200,
                    "us-west": 150,
                    "eu-central": 250,
                    "asia-east": 10,
                },
            },
        )

        # Network interfaces
        self.interfaces = {}  # node_id -> interface

        # Node regions
        self.node_regions = {}  # node_id -> region

        # Network partitions
        self.partitions = []  # List of (group1, group2) tuples

    def add_node(self, node_id: str, region: str, interface: str = "eth0"):
        """
        Add a node to the network simulation.

        Args:
            node_id: Node identifier
            region: Region identifier
            interface: Network interface
        """
        if region not in self.regions:
            print(f"Warning: Unknown region {region}, defaulting to us-east")
            region = "us-east"

        self.interfaces[node_id] = interface
        self.node_regions[node_id] = region

    def setup_network(self):
        """
        Set up the network simulation.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Clear existing tc rules
            self.clear_network()

            # Set up tc rules for each node
            for node_id, interface in self.interfaces.items():
                region = self.node_regions.get(node_id, "us-east")

                # Get region parameters
                params = self.regions.get(
                    region, {"latency": 10, "jitter": 2, "loss": 0.1}
                )

                # Set up tc rules
                self._setup_tc_rules(node_id, interface, params)

            return True
        except Exception as e:
            print(f"Error setting up network: {e}")
            return False

    def clear_network(self):
        """
        Clear all network simulation rules.

        Returns:
            True if successful, False otherwise
        """
        try:
            for node_id, interface in self.interfaces.items():
                self._clear_tc_rules(node_id, interface)

            # Clear partitions
            self.partitions = []

            return True
        except Exception as e:
            print(f"Error clearing network: {e}")
            return False

    def update_latency(self, node_id: str, target_latency: int, target_jitter: int = 0):
        """
        Update latency for a node.

        Args:
            node_id: Node identifier
            target_latency: Target latency in milliseconds
            target_jitter: Target jitter in milliseconds

        Returns:
            True if successful, False otherwise
        """
        try:
            interface = self.interfaces.get(node_id)
            if not interface:
                return False

            # Update tc rules
            cmd = f"sudo tc qdisc change dev {interface} root netem delay {target_latency}ms {target_jitter}ms"
            subprocess.run(cmd, shell=True, check=True)

            return True
        except Exception as e:
            print(f"Error updating latency: {e}")
            return False

    def update_packet_loss(self, node_id: str, loss_percent: float):
        """
        Update packet loss for a node.

        Args:
            node_id: Node identifier
            loss_percent: Packet loss percentage

        Returns:
            True if successful, False otherwise
        """
        try:
            interface = self.interfaces.get(node_id)
            if not interface:
                return False

            # Update tc rules
            cmd = (
                f"sudo tc qdisc change dev {interface} root netem loss {loss_percent}%"
            )
            subprocess.run(cmd, shell=True, check=True)

            return True
        except Exception as e:
            print(f"Error updating packet loss: {e}")
            return False

    def create_partition(self, group1: List[str], group2: List[str]):
        """
        Create a network partition between two groups of nodes.

        Args:
            group1: First group of node IDs
            group2: Second group of node IDs

        Returns:
            True if successful, False otherwise
        """
        try:
            # Add partition to list
            self.partitions.append((group1, group2))

            # Set up iptables rules to block traffic between groups
            for node1 in group1:
                for node2 in group2:
                    self._block_traffic(node1, node2)
                    self._block_traffic(node2, node1)

            return True
        except Exception as e:
            print(f"Error creating partition: {e}")
            return False

    def heal_partition(self, group1: List[str], group2: List[str]):
        """
        Heal a network partition between two groups of nodes.

        Args:
            group1: First group of node IDs
            group2: Second group of node IDs

        Returns:
            True if successful, False otherwise
        """
        try:
            # Remove partition from list
            if (group1, group2) in self.partitions:
                self.partitions.remove((group1, group2))
            elif (group2, group1) in self.partitions:
                self.partitions.remove((group2, group1))

            # Remove iptables rules to allow traffic between groups
            for node1 in group1:
                for node2 in group2:
                    self._unblock_traffic(node1, node2)
                    self._unblock_traffic(node2, node1)

            return True
        except Exception as e:
            print(f"Error healing partition: {e}")
            return False

    def heal_all_partitions(self):
        """
        Heal all network partitions.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Heal each partition
            for group1, group2 in self.partitions:
                self.heal_partition(group1, group2)

            return True
        except Exception as e:
            print(f"Error healing all partitions: {e}")
            return False

    def simulate_byzantine_behavior(self, node_id: str, behavior_type: str):
        """
        Simulate Byzantine behavior for a node.

        Args:
            node_id: Node identifier
            behavior_type: Type of Byzantine behavior
                - 'no-response': Drop all incoming packets
                - 'selective': Selectively drop packets
                - 'delay': Add significant delay to responses

        Returns:
            True if successful, False otherwise
        """
        try:
            interface = self.interfaces.get(node_id)
            if not interface:
                return False

            if behavior_type == "no-response":
                # Drop all incoming packets
                cmd = f"sudo iptables -A INPUT -i {interface} -j DROP"
                subprocess.run(cmd, shell=True, check=True)

            elif behavior_type == "selective":
                # Drop 50% of packets
                cmd = f"sudo tc qdisc change dev {interface} root netem loss 50%"
                subprocess.run(cmd, shell=True, check=True)

            elif behavior_type == "delay":
                # Add significant delay (1-5 seconds)
                delay = random.randint(1000, 5000)
                jitter = random.randint(100, 500)
                cmd = f"sudo tc qdisc change dev {interface} root netem delay {delay}ms {jitter}ms"
                subprocess.run(cmd, shell=True, check=True)

            return True
        except Exception as e:
            print(f"Error simulating Byzantine behavior: {e}")
            return False

    def reset_byzantine_behavior(self, node_id: str):
        """
        Reset Byzantine behavior for a node.

        Args:
            node_id: Node identifier

        Returns:
            True if successful, False otherwise
        """
        try:
            interface = self.interfaces.get(node_id)
            if not interface:
                return False

            # Remove iptables rules
            cmd = f"sudo iptables -D INPUT -i {interface} -j DROP 2>/dev/null || true"
            subprocess.run(cmd, shell=True)

            # Reset tc rules
            region = self.node_regions.get(node_id, "us-east")
            params = self.regions.get(region, {"latency": 10, "jitter": 2, "loss": 0.1})
            self._setup_tc_rules(node_id, interface, params)

            return True
        except Exception as e:
            print(f"Error resetting Byzantine behavior: {e}")
            return False

    def _setup_tc_rules(self, node_id: str, interface: str, params: Dict[str, Any]):
        """
        Set up tc rules for a node.

        Args:
            node_id: Node identifier
            interface: Network interface
            params: Network parameters
        """
        # Clear existing rules
        self._clear_tc_rules(node_id, interface)

        # Set up basic netem rules
        latency = params.get("latency", 10)
        jitter = params.get("jitter", 2)
        loss = params.get("loss", 0.1)

        cmd = f"sudo tc qdisc add dev {interface} root netem delay {latency}ms {jitter}ms loss {loss}%"
        subprocess.run(cmd, shell=True, check=True)

    def _clear_tc_rules(self, node_id: str, interface: str):
        """
        Clear tc rules for a node.

        Args:
            node_id: Node identifier
            interface: Network interface
        """
        cmd = f"sudo tc qdisc del dev {interface} root 2>/dev/null || true"
        subprocess.run(cmd, shell=True)

    def _block_traffic(self, source_node: str, target_node: str):
        """
        Block traffic between two nodes.

        Args:
            source_node: Source node identifier
            target_node: Target node identifier
        """
        # In a real implementation, this would use the actual IP addresses of the nodes
        # For simulation, we'll use node IDs as placeholders
        source_ip = f"10.0.0.{hash(source_node) % 254 + 1}"
        target_ip = f"10.0.0.{hash(target_node) % 254 + 1}"

        cmd = f"sudo iptables -A FORWARD -s {source_ip} -d {target_ip} -j DROP"
        subprocess.run(cmd, shell=True, check=True)

    def _unblock_traffic(self, source_node: str, target_node: str):
        """
        Unblock traffic between two nodes.

        Args:
            source_node: Source node identifier
            target_node: Target node identifier
        """
        # In a real implementation, this would use the actual IP addresses of the nodes
        # For simulation, we'll use node IDs as placeholders
        source_ip = f"10.0.0.{hash(source_node) % 254 + 1}"
        target_ip = f"10.0.0.{hash(target_node) % 254 + 1}"

        cmd = f"sudo iptables -D FORWARD -s {source_ip} -d {target_ip} -j DROP 2>/dev/null || true"
        subprocess.run(cmd, shell=True)


class DockerNetworkSimulator:
    """
    Simulates realistic WAN environment using Docker networks.
    Configures per-region latency, packet loss, and jitter for Docker containers.
    """

    def __init__(self, config_file: str = None):
        """
        Initialize the Docker network simulator.

        Args:
            config_file: Path to configuration file
        """
        self.config = {}
        if config_file and os.path.exists(config_file):
            with open(config_file, "r") as f:
                self.config = json.load(f)

        # Default configuration
        self.regions = self.config.get(
            "regions",
            {
                "us-east": {"latency": 10, "jitter": 2, "loss": 0.1},
                "us-west": {"latency": 80, "jitter": 5, "loss": 0.2},
                "eu-central": {"latency": 100, "jitter": 10, "loss": 0.5},
                "asia-east": {"latency": 200, "jitter": 20, "loss": 1.0},
            },
        )

        # Region latency matrix (ms)
        self.latency_matrix = self.config.get(
            "latency_matrix",
            {
                "us-east": {
                    "us-east": 10,
                    "us-west": 80,
                    "eu-central": 100,
                    "asia-east": 200,
                },
                "us-west": {
                    "us-east": 80,
                    "us-west": 10,
                    "eu-central": 150,
                    "asia-east": 150,
                },
                "eu-central": {
                    "us-east": 100,
                    "us-west": 150,
                    "eu-central": 10,
                    "asia-east": 250,
                },
                "asia-east": {
                    "us-east": 200,
                    "us-west": 150,
                    "eu-central": 250,
                    "asia-east": 10,
                },
            },
        )

        # Docker networks
        self.networks = {}  # region -> network_name

        # Node containers
        self.containers = {}  # node_id -> container_name

        # Node regions
        self.node_regions = {}  # node_id -> region

    def setup_networks(self):
        """
        Set up Docker networks for each region.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create a network for each region
            for region in self.regions:
                network_name = f"qtrust-{region}"

                # Check if network already exists
                cmd = f"docker network ls --filter name={network_name} --format '{{{{.Name}}}}'"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

                if network_name not in result.stdout:
                    # Create network
                    cmd = f"docker network create {network_name}"
                    subprocess.run(cmd, shell=True, check=True)

                self.networks[region] = network_name

            return True
        except Exception as e:
            print(f"Error setting up Docker networks: {e}")
            return False

    def add_container(self, node_id: str, container_name: str, region: str):
        """
        Add a container to a region network.

        Args:
            node_id: Node identifier
            container_name: Docker container name
            region: Region identifier

        Returns:
            True if successful, False otherwise
        """
        try:
            if region not in self.networks:
                if not self.setup_networks():
                    return False

            network_name = self.networks[region]

            # Connect container to network
            cmd = f"docker network connect {network_name} {container_name}"
            subprocess.run(cmd, shell=True, check=True)

            # Store container info
            self.containers[node_id] = container_name
            self.node_regions[node_id] = region

            return True
        except Exception as e:
            print(f"Error adding container to network: {e}")
            return False

    def setup_latency(self):
        """
        Set up latency between regions using tc and netem.

        Returns:
            True if successful, False otherwise
        """
        try:
            # For each container, set up tc rules for communication with other regions
            for node_id, container_name in self.containers.items():
                source_region = self.node_regions[node_id]

                for target_region, latency in self.latency_matrix[
                    source_region
                ].items():
                    if target_region == source_region:
                        continue  # Skip same region

                    # Get target network
                    target_network = self.networks[target_region]

                    # Set up tc rules inside the container
                    jitter = self.regions[target_region]["jitter"]
                    loss = self.regions[target_region]["loss"]

                    # Get the interface for the target network
                    cmd = f"docker exec {container_name} ip route | grep {target_network} | awk '{{print $3}}'"
                    result = subprocess.run(
                        cmd, shell=True, capture_output=True, text=True
                    )
                    interface = result.stdout.strip()

                    if interface:
                        # Set up tc rules
                        cmd = f"docker exec {container_name} tc qdisc add dev {interface} root netem delay {latency}ms {jitter}ms loss {loss}%"
                        subprocess.run(cmd, shell=True, check=True)

            return True
        except Exception as e:
            print(f"Error setting up latency: {e}")
            return False

    def create_partition(self, group1: List[str], group2: List[str]):
        """
        Create a network partition between two groups of nodes.

        Args:
            group1: First group of node IDs
            group2: Second group of node IDs

        Returns:
            True if successful, False otherwise
        """
        try:
            # Block traffic between containers in different groups
            for node1 in group1:
                container1 = self.containers.get(node1)
                if not container1:
                    continue

                for node2 in group2:
                    container2 = self.containers.get(node2)
                    if not container2:
                        continue

                    # Get container2's IP address
                    cmd = f"docker inspect -f '{{{{.NetworkSettings.IPAddress}}}}' {container2}"
                    result = subprocess.run(
                        cmd, shell=True, capture_output=True, text=True
                    )
                    ip2 = result.stdout.strip()

                    if ip2:
                        # Block traffic to container2
                        cmd = f"docker exec {container1} iptables -A OUTPUT -d {ip2} -j DROP"
                        subprocess.run(cmd, shell=True, check=True)

            return True
        except Exception as e:
            print(f"Error creating partition: {e}")
            return False

    def heal_partition(self, group1: List[str], group2: List[str]):
        """
        Heal a network partition between two groups of nodes.

        Args:
            group1: First group of node IDs
            group2: Second group of node IDs

        Returns:
            True if successful, False otherwise
        """
        try:
            # Unblock traffic between containers in different groups
            for node1 in group1:
                container1 = self.containers.get(node1)
                if not container1:
                    continue

                for node2 in group2:
                    container2 = self.containers.get(node2)
                    if not container2:
                        continue

                    # Get container2's IP address
                    cmd = f"docker inspect -f '{{{{.NetworkSettings.IPAddress}}}}' {container2}"
                    result = subprocess.run(
                        cmd, shell=True, capture_output=True, text=True
                    )
                    ip2 = result.stdout.strip()

                    if ip2:
                        # Unblock traffic to container2
                        cmd = f"docker exec {container1} iptables -D OUTPUT -d {ip2} -j DROP 2>/dev/null || true"
                        subprocess.run(cmd, shell=True)

            return True
        except Exception as e:
            print(f"Error healing partition: {e}")
            return False

    def simulate_byzantine_behavior(self, node_id: str, behavior_type: str):
        """
        Simulate Byzantine behavior for a node.

        Args:
            node_id: Node identifier
            behavior_type: Type of Byzantine behavior
                - 'no-response': Drop all incoming packets
                - 'selective': Selectively drop packets
                - 'delay': Add significant delay to responses

        Returns:
            True if successful, False otherwise
        """
        try:
            container_name = self.containers.get(node_id)
            if not container_name:
                return False

            if behavior_type == "no-response":
                # Drop all incoming packets
                cmd = f"docker exec {container_name} iptables -A INPUT -j DROP"
                subprocess.run(cmd, shell=True, check=True)

            elif behavior_type == "selective":
                # Drop 50% of packets
                cmd = f"docker exec {container_name} iptables -A INPUT -m statistic --mode random --probability 0.5 -j DROP"
                subprocess.run(cmd, shell=True, check=True)

            elif behavior_type == "delay":
                # Add significant delay to all interfaces
                cmd = f"docker exec {container_name} tc qdisc add dev eth0 root netem delay 3000ms 500ms"
                subprocess.run(cmd, shell=True, check=True)

            return True
        except Exception as e:
            print(f"Error simulating Byzantine behavior: {e}")
            return False

    def reset_byzantine_behavior(self, node_id: str):
        """
        Reset Byzantine behavior simulation for a node.

        Args:
            node_id: Node identifier

        Returns:
            True if successful, False otherwise
        """
        try:
            container_name = self.containers.get(node_id)
            if not container_name:
                return False

            # Get container
            container = self.docker_client.containers.get(container_name)

            # Reset traffic control rules
            cmd = "tc qdisc del dev eth0 root; tc qdisc add dev eth0 root netem delay 0ms"
            container.exec_run(cmd)

            # Remove container from byzantine list
            if node_id in self.byzantine_nodes:
                self.byzantine_nodes.remove(node_id)

            return True
        except Exception as e:
            print(f"Error resetting Byzantine behavior: {e}")
            return False

    def heal_all_partitions(self):
        """
        Heal all network partitions.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all partitions
            partitions = list(self.partitions)
            
            # Heal each partition
            for group1, group2 in partitions:
                self.heal_partition(group1, group2)
            
            # Clear partitions list
            self.partitions = []
            
            return True
        except Exception as e:
            print(f"Error healing all partitions: {e}")
            return False

    def cleanup(self):
        """
        Clean up all Docker networks.

        Returns:
            True if successful, False otherwise
        """
        try:
            for region, network_name in self.networks.items():
                cmd = f"docker network rm {network_name} 2>/dev/null || true"
                subprocess.run(cmd, shell=True)

            self.networks = {}

            return True
        except Exception as e:
            print(f"Error cleaning up Docker networks: {e}")
            return False


class NetworkSimulation:
    """
    High-level interface for network simulation that integrates with benchmark framework.
    This class wraps the low-level network simulators and provides a simple interface.
    """

    def __init__(self, num_regions=5, num_shards=64, nodes_per_shard=3, 
                 byzantine_ratio=0.2, sybil_ratio=0.1, **kwargs):
        """
        Initialize the network simulation.

        Args:
            num_regions: Number of network regions
            num_shards: Number of shards
            nodes_per_shard: Number of nodes per shard
            byzantine_ratio: Ratio of Byzantine nodes
            sybil_ratio: Ratio of Sybil nodes
            **kwargs: Additional parameters
        """
        self.num_regions = num_regions
        self.num_shards = num_shards
        self.nodes_per_shard = nodes_per_shard
        self.byzantine_ratio = byzantine_ratio
        self.sybil_ratio = sybil_ratio
        
        # Additional parameters
        self.config = kwargs
        
        # Use Docker if available, otherwise use direct network simulation
        try:
            import docker
            self.simulator = DockerNetworkSimulator()
        except ImportError:
            self.simulator = NetworkSimulator()
            
        # Node and shard mapping
        self.nodes = []
        self.shards = {}
        self.node_to_shard = {}
        
        # Node roles
        self.byzantine_nodes = []
        self.sybil_nodes = []
        
        # Simulation state
        self.is_running = False
        
    def initialize(self):
        """
        Initialize the simulation environment.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate regions
            regions = ["us-east", "us-west", "eu-central", "asia-east", "asia-south"][:self.num_regions]
            
            # Generate nodes and assign to shards
            node_id = 0
            for shard_id in range(self.num_shards):
                shard_nodes = []
                for _ in range(self.nodes_per_shard):
                    node_name = f"node_{node_id}"
                    region = random.choice(regions)
                    
                    # Add node to simulator
                    self.simulator.add_node(node_name, region)
                    
                    # Track node
                    self.nodes.append(node_name)
                    shard_nodes.append(node_name)
                    self.node_to_shard[node_name] = shard_id
                    
                    node_id += 1
                
                self.shards[shard_id] = shard_nodes
            
            # Designate Byzantine nodes
            num_byzantine = int(len(self.nodes) * self.byzantine_ratio)
            self.byzantine_nodes = random.sample(self.nodes, num_byzantine)
            
            # Designate Sybil nodes
            remaining_nodes = [n for n in self.nodes if n not in self.byzantine_nodes]
            num_sybil = int(len(self.nodes) * self.sybil_ratio)
            self.sybil_nodes = random.sample(remaining_nodes, min(num_sybil, len(remaining_nodes)))
            
            # Set up network
            return self.simulator.setup_network()
        
        except Exception as e:
            print(f"Error initializing simulation: {e}")
            return False
    
    def start_simulation(self):
        """
        Start the simulation.
        
        Returns:
            True if successful, False otherwise
        """
        if self.is_running:
            return True
            
        try:
            # Simulate Byzantine behavior for designated nodes
            for node in self.byzantine_nodes:
                behavior = random.choice([
                    "packet_drop", 
                    "message_delay", 
                    "message_corruption"
                ])
                self.simulator.simulate_byzantine_behavior(node, behavior)
            
            self.is_running = True
            return True
        
        except Exception as e:
            print(f"Error starting simulation: {e}")
            return False
    
    def stop_simulation(self):
        """
        Stop the simulation.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_running:
            return True
            
        try:
            # Reset Byzantine behavior
            for node in self.byzantine_nodes:
                self.simulator.reset_byzantine_behavior(node)
            
            # Clear network settings
            self.simulator.clear_network()
            
            self.is_running = False
            return True
        
        except Exception as e:
            print(f"Error stopping simulation: {e}")
            return False
    
    def create_network_partition(self):
        """
        Create a network partition.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Split shards into two groups
            shard_ids = list(self.shards.keys())
            mid = len(shard_ids) // 2
            group1_shards = shard_ids[:mid]
            group2_shards = shard_ids[mid:]
            
            # Get nodes in each group
            group1_nodes = []
            for shard_id in group1_shards:
                group1_nodes.extend(self.shards[shard_id])
                
            group2_nodes = []
            for shard_id in group2_shards:
                group2_nodes.extend(self.shards[shard_id])
            
            # Create partition
            return self.simulator.create_partition(group1_nodes, group2_nodes)
        
        except Exception as e:
            print(f"Error creating network partition: {e}")
            return False
    
    def heal_network_partition(self):
        """
        Heal all network partitions.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            return self.simulator.heal_all_partitions()
        
        except Exception as e:
            print(f"Error healing network partition: {e}")
            return False


if __name__ == "__main__":
    # Example usage
    simulator = NetworkSimulator()

    # Add nodes
    simulator.add_node("node1", "us-east")
    simulator.add_node("node2", "us-west")
    simulator.add_node("node3", "eu-central")
    simulator.add_node("node4", "asia-east")

    # Set up network
    simulator.setup_network()

    # Create a partition
    simulator.create_partition(["node1", "node2"], ["node3", "node4"])

    # Simulate Byzantine behavior
    simulator.simulate_byzantine_behavior("node2", "selective")

    # Wait for a while
    time.sleep(60)

    # Heal partition
    simulator.heal_partition(["node1", "node2"], ["node3", "node4"])

    # Reset Byzantine behavior
    simulator.reset_byzantine_behavior("node2")

    # Clear network
    simulator.clear_network()
