#!/bin/bash
# QTrust Artifact Evaluation Setup Script
# This script prepares the environment for evaluating the QTrust Blockchain Sharding Framework

set -e  # Exit on error

echo "=== QTrust Artifact Evaluation Setup ==="
echo "This script will prepare your environment for evaluating the QTrust artifact."

# Check Python version
echo -n "Checking Python version... "
python_version=$(python3 --version 2>&1 | awk '{print $2}')
python_major=$(echo $python_version | cut -d. -f1)
python_minor=$(echo $python_version | cut -d. -f2)

if [ "$python_major" -lt 3 ] || ([ "$python_major" -eq 3 ] && [ "$python_minor" -lt 8 ]); then
    echo "ERROR: Python 3.8 or higher is required."
    echo "Current version: $python_version"
    echo "Please install Python 3.8+ and try again."
    exit 1
else
    echo "OK (Python $python_version)"
fi

# Check for virtual environment
echo -n "Setting up virtual environment... "
if [ -d "venv" ]; then
    echo "Found existing virtual environment."
else
    python3 -m venv venv
    echo "Created new virtual environment."
fi

# Activate virtual environment
source venv/bin/activate
echo "Activated virtual environment."

# Install dependencies
echo "Installing dependencies (this may take a few minutes)..."
pip install --upgrade pip
pip install -r requirements.txt

# Check for GPU
echo -n "Checking for GPU support... "
if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo "GPU detected, installing PyTorch with CUDA support."
    pip install torch torchvision torchaudio
else
    echo "No GPU detected, installing CPU-only PyTorch."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p benchmark_results/{logs,data,charts}
mkdir -p demo_results

# Check system resources
echo "Checking system resources..."
cpu_count=$(python3 -c "import os; print(os.cpu_count())")
echo "CPU cores: $cpu_count"

if command -v free &> /dev/null; then
    total_ram=$(free -g | awk '/^Mem:/{print $2}')
    echo "Total RAM: ${total_ram}GB"
    
    if [ "$total_ram" -lt 8 ]; then
        echo "WARNING: At least 8GB RAM is recommended for optimal performance."
        echo "The system will use fallback modes for memory-intensive operations."
    fi
fi

# Set up pre-collected results for environments without sufficient resources
echo "Setting up pre-collected benchmark results..."
if [ -d "benchmark_results/data" ] && [ -d "benchmark_results/charts" ]; then
    echo "Pre-collected benchmark results are already available."
else
    echo "Copying pre-collected benchmark results..."
    cp -r benchmark_results_precollected/* benchmark_results/ 2>/dev/null || echo "Using empty benchmark results directory."
fi

# Install test dependencies
echo "Installing test dependencies..."
pip install pytest pytest-cov

# Run basic verification test
echo "Running basic verification test..."
python -c "
import sys
import os
sys.path.append('.')
from qtrust.trust.trust_vector import TrustVector
from qtrust.agents.mock_rainbow_agent import MockRainbowAgent

# Test trust vector
tv = TrustVector()
tv.update_dimension('transaction_validation', 0.8)
assert tv.get_dimension('transaction_validation') == 0.8

# Test mock agent
agent = MockRainbowAgent()
action = agent.select_action({'state': 'test'})
assert action is not None

print('Basic verification test passed!')
"

# Final setup
echo "Setting environment variables..."
export QTRUST_HOME=$(pwd)
export PYTHONPATH=$PYTHONPATH:$QTRUST_HOME

echo ""
echo "=== QTrust Artifact Evaluation Setup Complete ==="
echo ""
echo "To activate this environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "Quick start commands:"
echo "  python scripts/run_demo.py            # Run the demo"
echo "  python scripts/run_benchmark.py       # Run the benchmark"
echo "  python scripts/generate_visuals.py    # Generate visualizations"
echo ""
echo "For detailed instructions, see docs/artifact_evaluation.md"
echo ""
