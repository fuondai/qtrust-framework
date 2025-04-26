#!/bin/bash
# Setup environment for QTrust

set -e  # Exit on error

# Create benchmark results directories
echo "Creating necessary directories..."
mkdir -p benchmark_results/{logs,data,charts} demo_results

# Ensure Python 3.10 is installed
if command -v python3.10 &> /dev/null; then
    echo "Python 3.10 is already installed"
else
    echo "Python 3.10 is not installed. Please install Python 3.10 before continuing."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3.10 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install required packages
echo "Installing required packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Install PyTorch CPU version
echo "Installing PyTorch CPU version..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Create simulation module if it doesn't exist
echo "Ensuring simulation module exists..."
mkdir -p qtrust/simulation

# Check if fixed_network_simulation.py exists, create if not
if [ ! -f "qtrust/simulation/fixed_network_simulation.py" ]; then
    echo "Creating simplified network simulation module..."
    cp -f qtrust/simulation/network_simulation.py qtrust/simulation/fixed_network_simulation.py
fi

# Install the package in development mode
echo "Installing QTrust in development mode..."
pip install -e .

echo "Environment setup complete!"
echo "To activate the virtual environment, run: source venv/bin/activate"
