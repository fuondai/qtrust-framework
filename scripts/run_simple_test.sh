#!/bin/bash
# QTrust Simple Test Script
# This script runs a simplified test for the QTrust framework

set -e  # Exit on error

echo "=== QTrust Simple Test ==="
echo "This script will run a simplified test for the QTrust framework."

# Create necessary directories
mkdir -p demo_results

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Verify Python 3.10 is available
if command -v python3.10 &> /dev/null; then
    echo "Using Python 3.10"
    PYTHON="python3.10"
elif command -v python3 &> /dev/null; then
    echo "Using Python 3"
    PYTHON="python3"
else
    echo "Error: Python 3 not found. Please install Python 3.10."
    exit 1
fi

# Verify that the demo runs with simplified parameters
echo "Running QTrust demo..."
$PYTHON scripts/run_demo.py --verify

# Check if demo created any results
if [ -d "demo_results" ] && [ "$(ls -A demo_results)" ]; then
    echo "Demo completed successfully and generated results."
else
    echo "Warning: No results found in demo_results directory."
fi

echo ""
echo "=== Simple Test Complete ==="
echo "QTrust demo test completed."
echo "" 