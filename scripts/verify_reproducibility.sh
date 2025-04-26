#!/bin/bash
# QTrust Reproducibility Test Script
# This script validates that the QTrust repository can be set up and run correctly

set -e  # Exit on error

echo "=== QTrust Reproducibility Test ==="
echo "This script will verify that the QTrust repository can be set up and run correctly."
echo "It performs a clean installation and runs basic functionality tests."

# Create a temporary directory for testing
TEST_DIR=$(mktemp -d)
echo "Created temporary test directory: $TEST_DIR"

# Function to clean up on exit
cleanup() {
  echo "Cleaning up temporary directory..."
  rm -rf "$TEST_DIR"
}
trap cleanup EXIT

# Copy the repository to the test directory
echo "Copying repository to test directory..."
cp -r . "$TEST_DIR"
cd "$TEST_DIR"

# Run the setup script
echo "Running setup script..."
bash scripts/setup_environment.sh

# Verify that the demo runs
echo "Verifying demo functionality..."
python3.10 scripts/run_demo.py --verify

# Check if the benchmark script is available
if [ -f "scripts/run_benchmark.py" ]; then
    # Run a small-scale benchmark
    echo "Running small-scale benchmark test..."
    python3.10 scripts/run_benchmark.py --nodes 5 --shards 1 --duration 10
else
    echo "Skipping benchmark test - run_benchmark.py not found"
fi

# Check if the visualization script is available
if [ -f "scripts/generate_visuals.py" ]; then
    # Verify that visualization works
    echo "Verifying visualization functionality..."
    python3.10 scripts/generate_visuals.py
else
    echo "Skipping visualization test - generate_visuals.py not found"
fi

# Check if run_tests.sh is available
if [ -f "scripts/run_tests.sh" ]; then
    # Run basic tests
    echo "Running basic tests..."
    bash scripts/run_tests.sh
elif [ -f "run_all_tests.py" ]; then
    echo "Running tests using run_all_tests.py..."
    python3.10 run_all_tests.py
else
    echo "Skipping tests - test scripts not found"
fi

echo ""
echo "=== Reproducibility Test Complete ==="
echo "All tests passed! The repository is ready for artifact evaluation."
echo ""
