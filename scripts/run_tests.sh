#!/bin/bash
# QTrust Test Suite Runner
# This script runs the test suite for the QTrust Blockchain Sharding Framework

set -e  # Exit on error

echo "=== QTrust Test Suite Runner ==="

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Activated virtual environment."
fi

# Install test dependencies if needed
pip install pytest pytest-cov

# Run unit tests
echo "Running unit tests..."
python -m pytest tests/unit -v

# Run integration tests
echo "Running integration tests..."
python -m pytest tests/integration -v

# Run component tests
echo "Running component tests..."
python -m pytest tests/components -v

# Generate coverage report
echo "Generating test coverage report..."
python -m pytest --cov=qtrust tests/ --cov-report=term --cov-report=html

echo ""
echo "=== Test Suite Complete ==="
echo "Coverage report available in htmlcov/index.html"
echo ""
