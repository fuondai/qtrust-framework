#!/bin/bash

# Script to run all tests and generate coverage report
# This script verifies that test coverage exceeds 85% as required

# Install test dependencies if not already installed
pip install pytest pytest-cov coverage

# Create directory for test results
mkdir -p test_results

# Run unit tests with coverage
echo "Running unit tests with coverage..."
pytest -xvs tests/unit/ --cov=qtrust --cov-report=term --cov-report=xml:test_results/coverage.xml --cov-report=html:test_results/coverage_html

# Run integration tests
echo "Running integration tests..."
pytest -xvs tests/integration/

# Check if coverage meets the 85% threshold
COVERAGE=$(coverage report | grep TOTAL | awk '{print $NF}' | sed 's/%//')
echo "Total test coverage: $COVERAGE%"

if (( $(echo "$COVERAGE >= 85" | bc -l) )); then
    echo "✅ Test coverage threshold of 85% met or exceeded!"
else
    echo "❌ Test coverage below 85% threshold. Current coverage: $COVERAGE%"
    exit 1
fi

echo "All tests completed successfully!"
