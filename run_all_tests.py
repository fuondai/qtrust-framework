"""
Script to run all tests and generate coverage report.

This script runs all unit, integration, system, and security tests,
and generates a comprehensive coverage report.
"""

import os
import sys
import subprocess
import argparse
import json
from datetime import datetime

def run_command(command, cwd=None):
    """Run a shell command and return the output."""
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False, result.stdout, result.stderr
    
    return True, result.stdout, result.stderr

def create_directories():
    """Create necessary directories for test results."""
    directories = [
        'test_results',
        'test_results/unit',
        'test_results/integration',
        'test_results/system',
        'test_results/security',
        'test_results/coverage',
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def run_unit_tests():
    """Run unit tests with coverage."""
    print("\n=== Running Unit Tests ===\n")
    success, stdout, stderr = run_command([
        'python', '-m', 'pytest', 'tests/unit',
        '--cov=qtrust',
        '--cov-report=xml:test_results/coverage/unit_coverage.xml',
        '--cov-report=html:test_results/coverage/unit_coverage_html',
        '-v',
        '--junitxml=test_results/unit/results.xml'
    ])
    
    with open('test_results/unit/output.txt', 'w') as f:
        f.write(stdout)
        f.write(stderr)
    
    return success

def run_integration_tests():
    """Run integration tests."""
    print("\n=== Running Integration Tests ===\n")
    success, stdout, stderr = run_command([
        'python', '-m', 'pytest', 'tests/integration',
        '-v',
        '--junitxml=test_results/integration/results.xml'
    ])
    
    with open('test_results/integration/output.txt', 'w') as f:
        f.write(stdout)
        f.write(stderr)
    
    return success

def run_system_tests():
    """Run system tests."""
    print("\n=== Running System Tests ===\n")
    success, stdout, stderr = run_command([
        'python', '-m', 'pytest', 'tests/system',
        '-v',
        '--junitxml=test_results/system/results.xml'
    ])
    
    with open('test_results/system/output.txt', 'w') as f:
        f.write(stdout)
        f.write(stderr)
    
    return success

def run_security_tests():
    """Run security tests."""
    print("\n=== Running Security Tests ===\n")
    success, stdout, stderr = run_command([
        'python', '-m', 'pytest', 'tests/security',
        '-v',
        '--junitxml=test_results/security/results.xml'
    ])
    
    with open('test_results/security/output.txt', 'w') as f:
        f.write(stdout)
        f.write(stderr)
    
    return success

def run_code_quality_checks():
    """Run code quality checks."""
    print("\n=== Running Code Quality Checks ===\n")
    
    # Run pylint
    print("Running pylint...")
    pylint_success, pylint_stdout, pylint_stderr = run_command([
        'pylint', 'qtrust', '--output-format=text'
    ])
    
    with open('test_results/pylint_report.txt', 'w') as f:
        f.write(pylint_stdout)
        f.write(pylint_stderr)
    
    # Run mypy
    print("Running mypy...")
    mypy_success, mypy_stdout, mypy_stderr = run_command([
        'mypy', 'qtrust'
    ])
    
    with open('test_results/mypy_report.txt', 'w') as f:
        f.write(mypy_stdout)
        f.write(mypy_stderr)
    
    # Run bandit
    print("Running bandit...")
    bandit_success, bandit_stdout, bandit_stderr = run_command([
        'bandit', '-r', 'qtrust', '-f', 'json', '-o', 'test_results/bandit_report.json'
    ])
    
    return pylint_success and mypy_success and bandit_success

def generate_coverage_report():
    """Generate a comprehensive coverage report."""
    print("\n=== Generating Coverage Report ===\n")
    
    # Combine coverage data
    success, stdout, stderr = run_command([
        'coverage', 'combine'
    ])
    
    # Generate coverage report
    success, stdout, stderr = run_command([
        'coverage', 'report'
    ])
    
    with open('test_results/coverage/coverage_summary.txt', 'w') as f:
        f.write(stdout)
    
    # Generate HTML report
    success, stdout, stderr = run_command([
        'coverage', 'html', '-d', 'test_results/coverage/html_report'
    ])
    
    # Generate XML report
    success, stdout, stderr = run_command([
        'coverage', 'xml', '-o', 'test_results/coverage/coverage.xml'
    ])
    
    return success

def generate_summary_report(unit_success, integration_success, system_success, security_success, quality_success):
    """Generate a summary report of all tests."""
    print("\n=== Generating Summary Report ===\n")
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'tests': {
            'unit': {
                'success': unit_success,
                'report': 'test_results/unit/results.xml'
            },
            'integration': {
                'success': integration_success,
                'report': 'test_results/integration/results.xml'
            },
            'system': {
                'success': system_success,
                'report': 'test_results/system/results.xml'
            },
            'security': {
                'success': security_success,
                'report': 'test_results/security/results.xml'
            }
        },
        'code_quality': {
            'success': quality_success,
            'pylint_report': 'test_results/pylint_report.txt',
            'mypy_report': 'test_results/mypy_report.txt',
            'bandit_report': 'test_results/bandit_report.json'
        },
        'coverage': {
            'report': 'test_results/coverage/coverage_summary.txt',
            'html_report': 'test_results/coverage/html_report/index.html',
            'xml_report': 'test_results/coverage/coverage.xml'
        }
    }
    
    with open('test_results/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n=== Test Summary ===\n")
    print(f"Unit Tests: {'PASSED' if unit_success else 'FAILED'}")
    print(f"Integration Tests: {'PASSED' if integration_success else 'FAILED'}")
    print(f"System Tests: {'PASSED' if system_success else 'FAILED'}")
    print(f"Security Tests: {'PASSED' if security_success else 'FAILED'}")
    print(f"Code Quality Checks: {'PASSED' if quality_success else 'FAILED'}")
    
    overall_success = unit_success and integration_success and system_success and security_success and quality_success
    print(f"\nOverall: {'PASSED' if overall_success else 'FAILED'}")
    
    return overall_success

def main():
    """Main function to run all tests."""
    parser = argparse.ArgumentParser(description='Run all tests and generate coverage report.')
    parser.add_argument('--unit-only', action='store_true', help='Run only unit tests')
    parser.add_argument('--integration-only', action='store_true', help='Run only integration tests')
    parser.add_argument('--system-only', action='store_true', help='Run only system tests')
    parser.add_argument('--security-only', action='store_true', help='Run only security tests')
    parser.add_argument('--quality-only', action='store_true', help='Run only code quality checks')
    args = parser.parse_args()
    
    create_directories()
    
    # Determine which tests to run
    run_unit = not (args.integration_only or args.system_only or args.security_only or args.quality_only) or args.unit_only
    run_integration = not (args.unit_only or args.system_only or args.security_only or args.quality_only) or args.integration_only
    run_system = not (args.unit_only or args.integration_only or args.security_only or args.quality_only) or args.system_only
    run_security = not (args.unit_only or args.integration_only or args.system_only or args.quality_only) or args.security_only
    run_quality = not (args.unit_only or args.integration_only or args.system_only or args.security_only) or args.quality_only
    
    # Run tests
    unit_success = run_unit_tests() if run_unit else True
    integration_success = run_integration_tests() if run_integration else True
    system_success = run_system_tests() if run_system else True
    security_success = run_security_tests() if run_security else True
    quality_success = run_code_quality_checks() if run_quality else True
    
    # Generate coverage report if any tests were run
    if run_unit or run_integration or run_system or run_security:
        generate_coverage_report()
    
    # Generate summary report
    overall_success = generate_summary_report(
        unit_success, integration_success, system_success, security_success, quality_success
    )
    
    # Return exit code
    return 0 if overall_success else 1

if __name__ == '__main__':
    sys.exit(main())
