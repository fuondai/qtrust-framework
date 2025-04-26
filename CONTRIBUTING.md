## CONTRIBUTING.md

# Contributing to QTrust

Thank you for your interest in contributing to QTrust, a cross-shard blockchain sharding framework with reinforcement learning and hierarchical trust mechanisms. This document provides guidelines and instructions for contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Workflow](#development-workflow)
4. [Coding Standards](#coding-standards)
5. [Documentation Guidelines](#documentation-guidelines)
6. [Testing Requirements](#testing-requirements)
7. [Submission Process](#submission-process)
8. [Review Process](#review-process)
9. [Community Resources](#community-resources)

## Code of Conduct

All contributors are expected to adhere to our [Code of Conduct](./CODE_OF_CONDUCT.md). By participating in this project, you agree to abide by its terms.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.9 or higher (optional for full functionality)
- Git

### Setting Up Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/qtrust.git
   cd qtrust
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```
5. Install the package in development mode:
   ```bash
   pip install -e .
   ```

## Development Workflow

### Branching Strategy

- `main`: Stable release branch
- `develop`: Integration branch for features
- `feature/*`: Feature development branches
- `bugfix/*`: Bug fix branches
- `release/*`: Release preparation branches

### Creating a New Feature

1. Ensure your fork is up to date with the upstream repository
2. Create a new branch from `develop`:
   ```bash
   git checkout develop
   git pull upstream develop
   git checkout -b feature/your-feature-name
   ```
3. Implement your changes, following the coding standards
4. Write tests for your changes
5. Update documentation as necessary
6. Commit your changes with clear, descriptive commit messages

## Coding Standards

QTrust follows strict coding standards to maintain code quality and consistency:

### Python Style Guide

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with a maximum line length of 100 characters
- Use [Black](https://github.com/psf/black) for code formatting with default settings
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Use [flake8](https://flake8.pycqa.org/) for linting

### Code Quality

- Maximum cyclomatic complexity of 15 per function
- Maximum 50 lines per function, 500 lines per file
- Use type hints for all function parameters and return values
- Document all classes and functions with docstrings

### Naming Conventions

- Classes: `CamelCase`
- Functions and variables: `snake_case`
- Constants: `UPPER_CASE_WITH_UNDERSCORES`
- Private methods and variables: `_leading_underscore`
- Module-level private functions: `_leading_underscore`

## Documentation Guidelines

All contributions must include appropriate documentation:

### Code Documentation

- All public APIs must have docstrings following the [NumPy docstring format](https://numpydoc.readthedocs.io/en/latest/format.html)
- Include parameter descriptions, return types, and exceptions
- Document complex algorithms with explanatory comments

### Technical Documentation

- Follow the [Academic Style Guide](./docs/style_guide.md) for all technical documentation
- Update relevant documentation files when changing functionality
- Create new documentation for new features
- Use proper terminology as defined in the [Terminology Glossary](./docs/terminology_glossary.md)

## Testing Requirements

All contributions must include appropriate tests:

### Test Coverage

- Minimum 90% line coverage, 85% branch coverage
- Unit tests for all new functionality
- Integration tests for component interactions
- Performance tests for performance-critical code

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=qtrust

# Run specific test categories
python -m pytest tests/unit
python -m pytest tests/integration
```

## Submission Process

### Pull Request Guidelines

1. Ensure all tests pass locally
2. Ensure code meets all style and quality requirements
3. Create a pull request against the `develop` branch
4. Fill out the pull request template completely
5. Link any related issues in the pull request description

### Commit Message Format

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

Types:
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code changes that neither fix bugs nor add features
- `perf`: Performance improvements
- `test`: Adding or correcting tests
- `chore`: Changes to the build process or auxiliary tools

## Review Process

All submissions go through a review process:

1. Automated checks (CI/CD pipeline)
   - Code style and linting
   - Test coverage
   - Documentation building
2. Peer review
   - Technical accuracy
   - Code quality
   - Design considerations
   - Security implications
3. Maintainer review
   - Project alignment
   - Integration considerations
   - Release planning

## Community Resources

- **Issue Tracker**: [GitHub Issues](https://github.com/qtrust/qtrust/issues)
- **Discussion Forum**: [GitHub Discussions](https://github.com/qtrust/qtrust/discussions)
- **Documentation**: [Official Documentation](https://qtrust.readthedocs.io/)
- **Slack Channel**: [QTrust Community](https://qtrust-community.slack.com)

## Acknowledgments

We appreciate all contributions to QTrust, whether they come in the form of code, documentation, bug reports, feature requests, or community support. Thank you for helping to improve the project!
