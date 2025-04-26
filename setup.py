"""
Setup script for QTrust Blockchain Sharding Framework.
"""

from setuptools import setup, find_packages

setup(
    name="qtrust",
    version="1.0.0",
    description="QTrust: A Cross-Shard Blockchain Sharding Framework with Reinforcement Learning and Hierarchical Trust Mechanisms",
    author="Tuan-Dung Tran, Phuong-Dai Bui, Nguyen Tan Cam, Van-Hau Pham",
    author_email="info@qtrust.io",
    url="https://github.com/qtrust/qtrust",
    packages=find_packages(),
    install_requires=[
        "numpy>=2.2.0",
        "pandas>=2.2.0",
        "matplotlib>=3.8.0",
        "networkx>=3.4.0",
        "torch>=2.7.0",
        "torchvision>=0.22.0",
        "torchaudio>=2.7.0",
        "sympy>=1.13.0",
        "requests>=2.31.0",
        "docker>=7.1.0",
        "psutil>=7.0.0",
        "cryptography>=42.0.0",
        "pycryptodome>=3.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.3.0",
            "pytest-cov>=6.1.0",
            "black>=25.1.0",
            "flake8>=7.2.0",
            "mypy>=1.8.0",
            "pylint>=3.0.0",
            "bandit>=1.7.0",
            "interrogate>=1.5.0",
            "radon>=6.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
)
