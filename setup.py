#!/usr/bin/env python3
"""
Setup script for Bell Inequality Analysis package
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'docs', 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Bell Inequality Analysis for Financial Markets"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="bell-inequality-analysis",
    version="1.0.0",
    author="Bell Inequality Research Team",
    author_email="your-email@example.com",
    description="Bell Inequality Analysis for Financial Markets - Detecting Quantum-like Correlations",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/bell-inequality-analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Office/Business :: Financial",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "nbsphinx>=0.8",
        ],
        "research": [
            "jupyter>=1.0",
            "scikit-learn>=1.0",
            "networkx>=2.6",
            "statsmodels>=0.12",
        ]
    },
    entry_points={
        "console_scripts": [
            "bell-analysis=src.bell_inequality_analyzer:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.tex"],
    },
    keywords=[
        "bell inequality",
        "quantum finance",
        "financial markets",
        "quantum correlations",
        "econophysics",
        "market analysis",
        "financial data analysis",
        "quantum mechanics",
        "statistical analysis"
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/bell-inequality-analysis/issues",
        "Source": "https://github.com/yourusername/bell-inequality-analysis",
        "Documentation": "https://github.com/yourusername/bell-inequality-analysis/tree/main/docs",
        "Research": "https://github.com/yourusername/bell-inequality-analysis/discussions",
    },
)