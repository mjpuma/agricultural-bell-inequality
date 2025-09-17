#!/usr/bin/env python3
"""
Setup script for Agricultural Cross-Sector Bell Inequality Analysis
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="agricultural-bell-inequality",
    version="1.0.0",
    author="Agricultural Cross-Sector Analysis Team",
    author_email="your.email@example.com",
    description="Detecting quantum-like correlations in global food systems using Bell inequality tests",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/[username]/agricultural-bell-inequality",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "jupyter>=1.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "agricultural-bell-demo=examples.enhanced_s1_demo:main",
        ],
    },
    keywords=[
        "bell inequality",
        "quantum correlations", 
        "agricultural finance",
        "food systems",
        "crisis detection",
        "supply chain analysis",
        "zarifian methodology"
    ],
    project_urls={
        "Bug Reports": "https://github.com/[username]/agricultural-bell-inequality/issues",
        "Source": "https://github.com/[username]/agricultural-bell-inequality",
        "Documentation": "https://github.com/[username]/agricultural-bell-inequality/tree/main/docs",
    },
)