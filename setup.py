"""
Setup file for seamaware package.

Allows installation via pip install -e .
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="seamaware",
    version="0.2.0",
    author="Mac Mayo",
    author_email="macmayo1993@gmail.com",
    description="Non-orientable modeling for time series analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MacMayo1993/Seam-Aware-Modeling",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "statsmodels>=0.14.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.0.0",
            "black>=23.3.0",
            "mypy>=1.2.0",
            "jupyter>=1.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
    },
)
