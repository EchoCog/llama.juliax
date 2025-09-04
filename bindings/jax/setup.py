"""
Setup script for JAX Fusion bindings
"""

from setuptools import setup, find_packages

setup(
    name="llama-jax-fusion",
    version="0.1.0",
    description="JAX integration for llama.cpp Fusion Language",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="Fusion Language Team",
    author_email="fusion@example.com",
    url="https://github.com/EchoCog/llama.juliax",
    packages=find_packages(),
    package_dir={"": "src"},
    install_requires=[
        "jax>=0.4.0",
        "jaxlib>=0.4.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
        ],
        "gpu": [
            "jax[cuda]>=0.4.0",
        ],
        "tpu": [
            "jax[tpu]>=0.4.0", 
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        "jax",
        "automatic-differentiation", 
        "llama.cpp",
        "fusion-language",
        "mathematical-computing",
        "array-programming",
    ],
    project_urls={
        "Bug Reports": "https://github.com/EchoCog/llama.juliax/issues",
        "Source": "https://github.com/EchoCog/llama.juliax",
        "Documentation": "https://github.com/EchoCog/llama.juliax/docs",
    },
)