"""
ForgeNN Setup Script
For backward compatibility - use pyproject.toml for modern installations
"""

from setuptools import setup, find_packages
import os

# Read the README file
current_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(current_dir, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="forgenn",
    version="1.0.0",
    author="ForgeNN Contributors",
    author_email="nb652990@gmail.com",
    description="Modern neural networks in pure NumPy - Transformers, ResNet, and more",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Cobkgukgg/forgenn",
    project_urls={
        "Bug Tracker": "https://github.com/Cobkgukgg/forgenn/issues",
        "Documentation": "https://github.com/Cobkgukgg/forgenn#readme",
        "Source Code": "https://github.com/Cobkgukgg/forgenn",
        "Changelog": "https://github.com/Cobkgukgg/forgenn/blob/main/CHANGELOG.md",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*", "benchmarks", "benchmarks.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "matplotlib>=3.3.0",
            "jupyter>=1.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "viz": [
            "matplotlib>=3.3.0",
        ],
    },
    keywords="deep-learning neural-network machine-learning ai transformer resnet attention gelu numpy",
    include_package_data=True,
    zip_safe=False,
)