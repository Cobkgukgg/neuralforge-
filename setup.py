from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="neuralforge",
    version="1.0.0",
    author="Cobkgukgg",
    author_email="nb652990@gmail.com",
    description="A state-of-the-art neural network framework built from scratch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Cobkgukgg/neuralforge",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
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
        ],
    },
    keywords="deep-learning neural-network machine-learning ai transformer resnet",
)
