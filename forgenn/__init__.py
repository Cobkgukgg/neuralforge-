"""
ForgeNN - Neural networks in pure NumPy

Modern architectures (Transformers, ResNet) without the bloat.
Just NumPy, nothing else.

Quick start:
    >>> from forgenn import NeuralNetwork, Dense
    >>> model = NeuralNetwork()
    >>> model.add(Dense(10, 64, activation="gelu"))
    >>> model.add(Dense(64, 3, activation="softmax"))
    >>> model.compile(loss="categorical_crossentropy")
    >>> model.fit(X_train, y_train)

GitHub: https://github.com/Cobkgukgg/forgenn
"""

from .core import (
    # Core Classes
    NeuralNetwork,
    Layer,
    
    # Layers
    Dense,
    MultiHeadAttention,
    LayerNormalization,
    Conv2D,
    ResidualBlock,
    
    # Functions
    Activation,
    WeightInitializer,
    LossFunction,
    Optimizer,
    
    # Pre-built Architectures
    Architectures,
    
    # Configuration
    TrainingConfig,
    
    # Enums
    ActivationType,
    InitializationType,
)

__version__ = "1.0.0"
__author__ = "ForgeNN Contributors"
__license__ = "MIT"
__description__ = "Advanced Neural Network Framework with state-of-the-art architectures"

__all__ = [
    # Core
    "NeuralNetwork",
    "Layer",
    
    # Layers
    "Dense",
    "MultiHeadAttention",
    "LayerNormalization",
    "Conv2D",
    "ResidualBlock",
    
    # Functions
    "Activation",
    "WeightInitializer",
    "LossFunction",
    "Optimizer",
    
    # Architectures
    "Architectures",
    
    # Config
    "TrainingConfig",
    
    # Enums
    "ActivationType",
    "InitializationType",
]