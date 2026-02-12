"""
Simple test suite for ForgeNN
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from forgenn import (
    NeuralNetwork, Dense, Architectures,
    TrainingConfig, Activation, WeightInitializer
)

def test_activations():
    """Test activation functions"""
    print("Testing activations...")
    x = np.array([[1.0, -1.0, 0.0]])
    
    for act_name in ['relu', 'gelu', 'swish', 'mish']:
        act = getattr(Activation, act_name)
        output = act(x)
        assert output.shape == x.shape
    print("✓ Activations test passed")

def test_model_creation():
    """Test model creation"""
    print("Testing model creation...")
    model = NeuralNetwork()
    model.add(Dense(10, 20, activation="relu"))
    model.add(Dense(20, 5, activation="softmax"))
    assert len(model.layers) == 2
    print("✓ Model creation test passed")

def test_training():
    """Test basic training"""
    print("Testing training...")
    X = np.random.randn(100, 10)
    y = np.eye(3)[np.random.randint(0, 3, 100)]
    
    model = Architectures.mlp(10, [32], 3)
    model.compile(loss="categorical_crossentropy")
    
    config = TrainingConfig(epochs=5, batch_size=20, verbose=False)
    model.fit(X, y, config=config)
    
    print("✓ Training test passed")

if __name__ == "__main__":
    print("\nRunning ForgeNN Tests")
    print("="*50)
    test_activations()
    test_model_creation()
    test_training()
    print("="*50)
    print("All tests passed! ✓\n")