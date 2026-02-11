# Quick Start Guide - NeuralForge

## Installation

```bash
git clone https://github.com/yourusername/neuralforge.git
cd neuralforge
pip install -r requirements.txt
```

## 5-Minute Tutorial

### 1. Simple Classification

```python
from neuralforge import NeuralNetwork, Dense, TrainingConfig
import numpy as np

# Generate data
X = np.random.randn(1000, 10)
y = np.eye(3)[np.random.randint(0, 3, 1000)]

# Build model
model = NeuralNetwork()
model.add(Dense(10, 64, activation="relu"))
model.add(Dense(64, 3, activation="softmax"))

# Train
model.compile(loss="categorical_crossentropy")
model.fit(X, y, TrainingConfig(epochs=50))

# Predict
predictions = model.predict(X[:5])
```

### 2. Use Pre-built Architecture

```python
from neuralforge import Architectures

# ResNet
model = Architectures.resnet(
    input_dim=100,
    num_blocks=3,
    hidden_dim=128,
    output_dim=10
)

model.compile(loss="categorical_crossentropy")
model.fit(X_train, y_train)
```

### 3. Advanced Configuration

```python
from neuralforge import TrainingConfig

config = TrainingConfig(
    learning_rate=0.001,
    batch_size=64,
    epochs=100,
    dropout_rate=0.3,
    early_stopping=True,
    patience=10,
    validation_split=0.2
)

model.fit(X_train, y_train, config=config)
```

## Key Features

### Activation Functions
- `relu` - Standard ReLU
- `gelu` - Used in BERT, GPT
- `swish` - Used in EfficientNet
- `mish` - State-of-the-art
- `leaky_relu`, `elu`, `sigmoid`, `tanh`

### Loss Functions
- `mse` - Mean Squared Error
- `mae` - Mean Absolute Error
- `binary_crossentropy` - Binary classification
- `categorical_crossentropy` - Multi-class
- `huber` - Robust regression

### Optimizers
- `adam` - Adaptive learning rate (default)
- `sgd` - Stochastic Gradient Descent

## Examples

Check the `examples/` directory:
- `example_1_image_classification.py` - MNIST-style classification
- `example_2_transformer.py` - Transformer architecture
- `example_3_resnet.py` - ResNet for regression
- `example_4_advanced_architecture.py` - Custom advanced model

## Running Examples

```bash
cd examples
python example_1_image_classification.py
```

## Testing

```bash
cd tests
python test_neuralforge.py
```

## Benchmarking

```bash
cd benchmarks
python benchmark.py
```

## Next Steps

1. Read the full [README.md](README.md)
2. Check [CONTRIBUTING.md](CONTRIBUTING.md) to contribute
3. Explore the examples
4. Build your own architectures!

## Common Patterns

### Save/Load Models

```python
# Save
model.save("my_model.pkl")

# Load
model = NeuralNetwork().load("my_model.pkl")
```

### Model Summary

```python
model.summary()
```

### Access Training History

```python
history = model.fit(X, y, config)

print(history['train_loss'])
print(history['val_loss'])
```

## Tips

1. **Start simple** - Use pre-built architectures
2. **Monitor training** - Use `verbose=True` and validation split
3. **Use early stopping** - Prevent overfitting
4. **Normalize data** - Always normalize inputs
5. **Try different activations** - GELU often works well

## Help

- GitHub Issues: Report bugs or ask questions
- Examples: Check working code in `examples/`
- Tests: See `tests/` for usage patterns
