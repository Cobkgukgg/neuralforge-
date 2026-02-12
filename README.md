# ForgeNN

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![PyPI](https://img.shields.io/badge/pip-install%20forgenn-blue)

**Modern neural network framework built from scratch with NumPy**

I got tired of the bloated ML frameworks that hide everything behind abstractions, so I built this. It's a fully-functional deep learning library that implements modern architectures (Transformers, ResNet, attention mechanisms) using just NumPy. 

Why? Because sometimes you need to actually understand what's happening under the hood. And because I can.

---

## What's in here?

### The good stuff
- **Transformer encoder** - yeah, the attention mechanism everyone's talking about
- **ResNet blocks** - because deep networks are cool
- **Modern activations** - GELU (GPT uses this), Swish (EfficientNet), Mish, and the classics
- **Smart initialization** - Xavier, He, LeCun, Orthogonal (actually matters)

### Features I actually use
- Multi-head self-attention
- Layer normalization  
- Dropout (because overfitting is real)
- Early stopping (saves time)
- Adam optimizer (because it just works)
- Model save/load (obviously)

### What makes this different
No TensorFlow. No PyTorch. Just NumPy and math.

You can actually read the code and understand what's happening. Try doing that with PyTorch's C++ backend.

## Setup

### Install from PyPI (recommended)
```bash
pip install forgenn-ml
```

### Or install from source
```bash
git clone https://github.com/cobkgukgg/forgenn.git
cd forgenn
pip install -e .
```

That's it. Seriously, just NumPy.

```
numpy>=1.19.0
```

## Quick example

Build a network in like 10 lines:

```python
from forgenn import NeuralNetwork, Dense, TrainingConfig
import numpy as np

# some random data
X = np.random.randn(1000, 10)
y = np.random.randint(0, 2, (1000, 1))

# build it
model = NeuralNetwork("MyFirstModel")
model.add(Dense(10, 64, activation="relu"))
model.add(Dense(64, 32, activation="gelu"))  # gelu because why not
model.add(Dense(32, 1, activation="sigmoid"))

# train it
model.compile(loss="binary_crossentropy", optimizer="adam")
model.fit(X, y, TrainingConfig(epochs=100, batch_size=32))

# use it
predictions = model.predict(X[:5])
```

### Or use pre-built stuff

I already made some common architectures:

```python
from forgenn import Architectures

# ResNet for when you need to go deep
model = Architectures.resnet(
    input_dim=784,
    num_blocks=3,
    hidden_dim=128,
    output_dim=10
)

# Transformer because transformers are everywhere now
model = Architectures.transformer_encoder(
    input_dim=512,
    num_heads=8,
    ff_dim=2048,
    num_layers=6
)
```

### Config stuff

You can tweak things:

```python
from forgenn import TrainingConfig

config = TrainingConfig(
    learning_rate=0.001,      # standard
    batch_size=64,            # bigger = faster but needs more RAM
    epochs=200,               # or until early stopping kicks in
    dropout_rate=0.3,         # helps with overfitting
    early_stopping=True,      # stop when val loss stops improving
    patience=15,              # how long to wait
    validation_split=0.2      # use 20% for validation
)

history = model.fit(X_train, y_train, config)
```

## How it works

### Layers you can use

```python
Dense(input_size, output_size, 
      activation="relu",
      dropout_rate=0.0)

MultiHeadAttention(embed_dim, num_heads, dropout=0.1)

LayerNormalization(normalized_shape)

Conv2D(in_channels, out_channels, 
       kernel_size=3, stride=1, padding=0)

ResidualBlock(dim, activation="relu")
```

### Activations

| What | When to use |
|------|-------------|
| `relu` | Default choice, works most of the time |
| `gelu` | Transformers (GPT, BERT use this) |
| `swish` | Good for mobile/efficient networks |
| `mish` | Newer, slightly better than ReLU |
| `leaky_relu` | When you get dead neurons |

### Loss functions

- `mse` - regression
- `mae` - regression (robust to outliers)
- `binary_crossentropy` - binary classification  
- `categorical_crossentropy` - multi-class
- `huber` - regression with outliers

### Model Methods

```python
# Add layer
model.add(layer)

# Compile
model.compile(loss="mse", optimizer="adam")

# Train
model.fit(X_train, y_train, config=config)

# Predict
predictions = model.predict(X_test)

# Evaluate
results = model.evaluate(X_test, y_test)

# Save/Load
model.save("model.pkl")
model.load("model.pkl")

# Summary
model.summary()
```

## Examples

### MNIST-style classification

```python
import numpy as np
from forgenn import Architectures, TrainingConfig

# flatten those images
X_train = train_images.reshape(-1, 784) / 255.0
y_train = np.eye(10)[train_labels]

# build something that works
model = Architectures.mlp(
    input_dim=784,
    hidden_dims=[256, 128],
    output_dim=10,
    activation="gelu"
)

model.compile(loss="categorical_crossentropy", optimizer="adam")

config = TrainingConfig(
    learning_rate=0.001,
    batch_size=128,
    epochs=50,
    early_stopping=True
)

model.fit(X_train, y_train, config)

# check how we did
results = model.evaluate(X_test, y_test)
print(f"Accuracy: {results['accuracy']:.4f}")
```

### Custom architecture

Mix and match whatever you want:

```python
from forgenn import NeuralNetwork, Dense, ResidualBlock, LayerNormalization

model = NeuralNetwork("MyCustomNet")

model.add(Dense(100, 256, activation="gelu", dropout_rate=0.3))
model.add(LayerNormalization(256))

# throw in some residual blocks
for _ in range(3):
    model.add(ResidualBlock(256, activation="mish"))

model.add(Dense(256, 10, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam")
model.summary()
```

## Some notes

### Why GELU?

Used in GPT and BERT. Smoother than ReLU, works better for NLP stuff. The math is kinda cool:

```python
GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
```

### Residual connections

The thing that made deep networks actually work:

```
output = F(x) + x
```

Gradients can flow back easier. Without this, training deep networks is pain.

### Performance

Tested on my laptop (i7, 16GB RAM, no GPU):

| Dataset | Model | Accuracy | Time |
|---------|-------|----------|------|
| MNIST | MLP | ~98% | 2 min |
| MNIST | ResNet | ~99% | 4 min |
| CIFAR-10 | ResNet | ~75% | 15 min |

Not bad for pure Python/NumPy.

## Todo

Things I might add:

- [ ] Batch normalization
- [ ] LSTM/GRU layers  
- [ ] Better conv layers
- [ ] GPU support (CuPy?)
- [ ] Model visualization
- [ ] Data loaders
- [ ] More optimizers

Pull requests welcome.

## Contributing

Found a bug? Want to add something? PRs are open. 

Just keep it clean and add tests.

## License

MIT - do whatever you want with it

---

Made this because I was bored and wanted to actually understand how transformers work. Turned out pretty decent.

If you use this for something cool, let me know!
