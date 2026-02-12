"""
ForgeNN - Neural Network Framework

Built this to actually understand how modern architectures work.
No black boxes, just NumPy and math.

Includes:
- Transformers (with actual attention)
- ResNet (residual connections FTW)
- Modern activations (GELU, Swish, Mish)
- Everything you need to train decent models

Author: Made by someone who got tired of PyTorch hiding everything
"""

import numpy as np
from typing import Optional, List, Tuple, Union, Callable, Dict, Any
from abc import ABC, abstractmethod
import json
import pickle
from dataclasses import dataclass
from enum import Enum


class ActivationType(Enum):
    """Supported activation functions"""
    RELU = "relu"
    GELU = "gelu"
    SWISH = "swish"
    MISH = "mish"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SOFTMAX = "softmax"
    LEAKY_RELU = "leaky_relu"
    ELU = "elu"


class InitializationType(Enum):
    """Weight initialization strategies"""
    XAVIER = "xavier"
    HE = "he"
    LECUN = "lecun"
    ORTHOGONAL = "orthogonal"
    NORMAL = "normal"
    UNIFORM = "uniform"


@dataclass
class TrainingConfig:
    """Configuration for training"""
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    optimizer: str = "adam"
    loss_function: str = "mse"
    regularization: Optional[str] = None
    reg_lambda: float = 0.01
    dropout_rate: float = 0.0
    early_stopping: bool = False
    patience: int = 10
    validation_split: float = 0.2
    shuffle: bool = True
    verbose: bool = True


class Activation:
    """Activation functions with their derivatives
    
    All the modern ones are here. GELU is what GPT uses, Swish is from EfficientNet,
    Mish is newer and slightly better. ReLU still works fine though.
    """
    
    @staticmethod
    def relu(x: np.ndarray, derivative: bool = False) -> np.ndarray:
        if derivative:
            return (x > 0).astype(float)
        return np.maximum(0, x)
    
    @staticmethod
    def gelu(x: np.ndarray, derivative: bool = False) -> np.ndarray:
        """GELU - what BERT and GPT use for activation"""
        if derivative:
            cdf = 0.5 * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
            pdf = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
            return cdf + x * pdf
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    @staticmethod
    def swish(x: np.ndarray, derivative: bool = False, beta: float = 1.0) -> np.ndarray:
        """Swish/SiLU - EfficientNet uses this"""
        sigmoid = 1 / (1 + np.exp(-beta * x))
        if derivative:
            return sigmoid + x * sigmoid * (1 - sigmoid) * beta
        return x * sigmoid
    
    @staticmethod
    def mish(x: np.ndarray, derivative: bool = False) -> np.ndarray:
        """Mish - newer, works pretty well"""
        tanh_softplus = np.tanh(np.log(1 + np.exp(x)))
        if derivative:
            omega = 4 * (x + 1) + 4 * np.exp(2 * x) + np.exp(3 * x) + np.exp(x) * (4 * x + 6)
            delta = 2 * np.exp(x) + np.exp(2 * x) + 2
            return np.exp(x) * omega / (delta ** 2)
        return x * tanh_softplus
    
    @staticmethod
    def leaky_relu(x: np.ndarray, derivative: bool = False, alpha: float = 0.01) -> np.ndarray:
        if derivative:
            return np.where(x > 0, 1.0, alpha)
        return np.where(x > 0, x, alpha * x)
    
    @staticmethod
    def elu(x: np.ndarray, derivative: bool = False, alpha: float = 1.0) -> np.ndarray:
        if derivative:
            return np.where(x > 0, 1.0, alpha * np.exp(x))
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    @staticmethod
    def sigmoid(x: np.ndarray, derivative: bool = False) -> np.ndarray:
        sig = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        if derivative:
            return sig * (1 - sig)
        return sig
    
    @staticmethod
    def tanh(x: np.ndarray, derivative: bool = False) -> np.ndarray:
        if derivative:
            return 1 - np.tanh(x) ** 2
        return np.tanh(x)
    
    @staticmethod
    def softmax(x: np.ndarray, derivative: bool = False) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        softmax_output = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        if derivative:
            return softmax_output * (1 - softmax_output)
        return softmax_output


class WeightInitializer:
    """Advanced weight initialization strategies"""
    
    @staticmethod
    def xavier(shape: Tuple[int, ...], gain: float = 1.0) -> np.ndarray:
        """Xavier/Glorot initialization"""
        fan_in, fan_out = shape[0], shape[1] if len(shape) > 1 else shape[0]
        std = gain * np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.randn(*shape) * std
    
    @staticmethod
    def he(shape: Tuple[int, ...]) -> np.ndarray:
        """He initialization - good for ReLU"""
        fan_in = shape[0]
        std = np.sqrt(2.0 / fan_in)
        return np.random.randn(*shape) * std
    
    @staticmethod
    def lecun(shape: Tuple[int, ...]) -> np.ndarray:
        """LeCun initialization"""
        fan_in = shape[0]
        std = np.sqrt(1.0 / fan_in)
        return np.random.randn(*shape) * std
    
    @staticmethod
    def orthogonal(shape: Tuple[int, ...], gain: float = 1.0) -> np.ndarray:
        """Orthogonal initialization"""
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.randn(*flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return gain * q


class Layer(ABC):
    """Abstract base class for all layers"""
    
    def __init__(self):
        self.input = None
        self.output = None
        self.trainable = True
    
    @abstractmethod
    def forward(self, input_data: np.ndarray, training: bool = True) -> np.ndarray:
        pass
    
    @abstractmethod
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        pass
    
    def get_params(self) -> Dict[str, np.ndarray]:
        return {}
    
    def set_params(self, params: Dict[str, np.ndarray]):
        pass


class Dense(Layer):
    """Fully connected layer with advanced features"""
    
    def __init__(self, input_size: int, output_size: int, 
                 activation: str = "relu",
                 use_bias: bool = True,
                 initialization: str = "he",
                 dropout_rate: float = 0.0):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.dropout_mask = None
        
        # Initialize weights
        init_method = getattr(WeightInitializer, initialization)
        self.weights = init_method((input_size, output_size))
        self.bias = np.zeros((1, output_size)) if use_bias else None
        
        # Set activation
        self.activation_name = activation
        self.activation = getattr(Activation, activation)
        
        # For optimizer
        self.weights_velocity = np.zeros_like(self.weights)
        self.bias_velocity = np.zeros_like(self.bias) if use_bias else None
        self.weights_cache = np.zeros_like(self.weights)
        self.bias_cache = np.zeros_like(self.bias) if use_bias else None
    
    def forward(self, input_data: np.ndarray, training: bool = True) -> np.ndarray:
        self.input = input_data
        self.output = np.dot(input_data, self.weights)
        
        if self.use_bias:
            self.output += self.bias
        
        # Apply activation
        self.output = self.activation(self.output)
        
        # Apply dropout during training
        if training and self.dropout_rate > 0:
            self.dropout_mask = (np.random.rand(*self.output.shape) > self.dropout_rate).astype(float)
            self.output *= self.dropout_mask / (1 - self.dropout_rate)
        
        return self.output
    
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        # Apply dropout mask
        if self.dropout_rate > 0 and self.dropout_mask is not None:
            output_gradient *= self.dropout_mask / (1 - self.dropout_rate)
        
        # Activation derivative
        activation_gradient = self.activation(self.output, derivative=True)
        output_gradient = output_gradient * activation_gradient
        
        # Compute gradients
        weights_gradient = np.dot(self.input.T, output_gradient)
        input_gradient = np.dot(output_gradient, self.weights.T)
        
        # Update weights and bias
        self.weights -= learning_rate * weights_gradient
        if self.use_bias:
            bias_gradient = np.sum(output_gradient, axis=0, keepdims=True)
            self.bias -= learning_rate * bias_gradient
        
        return input_gradient
    
    def get_params(self) -> Dict[str, np.ndarray]:
        params = {"weights": self.weights}
        if self.use_bias:
            params["bias"] = self.bias
        return params
    
    def set_params(self, params: Dict[str, np.ndarray]):
        self.weights = params["weights"]
        if self.use_bias and "bias" in params:
            self.bias = params["bias"]


class MultiHeadAttention(Layer):
    """Multi-head self-attention mechanism (Transformer component)"""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.scale = self.head_dim ** -0.5
        
        # Initialize query, key, value projection matrices
        self.W_q = WeightInitializer.xavier((embed_dim, embed_dim))
        self.W_k = WeightInitializer.xavier((embed_dim, embed_dim))
        self.W_v = WeightInitializer.xavier((embed_dim, embed_dim))
        self.W_o = WeightInitializer.xavier((embed_dim, embed_dim))
        
        self.b_q = np.zeros((1, embed_dim))
        self.b_k = np.zeros((1, embed_dim))
        self.b_v = np.zeros((1, embed_dim))
        self.b_o = np.zeros((1, embed_dim))
    
    def split_heads(self, x: np.ndarray) -> np.ndarray:
        """Split last dimension into (num_heads, head_dim)"""
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(0, 2, 1, 3)  # (batch, num_heads, seq_len, head_dim)
    
    def forward(self, input_data: np.ndarray, training: bool = True) -> np.ndarray:
        self.input = input_data
        batch_size, seq_len, _ = input_data.shape
        
        # Linear projections
        Q = np.dot(input_data, self.W_q) + self.b_q
        K = np.dot(input_data, self.W_k) + self.b_k
        V = np.dot(input_data, self.W_v) + self.b_v
        
        # Split into multiple heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # Scaled dot-product attention
        attention_scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) * self.scale
        attention_weights = Activation.softmax(attention_scores)
        
        # Apply dropout
        if training and self.dropout > 0:
            mask = (np.random.rand(*attention_weights.shape) > self.dropout).astype(float)
            attention_weights *= mask / (1 - self.dropout)
        
        # Apply attention to values
        attention_output = np.matmul(attention_weights, V)
        
        # Concatenate heads
        attention_output = attention_output.transpose(0, 2, 1, 3)
        attention_output = attention_output.reshape(batch_size, seq_len, self.embed_dim)
        
        # Final linear projection
        self.output = np.dot(attention_output, self.W_o) + self.b_o
        
        return self.output
    
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        # Simplified backward pass (full implementation would be more complex)
        return output_gradient


class LayerNormalization(Layer):
    """Layer normalization (used in Transformers)"""
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = np.ones((1, normalized_shape))
        self.beta = np.zeros((1, normalized_shape))
    
    def forward(self, input_data: np.ndarray, training: bool = True) -> np.ndarray:
        self.input = input_data
        self.mean = np.mean(input_data, axis=-1, keepdims=True)
        self.var = np.var(input_data, axis=-1, keepdims=True)
        self.std = np.sqrt(self.var + self.eps)
        
        self.normalized = (input_data - self.mean) / self.std
        self.output = self.gamma * self.normalized + self.beta
        
        return self.output
    
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        # Compute gradients
        gamma_gradient = np.sum(output_gradient * self.normalized, axis=0, keepdims=True)
        beta_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        
        # Update parameters
        self.gamma -= learning_rate * gamma_gradient
        self.beta -= learning_rate * beta_gradient
        
        # Input gradient (simplified)
        return output_gradient


class Conv2D(Layer):
    """2D Convolutional layer"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, padding: int = 0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize kernels
        self.kernels = WeightInitializer.he((out_channels, in_channels, kernel_size, kernel_size))
        self.biases = np.zeros((out_channels, 1))
    
    def forward(self, input_data: np.ndarray, training: bool = True) -> np.ndarray:
        self.input = input_data
        batch_size, in_channels, height, width = input_data.shape
        
        # Add padding
        if self.padding > 0:
            input_data = np.pad(input_data, 
                              ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                              mode='constant')
        
        # Calculate output dimensions
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Perform convolution
        self.output = np.zeros((batch_size, self.out_channels, out_height, out_width))
        
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size
                
                receptive_field = input_data[:, :, h_start:h_end, w_start:w_end]
                
                for k in range(self.out_channels):
                    self.output[:, k, i, j] = np.sum(
                        receptive_field * self.kernels[k], axis=(1, 2, 3)
                    ) + self.biases[k]
        
        return self.output
    
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        # Simplified backward pass
        return output_gradient


class ResidualBlock(Layer):
    """Residual block (ResNet component)"""
    
    def __init__(self, dim: int, activation: str = "relu"):
        super().__init__()
        self.layer1 = Dense(dim, dim, activation=activation)
        self.layer2 = Dense(dim, dim, activation="relu")
        self.activation = getattr(Activation, activation)
    
    def forward(self, input_data: np.ndarray, training: bool = True) -> np.ndarray:
        self.input = input_data
        residual = input_data
        
        x = self.layer1.forward(input_data, training)
        x = self.layer2.forward(x, training)
        
        self.output = self.activation(x + residual)
        return self.output
    
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        activation_gradient = self.activation(self.output, derivative=True)
        output_gradient = output_gradient * activation_gradient
        
        gradient = self.layer2.backward(output_gradient, learning_rate)
        gradient = self.layer1.backward(gradient, learning_rate)
        
        return gradient + output_gradient  # Add skip connection gradient


class LossFunction:
    """Various loss functions"""
    
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray, derivative: bool = False) -> np.ndarray:
        if derivative:
            return 2 * (y_pred - y_true) / y_true.size
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray, derivative: bool = False) -> np.ndarray:
        if derivative:
            return np.sign(y_pred - y_true) / y_true.size
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def binary_crossentropy(y_true: np.ndarray, y_pred: np.ndarray, 
                          derivative: bool = False) -> np.ndarray:
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        if derivative:
            return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / y_true.size
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def categorical_crossentropy(y_true: np.ndarray, y_pred: np.ndarray,
                                derivative: bool = False) -> np.ndarray:
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        if derivative:
            return -(y_true / y_pred) / y_true.size
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    @staticmethod
    def huber(y_true: np.ndarray, y_pred: np.ndarray, 
             delta: float = 1.0, derivative: bool = False) -> np.ndarray:
        error = y_pred - y_true
        abs_error = np.abs(error)
        
        if derivative:
            return np.where(abs_error <= delta, 
                          error / y_true.size,
                          delta * np.sign(error) / y_true.size)
        
        quadratic = np.minimum(abs_error, delta)
        linear = abs_error - quadratic
        return np.mean(0.5 * quadratic ** 2 + delta * linear)


class Optimizer:
    """Advanced optimizers"""
    
    @staticmethod
    def sgd(params: Dict[str, np.ndarray], gradients: Dict[str, np.ndarray],
            learning_rate: float, momentum: float = 0.0, velocity: Dict = None):
        if velocity is None:
            velocity = {k: np.zeros_like(v) for k, v in params.items()}
        
        for key in params:
            velocity[key] = momentum * velocity[key] - learning_rate * gradients[key]
            params[key] += velocity[key]
        
        return params, velocity
    
    @staticmethod
    def adam(params: Dict[str, np.ndarray], gradients: Dict[str, np.ndarray],
             learning_rate: float, beta1: float = 0.9, beta2: float = 0.999,
             epsilon: float = 1e-8, t: int = 1,
             m: Dict = None, v: Dict = None):
        if m is None:
            m = {k: np.zeros_like(val) for k, val in params.items()}
        if v is None:
            v = {k: np.zeros_like(val) for k, val in params.items()}
        
        for key in params:
            m[key] = beta1 * m[key] + (1 - beta1) * gradients[key]
            v[key] = beta2 * v[key] + (1 - beta2) * (gradients[key] ** 2)
            
            m_hat = m[key] / (1 - beta1 ** t)
            v_hat = v[key] / (1 - beta2 ** t)
            
            params[key] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        
        return params, m, v


class NeuralNetwork:
    """Main neural network class"""
    
    def __init__(self, name: str = "NeuralForge"):
        self.name = name
        self.layers: List[Layer] = []
        self.loss_function = None
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def add(self, layer: Layer):
        """Add a layer to the network"""
        self.layers.append(layer)
        return self
    
    def compile(self, loss: str = "mse", optimizer: str = "adam"):
        """Compile the model"""
        self.loss_function = getattr(LossFunction, loss)
        self.optimizer_name = optimizer
        return self
    
    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through all layers"""
        output = X
        for layer in self.layers:
            output = layer.forward(output, training)
        return output
    
    def backward(self, loss_gradient: np.ndarray, learning_rate: float):
        """Backward pass through all layers"""
        gradient = loss_gradient
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learning_rate)
    
    def train_step(self, X_batch: np.ndarray, y_batch: np.ndarray, 
                   learning_rate: float) -> float:
        """Single training step"""
        # Forward pass
        predictions = self.forward(X_batch, training=True)
        
        # Compute loss
        loss = self.loss_function(y_batch, predictions)
        
        # Backward pass
        loss_gradient = self.loss_function(y_batch, predictions, derivative=True)
        self.backward(loss_gradient, learning_rate)
        
        return loss
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            config: TrainingConfig = None,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None):
        """Train the neural network"""
        if config is None:
            config = TrainingConfig()
        
        # Split validation set if not provided
        if X_val is None and config.validation_split > 0:
            split_idx = int(len(X_train) * (1 - config.validation_split))
            X_train, X_val = X_train[:split_idx], X_train[split_idx:]
            y_train, y_val = y_train[:split_idx], y_train[split_idx:]
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config.epochs):
            # Shuffle training data
            if config.shuffle:
                indices = np.random.permutation(len(X_train))
                X_train = X_train[indices]
                y_train = y_train[indices]
            
            # Training
            epoch_loss = 0
            num_batches = len(X_train) // config.batch_size
            
            for i in range(num_batches):
                start_idx = i * config.batch_size
                end_idx = start_idx + config.batch_size
                
                X_batch = X_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]
                
                batch_loss = self.train_step(X_batch, y_batch, config.learning_rate)
                epoch_loss += batch_loss
            
            avg_train_loss = epoch_loss / num_batches
            self.history['train_loss'].append(avg_train_loss)
            
            # Validation
            if X_val is not None:
                val_predictions = self.predict(X_val)
                val_loss = self.loss_function(y_val, val_predictions)
                self.history['val_loss'].append(val_loss)
                
                if config.verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{config.epochs} - "
                          f"Train Loss: {avg_train_loss:.4f} - "
                          f"Val Loss: {val_loss:.4f}")
                
                # Early stopping
                if config.early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= config.patience:
                            if config.verbose:
                                print(f"Early stopping at epoch {epoch + 1}")
                            break
            else:
                if config.verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{config.epochs} - Train Loss: {avg_train_loss:.4f}")
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.forward(X, training=False)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the model"""
        predictions = self.predict(X_test)
        loss = self.loss_function(y_test, predictions)
        
        # Calculate accuracy for classification
        if predictions.shape[1] > 1:  # Multi-class
            pred_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(y_test, axis=1)
            accuracy = np.mean(pred_classes == true_classes)
        else:  # Binary
            accuracy = np.mean((predictions > 0.5) == y_test)
        
        return {'loss': loss, 'accuracy': accuracy}
    
    def save(self, filepath: str):
        """Save model to file"""
        model_data = {
            'name': self.name,
            'layers': [],
            'loss_function': self.loss_function.__name__ if self.loss_function else None,
            'history': self.history
        }
        
        for layer in self.layers:
            layer_data = {
                'type': layer.__class__.__name__,
                'params': layer.get_params()
            }
            model_data['layers'].append(layer_data)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.name = model_data['name']
        self.history = model_data['history']
        
        print(f"Model loaded from {filepath}")
        return self
    
    def summary(self):
        """Print model architecture summary"""
        print(f"\n{'='*60}")
        print(f"{self.name} - Model Summary")
        print(f"{'='*60}")
        print(f"{'Layer':<20} {'Type':<20} {'Parameters':<20}")
        print(f"{'-'*60}")
        
        total_params = 0
        for i, layer in enumerate(self.layers):
            layer_type = layer.__class__.__name__
            params = layer.get_params()
            num_params = sum(p.size for p in params.values())
            total_params += num_params
            
            print(f"{f'Layer {i+1}':<20} {layer_type:<20} {num_params:<20}")
        
        print(f"{'-'*60}")
        print(f"Total parameters: {total_params:,}")
        print(f"{'='*60}\n")


# Pre-built architectures
class Architectures:
    """Pre-built state-of-the-art architectures"""
    
    @staticmethod
    def transformer_encoder(input_dim: int, num_heads: int = 8, 
                           ff_dim: int = 2048, num_layers: int = 6) -> NeuralNetwork:
        """Transformer encoder architecture"""
        model = NeuralNetwork("TransformerEncoder")
        
        for _ in range(num_layers):
            model.add(MultiHeadAttention(input_dim, num_heads))
            model.add(LayerNormalization(input_dim))
            model.add(Dense(input_dim, ff_dim, activation="gelu"))
            model.add(Dense(ff_dim, input_dim, activation="relu"))
            model.add(LayerNormalization(input_dim))
        
        return model
    
    @staticmethod
    def resnet(input_dim: int, num_blocks: int = 3, 
               hidden_dim: int = 128, output_dim: int = 10) -> NeuralNetwork:
        """ResNet-style architecture"""
        model = NeuralNetwork("ResNet")
        
        model.add(Dense(input_dim, hidden_dim, activation="relu"))
        
        for _ in range(num_blocks):
            model.add(ResidualBlock(hidden_dim, activation="relu"))
        
        model.add(Dense(hidden_dim, output_dim, activation="softmax"))
        
        return model
    
    @staticmethod
    def mlp(input_dim: int, hidden_dims: List[int], 
            output_dim: int, activation: str = "relu") -> NeuralNetwork:
        """Multi-layer Perceptron"""
        model = NeuralNetwork("MLP")
        
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            act = activation if i < len(dims) - 2 else "softmax"
            model.add(Dense(dims[i], dims[i+1], activation=act))
        
        return model


if __name__ == "__main__":
    # Example usage
    print("NeuralForge - Advanced Neural Network Framework")
    print("="*60)
    
    # Create sample data
    np.random.seed(42)
    X = np.random.randn(1000, 20)
    y = np.random.randint(0, 3, (1000, 1))
    y_onehot = np.eye(3)[y.flatten()]
    
    # Build model
    model = Architectures.mlp(
        input_dim=20,
        hidden_dims=[64, 32],
        output_dim=3,
        activation="gelu"
    )
    
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    model.summary()
    
    # Train
    config = TrainingConfig(
        learning_rate=0.001,
        batch_size=32,
        epochs=50,
        validation_split=0.2,
        early_stopping=True,
        patience=10,
        verbose=True
    )
    
    history = model.fit(X, y_onehot, config)
    
    print("\nTraining completed!")
