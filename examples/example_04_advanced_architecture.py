"""
Example 4: Custom Advanced Architecture
Building a custom architecture with all advanced features
"""

import numpy as np
import sys
sys.path.append('..')
from forgenn import (
    NeuralNetwork, Dense, ResidualBlock, LayerNormalization,
    TrainingConfig, Activation
)

def build_advanced_classifier(input_dim, num_classes):
    """
    Build an advanced classifier combining multiple techniques:
    - Residual connections
    - Layer normalization
    - Advanced activations (GELU, Mish)
    - Dropout regularization
    """
    model = NeuralNetwork("AdvancedClassifier")
    
    # Input processing with GELU
    model.add(Dense(input_dim, 256, activation="gelu", dropout_rate=0.2))
    model.add(LayerNormalization(256))
    
    # First residual block with Mish activation
    model.add(ResidualBlock(256, activation="mish"))
    model.add(LayerNormalization(256))
    
    # Deep feature extraction
    model.add(Dense(256, 512, activation="gelu", dropout_rate=0.3))
    model.add(LayerNormalization(512))
    
    # Second residual block
    model.add(ResidualBlock(512, activation="mish"))
    model.add(LayerNormalization(512))
    
    # Feature compression
    model.add(Dense(512, 256, activation="swish", dropout_rate=0.2))
    model.add(LayerNormalization(256))
    
    # Third residual block
    model.add(ResidualBlock(256, activation="gelu"))
    
    # Classification head
    model.add(Dense(256, 128, activation="gelu", dropout_rate=0.2))
    model.add(Dense(128, num_classes, activation="softmax"))
    
    return model

def generate_complex_data(n_samples=5000, n_features=100, n_classes=10):
    """Generate complex synthetic classification data"""
    # Create complex non-linear patterns
    X = np.random.randn(n_samples, n_features)
    
    # Create non-linear decision boundaries
    z = (np.sin(X[:, 0]) * np.cos(X[:, 1]) + 
         0.5 * X[:, 2]**2 + 
         0.3 * np.tanh(X[:, 3]) +
         np.random.randn(n_samples) * 0.5)
    
    # Convert to classes
    y = (z * 3).astype(int) % n_classes
    y_onehot = np.eye(n_classes)[y]
    
    return X, y_onehot

def main():
    print("="*70)
    print(" "*15 + "ADVANCED CUSTOM ARCHITECTURE")
    print("="*70)
    
    # Parameters
    N_FEATURES = 100
    N_CLASSES = 10
    N_SAMPLES = 8000
    
    # Generate data
    print("\n1. Generating complex synthetic data...")
    X, y = generate_complex_data(N_SAMPLES, N_FEATURES, N_CLASSES)
    
    # Split data
    split_idx = int(N_SAMPLES * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Normalize
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0) + 1e-8
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {N_FEATURES}")
    print(f"Classes: {N_CLASSES}")
    
    # Build model
    print("\n2. Building advanced architecture...")
    model = build_advanced_classifier(N_FEATURES, N_CLASSES)
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    
    print("\nModel Architecture:")
    model.summary()
    
    # Training configuration
    config = TrainingConfig(
        learning_rate=0.0005,
        batch_size=64,
        epochs=60,
        validation_split=0.2,
        early_stopping=True,
        patience=12,
        dropout_rate=0.2,
        verbose=True
    )
    
    # Train
    print("\n3. Training model...")
    print("-" * 70)
    history = model.fit(X_train, y_train, config=config)
    
    # Evaluate
    print("\n4. Evaluating model...")
    print("-" * 70)
    results = model.evaluate(X_test, y_test)
    
    print(f"\nFinal Test Results:")
    print(f"  Loss: {results['loss']:.4f}")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    
    # Analyze predictions
    print("\n5. Analyzing predictions...")
    predictions = model.predict(X_test)
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    
    # Per-class accuracy
    print("\nPer-class accuracy:")
    for i in range(N_CLASSES):
        class_mask = true_classes == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(pred_classes[class_mask] == i)
            print(f"  Class {i}: {class_acc:.4f}")
    
    # Confidence analysis
    print("\nPrediction confidence:")
    confidences = np.max(predictions, axis=1)
    print(f"  Mean confidence: {np.mean(confidences):.4f}")
    print(f"  Median confidence: {np.median(confidences):.4f}")
    print(f"  Min confidence: {np.min(confidences):.4f}")
    print(f"  Max confidence: {np.max(confidences):.4f}")
    
    # Save model
    print("\n6. Saving model...")
    model.save("advanced_classifier.pkl")
    
    print("\n" + "="*70)
    print("Training and evaluation complete!")
    print("="*70)
    
    print("\nKey Features Used:")
    print("  ✓ Residual connections (3 blocks)")
    print("  ✓ Layer normalization (6 layers)")
    print("  ✓ GELU activation (Transformer-style)")
    print("  ✓ Mish activation (State-of-the-art)")
    print("  ✓ Swish activation (EfficientNet-style)")
    print("  ✓ Dropout regularization")
    print("  ✓ Early stopping")
    print("  ✓ Adaptive learning rate (Adam)")
    
    return model, history, results

if __name__ == "__main__":
    model, history, results = main()
    
    # Additional analysis
    print("\n" + "="*70)
    print("Training History Summary:")
    print("="*70)
    print(f"Initial training loss: {history['train_loss'][0]:.4f}")
    print(f"Final training loss: {history['train_loss'][-1]:.4f}")
    print(f"Best validation loss: {min(history['val_loss']):.4f}")
    print(f"Improvement: {((history['train_loss'][0] - history['train_loss'][-1]) / history['train_loss'][0] * 100):.2f}%")