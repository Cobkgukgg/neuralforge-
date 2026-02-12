"""
Example 3: ResNet for Deep Learning
"""

import numpy as np
import sys
sys.path.append('..')
from forgenn import Architectures, NeuralNetwork, ResidualBlock, Dense, TrainingConfig

def generate_regression_data(n_samples=2000, n_features=50):
    """Generate synthetic regression data"""
    X = np.random.randn(n_samples, n_features)
    # Complex non-linear relationship
    y = (np.sin(X[:, 0]) + np.cos(X[:, 1]) + 
         0.5 * X[:, 2]**2 + 0.3 * X[:, 3] + 
         np.random.randn(n_samples) * 0.1)
    y = y.reshape(-1, 1)
    
    return X, y

def build_custom_resnet(input_dim, num_blocks=4, hidden_dim=128):
    """Build custom ResNet architecture"""
    model = NeuralNetwork("CustomResNet")
    
    # Initial projection
    model.add(Dense(input_dim, hidden_dim, activation="gelu"))
    
    # Residual blocks
    for i in range(num_blocks):
        model.add(ResidualBlock(hidden_dim, activation="mish"))
    
    # Final layers
    model.add(Dense(hidden_dim, 64, activation="gelu"))
    model.add(Dense(64, 1, activation="relu"))
    
    return model

def main():
    print("="*60)
    print("Example 3: ResNet for Regression")
    print("="*60)
    
    # Generate data
    print("\n1. Generating synthetic regression data...")
    X_train, y_train = generate_regression_data(5000, 50)
    X_test, y_test = generate_regression_data(1000, 50)
    
    # Normalize data
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0) + 1e-8
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std
    
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Create ResNet model
    print("\n2. Building ResNet architecture...")
    model = build_custom_resnet(
        input_dim=50,
        num_blocks=5,
        hidden_dim=128
    )
    
    model.compile(loss="huber", optimizer="adam")
    model.summary()
    
    # Configure training
    config = TrainingConfig(
        learning_rate=0.001,
        batch_size=64,
        epochs=50,
        validation_split=0.2,
        early_stopping=True,
        patience=10,
        dropout_rate=0.2,
        verbose=True
    )
    
    # Train
    print("\n3. Training ResNet...")
    history = model.fit(X_train, y_train, config=config)
    
    # Evaluate
    print("\n4. Evaluating on test set...")
    results = model.evaluate(X_test, y_test)
    print(f"\nTest Loss: {results['loss']:.4f}")
    
    # Make predictions
    print("\n5. Making predictions...")
    predictions = model.predict(X_test[:5])
    
    print("\nSample predictions vs actual:")
    print("-" * 40)
    for i in range(5):
        pred = predictions[i, 0] * y_std + y_mean
        actual = y_test[i, 0] * y_std + y_mean
        print(f"Sample {i+1}: Pred={pred:.4f}, Actual={actual:.4f}")
    
    # Save model
    print("\n6. Saving model...")
    model.save("resnet_regression.pkl")
    
    return model, history

if __name__ == "__main__":
    model, history = main()
    
    print("\n" + "="*60)
    print("ResNet training completed!")
    print("="*60)
    print("\nKey Features Demonstrated:")
    print("- Residual connections for deep networks")
    print("- Huber loss (robust to outliers)")
    print("- Mish activation (state-of-the-art)")
    print("- Dropout regularization")
    print("- Early stopping")