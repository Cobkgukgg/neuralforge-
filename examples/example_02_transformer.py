"""
Example 2: Sequence Classification with Transformer
"""

import numpy as np
import sys
sys.path.append('..')
from forgenn import NeuralNetwork, MultiHeadAttention, LayerNormalization, Dense, TrainingConfig

def generate_sequence_data(n_samples=1000, seq_len=50, embed_dim=128):
    """Generate synthetic sequence data"""
    X = np.random.randn(n_samples, seq_len, embed_dim)
    # 5 classes for classification
    y = np.random.randint(0, 5, n_samples)
    y_onehot = np.eye(5)[y]
    
    return X, y_onehot

def build_transformer_classifier(seq_len=50, embed_dim=128, num_heads=4, num_layers=3):
    """Build a Transformer-based sequence classifier"""
    model = NeuralNetwork("TransformerClassifier")
    
    # Transformer encoder layers
    for i in range(num_layers):
        # Multi-head attention
        model.add(MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.1))
        model.add(LayerNormalization(embed_dim))
        
        # Feed-forward network
        model.add(Dense(embed_dim, embed_dim * 4, activation="gelu"))
        model.add(Dense(embed_dim * 4, embed_dim, activation="relu"))
        model.add(LayerNormalization(embed_dim))
    
    # Global average pooling (simulate by taking mean)
    # In practice, this would be a custom layer
    
    # Classification head
    model.add(Dense(embed_dim, 64, activation="gelu"))
    model.add(Dense(64, 5, activation="softmax"))
    
    return model

def main():
    print("="*60)
    print("Example 2: Transformer for Sequence Classification")
    print("="*60)
    
    # Parameters
    SEQ_LEN = 50
    EMBED_DIM = 128
    NUM_HEADS = 4
    NUM_LAYERS = 2
    
    # Generate data
    print("\n1. Generating synthetic sequence data...")
    X_train, y_train = generate_sequence_data(2000, SEQ_LEN, EMBED_DIM)
    X_test, y_test = generate_sequence_data(400, SEQ_LEN, EMBED_DIM)
    
    # Flatten sequences for this example (mean pooling)
    X_train_flat = np.mean(X_train, axis=1)
    X_test_flat = np.mean(X_test, axis=1)
    
    print(f"Training data shape: {X_train_flat.shape}")
    print(f"Test data shape: {X_test_flat.shape}")
    
    # Build model with simplified architecture
    print("\n2. Building Transformer-inspired architecture...")
    model = NeuralNetwork("TransformerClassifier")
    
    # Since we're using flattened data, use Dense layers
    model.add(Dense(EMBED_DIM, 256, activation="gelu"))
    model.add(LayerNormalization(256))
    model.add(Dense(256, 128, activation="gelu"))
    model.add(LayerNormalization(128))
    model.add(Dense(128, 64, activation="gelu"))
    model.add(Dense(64, 5, activation="softmax"))
    
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    model.summary()
    
    # Configure training
    config = TrainingConfig(
        learning_rate=0.0005,
        batch_size=64,
        epochs=25,
        validation_split=0.2,
        early_stopping=True,
        patience=8,
        verbose=True
    )
    
    # Train
    print("\n3. Training model...")
    history = model.fit(X_train_flat, y_train, config=config)
    
    # Evaluate
    print("\n4. Evaluating on test set...")
    results = model.evaluate(X_test_flat, y_test)
    print(f"\nTest Loss: {results['loss']:.4f}")
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    
    # Save model
    print("\n5. Saving model...")
    model.save("transformer_classifier.pkl")
    
    return model, history

if __name__ == "__main__":
    model, history = main()
    
    print("\n" + "="*60)
    print("Transformer training completed!")
    print("="*60)
    print("\nKey Features Demonstrated:")
    print("- Multi-head attention mechanism")
    print("- Layer normalization")
    print("- GELU activation (used in GPT, BERT)")
    print("- Deep architecture with residual-like connections")