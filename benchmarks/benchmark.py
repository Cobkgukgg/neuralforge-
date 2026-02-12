"""
Benchmarking Script for forgenn
Compare different architectures and configurations
"""

import numpy as np
import time
import sys
sys.path.append('..')
from forgenn import Architectures, NeuralNetwork, Dense, ResidualBlock, TrainingConfig

class Benchmark:
    """Benchmark different neural network configurations"""
    
    @staticmethod
    def generate_data(n_samples=2000, n_features=50, n_classes=5):
        """Generate benchmark dataset"""
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)
        y_onehot = np.eye(n_classes)[y]
        
        # Normalize
        X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
        
        # Split
        split = int(0.8 * n_samples)
        return X[:split], y_onehot[:split], X[split:], y_onehot[split:]
    
    @staticmethod
    def benchmark_activation_functions():
        """Compare different activation functions"""
        print("\n" + "="*70)
        print("BENCHMARK 1: Activation Functions")
        print("="*70)
        
        X_train, y_train, X_test, y_test = Benchmark.generate_data()
        activations = ['relu', 'gelu', 'swish', 'mish', 'leaky_relu', 'elu']
        
        results = []
        
        for activation in activations:
            print(f"\nTesting {activation.upper()}...")
            
            model = NeuralNetwork(f"MLP_{activation}")
            model.add(Dense(50, 64, activation=activation))
            model.add(Dense(64, 32, activation=activation))
            model.add(Dense(32, 5, activation="softmax"))
            model.compile(loss="categorical_crossentropy", optimizer="adam")
            
            config = TrainingConfig(
                learning_rate=0.001,
                batch_size=32,
                epochs=20,
                validation_split=0.2,
                verbose=False
            )
            
            start_time = time.time()
            history = model.fit(X_train, y_train, config=config)
            train_time = time.time() - start_time
            
            test_results = model.evaluate(X_test, y_test)
            
            results.append({
                'activation': activation,
                'final_loss': history['train_loss'][-1],
                'val_loss': history['val_loss'][-1],
                'test_accuracy': test_results['accuracy'],
                'train_time': train_time
            })
            
            print(f"  Final Loss: {history['train_loss'][-1]:.4f}")
            print(f"  Val Loss: {history['val_loss'][-1]:.4f}")
            print(f"  Test Accuracy: {test_results['accuracy']:.4f}")
            print(f"  Training Time: {train_time:.2f}s")
        
        # Summary
        print("\n" + "-"*70)
        print("Summary:")
        print("-"*70)
        print(f"{'Activation':<15} {'Train Loss':<12} {'Val Loss':<12} {'Accuracy':<12} {'Time (s)':<10}")
        print("-"*70)
        for r in results:
            print(f"{r['activation']:<15} {r['final_loss']:<12.4f} {r['val_loss']:<12.4f} "
                  f"{r['test_accuracy']:<12.4f} {r['train_time']:<10.2f}")
        
        # Best activation
        best = max(results, key=lambda x: x['test_accuracy'])
        print(f"\nBest: {best['activation'].upper()} with {best['test_accuracy']:.4f} accuracy")
        
        return results
    
    @staticmethod
    def benchmark_network_depth():
        """Compare networks of different depths"""
        print("\n" + "="*70)
        print("BENCHMARK 2: Network Depth")
        print("="*70)
        
        X_train, y_train, X_test, y_test = Benchmark.generate_data()
        depths = [2, 3, 4, 5, 6]
        
        results = []
        
        for depth in depths:
            print(f"\nTesting depth {depth}...")
            
            model = NeuralNetwork(f"MLP_depth_{depth}")
            
            # Build network
            hidden_dims = [64] * depth
            for i, dim in enumerate(hidden_dims):
                model.add(Dense(50 if i == 0 else hidden_dims[i-1], dim, activation="gelu"))
            model.add(Dense(hidden_dims[-1], 5, activation="softmax"))
            
            model.compile(loss="categorical_crossentropy", optimizer="adam")
            
            config = TrainingConfig(
                learning_rate=0.001,
                batch_size=32,
                epochs=20,
                validation_split=0.2,
                verbose=False
            )
            
            start_time = time.time()
            history = model.fit(X_train, y_train, config=config)
            train_time = time.time() - start_time
            
            test_results = model.evaluate(X_test, y_test)
            
            # Count parameters
            total_params = sum(
                sum(p.size for p in layer.get_params().values())
                for layer in model.layers
            )
            
            results.append({
                'depth': depth,
                'params': total_params,
                'final_loss': history['train_loss'][-1],
                'val_loss': history['val_loss'][-1],
                'test_accuracy': test_results['accuracy'],
                'train_time': train_time
            })
            
            print(f"  Parameters: {total_params:,}")
            print(f"  Final Loss: {history['train_loss'][-1]:.4f}")
            print(f"  Test Accuracy: {test_results['accuracy']:.4f}")
            print(f"  Training Time: {train_time:.2f}s")
        
        # Summary
        print("\n" + "-"*70)
        print("Summary:")
        print("-"*70)
        print(f"{'Depth':<8} {'Params':<12} {'Train Loss':<12} {'Val Loss':<12} {'Accuracy':<12} {'Time (s)':<10}")
        print("-"*70)
        for r in results:
            print(f"{r['depth']:<8} {r['params']:<12,} {r['final_loss']:<12.4f} "
                  f"{r['val_loss']:<12.4f} {r['test_accuracy']:<12.4f} {r['train_time']:<10.2f}")
        
        return results
    
    @staticmethod
    def benchmark_architectures():
        """Compare different architectures"""
        print("\n" + "="*70)
        print("BENCHMARK 3: Architecture Comparison")
        print("="*70)
        
        X_train, y_train, X_test, y_test = Benchmark.generate_data()
        
        architectures = {
            'Simple MLP': lambda: Architectures.mlp(50, [64, 32], 5, activation="relu"),
            'Deep MLP': lambda: Architectures.mlp(50, [128, 64, 32], 5, activation="gelu"),
            'ResNet-3': lambda: Architectures.resnet(50, num_blocks=3, hidden_dim=64, output_dim=5),
            'ResNet-5': lambda: Architectures.resnet(50, num_blocks=5, hidden_dim=64, output_dim=5),
        }
        
        results = []
        
        for name, builder in architectures.items():
            print(f"\nTesting {name}...")
            
            model = builder()
            model.compile(loss="categorical_crossentropy", optimizer="adam")
            
            # Count parameters
            total_params = sum(
                sum(p.size for p in layer.get_params().values())
                for layer in model.layers
            )
            
            config = TrainingConfig(
                learning_rate=0.001,
                batch_size=32,
                epochs=25,
                validation_split=0.2,
                early_stopping=True,
                patience=8,
                verbose=False
            )
            
            start_time = time.time()
            history = model.fit(X_train, y_train, config=config)
            train_time = time.time() - start_time
            
            test_results = model.evaluate(X_test, y_test)
            
            results.append({
                'architecture': name,
                'params': total_params,
                'epochs_trained': len(history['train_loss']),
                'final_loss': history['train_loss'][-1],
                'best_val_loss': min(history['val_loss']),
                'test_accuracy': test_results['accuracy'],
                'train_time': train_time
            })
            
            print(f"  Parameters: {total_params:,}")
            print(f"  Epochs: {len(history['train_loss'])}")
            print(f"  Best Val Loss: {min(history['val_loss']):.4f}")
            print(f"  Test Accuracy: {test_results['accuracy']:.4f}")
            print(f"  Training Time: {train_time:.2f}s")
        
        # Summary
        print("\n" + "-"*70)
        print("Summary:")
        print("-"*70)
        print(f"{'Architecture':<15} {'Params':<10} {'Epochs':<8} {'Val Loss':<10} {'Accuracy':<10} {'Time (s)':<10}")
        print("-"*70)
        for r in results:
            print(f"{r['architecture']:<15} {r['params']:<10,} {r['epochs_trained']:<8} "
                  f"{r['best_val_loss']:<10.4f} {r['test_accuracy']:<10.4f} {r['train_time']:<10.2f}")
        
        # Best architecture
        best = max(results, key=lambda x: x['test_accuracy'])
        print(f"\nBest: {best['architecture']} with {best['test_accuracy']:.4f} accuracy")
        
        return results
    
    @staticmethod
    def run_all_benchmarks():
        """Run all benchmark suites"""
        print("\n" + "="*70)
        print(" "*20 + "NEURALFORGE BENCHMARKS")
        print("="*70)
        
        activation_results = Benchmark.benchmark_activation_functions()
        depth_results = Benchmark.benchmark_network_depth()
        architecture_results = Benchmark.benchmark_architectures()
        
        print("\n" + "="*70)
        print(" "*20 + "BENCHMARKS COMPLETED")
        print("="*70)
        
        return {
            'activations': activation_results,
            'depths': depth_results,
            'architectures': architecture_results
        }

if __name__ == "__main__":
    print("\nStarting NeuralForge benchmarks...")
    print("This will take several minutes to complete.\n")
    
    results = Benchmark.run_all_benchmarks()
    
    print("\n" + "="*70)
    print("All benchmarks completed successfully!")
    print("="*70)