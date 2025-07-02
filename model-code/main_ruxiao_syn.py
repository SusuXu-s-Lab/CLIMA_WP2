#!/usr/bin/env python3
"""
Main training script for disaster recovery co-evolution model.

This script orchestrates the complete training pipeline:
1. Data generation/loading
2. Model creation
3. Training with ELBO optimization
4. Model saving for later use

Usage:
    python main.py
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any
import json

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import our modules
from data import DataLoader, DisasterRecoveryDataGenerator
from models import (
    NetworkTypeNN, SelfActivationNN, InfluenceNN, InteractionFormationNN,
    NetworkEvolution, StateTransition
)
from inference import (
    MeanFieldPosterior, GumbelSoftmaxSampler, ELBOComputation, NetworkStateTrainer,
    TRAINING_CONFIG
)

import torch.nn as nn

def apply_xavier_init(trainer):
    """Apply Xavier initialization to all model components."""
    
    # List of all neural networks in the trainer
    networks = [
        trainer.mean_field_posterior.network_type_nn,
        trainer.elbo_computer.state_transition.self_nn,
        trainer.elbo_computer.state_transition.influence_nn,
        trainer.elbo_computer.network_evolution.interaction_nn
    ]
    
    # Apply Xavier initialization
    for network in networks:
        for module in network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)


def setup_project():
    """Setup project directories and working directory."""
    # Set working directory to project root
    os.chdir(project_root)
    print(f"Setting working directory to: {project_root}")
    
    # Create necessary directories
    data_dir = project_root / 'data/syn_data_ruxiao_v2'
    models_dir = project_root / 'saved_models'
    logs_dir = project_root / 'logs'
    
    data_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    
    print(f"Project root: {project_root}")
    print(f"Working directory: {os.getcwd()}")


def generate_or_load_data(data_dir: str = 'data/syn_data_ruxiao_v2', regenerate: bool = False) -> Dict[str, Any]:
    """Generate synthetic data or load existing data."""
    
    print("=== Data Preparation ===")
    
    # Check if data files exist
    data_files = [
        'household_states_community_one_hot.csv',
        'household_features_community_one_hot.csv', 
        'household_locations_community_one_hot.csv',
        'observed_network_community_one_hot.csv',
        'ground_truth_network_community_one_hot.csv'
    ]
    
    data_exists = all((Path(data_dir) / file).exists() for file in data_files)
    
    if not data_exists or regenerate:
        print("Generating synthetic data...")
        generator = DisasterRecoveryDataGenerator(
            n_households=50,
            n_timesteps=25,
            observation_rate=0.3
        )
        generator.generate_all_data(data_dir)
        print("✓ Synthetic data generated")
    else:
        print("✓ Using existing data files")
    
    # Load data
    print("Loading data...")
    loader = DataLoader(data_dir, device='cpu', file_list=data_files)
    data = loader.load_data()
    
    print(f"✓ Loaded {data['n_households']} households, {data['n_timesteps']} timesteps")
    print(f"✓ Features: {data['feature_names']}")
    
    return data, loader


def create_models(data: Dict[str, Any], L: int) -> Dict[str, Any]:
    """Create all neural networks and model components."""
    
    print("=== Model Creation ===")
    
    feature_dim = len(data['feature_names'])
    
    print(f"Feature dimension: {feature_dim}")
    print(f"History length: {L}")
    
    # Create neural networks
    print("Creating neural networks...")
    network_type_nn = NetworkTypeNN(feature_dim=feature_dim, L=L, hidden_dim=128)
    self_nn = SelfActivationNN(feature_dim=feature_dim, L=L, hidden_dim=64)
    influence_nn = InfluenceNN(feature_dim=feature_dim, L=L, hidden_dim=64)
    interaction_nn = InteractionFormationNN(feature_dim=feature_dim, hidden_dim=32)
    
    print("✓ Neural networks created")
    
    # Create network evolution model
    print("Creating network evolution model...")
    network_evolution = NetworkEvolution(interaction_nn)
    
    # Set normalization factors using median of demographic/geographic distances
    print("Computing normalization factors...")
    network_evolution.set_normalization_factors(data['features'], data['distances'])
    print("✓ Network evolution model created")
    
    # Create state transition model
    print("Creating state transition model...")
    state_transition = StateTransition(self_nn, influence_nn, L=L)
    print("✓ State transition model created")
    
    models = {
        'network_type_nn': network_type_nn,
        'self_nn': self_nn,
        'influence_nn': influence_nn,
        'interaction_nn': interaction_nn,
        'network_evolution': network_evolution,
        'state_transition': state_transition
    }
    
    # Count total parameters
    total_params = sum(sum(p.numel() for p in model.parameters()) 
                      for model in models.values() if hasattr(model, 'parameters'))
    print(f"✓ Total parameters: {total_params:,}")
    
    return models


def create_inference_components(models: Dict[str, Any], L: int, use_sparsity: bool = True) -> Dict[str, Any]:
    """Create inference components for training."""
    
    print("=== Inference Setup ===")
    
    # Create inference components
    mean_field_posterior = MeanFieldPosterior(models['network_type_nn'], L=L)
    gumbel_sampler = GumbelSoftmaxSampler()
    
    # Create ELBO computer with optional sparsity regularization
    confidence_weight = TRAINING_CONFIG['confidence_weight'] if use_sparsity else 0.0
    elbo_computer = ELBOComputation(
        models['network_evolution'], 
        models['state_transition'],
        rho_1=TRAINING_CONFIG['rho_1'],
        rho_2=TRAINING_CONFIG['rho_2'],
        confidence_weight=confidence_weight
    )
    
    print(f"✓ Sparsity regularization: {'enabled' if use_sparsity else 'disabled'} (weight={confidence_weight})")
    
    # Create trainer
    trainer = NetworkStateTrainer(
        mean_field_posterior=mean_field_posterior,
        gumbel_sampler=gumbel_sampler,
        elbo_computer=elbo_computer,
        learning_rate=TRAINING_CONFIG['learning_rate'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    apply_xavier_init(trainer)
    
    print("✓ Trainer created")
    
    return {
        'mean_field_posterior': mean_field_posterior,
        'gumbel_sampler': gumbel_sampler,
        'elbo_computer': elbo_computer,
        'trainer': trainer
    }


def train_model(data: Dict[str, Any], loader: DataLoader, inference_components: Dict[str, Any],
                train_end_time: int = 15, max_epochs: int = 1000) -> Dict[str, Any]:
    """Train the model with appropriate batching strategy."""
    
    print("=== Training ===")
    
    # Create train/test split
    print(f"Creating train/test split (train: 0-{train_end_time}, test: {train_end_time+1}-{data['n_timesteps']-1})")
    train_data, test_data = loader.train_test_split(data, train_end_time=train_end_time)
    
    trainer = inference_components['trainer']
    
    # Decide on batching strategy
    n_households = data['n_households']
    use_mini_batch = n_households > 100  # Threshold for mini-batching
    
    if use_mini_batch:
        print(f"Using mini-batch training (network size: {n_households})")
        batch_size = min(100, n_households // 2)  # Adaptive batch size
        node_batches = loader.create_node_batches(n_households, batch_size=batch_size)
        print(f"✓ Created {len(node_batches)} batches of size ~{batch_size}")
        
        # Train with mini-batches
        history = trainer.train(
            features=train_data['features'],
            states=train_data['states'],
            distances=train_data['distances'],
            network_data=train_data['observed_network'],
            ground_truth_network=data['ground_truth_network'],
            max_timestep=train_data['n_timesteps'] - 1,
            max_epochs=max_epochs,
            node_batches=node_batches,
            verbose=True,
            early_stopping=True,
            patience=200
        )
    else:
        print(f"Using full-batch training (network size: {n_households})")
        
        # Train with full batch
        history = trainer.train(
            features=train_data['features'],
            states=train_data['states'],
            distances=train_data['distances'],
            network_data=train_data['observed_network'],
            ground_truth_network=data['ground_truth_network'],
            max_timestep=train_data['n_timesteps'] - 1,
            max_epochs=max_epochs,
            verbose=True,
            early_stopping=False,
            patience=200
        )
    
    print("✓ Training completed")
    
    return {
        'history': history,
        'train_data': train_data,
        'test_data': test_data
    }


def save_model_and_results(models: Dict[str, Any], inference_components: Dict[str, Any],
                          training_results: Dict[str, Any], data: Dict[str, Any], L: int):
    """Save trained model and training results."""
    
    print("=== Saving Results ===")
    
    # Prepare model state dictionaries
    model_state = {
        'network_type_nn': models['network_type_nn'].state_dict(),
        'self_nn': models['self_nn'].state_dict(),
        'influence_nn': models['influence_nn'].state_dict(),
        'interaction_nn': models['interaction_nn'].state_dict(),
        'network_evolution': models['network_evolution'].state_dict(),
    }
    
    # ELBO computer parameters
    elbo_params = {
        'rho_1': inference_components['elbo_computer'].rho_1.item(),
        'rho_2': inference_components['elbo_computer'].rho_2.item(),
        'confidence_weight': inference_components['elbo_computer'].confidence_weight
    }
    
    # Training metadata
    history = training_results['history']
    final_metrics = history[-1] if history else {}
    
    metadata = {
        'data_info': {
            'feature_names': data['feature_names'],
            'n_households': data['n_households'],
            'n_timesteps': data['n_timesteps']
        },
        'model_config': {
            'feature_dim': len(data['feature_names']),
            'L': L,
            'hidden_dim': 64
        },
        'training_config': TRAINING_CONFIG,
        'final_metrics': final_metrics,
        'training_epochs': len(history)
    }
    
    # Save complete model
    model_path = 'saved_models/trained_model_ruxiao_density_info_penalty3_rho50_overfit1_epoch_400_q128_64_64_32_seed22.pth' 
    torch.save({
        'model_state': model_state,
        'elbo_params': elbo_params,
        'metadata': metadata,
        'training_history': history
    }, model_path)
    
    print(f"✓ Model saved to {model_path}")
    
    # Save training history as JSON for analysis
    history_path = 'logs/training_history_ruxiao_density_info_penalty3_rho50_overfit1_epoch_400_q128_64_64_32_seed22.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"✓ Training history saved to {history_path}")
    
    # Print final results
    print("\n" + "="*50)
    print("TRAINING RESULTS")
    print("="*50)
    
    if final_metrics:
        print(f"Final ELBO: {final_metrics['total_elbo']:.4f}")
        print(f"  State likelihood: {final_metrics['state_likelihood']:.4f}")
        print(f"  Observation likelihood: {final_metrics['observation_likelihood']:.4f}")
        print(f"  Prior likelihood: {final_metrics['prior_likelihood']:.4f}")
        print(f"  Posterior entropy: {final_metrics['posterior_entropy']:.4f}")
        
        if 'sparsity_regularization' in final_metrics:
            print(f"  Sparsity regularization: {final_metrics['sparsity_regularization']:.4f}")
        
        # print(f"\nObservation model parameters:")
        # print(f"  ρ₁ (bonding miss rate): {final_metrics['rho_1']:.4f}")
        # print(f"  ρ₂ (bridging miss rate): {final_metrics['rho_2']:.4f}")
        
        print(f"\nTraining info:")
        print(f"  Total epochs: {len(history)}")
        print(f"  Final temperature: {final_metrics['temperature']:.4f}")
        print(f"  Final samples: {final_metrics['num_samples']}")
    
    print("\n✓ All results saved successfully!")
    
    return model_path, history_path


def main():
    """Main training pipeline."""
    
    print("🚀 Starting Disaster Recovery Co-Evolution Model Training")
    print("="*60)
    
    try:
        # 1. Setup
        setup_project()
        
        # 2. Data preparation
        data, loader = generate_or_load_data(regenerate=False)  # Set True to regenerate data
        
        # 3. Model creation
        models = create_models(data, L=3)  # L is history length
        
        # 4. Inference setup
        inference_components = create_inference_components(models, L=3, use_sparsity=True)
        
        # 5. Training
        training_results = train_model(
            data=data,
            loader=loader,
            inference_components=inference_components,
            train_end_time=15,  # Train on timesteps 0-15, test on 16-24
            max_epochs=400
        )
        
        # 6. Save results
        model_path, history_path = save_model_and_results(
            models, inference_components, training_results, data, L=3
        )
        
        print("\n🎉 Training pipeline completed successfully!")
        print(f"📁 Model saved: {model_path}")
        print(f"📊 History saved: {history_path}")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(22)
    np.random.seed(22)

    # Run main pipeline
    exit_code = main()
    sys.exit(exit_code)

import torch
torch.set_num_threads(4)  # 设置CPU线程数
print(f"CPU线程数设置为: {torch.get_num_threads()}")