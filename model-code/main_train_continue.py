#!/usr/bin/env python3
"""
Flexible model continuation training script.

import torch.nn as nn

This script allows you to:
1. Load a previously trained model
2. Continue training with different parameters
3. Switch datasets
4. Modify training configuration
5. Resume from any checkpoint

Usage:

    python main_train_continue.py \
    --model_path saved_models/G2_Sparse_HighSeed_obs20_prior1_entropy1_density1_penalize_Wrong_observation_epoch550.pth \
    --data_dir data/syn_data_ruxiao_v4_config/dataset/G2_Sparse_HighSeed \
    --freeze_network \
    --retrain_prediction \
    --epochs 300 \
    --lr 1e-3 \
    --patience 20 \
    --train_end_time 12 \
    --reset_epoch 550 \
    --save_suffix gru_retrain
    
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import json

# torch.autograd.set_detect_anomaly(True)

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import our modules
from data import DataLoader, DisasterRecoveryDataGenerator
from models import (
    NetworkTypeNN, SeqSelfNN, SeqPairInfluenceNN, InteractionFormationNN,
    NetworkEvolution, StateTransition
)
from inference import (
    MeanFieldPosterior, GumbelSoftmaxSampler, ELBOComputation, NetworkStateTrainer
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Continue training a disaster recovery model')
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the saved model checkpoint')
    
    # Optional data arguments
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Data directory (if None, use same as original training)')
    parser.add_argument('--train_end_time', type=int, default=None,
                       help='Training end time (if None, use same as original)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Additional epochs to train')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (if None, use original)')
    parser.add_argument('--weight_decay', type=float, default=None,
                       help='Weight decay (if None, use original)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (if None, use full batch for small datasets)')
    
    # Component weights for ELBO balancing
    parser.add_argument('--state_weight', type=float, default=None,
                       help='Weight for state likelihood component')
    parser.add_argument('--obs_weight', type=float, default=None,
                       help='Weight for observation likelihood component')
    parser.add_argument('--prior_weight', type=float, default=None,
                       help='Weight for prior likelihood component')
    parser.add_argument('--entropy_weight', type=float, default=None,
                       help='Weight for posterior entropy component')
    
    # Prediction NN retrain option (NEW)
    parser.add_argument('--retrain_prediction', action='store_true', default=False,
                       help='Re-initialize prediction components (SelfActivationNN, InfluenceNN) for retraining')
    parser.add_argument('--freeze_network', action='store_true', default=False,
                       help='Freeze network inference components (NetworkTypeNN, InteractionFormationNN)')
    
    # Influence limitation (NEW)
    parser.add_argument('--max_neighbor_influences', type=int, default=None,
                       help='Max times a neighbor can influence another (K value). If None, use checkpoint value or default=100')
    
    # Training settings
    parser.add_argument('--early_stopping', action='store_true', default=True,
                       help='Use early stopping')
    parser.add_argument('--patience', type=int, default=30,
                       help='Early stopping patience')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Verbose training output')
    
    # Output settings
    parser.add_argument('--save_suffix', type=str, default='continued',
                       help='Suffix for saved model filename')
    parser.add_argument('--save_dir', type=str, default='saved_models',
                       help='Directory to save continued model')
    
    parser.add_argument('--reset_epoch', type=int, default=None,
                    help='Temporarily set trainer.epoch to this value before continuing training')
    
    return parser.parse_args()


def load_model_and_data(model_path: str, data_dir: Optional[str] = None, 
                       train_end_time: Optional[int] = None, device: str = 'cpu',
                       retrain_prediction: bool = False, max_neighbor_influences: Optional[int] = None):
    """Load trained model and optionally switch to new data or upgrade architecture."""
    
    print(f"Loading model from: {model_path}")
    
    # Load checkpoint
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract configuration
    metadata = checkpoint['metadata']
    model_config = metadata['model_config']
    feature_dim = model_config['feature_dim']
    L = model_config['L']
    
    # Determine max_neighbor_influences (K value)
    # Priority: command-line arg > checkpoint > TRAINING_CONFIG default
    if max_neighbor_influences is not None:
        K = max_neighbor_influences
        print(f"Using K from command-line argument: {K}")
    elif 'max_neighbor_influences' in model_config:
        K = model_config['max_neighbor_influences']
        print(f"Using K from checkpoint: {K}")
    else:
        from inference import TRAINING_CONFIG
        K = TRAINING_CONFIG['max_neighbor_influences']
        print(f"Using K from TRAINING_CONFIG: {K}")
    
    # Determine lingering effect parameters
    # Priority: checkpoint metadata > TRAINING_CONFIG default
    training_config = metadata.get('training_config', {})
    # L_linger = training_config.get('L_linger', 3)
    decay_type = training_config.get('decay_type', 'exponential')
    decay_rate = training_config.get('decay_rate', 0.5)
    
    print(f"Original model config: feature_dim={feature_dim}, L={L}, K={K}")
    # print(f"Lingering effect config: L_linger={L_linger}, decay_type={decay_type}, decay_rate={decay_rate}")
    
    print("\nRecreating model architecture...")
    network_type_nn = NetworkTypeNN(feature_dim, L, hidden_dim=128)
    interaction_nn = InteractionFormationNN(feature_dim, hidden_dim=128)
    if retrain_prediction:
        print("[Retrain] Re-initializing prediction GRU networks")
        seq_self_nn = SeqSelfNN(feature_dim=feature_dim, hidden_dim=32)
        seq_pair_infl_nn = SeqPairInfluenceNN(pairwise_feature_dim=feature_dim, hidden_dim=32)
        print("   ⚠️  GRU networks will be randomly initialized on first forward pass (lazy init)")
    else:
        print("[Load] Detecting and loading GRU networks from checkpoint")
        model_state = checkpoint['model_state']
        
        detected_self_hidden_dim = 32
        detected_pair_hidden_dim = 32
        
        if 'seq_self_nn' in model_state:
            saved_self_state = model_state['seq_self_nn']
            if 'gru.weight_ih_l0' in saved_self_state:
                detected_self_hidden_dim = saved_self_state['gru.weight_ih_l0'].shape[0] // 3
                print(f"✓ Detected seq_self_nn hidden_dim: {detected_self_hidden_dim}")
        
        if 'seq_pair_infl_nn' in model_state:
            saved_pair_state = model_state['seq_pair_infl_nn']
            if 'gru.weight_ih_l0' in saved_pair_state:
                detected_pair_hidden_dim = saved_pair_state['gru.weight_ih_l0'].shape[0] // 3
                print(f"✓ Detected seq_pair_infl_nn hidden_dim: {detected_pair_hidden_dim}")
        
        seq_self_nn = SeqSelfNN(feature_dim=feature_dim, hidden_dim=detected_self_hidden_dim)
        seq_pair_infl_nn = SeqPairInfluenceNN(pairwise_feature_dim=feature_dim, hidden_dim=detected_pair_hidden_dim)
        

        seq_self_nn.load_state_dict(model_state['seq_self_nn'])
        seq_pair_infl_nn.load_state_dict(model_state['seq_pair_infl_nn'])
        print("✓ GRU networks loaded from checkpoint")
    
    network_evolution = NetworkEvolution(interaction_nn)

    state_transition = StateTransition(seq_self_nn=seq_self_nn, seq_pair_infl_nn=seq_pair_infl_nn)
    mean_field_posterior = MeanFieldPosterior(network_type_nn, L)
    gumbel_sampler = GumbelSoftmaxSampler()
    elbo_computer = ELBOComputation(network_evolution, state_transition, variational_posterior=mean_field_posterior)
    
    trainer = NetworkStateTrainer(mean_field_posterior, gumbel_sampler, elbo_computer)
    
    # Load model weights - with error handling for incompatible architectures
    print("\nLoading model weights...")
    model_state = checkpoint['model_state']
    trainer.mean_field_posterior.network_type_nn.load_state_dict(model_state['network_type_nn'])
    trainer.elbo_computer.network_evolution.interaction_nn.load_state_dict(model_state['interaction_nn'])
    trainer.elbo_computer.network_evolution.load_state_dict(model_state['network_evolution'])
    print("✓ Network inference components loaded")
    
    # Load ELBO parameters
    elbo_params = checkpoint['elbo_params']
    trainer.elbo_computer.rho_1.data = torch.tensor(elbo_params['rho_1'])
    trainer.elbo_computer.rho_2.data = torch.tensor(elbo_params['rho_2'])
    trainer.elbo_computer.confidence_weight = elbo_params['confidence_weight']
    
    # Load training history
    training_history = checkpoint.get('training_history', [])
    trainer.training_history = training_history
    trainer.epoch = len(training_history)
    
    print(f"Model loaded successfully. Previous training: {trainer.epoch} epochs")
    
    # Load data
    original_data_dir = data_dir if data_dir else 'data'  # Default fallback
    print(f"Loading data from: {original_data_dir}")

    data_files = [
        'household_states_community_one_hot_with_log2.csv',
        'household_features_community_one_hot_with_log2.csv', 
        'household_locations_community_one_hot_with_log2.csv',
        'observed_network_community_one_hot_with_log2.csv',
        'ground_truth_network_community_one_hot_with_log2.csv'
    ]
    
    loader = DataLoader(original_data_dir, device=device, file_list=data_files)
    data = loader.load_data()
    
    # Check feature compatibility
    if data['features'].shape[1] != feature_dim:
        raise ValueError(f"Feature dimension mismatch: model expects {feature_dim}, "
                        f"data has {data['features'].shape[1]}")
    
    # Train/test split
    original_train_end = train_end_time if train_end_time else 15  # Default fallback
    train_data, test_data = loader.train_test_split(data, train_end_time=original_train_end)
    
    # Set normalization factors if using same model architecture
    trainer.elbo_computer.network_evolution.set_normalization_factors(
        data['features'], data['distances']
    )
    
    print(f"Data loaded: {data['n_households']} households, {data['n_timesteps']} timesteps")
    print(f"Train/test split: train=0-{original_train_end}, test={original_train_end+1}-{data['n_timesteps']-1}")
    
    return trainer, train_data, test_data, data, loader


def update_training_parameters(trainer: NetworkStateTrainer, args):
    """Update training parameters based on command line arguments."""
    
    # Update learning rate
    if args.lr is not None:
        print(f"Updating learning rate: {trainer.optimizer.param_groups[0]['lr']} -> {args.lr}")
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = args.lr
    
    # Update weight decay
    if args.weight_decay is not None:
        print(f"Updating weight decay to: {args.weight_decay}")
        for param_group in trainer.optimizer.param_groups:
            param_group['weight_decay'] = args.weight_decay
    
    # Print current optimizer settings
    print(f"Current optimizer settings:")
    print(f"  Learning rate: {trainer.optimizer.param_groups[0]['lr']}")
    print(f"  Weight decay: {trainer.optimizer.param_groups[0]['weight_decay']}")


def freeze_network_components(trainer: NetworkStateTrainer):
    """Freeze network inference components, keep prediction components trainable."""
    
    print("\n🔒 Freezing network inference components...")
    
    # Freeze NetworkTypeNN (network posterior q(A|...))
    frozen_params = 0
    for param in trainer.mean_field_posterior.network_type_nn.parameters():
        param.requires_grad = False
        frozen_params += param.numel()
    print(f"  ✓ NetworkTypeNN frozen ({frozen_params:,} parameters)")
    
    # Freeze InteractionFormationNN (link formation prior)
    frozen_params = 0
    for param in trainer.elbo_computer.network_evolution.interaction_nn.parameters():
        param.requires_grad = False
        frozen_params += param.numel()
    print(f"  ✓ InteractionFormationNN frozen ({frozen_params:,} parameters)")
    
    # Keep prediction components trainable
    print("\n✅ GRU prediction components remain trainable:")
    trainable_params = 0
    for param in trainer.elbo_computer.state_transition.seq_self_nn.parameters():
        trainable_params += param.numel()
    print(f"  ✓ SeqSelfNN ({trainable_params:,} parameters)")
    
    trainable_params = 0
    for param in trainer.elbo_computer.state_transition.seq_pair_infl_nn.parameters():
        trainable_params += param.numel()
    print(f"  ✓ SeqPairInfluenceNN ({trainable_params:,} parameters)")
    
    # Recreate optimizer with only trainable parameters
    trainable_params_list = [
        {'params': list(trainer.elbo_computer.state_transition.seq_self_nn.parameters())},
        {'params': list(trainer.elbo_computer.state_transition.seq_pair_infl_nn.parameters())},
    ]
    
    # Get current optimizer settings
    old_lr = trainer.optimizer.param_groups[0]['lr']
    old_wd = trainer.optimizer.param_groups[0]['weight_decay']
    
    # Create new optimizer with only trainable parameters
    trainer.optimizer = torch.optim.Adam(
        trainable_params_list,
        lr=old_lr,
        weight_decay=old_wd
    )
    
    print(f"\n✓ Optimizer updated with {sum(p.numel() for group in trainable_params_list for p in group['params']):,} trainable parameters")
    print(f"  (Network inference parameters excluded from optimization)")



def update_elbo_weights(trainer: NetworkStateTrainer, args):
    """Update ELBO component weights if specified."""
    
    weights_updated = False
    
    if hasattr(trainer.elbo_computer, 'component_weights'):
        current_weights = trainer.elbo_computer.component_weights
    else:
        # Add component weights to ELBO computer if not present
        trainer.elbo_computer.component_weights = {
            'state': 1.0,
            'obs': 1.0, 
            'prior': 1.0,
            'entropy': 1.0
        }
        current_weights = trainer.elbo_computer.component_weights
    
    if args.state_weight is not None:
        current_weights['state'] = args.state_weight
        weights_updated = True
    
    if args.obs_weight is not None:
        current_weights['obs'] = args.obs_weight
        weights_updated = True
    
    if args.prior_weight is not None:
        current_weights['prior'] = args.prior_weight
        weights_updated = True
    
    if args.entropy_weight is not None:
        current_weights['entropy'] = args.entropy_weight
        weights_updated = True
    
    if weights_updated:
        print(f"Updated ELBO component weights:")
        for component, weight in current_weights.items():
            print(f"  {component}: {weight}")
    
    return weights_updated


def continue_training(trainer: NetworkStateTrainer, train_data: Dict, data: Dict, args):
    """Continue training with specified parameters."""
    
    print(f"\n=== CONTINUING TRAINING ===")
    print(f"Starting from epoch: {trainer.epoch}")
    print(f"Additional epochs: {args.epochs}")
    print(f"Total target epochs: {trainer.epoch + args.epochs}")
    
    # Determine batching strategy
    n_households = train_data['n_households']
    if args.batch_size is not None:
        print(f"Using mini-batch training with batch size: {args.batch_size}")
        from data import DataLoader
        temp_loader = DataLoader('')
        node_batches = temp_loader.create_node_batches(n_households, batch_size=args.batch_size)
    elif n_households > 100:
        print(f"Using automatic mini-batch training for large network (size: {n_households})")
        batch_size = min(200, 2*n_households/2)
        from data import DataLoader
        temp_loader = DataLoader('')
        node_batches = temp_loader.create_node_batches(n_households, batch_size=batch_size)
    else:
        print(f"Using full-batch training for small network (size: {n_households})")
        node_batches = None
    
    # Continue training
    max_total_epochs = trainer.epoch + args.epochs
    
    history = trainer.train(
        features=train_data['features'],
        states=train_data['states'],
        distances=train_data['distances'],
        network_data=train_data['observed_network'],
        ground_truth_network=data['ground_truth_network'],
        max_timestep=train_data['n_timesteps'] - 1,
        max_epochs= args.epochs,
        node_batches=node_batches,
        verbose=args.verbose,
        early_stopping=args.early_stopping,
        patience=args.patience
    )
    
    return history


def save_continued_model(trainer: NetworkStateTrainer, original_checkpoint: Dict, 
                        args, train_data: Dict):
    """Save the continued model with updated metadata."""
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Generate save filename
    original_filename = Path(args.model_path).stem
    save_filename = f"{original_filename}_{args.save_suffix}.pth"
    save_path = save_dir / save_filename
    
    print(f"Saving continued model to: {save_path}")
    
    # Update model state
    model_state = {
        'network_type_nn': trainer.mean_field_posterior.network_type_nn.state_dict(),
        'seq_self_nn': trainer.elbo_computer.state_transition.seq_self_nn.state_dict(),
        'seq_pair_infl_nn': trainer.elbo_computer.state_transition.seq_pair_infl_nn.state_dict(),
        'interaction_nn': trainer.elbo_computer.network_evolution.interaction_nn.state_dict(),
        'network_evolution': trainer.elbo_computer.network_evolution.state_dict(),
    }
    
    # Update ELBO parameters
    elbo_params = {
        'rho_1': trainer.elbo_computer.rho_1.item(),
        'rho_2': trainer.elbo_computer.rho_2.item(),
        'confidence_weight': trainer.elbo_computer.confidence_weight
    }
    
    # Update metadata
    metadata = original_checkpoint['metadata'].copy()
    metadata.update({
        'continued_training': {
            'original_model': str(args.model_path),
            'additional_epochs': args.epochs,
            'total_epochs': trainer.epoch,
            'continuation_args': vars(args)
        },
        'final_metrics': trainer.training_history[-1] if trainer.training_history else {}
    })
    
    # Save complete checkpoint
    torch.save({
        'model_state': model_state,
        'elbo_params': elbo_params,
        'metadata': metadata,
        'training_history': trainer.training_history
    }, save_path)
    
    # Save training history separately
    history_path = save_dir / f"{original_filename}_{args.save_suffix}_history.json"
    with open(history_path, 'w') as f:
        json.dump(trainer.training_history, f, indent=2)
    
    print(f"✓ Model saved: {save_path}")
    print(f"✓ History saved: {history_path}")
    
    return save_path, history_path


def print_training_summary(trainer: NetworkStateTrainer, original_epochs: int):
    """Print summary of continued training."""
    
    print("\n" + "="*60)
    print("CONTINUED TRAINING SUMMARY")
    print("="*60)
    
    if trainer.training_history:
        final_metrics = trainer.training_history[-1]
        
        print(f"Training Progress:")
        print(f"  Original epochs: {original_epochs}")
        print(f"  Additional epochs: {trainer.epoch - original_epochs}")
        print(f"  Total epochs: {trainer.epoch}")
        
        print(f"\nFinal Metrics:")
        print(f"  ELBO: {final_metrics['total_elbo']:.4f}")
        print(f"  State likelihood: {final_metrics['state_likelihood']:.4f}")
        print(f"  Observation likelihood: {final_metrics['observation_likelihood']:.4f}")
        print(f"  Prior likelihood: {final_metrics['prior_likelihood']:.4f}")
        print(f"  Posterior entropy: {final_metrics['posterior_entropy']:.4f}")
        
        # print(f"\nFinal Parameters:")
        # print(f"  ρ₁ (bonding miss rate): {final_metrics['rho_1']:.4f}")
        # print(f"  ρ₂ (bridging miss rate): {final_metrics['rho_2']:.4f}")
        # print(f"  Temperature: {final_metrics['temperature']:.4f}")
        # print(f"  Samples: {final_metrics['num_samples']}")
    
    print("\n" + "="*60)


def main():
    """Main continuation training pipeline."""
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup dual logging (print to both console and file)
    from utils.dual_logger import setup_dual_logging, restore_normal_output
    from datetime import datetime
    
    # Create log filename with timestamp and model name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = Path(args.model_path).stem
    log_file = f"logs/continue_training_{model_name}_{timestamp}.log"
    
    os.makedirs("logs", exist_ok=True)
    dual_logger = setup_dual_logging(log_file, mode='w')
    
    print("🔄 Starting Model Continuation Training")
    print("="*60)
    print(f"📝 Logging to: {log_file}")
    
    # Print configuration summary
    if args.retrain_prediction:
        print("\n🔧 RETRAIN PREDICTION MODE")
        print("  Prediction NN will be re-initialized: hidden_dim=512, num_layers=3")
    if args.freeze_network:
        print("\n🔒 NETWORK FREEZING MODE")
        print("  Network components will be frozen")
        print("  Only prediction components will be trained")

    try:
        # Load model and data (with retrain_prediction option)
        trainer, train_data, test_data, full_data, loader = load_model_and_data(
            model_path=args.model_path,
            data_dir=args.data_dir,
            train_end_time=args.train_end_time,
            retrain_prediction=args.retrain_prediction,
            max_neighbor_influences=args.max_neighbor_influences
        )

        if args.reset_epoch is not None:
            trainer.epoch = args.reset_epoch
        original_epochs = trainer.epoch
        if args.freeze_network:
            freeze_network_components(trainer)
        update_training_parameters(trainer, args)
        weights_updated = update_elbo_weights(trainer, args)
        history = continue_training(trainer, train_data, full_data, args)
        save_path, history_path = save_continued_model(
            trainer, torch.load(args.model_path), args, train_data
        )
        print_training_summary(trainer, original_epochs)
        print(f"\n🎉 Continuation training completed successfully!")
        print(f"📁 Model saved: {save_path}")
        print(f"📊 History saved: {history_path}")
        restore_normal_output(dual_logger)
    except Exception as e:
        print(f"\n❌ Error during continuation training: {e}")
        import traceback
        traceback.print_exc()
        try:
            restore_normal_output(dual_logger)
        except:
            pass
        return 1
    return 0


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run continuation training
    exit_code = main()
    sys.exit(exit_code)