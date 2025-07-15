"""
Simple model loading and evaluation script.
"""

from pathlib import Path
import sys
import pdb
project_root = Path(__file__).parent
sys.path.append(str(project_root))


"""
Simple model loading and evaluation script.
"""

import torch
import pickle
from data import DataLoader
from models import NetworkTypeNN, SelfActivationNN, InfluenceNN, InteractionFormationNN
from models import NetworkEvolution, StateTransition
from inference import MeanFieldPosterior, GumbelSoftmaxSampler, ELBOComputation, NetworkStateTrainer
from evaluation.evaluation_2 import evaluate_model_corrected, print_evaluation_results
import warnings
warnings.filterwarnings("ignore")

def load_model(model_path, device='cpu'):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get config from metadata (based on your main.py save format)
    metadata = checkpoint['metadata']
    model_config = metadata['model_config']
    feature_dim = model_config['feature_dim']
    L = model_config['L']
    
    # Recreate model components
    network_type_nn = NetworkTypeNN(feature_dim, L,hidden_dim=128)
    self_nn = SelfActivationNN(feature_dim, L,hidden_dim=64)
    influence_nn = InfluenceNN(feature_dim, L, hidden_dim=64)
    interaction_nn = InteractionFormationNN(feature_dim, hidden_dim=32)
    
    network_evolution = NetworkEvolution(interaction_nn)
    state_transition = StateTransition(self_nn, influence_nn, L)
    mean_field_posterior = MeanFieldPosterior(network_type_nn, L)
    gumbel_sampler = GumbelSoftmaxSampler()
    elbo_computer = ELBOComputation(network_evolution, state_transition)
    
    trainer = NetworkStateTrainer(mean_field_posterior, gumbel_sampler, elbo_computer)
    
    # Load weights (based on your main.py save format)
    model_state = checkpoint['model_state']
    trainer.mean_field_posterior.network_type_nn.load_state_dict(model_state['network_type_nn'])
    trainer.elbo_computer.state_transition.self_nn.load_state_dict(model_state['self_nn'])
    trainer.elbo_computer.state_transition.influence_nn.load_state_dict(model_state['influence_nn'])
    trainer.elbo_computer.network_evolution.interaction_nn.load_state_dict(model_state['interaction_nn'])
    trainer.elbo_computer.network_evolution.load_state_dict(model_state['network_evolution'])
    # trainer.elbo_computer.network_evolution.sigma_demo_sq = 14.5703
    # trainer.elbo_computer.network_evolution.sigma_geo_sq = 1.9978

    
    # Load ELBO parameters
    elbo_params = checkpoint['elbo_params']
    trainer.elbo_computer.rho_1.data = torch.tensor(elbo_params['rho_1'])
    trainer.elbo_computer.rho_2.data = torch.tensor(elbo_params['rho_2'])
    # trainer.elbo_computer.sparsity_weight = elbo_params['sparsity_weight']
    trainer.elbo_computer.confidence_weight = elbo_params['confidence_weight']
    
    return trainer

def main():
    # Load data
    data_files = [
        'household_states_community_one_hot.csv',
        'household_features_community_one_hot.csv', 
        'household_locations_community_one_hot.csv',
        'observed_network_community_one_hot.csv',
        'ground_truth_network_community_one_hot.csv'
    ]    
    data_path='data/syn_50house_300bri_70bond_alpha0.5_fastgrow'
    loader = DataLoader(data_path, file_list=data_files)
    # loader = DataLoader(project_root/'data/syn_data_v3')
    data = loader.load_data()
    train_data, test_data = loader.train_test_split(data, train_end_time=15)

    # Load model
    trainer = load_model('saved_models/syn_50house_300bri_70bond_alpha0.5_fastgrow.pth')
    
    # Evaluate
    results = evaluate_model_corrected(trainer, test_data, test_end_time=23)
    
    # Print results
    print_evaluation_results(results, data_path+'/household_states_raw.csv', trainer, test_data)
    
    # # Save results
    # with open('results.pkl', 'wb') as f:
    #     pickle.dump(results, f)
    
    # print("\nResults saved to results.pkl")

if __name__ == "__main__":
    main()