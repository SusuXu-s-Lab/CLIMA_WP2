"""
Simple model loading and evaluation script.
"""

from pathlib import Path
import sys
from datetime import datetime
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
from evaluation.evaluation_corrected import evaluate_model_corrected, print_evaluation_results

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
    self_nn = SelfActivationNN(feature_dim, L,hidden_dim=256)
    influence_nn = InfluenceNN(feature_dim, L, hidden_dim=256)
    interaction_nn = InteractionFormationNN(feature_dim, hidden_dim=128)
    
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
        'household_states_community_one_hot_with_log2.csv',
        'household_features_community_one_hot_with_log2.csv', 
        'household_locations_community_one_hot_with_log2.csv',
        'observed_network_community_one_hot_with_log2.csv',
        'ground_truth_network_community_one_hot_with_log2.csv'
    ]    

    # data_files = [
    #     'household_states_community_one_hot.csv',
    #     'household_features_community_one_hot.csv', 
    #     'household_locations_community_one_hot.csv',
    #     'observed_network_community_one_hot.csv',
    #     'ground_truth_network_community_one_hot.csv'
    # ]

    log_filename = f"evaluation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    
    log_file = open(log_filename, 'w', encoding='utf-8')
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, log_file)
    
    try:
        # loader = DataLoader(project_root/'data/syn_data_ruxiao_v2/syn_50house_200bri_20bond', file_list=data_files)
        loader = DataLoader(project_root/'data/syn_data_ruxiao_v3/syn_data1_200node', file_list=data_files)
        data = loader.load_data()
        train_data, test_data = loader.train_test_split(data, train_end_time=7)
        
        trainer = load_model(project_root/'saved_models/trained_model_ruxiao3_data1_q128_256_256_128_with_log_200node_seed123_small_pos_penalty5_epoch250_topk50.pth')
        
        results = evaluate_model_corrected(trainer, test_data, train_end_time=7, test_end_time=11)
        
        print_evaluation_results(results, test_data['ground_truth_network'], trainer)
        
        print(f"\n✅ Log saved to: {log_filename}")
        
    finally:
        sys.stdout = original_stdout
        log_file.close()
        print(f"Log saved to: {log_filename}")

if __name__ == "__main__":
    main()