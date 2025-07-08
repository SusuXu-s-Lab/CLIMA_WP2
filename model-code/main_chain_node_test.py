import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Dict, List, Tuple

from data import DataLoader
from models import NetworkTypeNN, SelfActivationNN, InfluenceNN, InteractionFormationNN, NetworkEvolution, StateTransition
from inference import MeanFieldPosterior, GumbelSoftmaxSampler, ELBOComputation, NetworkStateTrainer

# Global recorder instance
TRAINING_RECORDER = None

class TrainingVariableRecorder:
    """捕获训练过程中真实中间变量的记录器"""
    
    def __init__(self):
        self.records = {}
        self.current_epoch = None
        self._temp_node_records = {}  # 临时存储当前计算的节点记录
        # print("TrainingVariableRecorder initialized")
        
    def start_epoch(self, epoch):
        """开始记录某个epoch"""
        # print(f"start_epoch called with epoch {epoch}")
        self.current_epoch = epoch
        if epoch not in self.records:
            self.records[epoch] = {}
        print(f"current_epoch set to: {self.current_epoch}")
            
    def record_timestep_data(self, timestep, gumbel_samples, node_computations):
        """记录某个timestep的完整数据"""
        # print(f"record_timestep_data called: timestep={timestep}, current_epoch={self.current_epoch}")
        
        if self.current_epoch is None:
            print("WARNING: current_epoch is None, not recording")
            return
            
        # print(f"Recording data for epoch {self.current_epoch}, timestep {timestep}")
        # print(f"node_computations keys: {list(node_computations.keys())}")
        
        # 转换Gumbel samples为可序列化格式
        serializable_samples = []
        for sample_idx, sample in enumerate(gumbel_samples):
            serializable_sample = {}
            for pair_key, tensor in sample.items():
                # 修复筛选条件：确保只选择当前timestep的键
                if pair_key.endswith(f"_{timestep}"):  # 只记录当前timestep的
                    serializable_sample[pair_key] = {
                        'values': tensor.detach().cpu().numpy().tolist(),
                        'most_likely': int(torch.argmax(tensor).item())
                    }
            serializable_samples.append(serializable_sample)
        
        # print(f"serializable_samples length: {len(serializable_samples)}")
        
        self.records[self.current_epoch][timestep] = {
            'gumbel_samples': serializable_samples,
            'node_computations': node_computations
        }
        
        # print(f"Data recorded. Current records structure: {list(self.records.keys())}")
    
    def save_records(self, output_dir):
        """保存记录到文件"""
        output_path = Path(output_dir)
        
        # print(f"save_records called. Total records: {len(self.records)}")
        # for epoch, epoch_data in self.records.items():
        #     print(f"  Epoch {epoch}: {len(epoch_data)} timesteps")
        #     for timestep, timestep_data in epoch_data.items():
        #         print(f"    Timestep {timestep}: {len(timestep_data.get('node_computations', {}))} nodes")
        
        # JSON格式
        with open(output_path / 'real_training_variables.json', 'w') as f:
            json.dump(self.records, f, indent=2)
        
        # 人类可读格式
        self._save_readable(output_path / 'real_training_variables.txt')
        
        print(f"💾 Real training variables saved!")
        print(f"    📊 JSON: real_training_variables.json")
        print(f"    📄 TXT: real_training_variables.txt")
    
    def _save_readable(self, txt_path):
        """保存人类可读的格式"""
        lines = []
        lines.append("REAL TRAINING INTERMEDIATE VARIABLES")
        lines.append("="*80)
        lines.append("")
        lines.append("This file contains the EXACT variables from model training:")
        lines.append("- Gumbel-Softmax samples generated during training")
        lines.append("- Self-activation probabilities from neural networks")
        lines.append("- Neighbor influence probabilities from neural networks")
        lines.append("- Final FR-SIC probabilities: 1 - (1-p_self) * ∏(1-p_neighbor)")
        lines.append("- Only inactive nodes are shown (only these need prediction)")
        lines.append("")
        
        if not self.records:
            lines.append("NO DATA RECORDED!")
            lines.append("This indicates that the recording logic was not triggered.")
            lines.append("")
        
        for epoch in sorted(self.records.keys()):
            lines.append("="*60)
            lines.append(f"EPOCH {epoch}")
            lines.append("="*60)
            lines.append("")
            
            epoch_data = self.records[epoch]
            if not epoch_data:
                lines.append("No data for this epoch")
                lines.append("")
                continue
            
            for timestep in sorted(epoch_data.keys()):
                data = epoch_data[timestep]
                lines.append(f"TIMESTEP {timestep}:")
                lines.append("-" * 40)
                
                # Gumbel samples
                num_samples = len(data['gumbel_samples'])
                lines.append(f"Gumbel Samples ({num_samples}):")
                for sample_idx, sample in enumerate(data['gumbel_samples']):
                    lines.append(f"  Sample {sample_idx}:")
                    for pair_key, gumbel_data in sample.items():
                        vals = gumbel_data['values']
                        most_likely = gumbel_data['most_likely']
                        lines.append(f"    {pair_key}: [{vals[0]:.4f}, {vals[1]:.4f}, {vals[2]:.4f}] → type_{most_likely}")
                    if not sample:
                        lines.append("    (no hidden pairs at this timestep)")
                lines.append("")
                
                # Node computations
                node_computations = data['node_computations']
                if node_computations:
                    lines.append("Node Computations (inactive nodes only):")
                    
                    for node_key in sorted(node_computations.keys()):
                        node_idx = node_key.split('_')[1]
                        computations = node_computations[node_key]
                        
                        lines.append(f"  Node {node_idx}:")
                        
                        # 计算跨样本的平均概率
                        sample_final_probs = [comp['final'] for comp in computations]
                        avg_final_prob = sum(sample_final_probs) / len(sample_final_probs)
                        
                        lines.append(f"    Average final probability: {avg_final_prob:.6f}")
                        lines.append(f"    Details by sample:")
                        
                        for comp in computations:
                            sample_idx = comp['sample']
                            self_prob = comp['self']
                            final_prob = comp['final']
                            neighbors = comp['neighbors']
                            
                            lines.append(f"      Sample {sample_idx}:")
                            lines.append(f"        Self-activation: {self_prob:.6f}")
                            
                            if neighbors:
                                lines.append(f"        Neighbor influences:")
                                product_term = 1.0
                                for neighbor in neighbors:
                                    source = neighbor['source_node']
                                    prob = neighbor['influence_prob']
                                    link_type = neighbor['link_type_info']
                                    product_term *= (1.0 - prob)
                                    lines.append(f"          {source}→{node_idx}: {prob:.6f} ({link_type})")
                                
                                lines.append(f"        Product ∏(1-p_neighbor): {product_term:.6f}")
                                lines.append(f"        FR-SIC: 1 - (1-{self_prob:.6f}) * {product_term:.6f} = {final_prob:.6f}")
                            else:
                                lines.append(f"        No active neighbors")
                                lines.append(f"        Final = Self: {final_prob:.6f}")
                            lines.append("")
                else:
                    lines.append("No inactive nodes at this timestep")
                
                lines.append("")
                lines.append("")
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))


class ChainNetworkTester:
    """
    专注的Chain Network测试器，现在带有真实训练变量捕获功能和网络结构分析
    """
    
    def __init__(self, data_dir='CODE 6/data/syn_data_chain_node'):
        self.data_dir = Path(data_dir)
        
        # 初始化全局记录器
        global TRAINING_RECORDER
        TRAINING_RECORDER = TrainingVariableRecorder()
        self.recorder = TRAINING_RECORDER
        
        # 设置到其他模块
        import models.state_dynamics as state_dynamics_module
        import inference.elbo_computation as elbo_module
        state_dynamics_module.TRAINING_RECORDER = TRAINING_RECORDER
        elbo_module.TRAINING_RECORDER = TRAINING_RECORDER
        
        print(f"Global TRAINING_RECORDER set to: {id(TRAINING_RECORDER)}")
        
        self.monitoring_data = {
            'epochs': [],
            'elbo_components': [],
            'network_probabilities': [],  # 保留structure分析
            'parameters': []
        }
        
    def setup_model(self, L=1):
        """Initialize all model components"""
        print("=== Setting Up Model Components ===")
        
        data = self._load_test_data()
        
        feature_dim = data['features'].shape[1]
        print(f"Feature dimension: {feature_dim}")
        print(f"Number of households: {data['n_households']}")
        print(f"Number of timesteps: {data['n_timesteps']}")
        
        # Initialize neural networks
        network_type_nn = NetworkTypeNN(feature_dim=feature_dim, L=L, hidden_dim=32)
        self_nn = SelfActivationNN(feature_dim=feature_dim, L=L, hidden_dim=32)
        influence_nn = InfluenceNN(feature_dim=feature_dim, L=L, hidden_dim=32)
        interaction_nn = InteractionFormationNN(feature_dim=feature_dim, hidden_dim=32)
        
        # Initialize model components
        network_evolution = NetworkEvolution(interaction_nn)
        network_evolution.set_normalization_factors(data['features'], data['distances'])
        
        state_transition = StateTransition(self_nn, influence_nn, L=L)
        
        # Initialize training components
        mean_field_posterior = MeanFieldPosterior(network_type_nn, L=L)
        gumbel_sampler = GumbelSoftmaxSampler()
        elbo_computer = ELBOComputation(network_evolution, state_transition, sparsity_weight=0.0)
        
        trainer = NetworkStateTrainer(
            mean_field_posterior=mean_field_posterior,
            gumbel_sampler=gumbel_sampler,
            elbo_computer=elbo_computer,
            learning_rate=1e-3,
            weight_decay=1e-4
        )
        
        self.data = data
        self.trainer = trainer
        self.components = {
            'mean_field_posterior': mean_field_posterior,
            'network_evolution': network_evolution,
            'state_transition': state_transition,
            'elbo_computer': elbo_computer,
            'gumbel_sampler': gumbel_sampler
        }
        
        return self.data, self.trainer, self.components
    
    def _load_test_data(self):
        """Load test data using ChainNetworkTestGenerator"""
        from data import ChainNetworkTestGenerator
        
        # Generate the test data if it doesn't exist
        generator = ChainNetworkTestGenerator()
        generator.generate_all_data(output_dir=self.data_dir)
        
        # Now load it with your actual DataLoader
        loader = DataLoader(self.data_dir, device='cpu')
        data = loader.load_data()
        
        return data
    
    def analyze_initial_state(self):
        """初始状态分析"""
        print("\n" + "="*80)
        print("INITIAL MODEL STATE ANALYSIS")
        print("="*80)
        
        # 1. Parameter initialization
        self._analyze_parameters("INITIAL")
        
        # 2. Network structure inference (enhanced)
        self._analyze_complete_network_inference("INITIAL")
    
    def _analyze_parameters(self, stage):
        """Analyze all model parameters"""
        print(f"\n--- {stage} PARAMETERS ---")
        
        # Network evolution parameters
        ne = self.components['network_evolution']
        print(f"Network Evolution:")
        print(f"  alpha_0: {ne.alpha_0.item():.6f}")
        print(f"  sigma_demo_sq: {ne.sigma_demo_sq.item():.6f}")
        print(f"  sigma_geo_sq: {ne.sigma_geo_sq.item():.6f}")
        
        # Observation parameters
        ec = self.components['elbo_computer']
        print(f"Observation Model:")
        print(f"  rho_1 (bonding miss rate): {ec.rho_1.item():.6f}")
        print(f"  rho_2 (bridging miss rate): {ec.rho_2.item():.6f}")
        
        # Store for tracking
        param_dict = {
            'stage': stage,
            'alpha_0': ne.alpha_0.item(),
            'rho_1': ec.rho_1.item(),
            'rho_2': ec.rho_2.item()
        }
        self.monitoring_data['parameters'].append(param_dict)
    
    def _analyze_complete_network_inference(self, stage):
        """Complete analysis of network inference - ALL hidden pairs, ALL timesteps (from old_main.py)"""
        print(f"\n--- {stage} COMPLETE NETWORK INFERENCE ---")
        
        try:
            # Compute variational probabilities for all hidden pairs
            node_batch = torch.arange(self.data['n_households'])
            conditional_probs, marginal_probs = self.components['mean_field_posterior'].compute_probabilities_batch(
                self.data['features'], self.data['states'], self.data['distances'],
                node_batch, self.data['observed_network'], self.data['n_timesteps'] - 1
            )
            
            # Save complete analysis to file (from old_main.py)
            self._save_complete_network_analysis(stage, conditional_probs, marginal_probs)
            
            # Print brief summary to console (from old_main.py)
            print("Brief console summary (detailed analysis saved to file):")
            
            # Get all hidden pairs
            all_hidden_pairs = set()
            for key in marginal_probs.keys():
                i, j, t = key.split('_')
                all_hidden_pairs.add((int(i), int(j)))
            
            sorted_hidden_pairs = sorted(list(all_hidden_pairs))
            
            print(f"Total hidden pairs: {len(sorted_hidden_pairs)}")
            print(f"Pairs: {sorted_hidden_pairs}")
            
            # Show accuracy for first and last timestep
            for t in [0, self.data['n_timesteps'] - 1]:
                print(f"\nTimestep {t} accuracy:")
                for i, j in sorted_hidden_pairs:
                    pair_key = f"{i}_{j}_{t}"
                    if pair_key in marginal_probs:
                        prob = marginal_probs[pair_key].detach().numpy()
                        predicted_idx = np.argmax(prob)
                        confidence = prob[predicted_idx]
                        link_types = ['none', 'bonding', 'bridging']
                        predicted_type = link_types[predicted_idx]
                        print(f"  ({i},{j}): {predicted_type} (conf: {confidence:.3f})")
            
            # Store for tracking
            prob_dict = {
                'stage': stage,
                'marginal_probs': {k: v.detach().clone() for k, v in marginal_probs.items()},
                'conditional_probs': {k: v.detach().clone() for k, v in conditional_probs.items()}
            }
            self.monitoring_data['network_probabilities'].append(prob_dict)
            
        except Exception as e:
            print(f"Error in complete network inference analysis: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_complete_network_analysis(self, stage, conditional_probs, marginal_probs):
        """Save COMPLETE network analysis - all pairs, all timesteps, all details (from old_main.py)"""
        
        # Prepare filename
        stage_clean = stage.lower().replace(' ', '_').replace(':', '')
        filename = f"complete_network_analysis_{stage_clean}.txt"
        filepath = self.data_dir / filename
        
        lines = []
        lines.append("="*100)
        lines.append(f"COMPLETE NETWORK ANALYSIS - {stage}")
        lines.append("="*100)
        lines.append("")
        
        # Get all hidden pairs
        all_hidden_pairs = set()
        for key in marginal_probs.keys():
            i, j, t = key.split('_')
            all_hidden_pairs.add((int(i), int(j)))
        
        sorted_hidden_pairs = sorted(list(all_hidden_pairs))
        
        # Expected structure (ground truth)
        lines.append("EXPECTED GROUND TRUTH:")
        lines.append("- Chain structure: A(0)-B(1)-C(2)-D(3)-E(4)")
        lines.append("- Should have bridging links: (0,1), (1,2), (2,3), (3,4)")
        lines.append("- Should have NO links: all other pairs")
        lines.append("- Observed pairs: (0,1) and (2,3)")
        lines.append(f"- Hidden pairs to infer: {sorted_hidden_pairs}")
        lines.append("")
        
        # For each timestep, show complete analysis
        for t in range(self.data['n_timesteps']):
            lines.append("="*80)
            lines.append(f"TIMESTEP {t}")
            lines.append("="*80)
            lines.append("")
            
            # For each hidden pair, show complete analysis
            for i, j in sorted_hidden_pairs:
                pair_key = f"{i}_{j}_{t}"
                if pair_key in marginal_probs:
                    
                    # Determine expected type
                    if (i, j) in [(0,1), (1,2), (2,3), (3,4)]:
                        expected = "bridging"
                        pair_type = "CHAIN"
                    else:
                        expected = "none"
                        pair_type = "NON-CHAIN"
                    
                    lines.append(f"Pair ({i},{j}) [{pair_type}] - Expected: {expected}")
                    lines.append("-" * 60)
                    
                    # Marginal probabilities
                    marginal_prob = marginal_probs[pair_key].detach().numpy()
                    predicted_idx = np.argmax(marginal_prob)
                    confidence = marginal_prob[predicted_idx]
                    link_types = ['none', 'bonding', 'bridging']
                    predicted_type = link_types[predicted_idx]
                    is_correct = predicted_type == expected
                    status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
                    
                    lines.append(f"MARGINAL PROBABILITIES π̄_{i}{j}(t={t}):")
                    lines.append(f"  Values: [{marginal_prob[0]:.6f}, {marginal_prob[1]:.6f}, {marginal_prob[2]:.6f}]")
                    lines.append(f"  Predicted: {predicted_type} (confidence: {confidence:.6f})")
                    lines.append(f"  Expected:  {expected}")
                    lines.append(f"  Result: {status}")
                    lines.append("")
                    
                    # Conditional probabilities (full 3x3 matrix)
                    if pair_key in conditional_probs:
                        conditional_prob = conditional_probs[pair_key].detach().numpy()
                        lines.append(f"CONDITIONAL PROBABILITIES π_{i}{j}(t={t}|k') [3x3 matrix]:")
                        lines.append("                    Current State")
                        lines.append("Prev State    |   none   bonding bridging")
                        lines.append("------------- | -------- ------- --------")
                        
                        for k_prev in range(3):
                            row = conditional_prob[k_prev, :]
                            lines.append(f"{link_types[k_prev]:>13} | {row[0]:8.6f} {row[1]:7.6f} {row[2]:8.6f}")
                        
                        lines.append("")
                        lines.append("Consistency Check:")
                        all_good = True
                        for k_prev in range(3):
                            row_sum = np.sum(conditional_prob[k_prev, :])
                            status_check = "✓" if 0.99 <= row_sum <= 1.01 else "✗"
                            if status_check == "✗":
                                all_good = False
                            lines.append(f"  Row {k_prev} ({link_types[k_prev]}): sum = {row_sum:.6f} {status_check}")
                        
                        if all_good:
                            lines.append("  Overall: ✓ All rows sum to 1.0")
                        else:
                            lines.append("  Overall: ✗ Some rows don't sum to 1.0 - NUMERICAL ISSUE!")
                    else:
                        lines.append(f"CONDITIONAL PROBABILITIES: Not available")
                    
                    lines.append("")
                    lines.append("")
        
        # Summary statistics
        lines.append("="*80)
        lines.append("SUMMARY STATISTICS")
        lines.append("="*80)
        lines.append("")
        
        # Overall accuracy by timestep
        lines.append("Accuracy by Timestep:")
        for t in range(self.data['n_timesteps']):
            correct = 0
            total = 0
            
            for i, j in sorted_hidden_pairs:
                pair_key = f"{i}_{j}_{t}"
                if pair_key in marginal_probs:
                    # Determine expected
                    expected = "bridging" if (i, j) in [(0,1), (1,2), (2,3), (3,4)] else "none"
                    
                    # Get predicted
                    prob = marginal_probs[pair_key].detach().numpy()
                    predicted_idx = np.argmax(prob)
                    link_types = ['none', 'bonding', 'bridging']
                    predicted_type = link_types[predicted_idx]
                    
                    if predicted_type == expected:
                        correct += 1
                    total += 1
            
            accuracy = correct / total if total > 0 else 0
            lines.append(f"  t={t:2d}: {correct:2d}/{total:2d} = {accuracy:6.1%}")
        
        lines.append("")
        
        # Error analysis
        lines.append("Error Analysis:")
        lines.append("False Positives (predicted link but should be none):")
        lines.append("False Negatives (predicted none but should be link):")
        
        for t in [0, self.data['n_timesteps'] - 1]:  # First and last timestep
            lines.append(f"\n  Timestep {t}:")
            false_pos = []
            false_neg = []
            
            for i, j in sorted_hidden_pairs:
                pair_key = f"{i}_{j}_{t}"
                if pair_key in marginal_probs:
                    expected = "bridging" if (i, j) in [(0,1), (1,2), (2,3), (3,4)] else "none"
                    
                    prob = marginal_probs[pair_key].detach().numpy()
                    predicted_idx = np.argmax(prob)
                    link_types = ['none', 'bonding', 'bridging']
                    predicted_type = link_types[predicted_idx]
                    confidence = prob[predicted_idx]
                    
                    if expected == "none" and predicted_type != "none":
                        false_pos.append(f"({i},{j}): predicted {predicted_type} (conf: {confidence:.3f})")
                    elif expected == "bridging" and predicted_type == "none":
                        false_neg.append(f"({i},{j}): predicted none (conf: {confidence:.3f})")
            
            if false_pos:
                lines.append("    False Positives:")
                for fp in false_pos:
                    lines.append(f"      {fp}")
            else:
                lines.append("    False Positives: None ✓")
            
            if false_neg:
                lines.append("    False Negatives:")
                for fn in false_neg:
                    lines.append(f"      {fn}")
            else:
                lines.append("    False Negatives: None ✓")
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(line + '\n')
        
        print(f"💾 Complete network analysis saved to: {filename}")
    
    def _analyze_network_inference(self, stage):
        """简化版本的网络结构推断分析（为了向后兼容）"""
        print(f"\n--- {stage} NETWORK STRUCTURE INFERENCE ---")
        
        try:
            # Compute variational probabilities for all hidden pairs
            node_batch = torch.arange(self.data['n_households'])
            conditional_probs, marginal_probs = self.components['mean_field_posterior'].compute_probabilities_batch(
                self.data['features'], self.data['states'], self.data['distances'],
                node_batch, self.data['observed_network'], self.data['n_timesteps'] - 1
            )
            
            # Print brief summary
            print("Brief summary:")
            
            # Get all hidden pairs
            all_hidden_pairs = set()
            for key in marginal_probs.keys():
                i, j, t = key.split('_')
                all_hidden_pairs.add((int(i), int(j)))
            
            sorted_hidden_pairs = sorted(list(all_hidden_pairs))
            print(f"Hidden pairs to infer: {sorted_hidden_pairs}")
            
            # Show accuracy for first timestep
            print(f"\nTimestep 0 predictions:")
            for i, j in sorted_hidden_pairs:
                pair_key = f"{i}_{j}_0"
                if pair_key in marginal_probs:
                    prob = marginal_probs[pair_key].detach().numpy()
                    predicted_idx = np.argmax(prob)
                    confidence = prob[predicted_idx]
                    link_types = ['none', 'bonding', 'bridging']
                    predicted_type = link_types[predicted_idx]
                    
                    expected = "bridging" if (i, j) in [(1,2), (3,4)] else "none"
                    status = "✓" if predicted_type == expected else "✗"
                    print(f"  {status} ({i},{j}): {predicted_type} (conf: {confidence:.3f})")
            
            # Store for tracking
            prob_dict = {
                'stage': stage,
                'marginal_probs': {k: v.detach().clone() for k, v in marginal_probs.items()},
                'conditional_probs': {k: v.detach().clone() for k, v in conditional_probs.items()}
            }
            self.monitoring_data['network_probabilities'].append(prob_dict)
            
        except Exception as e:
            print(f"Error in network inference analysis: {e}")
    
    def monitor_training_epoch(self, epoch, metrics):
        """Monitor training with real variable recording and enhanced network analysis"""
        
        # 通知记录器开始新epoch (只记录特定epoch)
        should_record = (epoch % 20 == 0 or epoch < 5)
        print(f"Epoch {epoch}: should_record = {should_record}")
        
        if should_record:
            print(f"Starting recording for epoch {epoch}")
            self.recorder.start_epoch(epoch)
            print(f"Recorder current_epoch set to: {self.recorder.current_epoch}")
            print(f"Global TRAINING_RECORDER current_epoch: {TRAINING_RECORDER.current_epoch}")
        
        # 存储基本metrics
        self.monitoring_data['epochs'].append(epoch)
        self.monitoring_data['elbo_components'].append(metrics.copy())
        
        # 详细分析每20个epoch
        if epoch % 20 == 0:
        #     print(f"\n" + "="*50 + f" EPOCH {epoch} " + "="*50)
            
        #     # ELBO components
        #     print(f"ELBO Components:")
        #     print(f"  Total ELBO: {metrics['total_elbo']:.6f}")
        #     print(f"  State likelihood: {metrics['state_likelihood']:.6f}")
        #     print(f"  Observation likelihood: {metrics['observation_likelihood']:.6f}")
        #     print(f"  Prior likelihood: {metrics['prior_likelihood']:.6f}")
        #     print(f"  MeanFieldPosterior entropy: {metrics['posterior_entropy']:.6f}")
        #     if 'constraint_penalty' in metrics:
        #         print(f"  Constraint penalty: {metrics['constraint_penalty']:.6f}")
            
            # Model parameters
            self._analyze_parameters(f"EPOCH_{epoch}")
            
            # Complete network analysis (enhanced version from old_main.py)
            print("Running complete network structure analysis (saved to files)...")
            self._analyze_complete_network_inference(f"EPOCH_{epoch}")
            
            print("Training variables are being recorded automatically...")
        
        # 简要更新每10个epoch
        elif epoch % 10 == 0:
            print(f"\nEpoch {epoch}: ELBO = {metrics['total_elbo']:.4f}, Temp = {metrics['temperature']:.4f}")
    
    def run_complete_test(self, L=1, max_epochs=100):
        """Run complete test with real variable recording and enhanced network analysis"""
        # print("="*80)
        # print("CHAIN NETWORK TEST WITH REAL VARIABLE RECORDING & NETWORK ANALYSIS")
        # print("="*80)
        
        # 1. Setup
        data, trainer, components = self.setup_model(L)
        
        # 2. Initial analysis (with enhanced network analysis)
        self.analyze_initial_state()
        
        # 3. Training with automatic variable recording and network analysis
        # print(f"\n" + "="*50)
        # print("STARTING TRAINING WITH VARIABLE RECORDING & NETWORK ANALYSIS")
        # print("="*50)
        # print("Note: Training variables will be automatically captured during training")
        # print("Note: Complete network analysis will be saved to files every 20 epochs")
        # print("Only inactive nodes (those needing prediction) will be recorded")
        
        # Create custom training loop with monitoring
        for epoch in range(max_epochs):
            try:
                # print(f"\n--- Starting epoch {epoch} ---")
                
                # 通知记录器开始记录这个epoch (如果需要记录的话)
                should_record = (epoch % 100 == 0 or epoch < 5)
                if should_record:
                    # print(f"This epoch will be recorded")
                    self.recorder.start_epoch(epoch)
                
                # 正常训练步骤 - 变量记录在训练过程中自动发生
                # print("Calling trainer.train_epoch_full...")
                metrics = trainer.train_epoch_full(
                    features=data['features'],
                    states=data['states'],
                    distances=data['distances'],
                    network_data=data['observed_network'],
                    max_timestep=data['n_timesteps'] - 1,
                    max_epochs=max_epochs,
                    lambda_constraint=0.01
                )
                # print("trainer.train_epoch_full completed")
                
                # 监控 (记录在训练步骤中自动发生，网络分析也在这里)
                self.monitor_training_epoch(epoch, metrics)
                
                # 检查收敛
                if epoch > 10:
                    recent_elbos = [m['total_elbo'] for m in self.monitoring_data['elbo_components'][-5:]]
                    if all(abs(elbo - recent_elbos[0]) < 1e-6 for elbo in recent_elbos):
                        print(f"Early stopping: ELBO converged at epoch {epoch}")
                        break
                        
                if epoch > 0 and metrics['total_elbo'] < -1000:
                    print(f"Warning: ELBO became very negative ({metrics['total_elbo']:.2f})")
                    
            except Exception as e:
                print(f"Error in training epoch {epoch}: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # 4. 保存记录的真实训练变量
        # print(f"\n" + "="*50)
        # print("SAVING REAL TRAINING VARIABLES")
        # print("="*50)
        self.recorder.save_records(self.data_dir)
        
        # 5. Final analysis (with enhanced network analysis)
        # print(f"\n" + "="*50)
        # print("FINAL ANALYSIS")
        # print("="*50)
        
        self._analyze_parameters("FINAL")
        self._analyze_complete_network_inference("FINAL")
        
        # 6. Create summary
        self.save_summary_report()
        
        return self.monitoring_data
    
    def save_summary_report(self):
        """Save summary report with enhanced network analysis info"""
        report_path = self.data_dir / 'summary_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("CHAIN NETWORK TEST WITH REAL VARIABLE RECORDING & NETWORK ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            # Test setup
            f.write("TEST SETUP:\n")
            f.write("- 5 households in chain structure: A(0)-B(1)-C(2)-D(3)-E(4)\n")
            f.write("- Sequential activation: A(t=1), B(t=2), C(t=3), D(t=4), E(t=5)\n")
            f.write("- Observed pairs: (0,1) and (2,3)\n")
            f.write("- Hidden pairs to infer: (1,2) and (3,4) should be bridging\n")
            f.write("- Only 'vacant' decision is active\n\n")
            
            # Training results summary
            if self.monitoring_data['elbo_components']:
                initial_elbo = self.monitoring_data['elbo_components'][0]['total_elbo']
                final_elbo = self.monitoring_data['elbo_components'][-1]['total_elbo']
                f.write(f"TRAINING RESULTS:\n")
                f.write(f"- Initial ELBO: {initial_elbo:.6f}\n")
                f.write(f"- Final ELBO: {final_elbo:.6f}\n")
                f.write(f"- Improvement: {final_elbo - initial_elbo:+.6f}\n\n")
            
            # Generated files
            f.write("GENERATED FILES:\n")
            f.write("1. REAL TRAINING VARIABLES (MAIN OUTPUT):\n")
            f.write("   - real_training_variables.json (complete machine-readable data)\n")
            f.write("   - real_training_variables.txt (human-readable analysis)\n\n")
            
            f.write("2. COMPLETE NETWORK STRUCTURE ANALYSIS (NEW):\n")
            f.write("   - complete_network_analysis_initial.txt\n")
            f.write("   - complete_network_analysis_epoch_*.txt (every 20 epochs)\n")
            f.write("   - complete_network_analysis_final.txt\n\n")
            
            f.write("3. SUPPORTING ANALYSIS:\n")
            f.write("   - summary_report.txt (this file)\n\n")
            
            f.write("WHAT THE REAL TRAINING VARIABLES CONTAIN:\n")
            f.write("For each recorded epoch and timestep:\n")
            f.write("1. Exact Gumbel-Softmax samples generated during training\n")
            f.write("2. Self-activation probabilities from neural networks\n")
            f.write("3. Neighbor influence probabilities from neural networks\n")
            f.write("4. Final FR-SIC probabilities: 1 - (1-p_self) * ∏(1-p_neighbor)\n")
            f.write("5. Only inactive nodes are included (only these need prediction)\n\n")
            
            f.write("WHAT THE NETWORK STRUCTURE ANALYSIS CONTAINS:\n")
            f.write("For each analysis stage (initial, every 20 epochs, final):\n")
            f.write("1. Complete marginal probabilities π̄_ij(t) for all hidden pairs and timesteps\n")
            f.write("2. Complete conditional probabilities π_ij(t|k') [3x3 matrices] for all hidden pairs\n")
            f.write("3. Predicted vs expected link types with confidence scores\n")
            f.write("4. Accuracy statistics by timestep\n")
            f.write("5. Error analysis (false positives/negatives)\n")
            f.write("6. Numerical consistency checks for probability matrices\n\n")
            
            f.write("KEY POINTS:\n")
            f.write("- All data comes directly from training process, no re-computation\n")
            f.write("- Variables are captured at the exact moment of computation\n")
            f.write("- Each sample shows the complete computation chain\n")
            f.write("- Averaged probabilities across samples are also provided\n")
            f.write("- Network analysis shows how model learns the chain structure over time\n")
            f.write("- Expected: pairs (1,2) and (3,4) should be inferred as bridging links\n")
            f.write("- Expected: all other hidden pairs should be inferred as no connection\n")
        
        print(f"💾 Summary report saved to: summary_report.txt")


# Main function to run the test
def run_real_variable_capture_test():
    """运行带有真实变量捕获和完整网络分析的测试"""
    
    # Run the test
    tester = ChainNetworkTester()
    results = tester.run_complete_test(L=3, max_epochs=3)  # 增加到300个epoch进行完整测试

    print(f"\n{'='*80}")
    print("REAL VARIABLE CAPTURE & NETWORK ANALYSIS TEST COMPLETE!")
    print(f"{'='*80}")
    print(f"Generated files contain:")
    print(f"🔍 Exact Gumbel-Softmax samples from training")
    print(f"🧮 Real self-activation probabilities from neural networks")
    print(f"⚡ Real neighbor influence probabilities from neural networks") 
    print(f"📊 Real FR-SIC final probabilities")
    print(f"📋 Only inactive nodes (those needing prediction)")
    print(f"🔗 Complete network structure analysis for all hidden pairs")
    print(f"📈 Marginal & conditional probabilities for all timesteps")
    print(f"✅ Accuracy tracking and error analysis")
    
    return results


if __name__ == "__main__":
    print("Starting chain network test with real variable capture and network analysis...")
    results = run_real_variable_capture_test()