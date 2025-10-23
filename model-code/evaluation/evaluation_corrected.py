"""
Simplified evaluation module with unified train/test logic.
Core improvements:
1. Unified evaluation for train/test (only differ by n_samples)
2. Proper Time → Sample aggregation for all metrics
3. Full network inference (no observed/hidden distinction)
4. Step-by-step statistics with sample averaging
5. Parallel sample evaluation for performance
6. Automatic logging to file with timestamps

Usage:
    # Method 1: Use evaluate_and_log (recommended)
    from evaluation.evaluation_corrected import evaluate_and_log
    
    results = evaluate_and_log(
        trainer, test_data, 
        model_name='my_model',
        train_end_time=10,
        test_end_time=23,
        n_train_samples=1,
        n_test_samples=5
    )
    # Automatically saves to: evaluation_logs/my_model_YYYYMMDD_HHMMSS.log
    
    # Method 2: Use EvaluationLogger context manager
    from evaluation.evaluation_corrected import EvaluationLogger, evaluate_model_corrected, print_evaluation_results
    
    with EvaluationLogger(model_name='my_model'):
        results = evaluate_model_corrected(trainer, test_data, ...)
        print_evaluation_results(results, ground_truth_network, trainer)
"""

import torch
import torch.nn.functional as F
import numpy as np
import gc
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, 
                            confusion_matrix, roc_auc_score, average_precision_score)
from data.data_loader import NetworkData
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from .probability_diagnostic import ProbabilityDiagnostic


# ============================================================================
# Logging Utility
# ============================================================================

class EvaluationLogger:
    """Context manager for logging evaluation results to file"""
    def __init__(self, log_file=None, model_name=None):
        """
        Args:
            log_file: Path to log file (if None, auto-generate)
            model_name: Model name for auto-generated log file
        """
        self.terminal = sys.stdout
        self.log_file = None
        self.log = None
        
        if log_file is None and model_name is not None:
            # Auto-generate log file name
            log_dir = Path.cwd() / 'evaluation_logs'
            log_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.log_file = log_dir / f'{model_name}_{timestamp}.log'
        elif log_file is not None:
            self.log_file = Path(log_file)
        
    def __enter__(self):
        if self.log_file is not None:
            self.log = open(self.log_file, 'w', encoding='utf-8')
            sys.stdout = self
            print(f"{'='*80}")
            print(f"EVALUATION LOG")
            print(f"{'='*80}")
            print(f"Log file: {self.log_file}")
            print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*80}\n")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.log is not None:
            print(f"\n{'='*80}")
            print(f"Log saved to: {self.log_file}")
            print(f"{'='*80}")
            sys.stdout = self.terminal
            self.log.close()
            print(f"✓ Evaluation results saved to: {self.log_file}")
    
    def write(self, message):
        self.terminal.write(message)
        if self.log is not None:
            self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        if self.log is not None:
            self.log.flush()


# ============================================================================
# Configuration
# ============================================================================

class EvalConfig:
    """Evaluation configuration"""
    def __init__(self):
        self.train_thresholds = {'vacant': 0.8, 'repair': 0.8, 'sell': 0.8}
        self.test_thresholds = {'vacant': 0.6, 'repair': 0.6, 'sell': 0.6}
        self.k_values = [5, 10, 20, 50]
        
        # Parallel processing: DISABLED due to performance issues
        # (后2个样本变得奇慢，可能是资源争抢/缓存污染)
        self.enable_parallel_samples = False  # Temporarily disabled - sequential is more stable
        self.max_parallel_workers = 3  # Max number of parallel workers (not used when disabled)


# ============================================================================
# Utility Functions
# ============================================================================

def safe_auc(y_true, y_score):
    """Safely compute AUC"""
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return roc_auc_score(y_true, y_score)
    except:
        return None


def safe_ap(y_true, y_score):
    """Safely compute Average Precision"""
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return average_precision_score(y_true, y_score)
    except:
        return None


def compute_top_k_recall(y_true, y_score, k_values):
    """Compute Top-k recall"""
    results = {}
    n_positives = np.sum(y_true)
    
    if n_positives == 0:
        return {k: None for k in k_values}
    
    for k in k_values:
        if k > len(y_true):
            results[k] = None
            continue
        top_k_indices = np.argsort(y_score)[-k:]
        top_k_true = np.array(y_true)[top_k_indices]
        results[k] = np.sum(top_k_true) / n_positives
    
    return results


def aggregate_top_k_temporal(step_results, step_n_positives, k_values):
    """Aggregate Top-k recall across timesteps with temporal weighting"""
    aggregated = {}
    total_positives = sum(step_n_positives)
    
    if total_positives == 0:
        return {k: None for k in k_values}
    
    for k in k_values:
        weighted_sum = 0.0
        valid_steps = 0
        
        for step_result, n_pos in zip(step_results, step_n_positives):
            if step_result.get(k) is not None and n_pos > 0:
                weighted_sum += step_result[k] * n_pos
                valid_steps += n_pos
        
        aggregated[k] = weighted_sum / valid_steps if valid_steps > 0 else None
    
    return aggregated


# ============================================================================
# Network Wrapper
# ============================================================================

class SimpleNetworkWrapper:
    """Simple wrapper for inferred network structure"""
    def __init__(self, structure_dict, base_network=None):
        """
        Args:
            structure_dict: {(i,j): link_type}
            base_network: Optional base network for neighbor_index
        """
        self.structure = structure_dict
        self.base_network = base_network
    
    def get_link_type(self, i, j, t):
        """Get link type for pair (i,j)"""
        return self.structure.get((min(i,j), max(i,j)), 0)
    
    def is_observed(self, i, j, t):
        """For compatibility - all links are 'inferred'"""
        return False
    
    def get_observed_edges_at_time(self, t):
        """For compatibility"""
        return []
    
    def get_hidden_pairs(self, t):
        """For compatibility"""
        return list(self.structure.keys())


# ============================================================================
# Core Evaluator
# ============================================================================

class ModelEvaluator:
    """Unified model evaluator with Time → Sample aggregation"""
    
    def __init__(self, mean_field_posterior, state_transition, config=None, enable_diagnostic=True):
        self.posterior = mean_field_posterior
        self.state_transition = state_transition
        self.config = config or EvalConfig()
        self.inference_history = {}
        self.global_bonding_pairs = set()
        
        # Probability diagnostic tool
        self.enable_diagnostic = enable_diagnostic
        if enable_diagnostic:
            self.prob_diagnostic = ProbabilityDiagnostic()
        else:
            self.prob_diagnostic = None
    
    # ========================================================================
    # Main Evaluation Entry
    # ========================================================================
    
    def evaluate(self, features, ground_truth_states, distances, observed_network,
                ground_truth_network, train_end_time=15, test_end_time=24, 
                n_train_samples=1, n_test_samples=10):
        """
        Main evaluation entry point.
        
        Args:
            n_train_samples: Number of samples for train period (default=1)
            n_test_samples: Number of samples for test period (default=10)
        """
        print(f"=== Model Evaluation ===")
        print(f"Train: 0 to {train_end_time} (samples={n_train_samples})")
        print(f"Test: {train_end_time+1} to {test_end_time} (samples={n_test_samples})")
        
        device = features.device
        n_households = features.shape[0]
        all_nodes = torch.arange(n_households, dtype=torch.long, device=device)
        
        # Identify global bonding pairs
        self.global_bonding_pairs = self._identify_global_bonding_pairs(
            observed_network, train_end_time
        )
        print(f"Global bonding pairs: {len(self.global_bonding_pairs)}")
        
        # Train period
        print("\n=== TRAIN PERIOD ===")
        train_results = self._evaluate_period(
            features, ground_truth_states, distances, all_nodes,
            observed_network, ground_truth_network,
            start_time=0,
            end_time=train_end_time,
            n_samples=n_train_samples,
            target_households=None,
            is_train=True
        )
        
        # Test period
        print("\n=== TEST PERIOD ===")
        target_households = self._identify_target_households(ground_truth_states, train_end_time)
        test_results = self._evaluate_period(
            features, ground_truth_states, distances, all_nodes,
            observed_network, ground_truth_network,
            start_time=train_end_time,
            end_time=test_end_time,
            n_samples=n_test_samples,
            target_households=target_households,
            is_train=False
        )
        
        # Print probability diagnostic if enabled
        if self.enable_diagnostic and self.prob_diagnostic is not None:
            print("\n" + "="*80)
            print("🔬 PROBABILITY DIAGNOSTIC SUMMARY")
            print("="*80)
            
            # Train period summary
            self.prob_diagnostic.print_period_summary(0, train_end_time, "TRAIN")
            
            # Test period summary
            self.prob_diagnostic.print_period_summary(train_end_time+1, test_end_time, "TEST")
            
            # Detailed timestep analysis for first few test steps
            print("\n" + "="*80)
            print("🔍 DETAILED TIMESTEP ANALYSIS")
            print("="*80)
            for t in range(train_end_time+1, min(train_end_time+4, test_end_time+1)):
                self.prob_diagnostic.print_timestep_summary(t, f"TEST t={t}")
        
        return {
            'train_evaluation': train_results,
            'test_evaluation': test_results,
            'summary': self._create_summary(train_results, test_results)
        }
    
    # ========================================================================
    # Unified Evaluation Logic
    # ========================================================================
    
    def _evaluate_period(self, features, ground_truth_states, distances, all_nodes,
                        observed_network, ground_truth_network,
                        start_time, end_time, n_samples, target_households, is_train):
        """
        Unified evaluation logic for both train and test periods.
        
        Key difference: is_train determines whether to use ground_truth_states directly
        
        Performance: Uses parallel processing when n_samples > 1 and enabled in config
        """
        device = features.device
        
        # Determine if we should use parallel processing
        # Only parallelize test samples (train uses n_samples=1 typically)
        use_parallel = (n_samples > 1 and 
                       self.config.enable_parallel_samples and 
                       not is_train)
        
        if use_parallel:
            print(f"  💡 Using parallel processing with up to {self.config.max_parallel_workers} workers")
            all_sample_results = self._run_samples_parallel(
                features, ground_truth_states, distances, all_nodes,
                observed_network, ground_truth_network,
                start_time, end_time, n_samples, target_households, is_train, device
            )
        else:
            # Sequential processing (original logic)
            all_sample_results = self._run_samples_sequential(
                features, ground_truth_states, distances, all_nodes,
                observed_network, ground_truth_network,
                start_time, end_time, n_samples, target_households, is_train, device
            )
        
        # Aggregate results
        pred_metrics = self._aggregate_samples_properly(
            all_sample_results, target_households, is_train
        )
        struct_metrics = self._aggregate_structure_samples_properly(
            all_sample_results, ground_truth_network, observed_network
        )
        step_by_step_metrics = self._compute_step_by_step_statistics(
            all_sample_results, n_samples, ground_truth_network
        )
        
        results = {
            'prediction_metrics': pred_metrics,
            'structure_metrics': struct_metrics,
            'step_by_step_metrics': step_by_step_metrics,
            'n_samples': n_samples
        }
        
        # Add final state evaluation for test period
        if not is_train and target_households is not None:
            final_eval = self._aggregate_final_evals(
                [r['final_eval'] for r in all_sample_results]
            )
            results['final_and_timing_evaluation'] = final_eval
        
        return results
    
    def _run_samples_sequential(self, features, ground_truth_states, distances, all_nodes,
                               observed_network, ground_truth_network,
                               start_time, end_time, n_samples, target_households, is_train, device):
        """
        Run samples sequentially (original logic).
        Used for train period or when parallel processing is disabled.
        """
        all_sample_results = []
        
        with torch.no_grad():
            for sample_idx in range(n_samples):
                # ✅ Always set seed for reproducibility (even when n_samples=1)
                sample_seed = 42 + sample_idx
                torch.manual_seed(sample_seed)
                np.random.seed(sample_seed)
                
                if n_samples > 1:
                    print(f"  Running simulation {sample_idx+1}/{n_samples}")
                else:
                    print(f"  Running simulation (seed={sample_seed})")
                
                sample_result = self._run_single_simulation(
                    features, ground_truth_states, distances, all_nodes,
                    observed_network, ground_truth_network,
                    start_time, end_time, target_households, is_train
                )
                
                all_sample_results.append(sample_result)
                
                # Memory cleanup
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                elif device.type == 'mps':
                    torch.mps.empty_cache()
                gc.collect()
        
        return all_sample_results
    
    def _run_samples_parallel(self, features, ground_truth_states, distances, all_nodes,
                             observed_network, ground_truth_network,
                             start_time, end_time, n_samples, target_households, is_train, device):
        """
        Run samples in parallel using ThreadPoolExecutor.
        
        Note: Uses threads instead of processes because:
        1. PyTorch models are not easily picklable for multiprocessing
        2. Most computation is in PyTorch which releases GIL during operations
        3. Threads share memory, reducing overhead
        
        This provides near-linear speedup for multiple samples.
        """
        all_sample_results = [None] * n_samples
        max_workers = min(self.config.max_parallel_workers, n_samples)
        
        def run_single_sample(sample_idx):
            """Wrapper function for parallel execution"""
            # Set seed for reproducibility
            sample_seed = 42 + sample_idx
            torch.manual_seed(sample_seed)
            np.random.seed(sample_seed)
            
            print(f"  Running simulation {sample_idx+1}/{n_samples} (worker thread)")
            
            with torch.no_grad():
                result = self._run_single_simulation(
                    features, ground_truth_states, distances, all_nodes,
                    observed_network, ground_truth_network,
                    start_time, end_time, target_households, is_train
                )
            
            # Memory cleanup
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            elif device.type == 'mps':
                torch.mps.empty_cache()
            gc.collect()
            
            return sample_idx, result
        
        # Execute samples in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(run_single_sample, idx): idx 
                      for idx in range(n_samples)}
            
            # Collect results as they complete
            for future in as_completed(futures):
                sample_idx, result = future.result()
                all_sample_results[sample_idx] = result
        
        print(f"  ✓ All {n_samples} samples completed")
        return all_sample_results
    
    def _run_single_simulation(self, features, ground_truth_states, distances, all_nodes,
                               observed_network, ground_truth_network,
                               start_time, end_time, target_households, is_train):
        """
        Run a single simulation (one sample).
        
        Returns:
            {
                'predictions': [step_results...],
                'structures': [struct_results...],
                'final_eval': {...}  # Only for test
            }
        """
        device = features.device
        
        # Initialize model states
        if is_train:
            model_states = ground_truth_states
        else:
            model_states = ground_truth_states[:, :start_time+1, :].clone()
        
        pred_records = []
        struct_records = []
        
        self.state_transition.broken_links_history.clear()
        self.state_transition.clear_influence_tracking()
        
        # Initial structure
        struct_t0 = self._infer_structure(
            features, model_states, distances, all_nodes, observed_network, start_time
        )
        struct_records.append(struct_t0)
        self.inference_history[start_time] = struct_t0['inferred_structure']
        
        # Temporal steps
        for t in range(start_time, end_time):
            
            # Predict states
            inferred_network = SimpleNetworkWrapper(
                self.inference_history[t], observed_network
            )
            pred_result = self._predict_states(
                features, model_states, distances, all_nodes,
                inferred_network, ground_truth_states, t,
                is_train, target_households
            )
            pred_records.append(pred_result)
            
            # Update model states for test
            if not is_train:
                model_states = torch.cat([
                    model_states,
                    pred_result['next_states'].unsqueeze(1)
                ], dim=1)
            
            # Infer structure at t+1
            struct_result = self._infer_structure(
                features, model_states, distances, all_nodes, observed_network, t+1
            )
            
            # Apply bonding persistence
            struct_result = self._apply_bonding_persistence(
                struct_result, t+1, observed_network
            )
            
            struct_records.append(struct_result)
            self.inference_history[t+1] = struct_result['inferred_structure']
        
        result = {
            'predictions': pred_records,
            'structures': struct_records
        }
        
        # Evaluate final state for test
        if not is_train and target_households is not None:
            result['final_eval'] = self._evaluate_final_and_timing(
                target_households, model_states, ground_truth_states,
                start_time, end_time
            )
        
        return result
    
    # ========================================================================
    # State Prediction
    # ========================================================================
    
    def _predict_states(self, features, states, distances, all_nodes,
                    network, ground_truth_states, t, is_train, target_households):
        """Predict states for all decision types"""
        device = features.device
        
        # Get current states
        if is_train:
            current_states = states[:, t, :]
            next_states = None
        else:
            current_states = states[:, -1, :]
            next_states = current_states.clone()
        
        predictions_by_decision = {}
        
        for decision_k in range(3):
            decision_name = ['vacant', 'repair', 'sell'][decision_k]
            
            # Identify inactive households
            if is_train:
                inactive_mask = states[:, t, decision_k] == 0
                inactive = torch.where(inactive_mask)[0]
                ground_truth = states[inactive, t+1, decision_k] if len(inactive) > 0 else torch.tensor([], device=device)
            else:
                target = target_households[decision_name]
                target_tensor = torch.tensor(target, dtype=torch.long, device=device)
                
                inactive_mask = current_states[target_tensor, decision_k] == 0
                inactive = target_tensor[inactive_mask]
                active = target_tensor[~inactive_mask]
                
                # Ground truth for all targets
                ground_truth = ground_truth_states[target_tensor, t+1, decision_k]
            
            if is_train and len(inactive) == 0:
                # No inactive households at this timestep for this decision type
                # This is normal: all target households are already in active state
                predictions_by_decision[decision_name] = self._empty_prediction(
                    reason=f"all_target_households_already_active_at_t{t}"
                )
                continue
            
            # Compute predictions for inactive
            all_pred = []
            all_prob = []
            
            if not is_train:
                if len(inactive) > 0:
                    temp_states = torch.cat([states[:, :t+1], next_states.unsqueeze(1)], dim=1)
                    # Enable diagnostic collection in test period
                    probs = self._compute_activation_probs(
                        inactive, decision_k, features, temp_states, distances, network, t,
                        collect_diagnostic=(not is_train and self.enable_diagnostic)
                    )
                    threshold = self._get_threshold(decision_k, t, is_train)
                    predictions = (probs > threshold).float()
                    
                    # Update next states
                    next_states[inactive, decision_k] = predictions
                else:
                    probs = torch.tensor([], device=device)
                    predictions = torch.tensor([], device=device)
                
                # inactive_idx = 0
                # active_idx = 0
                
                # for h in target:
                #     if h in inactive.tolist():
                #         all_pred.append(predictions[inactive_idx].item())
                #         all_prob.append(probs[inactive_idx].item())
                #         inactive_idx += 1
                #     else:
                #         all_pred.append(1.0)
                #         all_prob.append(1.0)
                #         active_idx += 1
                
                # pred_np = np.array(all_pred)
                # prob_np = np.array(all_prob)
                # gt = ground_truth.cpu().numpy()
                
                # predictions_by_decision[decision_name] = {
                #     'predictions': pred_np,
                #     'probabilities': prob_np,
                #     'ground_truth': gt,
                #     'n_households': len(gt),
                #     'n_switched': int(gt.sum())
                # }
            
                all_pred = predictions.cpu().numpy() 
                all_prob = probs.cpu().numpy()
                gt = ground_truth[inactive_mask].cpu().numpy()  

                predictions_by_decision[decision_name] = {
                    'predictions': all_pred,
                    'probabilities': all_prob,
                    'ground_truth': gt,
                    'n_households': len(all_pred),  
                    'n_switched': int(gt.sum())
                }
                continue
            
            temp_states = states
            # Enable diagnostic collection in train period
            probs = self._compute_activation_probs(
                inactive, decision_k, features, temp_states, distances, network, t,
                collect_diagnostic=(is_train and self.enable_diagnostic)
            )
            
            threshold = self._get_threshold(decision_k, t, is_train)
            predictions = (probs > threshold).float()
            
            gt = ground_truth.cpu().numpy()
            pred_np = predictions.cpu().numpy()
            prob_np = probs.cpu().numpy()
            
            predictions_by_decision[decision_name] = {
                'predictions': pred_np,
                'probabilities': prob_np,
                'ground_truth': gt,
                'n_households': len(gt),
                'n_switched': int(gt.sum())
            }
        
        result = {
            'timestep_from': t,
            'timestep_to': t+1,
            'predictions_by_decision': predictions_by_decision
        }
        
        if not is_train:
            result['next_states'] = next_states
        
        return result
    
    def _compute_activation_probs(self, households, decision_k, features,
                                  states, distances, network, t, collect_diagnostic=False):
        """
        Compute activation probabilities using deterministic network.
        
        🚀 OPTIMIZED: Vectorized sample creation (still samples every link each timestep)
        
        Args:
            collect_diagnostic: If True, collect probability components for diagnostic
        """
        device = features.device
        n_households = features.shape[0]
        
        # 🔧 OPTIMIZATION 1: Pre-compute pair indices (cache on first call)
        if not hasattr(self, '_pair_indices_cache'):
            i_list = []
            j_list = []
            for i in range(n_households):
                for j in range(i + 1, n_households):
                    i_list.append(i)
                    j_list.append(j)
            self._pair_indices_cache = (i_list, j_list)
            self._n_pairs = len(i_list)
        
        i_indices, j_indices = self._pair_indices_cache
        
        # 🔧 OPTIMIZATION 2: Batch get link types (still queries every pair)
        link_types = [network.get_link_type(i, j, t) for i, j in zip(i_indices, j_indices)]
        link_types_tensor = torch.tensor(link_types, dtype=torch.long, device=device)
        
        # 🔧 OPTIMIZATION 3: Batch one-hot encoding (GPU parallel, single call)
        samples_tensor = F.one_hot(link_types_tensor, num_classes=3).float()  # [n_pairs, 3]
        
        # 🔧 OPTIMIZATION 4: Construct dictionary (unavoidable for API compatibility)
        samples = {
            f"{i}_{j}_{t}": samples_tensor[idx]
            for idx, (i, j) in enumerate(zip(i_indices, j_indices))
        }
        
        # Decide which function to call based on diagnostic needs
        if collect_diagnostic and self.prob_diagnostic is not None:
            # Use simplified version that returns components
            probs, components = self.state_transition.compute_activation_probability(
                household_idx=households,
                decision_type=decision_k,
                features=features,
                states=states,
                distances=distances,
                network_data=network,
                gumbel_samples=samples,
                time=t,
                return_components=True
            )
            
            # Record to diagnostic tool
            self.prob_diagnostic.record_probabilities(
                timestep=t,
                decision_type=decision_k,
                p_self=components['p_self'],
                current_neighbor_product=components['current_neighbor_product'],
                lingering_product=components['lingering_product'],
                final_probs=components['final_probs']
            )
        else:
            # Use original detailed version for compatibility
            probs, _ = self.state_transition.compute_detailed_activation_probability(
                household_idx=households,
                decision_type=decision_k,
                features=features,
                states=states,
                distances=distances,
                network_data=network,
                gumbel_samples=samples,
                time=t
            )
        
        return probs
    
    def _get_threshold(self, decision_k, t, is_train):
        """Get decision threshold"""
        decision_name = ['vacant', 'repair', 'sell'][decision_k]
        if is_train:
            return self.config.train_thresholds[decision_name]
        else:
            base = self.config.test_thresholds[decision_name]
            return max(0.3, base - max(0, t - 15) * 0.02)
    
    def _empty_prediction(self, reason="no inactive households"):
        """
        Empty prediction result when no households to predict.
        
        Args:
            reason: Explanation for why prediction is empty (for logging/debugging)
        """
        return {
            'predictions': np.array([]),
            'probabilities': np.array([]),
            'ground_truth': np.array([]),
            'n_households': 0,
            'n_switched': 0,
            'empty_reason': reason  # Add reason for tracking
        }
    
    # ========================================================================
    # Structure Inference (Full Network)
    # ========================================================================
    
    def _infer_structure(self, features, states, distances, all_nodes,
                        observed_network, t):
        """
        Infer network structure for ALL pairs (no observed/hidden distinction).
        """
        device = features.device
            
        # Get marginal probabilities for all pairs
        _, marginal_probs = self.posterior.compute_probabilities_batch(
            features, states, distances, all_nodes, observed_network, t
        )
        
        # Infer structure from marginal probabilities
        inferred = {}
        probs_dict = {}
        
        for pair_key, probs in marginal_probs.items():
            parts = pair_key.split('_')
            if len(parts) != 3:
                continue
            i, j, time = int(parts[0]), int(parts[1]), int(parts[2])
            if time != t:
                continue
            
            # Infer link type
            inferred_type = self._confident_inference(probs)
            inferred[(min(i,j), max(i,j))] = inferred_type
            probs_dict[(min(i,j), max(i,j))] = probs
        
        return {
            'timestep': t,
            'inferred_structure': inferred,
            'structure_probabilities': probs_dict
        }
    
    def _confident_inference(self, probabilities, confidence_threshold=0.75):
        """Infer link type from probabilities"""
        if hasattr(probabilities, 'detach'):
            probs = probabilities.detach().cpu().numpy()
        else:
            probs = probabilities
        
        # Too uniform -> no connection
        if max(probs) - min(probs) < 0.2:
            return 0
        
        # No connection dominant
        if np.argmax(probs) == 0:
            return 0
        
        # Low confidence
        if max(probs[1], probs[2]) < confidence_threshold:
            return 0
        
        # Close between bonding/bridging -> prefer bridging
        if abs(probs[1] - probs[2]) < 0.1:
            return 2
        
        return np.argmax(probs)
    
    # ========================================================================
    # Bonding Persistence
    # ========================================================================
    
    def _identify_global_bonding_pairs(self, observed_network, train_end_time):
        """Identify all bonding pairs observed during training"""
        bonding_pairs = set()
        n_households = observed_network.n_households
        
        for t in range(train_end_time + 1):
            for i in range(n_households):
                for j in range(i + 1, n_households):
                    if (observed_network.is_observed(i, j, t) and 
                        observed_network.get_link_type(i, j, t) == 1):
                        bonding_pairs.add((i, j))
        
        return bonding_pairs
    
    def _apply_bonding_persistence(self, struct_result, t, observed_network):
        """Apply bonding persistence rules"""
        inferred = struct_result['inferred_structure'].copy()
        
        # Global bonding persistence
        for (i, j) in self.global_bonding_pairs:
            if (i, j) in inferred:
                inferred[(i, j)] = 1
        
        # Local bonding persistence
        if t > 0:
            prev_struct = self.inference_history.get(t-1, {})
            for (i, j), prev_type in prev_struct.items():
                if prev_type == 1 and not observed_network.is_observed(i, j, t):
                    inferred[(i, j)] = 1
        
        struct_result['inferred_structure'] = inferred
        return struct_result
    
    # ========================================================================
    # Metrics Aggregation: Time → Sample
    # ========================================================================
    
    def _aggregate_samples_properly(self, all_sample_results, target_households, is_train):
        """
        Proper aggregation: Time → Sample
        
        Steps:
        1. For each sample: aggregate all time → compute metrics
        2. Across samples: compute mean ± std
        """
        n_samples = len(all_sample_results)
        k_values = self.config.k_values
        
        # For each sample, compute temporal-aggregated metrics
        sample_metrics_list = []
        for sample_result in all_sample_results:
            sample_records = sample_result['predictions']
            sample_metric = self._aggregate_one_sample_temporal(sample_records, k_values, is_train)
            sample_metrics_list.append(sample_metric)
        
        # Aggregate across samples
        aggregated = {'by_decision': {}, 'overall': {}, 'n_samples': n_samples}
        
        for decision_name in ['vacant', 'repair', 'sell']:
            # Collect values across samples
            auc_values = [s['by_decision'][decision_name]['auc'] 
                         for s in sample_metrics_list 
                         if s['by_decision'][decision_name]['auc'] is not None]
            ap_values = [s['by_decision'][decision_name]['average_precision'] 
                        for s in sample_metrics_list
                        if s['by_decision'][decision_name]['average_precision'] is not None]
            
            ap_baseline_values = [s['by_decision'][decision_name]['ap_baseline'] 
                             for s in sample_metrics_list
                             if s['by_decision'][decision_name]['ap_baseline'] is not None]
            ap_lift_values = [s['by_decision'][decision_name]['ap_lift'] 
                            for s in sample_metrics_list
                            if s['by_decision'][decision_name]['ap_lift'] is not None]
            
            recall_at_k_samples = {k: [] for k in k_values}
            precision_at_k_samples = {k: [] for k in k_values}
            lift_at_k_samples = {k: [] for k in k_values}
            
            for s in sample_metrics_list:
                for k in k_values:
                    if s['by_decision'][decision_name]['recall_at_k'].get(k) is not None:
                        recall_at_k_samples[k].append(s['by_decision'][decision_name]['recall_at_k'][k])
                    if s['by_decision'][decision_name]['precision_at_k'].get(k) is not None:
                        precision_at_k_samples[k].append(s['by_decision'][decision_name]['precision_at_k'][k])
                    if s['by_decision'][decision_name]['lift_at_k'].get(k) is not None:
                        lift_at_k_samples[k].append(s['by_decision'][decision_name]['lift_at_k'][k])
            
            # Store aggregated results
            aggregated['by_decision'][decision_name] = {
                'auc': np.mean(auc_values) if auc_values else None,
                'auc_std': np.std(auc_values) if len(auc_values) > 1 else None,
                'average_precision': np.mean(ap_values) if ap_values else None,
                'ap_std': np.std(ap_values) if len(ap_values) > 1 else None,
                'ap_baseline': np.mean(ap_baseline_values) if ap_baseline_values else None,
                'ap_baseline_std': np.std(ap_baseline_values) if len(ap_baseline_values) > 1 else None,
                'ap_lift': np.mean(ap_lift_values) if ap_lift_values else None,
                'ap_lift_std': np.std(ap_lift_values) if len(ap_lift_values) > 1 else None,
                'recall_at_k': {k: np.mean(recall_at_k_samples[k]) if recall_at_k_samples[k] else None 
                            for k in k_values},
                'recall_at_k_std': {k: np.std(recall_at_k_samples[k]) if len(recall_at_k_samples[k]) > 1 else None
                                for k in k_values},
                'precision_at_k': {k: np.mean(precision_at_k_samples[k]) if precision_at_k_samples[k] else None
                                for k in k_values},
                'precision_at_k_std': {k: np.std(precision_at_k_samples[k]) if len(precision_at_k_samples[k]) > 1 else None
                                    for k in k_values},
                'lift_at_k': {k: np.mean(lift_at_k_samples[k]) if lift_at_k_samples[k] else None
                            for k in k_values},
                'lift_at_k_std': {k: np.std(lift_at_k_samples[k]) if len(lift_at_k_samples[k]) > 1 else None
                                for k in k_values}
            }
            
            # Add confusion matrix for train only
            if is_train and n_samples == 1:
                aggregated['by_decision'][decision_name]['confusion_matrix'] = \
                    sample_metrics_list[0]['by_decision'][decision_name].get('confusion_matrix')
        
        # Overall metrics (average across decisions)
        aggregated['overall'] = self._compute_overall_metrics(aggregated['by_decision'])
        
        return aggregated
    

    
    def _aggregate_one_sample_temporal(self, sample_records, k_values, is_train):
        """
        Aggregate one sample across time using temporal-weighted averaging.
        
        Strategy:
        1. Compute metrics per timestep
        2. Weight by number of positive cases (n_switched)
        3. Average across timesteps
        
        This avoids the problem of later timesteps (with more samples but worse performance)
        dominating the overall metric.
        """
        aggregated = {'by_decision': {}}
        
        for decision_name in ['vacant', 'repair', 'sell']:
            step_data_list = []
            
            for step_result in sample_records:
                step_data = step_result['predictions_by_decision'][decision_name]
                if step_data['n_households'] > 0:
                    step_data_list.append({
                        'y_true': np.array(step_data['ground_truth']),
                        'y_score': np.array(step_data['probabilities']),
                        'y_pred': np.array(step_data['predictions']),
                        'weight': step_data['n_switched']
                    })
            
            if not step_data_list:
                aggregated['by_decision'][decision_name] = {
                    'auc': None,
                    'average_precision': None,
                    'ap_baseline': None,        
                    'ap_lift': None,            
                    'recall_at_k': {k: None for k in k_values},
                    'precision_at_k': {k: None for k in k_values},
                    'lift_at_k': {k: None for k in k_values}
                }
                continue
            
            # ========================================================================
            # NEW: Temporal-weighted AUC and AP (per-timestep then average)
            # ========================================================================
            auc_per_step = []
            ap_per_step = []
            weights = []
            
            for step_data in step_data_list:
                step_auc = safe_auc(step_data['y_true'], step_data['y_score'])
                step_ap = safe_ap(step_data['y_true'], step_data['y_score'])
                
                if step_auc is not None:
                    auc_per_step.append(step_auc)
                if step_ap is not None:
                    ap_per_step.append(step_ap)
                
                # Weight by number of positive cases in this timestep
                weights.append(step_data['weight'])
            
            # Weighted average
            if auc_per_step and sum(weights) > 0:
                auc = np.average(auc_per_step, weights=weights[:len(auc_per_step)])
            else:
                auc = None
            
            if ap_per_step and sum(weights) > 0:
                ap = np.average(ap_per_step, weights=weights[:len(ap_per_step)])
            else:
                ap = None
            
            # Baseline: overall positive rate across all timesteps
            all_y_true = np.concatenate([s['y_true'] for s in step_data_list])
            ap_baseline = np.sum(all_y_true) / len(all_y_true) if len(all_y_true) > 0 else None
            ap_lift = ap / ap_baseline if (ap is not None and ap_baseline and ap_baseline > 0) else None

            # ========================================================================
            # Top-K metrics (temporal weighted - unchanged)
            # ========================================================================
            step_recall_at_k = []
            step_n_positives = []
            for step_data in step_data_list:
                recall_k = compute_top_k_recall(step_data['y_true'], step_data['y_score'], k_values)
                step_recall_at_k.append(recall_k)
                step_n_positives.append(np.sum(step_data['y_true']))
            
            recall_at_k = aggregate_top_k_temporal(step_recall_at_k, step_n_positives, k_values)
            precision_at_k = self._compute_temporal_precision_at_k(step_data_list, k_values)
            lift_at_k = self._compute_temporal_lift_at_k(step_data_list, k_values)
            
            result = {
                'auc': auc,
                'average_precision': ap,
                'ap_baseline': ap_baseline,     
                'ap_lift': ap_lift,            
                'recall_at_k': recall_at_k,
                'precision_at_k': precision_at_k,
                'lift_at_k': lift_at_k
            }
            
            # Add confusion matrix for train only (concatenate all timesteps)
            if is_train:
                all_y_true = np.concatenate([s['y_true'] for s in step_data_list])
                all_y_pred = np.concatenate([s['y_pred'] for s in step_data_list])
                result['confusion_matrix'] = confusion_matrix(all_y_true, all_y_pred).tolist()
            
            aggregated['by_decision'][decision_name] = result
        
        return aggregated
    
    def _compute_temporal_precision_at_k(self, step_data_list, k_values):
        """Compute temporal-weighted Precision@K"""
        results = {}
        
        for k in k_values:
            weighted_sum = 0.0
            total_weight = 0.0
            
            for step_data in step_data_list:
                y_true = step_data['y_true']
                y_score = step_data['y_score']
                weight = step_data['weight']
                
                if len(y_true) >= k:
                    top_k_indices = np.argsort(y_score)[-k:]
                    n_true_in_top_k = np.sum(y_true[top_k_indices])
                    prec_k = n_true_in_top_k / k
                    
                    weighted_sum += prec_k * weight
                    total_weight += weight
            
            results[k] = weighted_sum / total_weight if total_weight > 0 else None
        
        return results
    
    def _compute_temporal_lift_at_k(self, step_data_list, k_values):
        """Compute temporal-weighted Lift@K"""
        results = {}
        
        precision_at_k = self._compute_temporal_precision_at_k(step_data_list, k_values)
        
        total_positives = sum(np.sum(step['y_true']) for step in step_data_list)
        total_samples = sum(len(step['y_true']) for step in step_data_list)
        baseline = total_positives / total_samples if total_samples > 0 else None
        
        for k in k_values:
            prec_k = precision_at_k.get(k)
            if prec_k is not None and baseline and baseline > 0:
                results[k] = prec_k / baseline
            else:
                results[k] = None
        
        return results
    
    def _compute_overall_metrics(self, by_decision):
        """Average metrics across decision types"""
        overall = {}
        
        # Nullable metrics
        for metric in ['auc', 'average_precision']:
            values = [d[metric] for d in by_decision.values() 
                    if metric in d and d[metric] is not None]
            overall[metric] = np.mean(values) if values else None
        
        # Top-K metrics
        k_values = self.config.k_values
        for k in k_values:
            for metric_name in ['recall_at_k', 'precision_at_k', 'lift_at_k']:
                values = [d[metric_name][k] for d in by_decision.values()
                        if d[metric_name].get(k) is not None]
                key = f'{metric_name.replace("_at_k", "")}_{k}'
                overall[key] = np.mean(values) if values else None
        
        return overall
    
    def _aggregate_structure_samples_properly(self, all_sample_results, 
                                             ground_truth_network, observed_network):
        """Aggregate structure metrics: Time → Sample"""
        n_samples = len(all_sample_results)
        
        sample_metrics_list = []
        for sample_result in all_sample_results:
            sample_struct_records = sample_result['structures']
            sample_metric = self._aggregate_structure_one_sample(
                sample_struct_records, ground_truth_network
            )
            sample_metrics_list.append(sample_metric)
        
        aggregated = {}
        
        # Link existence metrics
        if all('link_existence' in s for s in sample_metrics_list):
            auc_values = [s['link_existence']['auc'] for s in sample_metrics_list 
                        if s['link_existence']['auc'] is not None]
            ap_values = [s['link_existence']['average_precision'] for s in sample_metrics_list
                        if s['link_existence']['average_precision'] is not None]
            
            aggregated['link_existence'] = {
                'auc': np.mean(auc_values) if auc_values else None,
                'auc_std': np.std(auc_values) if len(auc_values) > 1 else None,
                'average_precision': np.mean(ap_values) if ap_values else None,
                'ap_std': np.std(ap_values) if len(ap_values) > 1 else None,
                'n_samples': n_samples
            }
        
        # Link type metrics
        if all('link_type' in s for s in sample_metrics_list):
            type_acc_values = [s['link_type']['accuracy'] for s in sample_metrics_list]
            aggregated['link_type'] = {
                'accuracy': np.mean(type_acc_values),
                'accuracy_std': np.std(type_acc_values) if len(type_acc_values) > 1 else None
            }
        
        return aggregated
    

    def _compute_step_by_step_statistics(self, all_sample_results, n_samples, ground_truth_network):
        """
        Compute step-by-step statistics with sample averaging.
        
        For each timestep:
        1. Compute metrics for each sample
        2. Average across samples
        
        Args:
            all_sample_results: List of sample results from _run_single_simulation
            n_samples: Number of samples
            ground_truth_network: Ground truth network for structure evaluation
            
        Returns:
            List of step-by-step statistics
        """
        if n_samples == 0:
            return []
        
        n_steps = len(all_sample_results[0]['predictions'])
        k_values = self.config.k_values
        step_by_step_stats = []
        
        for step_idx in range(n_steps):
            # Collect prediction data from all samples at this timestep
            step_samples = [
                sample_result['predictions'][step_idx]
                for sample_result in all_sample_results
            ]
            
            # Collect structure data from all samples at this timestep
            # +1 because structures list includes t=0
            step_structures = [
                sample_result['structures'][step_idx + 1]
                for sample_result in all_sample_results
            ]
            
            step_stats = {'by_decision': {}}
            
            # ========================================
            # Prediction Metrics
            # ========================================
            for decision_name in ['vacant', 'repair', 'sell']:
                # Compute metrics for each sample
                sample_metrics = []
                n_households_list = []  # Track n_households for each sample
                n_switched_list = []    # Track n_switched for each sample
                
                for step_sample in step_samples:
                    data = step_sample['predictions_by_decision'][decision_name]
                    n_households_list.append(data['n_households'])
                    n_switched_list.append(data['n_switched'])
                    
                    if data['n_households'] > 0:
                        y_true = np.array(data['ground_truth'])
                        y_score = np.array(data['probabilities'])
                        
                        auc = safe_auc(y_true, y_score)
                        ap = safe_ap(y_true, y_score)
                        recall_at_k = compute_top_k_recall(y_true, y_score, k_values)

                        precision_at_k = {}
                        for k in k_values:
                            if k <= len(y_true):
                                top_k_indices = np.argsort(y_score)[-k:]
                                n_true_in_top_k = np.sum(y_true[top_k_indices])
                                precision_at_k[k] = n_true_in_top_k / k
                            else:
                                precision_at_k[k] = None
                        
                        sample_metrics.append({
                            'auc': auc,
                            'ap': ap,
                            'recall_at_k': recall_at_k,
                            'precision_at_k': precision_at_k
                        })
                
                # Aggregate across samples
                if sample_metrics:
                    auc_values = [m['auc'] for m in sample_metrics if m['auc'] is not None]
                    ap_values = [m['ap'] for m in sample_metrics if m['ap'] is not None]
                    
                    recall_at_k_values = {k: [] for k in k_values}
                    precision_at_k_values = {k: [] for k in k_values}

                    for m in sample_metrics:
                        for k in k_values:
                            if m['recall_at_k'].get(k) is not None:
                                recall_at_k_values[k].append(m['recall_at_k'][k])
                            if m['precision_at_k'].get(k) is not None:
                                precision_at_k_values[k].append(m['precision_at_k'][k])

                    
                    step_stats['by_decision'][decision_name] = {
                        'auc_mean': np.mean(auc_values) if auc_values else None,
                        'auc_std': np.std(auc_values) if len(auc_values) > 1 else None,
                        'ap_mean': np.mean(ap_values) if ap_values else None,
                        'ap_std': np.std(ap_values) if len(ap_values) > 1 else None,
                        'recall_at_k_mean': {k: np.mean(recall_at_k_values[k]) if recall_at_k_values[k] else None
                                        for k in k_values},
                        'recall_at_k_std': {k: np.std(recall_at_k_values[k]) if len(recall_at_k_values[k]) > 1 else None
                                        for k in k_values},
                        'precision_at_k_mean': {k: np.mean(precision_at_k_values[k]) if precision_at_k_values[k] else None
                                            for k in k_values},
                        'precision_at_k_std': {k: np.std(precision_at_k_values[k]) if len(precision_at_k_values[k]) > 1 else None
                                            for k in k_values},
                        # NEW: Add raw data for better diagnostics
                        'n_households': np.mean(n_households_list) if n_households_list else 0,
                        'n_switched': np.mean(n_switched_list) if n_switched_list else 0
                    }
                else:
                    # No sample_metrics computed
                    step_stats['by_decision'][decision_name] = {
                        'auc_mean': None,
                        'ap_mean': None,
                        'n_households': np.mean(n_households_list) if n_households_list else 0,
                        'n_switched': np.mean(n_switched_list) if n_switched_list else 0
                    }
            
            # ========================================
            # Structure Metrics
            # ========================================
            step_stats['structure_metrics'] = self._aggregate_structure_one_sample_step(
                step_structures, n_samples, ground_truth_network
            )
            
            # Store timestep info
            step_stats['timestep_from'] = step_samples[0]['timestep_from']
            step_stats['timestep_to'] = step_samples[0]['timestep_to']
            step_by_step_stats.append(step_stats)
        
        return step_by_step_stats
    
    def _aggregate_structure_one_sample(self, struct_records, ground_truth_network):
        """Aggregate structure metrics for one sample"""
        all_exist_pred, all_exist_true = [], []
        all_type_pred, all_type_true = [], []
        all_conn_probs = []
        
        for struct in struct_records:
            t = struct['timestep']
            inferred = struct['inferred_structure']
            probs = struct.get('structure_probabilities', {})
            
            for (i, j), pred_type in inferred.items():
                true_type = ground_truth_network.get_link_type(i, j, t)
                
                # Existence
                all_exist_pred.append(pred_type > 0)
                all_exist_true.append(true_type > 0)
                
                # Connection probability
                if (i, j) in probs:
                    p = probs[(i, j)]
                    if hasattr(p, '__getitem__') and len(p) >= 3:
                        conn_prob = float(p[1]) + float(p[2])
                        if hasattr(conn_prob, 'item'):
                            conn_prob = conn_prob.item()
                        all_conn_probs.append(conn_prob)
                    else:
                        all_conn_probs.append(float(pred_type > 0))
                else:
                    all_conn_probs.append(float(pred_type > 0))
                
                # Type (only for truly existing links)
                if true_type > 0:
                    all_type_pred.append(pred_type)
                    all_type_true.append(true_type)
        
        results = {}
        
        # Link existence
        if len(all_exist_pred) > 0:
            auc = safe_auc(all_exist_true, all_conn_probs) if len(all_conn_probs) == len(all_exist_true) else None
            ap = safe_ap(all_exist_true, all_conn_probs) if len(all_conn_probs) == len(all_exist_true) else None
            
            results['link_existence'] = {
                'auc': auc,
                'average_precision': ap
            }
        
        # Link type
        if len(all_type_pred) > 0:
            type_acc = accuracy_score(all_type_true, all_type_pred)
            results['link_type'] = {
                'accuracy': type_acc
            }
        
        return results
    
    # ========================================================================
    # Step-by-Step Statistics
    # ========================================================================

    def _aggregate_structure_one_sample_step(self, step_structures, n_samples, 
                                         ground_truth_network):
        """Aggregate structure metrics for one timestep across samples"""
        
        sample_metrics = []
        
        for struct in step_structures:
            t = struct['timestep']
            inferred = struct['inferred_structure']
            probs = struct.get('structure_probabilities', {})

            print(f"  inferred pairs: {len(inferred)}")

            
            exist_pred, exist_true, conn_probs = [], [], []
            type_pred, type_true = [], []
            bonding_pred, bonding_true = [], []
            bridging_pred, bridging_true = [], []
            
            for (i, j), pred_type in inferred.items():
                true_type = ground_truth_network.get_link_type(i, j, t)
                
                # Link existence
                exist_pred.append(pred_type > 0)
                exist_true.append(true_type > 0)
                
                # Connection probability
                if (i, j) in probs:
                    p = probs[(i, j)]
                    if hasattr(p, '__getitem__') and len(p) >= 3:
                        conn_prob = float(p[1]) + float(p[2])
                        if hasattr(conn_prob, 'item'):
                            conn_prob = conn_prob.item()
                        conn_probs.append(conn_prob)
                    else:
                        conn_probs.append(float(pred_type > 0))
                else:
                    conn_probs.append(float(pred_type > 0))

                
                # Link type (only for existing links)
                if true_type > 0:
                    type_pred.append(pred_type)
                    type_true.append(true_type)
                    
                    # Per-type metrics
                    if true_type == 1:  # Bonding
                        bonding_pred.append(pred_type)
                        bonding_true.append(1)
                    elif true_type == 2:  # Bridging
                        bridging_pred.append(pred_type)
                        bridging_true.append(2)

            
            # Compute metrics for this sample
            metrics = {}
            
            if len(exist_pred) > 0:
                auc = safe_auc(exist_true, conn_probs) if len(conn_probs) == len(exist_true) else None
                ap = safe_ap(exist_true, conn_probs) if len(conn_probs) == len(exist_true) else None
                metrics['link_existence'] = {'auc': auc, 'ap': ap}
            
            if len(type_pred) > 0:
                acc = accuracy_score(type_true, type_pred)
                metrics['link_type'] = {'accuracy': acc}
                
                # ✅ Per-type metrics
                if len(bonding_true) > 0:
                    bonding_correct = sum(p == 1 for p in bonding_pred)
                    metrics['bonding'] = {
                        'precision': bonding_correct / len([p for p in bonding_pred if p == 1]) if any(p == 1 for p in bonding_pred) else 0,
                        'recall': bonding_correct / len(bonding_true)
                    }
                
                if len(bridging_true) > 0:
                    bridging_correct = sum(p == 2 for p in bridging_pred)
                    metrics['bridging'] = {
                        'precision': bridging_correct / len([p for p in bridging_pred if p == 2]) if any(p == 2 for p in bridging_pred) else 0,
                        'recall': bridging_correct / len(bridging_true)
                    }
            
            # ✅ Confusion matrix (only for train with 1 sample)
            if n_samples == 1 and len(exist_pred) > 0:
                metrics['confusion_matrix'] = confusion_matrix(exist_true, exist_pred).tolist()
            
            sample_metrics.append(metrics)
        
        # Aggregate across samples
        aggregated = {}
        
        # Link existence
        if all('link_existence' in m for m in sample_metrics):
            auc_values = [m['link_existence']['auc'] for m in sample_metrics if m['link_existence']['auc'] is not None]
            ap_values = [m['link_existence']['ap'] for m in sample_metrics if m['link_existence']['ap'] is not None]
            
            aggregated['link_existence'] = {
                'auc_mean': np.mean(auc_values) if auc_values else None,
                'auc_std': np.std(auc_values) if len(auc_values) > 1 else None,
                'ap_mean': np.mean(ap_values) if ap_values else None,
                'ap_std': np.std(ap_values) if len(ap_values) > 1 else None
            }
        
        # Link type
        if all('link_type' in m for m in sample_metrics):
            acc_values = [m['link_type']['accuracy'] for m in sample_metrics]
            aggregated['link_type'] = {
                'accuracy_mean': np.mean(acc_values),
                'accuracy_std': np.std(acc_values) if len(acc_values) > 1 else None
            }
        
        # Per-type metrics
        for link_type in ['bonding', 'bridging']:
            if all(link_type in m for m in sample_metrics):
                prec_values = [m[link_type]['precision'] for m in sample_metrics]
                rec_values = [m[link_type]['recall'] for m in sample_metrics]
                aggregated[link_type] = {
                    'precision_mean': np.mean(prec_values),
                    'recall_mean': np.mean(rec_values),
                    'precision_std': np.std(prec_values) if len(prec_values) > 1 else None,
                    'recall_std': np.std(rec_values) if len(rec_values) > 1 else None
                }
        
        # Confusion matrix (only for single sample)
        if n_samples == 1 and 'confusion_matrix' in sample_metrics[0]:
            aggregated['confusion_matrix'] = sample_metrics[0]['confusion_matrix']
        
        return aggregated
    
    # ========================================================================
    # Final State & Timing Evaluation
    # ========================================================================
    
    def _evaluate_final_and_timing(self, target_households, model_states,
                                   ground_truth_states, train_end_time, test_end_time):
        """Evaluate final state and timing for one sample"""
        results = {
            'final_state_evaluation': {},
            'timing_evaluation': {}
        }
        
        for decision_k in range(3):
            decision_name = ['vacant', 'repair', 'sell'][decision_k]
            target = target_households[decision_name]
            
            if len(target) == 0:
                continue
            
            target_tensor = torch.tensor(target)
            
            # Final state
            our_final = model_states[target_tensor, test_end_time, decision_k]
            true_final = ground_truth_states[target_tensor, test_end_time, decision_k]
            
            accuracy = (our_final == true_final).float().mean().item()
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_final.cpu().numpy(), our_final.cpu().numpy(),
                average='binary', zero_division=0
            )
            
            results['final_state_evaluation'][decision_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'n_households': len(target)
            }
            
            # Timing
            timing_errors = []
            correct_timing = 0
            early_predictions = 0
            late_predictions = 0
            
            for hh in target:
                true_trans = ground_truth_states[hh, train_end_time+1:test_end_time+1, decision_k]
                true_first = torch.where(true_trans == 1)[0]
                true_time = true_first[0].item() + train_end_time + 1 if len(true_first) > 0 else None
                
                our_trans = model_states[hh, train_end_time+1:test_end_time+1, decision_k]
                our_first = torch.where(our_trans == 1)[0]
                our_time = our_first[0].item() + train_end_time + 1 if len(our_first) > 0 else None
                
                if true_time is not None and our_time is not None:
                    error = abs(true_time - our_time)
                    timing_errors.append(error)
                    
                    if error == 0:
                        correct_timing += 1
                    elif our_time < true_time:
                        early_predictions += 1
                    else:
                        late_predictions += 1
            
            n_timed = len(timing_errors)
            if n_timed > 0:
                results['timing_evaluation'][decision_name] = {
                    'correct_timing_rate': correct_timing / n_timed,
                    'average_timing_error': float(np.mean(timing_errors)),
                    'early_prediction_rate': early_predictions / n_timed,
                    'late_prediction_rate': late_predictions / n_timed,
                    'n_timed_predictions': n_timed
                }
            else:
                results['timing_evaluation'][decision_name] = {
                    'correct_timing_rate': np.nan,
                    'average_timing_error': np.nan,
                    'early_prediction_rate': np.nan,
                    'late_prediction_rate': np.nan,
                    'n_timed_predictions': 0
                }
        
        return results
    
    def _aggregate_final_evals(self, all_final_evals):
        """Aggregate final evaluations across samples"""
        n_samples = len(all_final_evals)
        
        aggregated = {
            'final_state_evaluation': {},
            'timing_evaluation': {},
            'n_samples': n_samples
        }
        
        for decision_name in ['vacant', 'repair', 'sell']:
            # Final state
            metrics = [e['final_state_evaluation'].get(decision_name) 
                      for e in all_final_evals if decision_name in e['final_state_evaluation']]
            
            if metrics:
                aggregated['final_state_evaluation'][decision_name] = {
                    'accuracy': np.mean([m['accuracy'] for m in metrics]),
                    'accuracy_std': np.std([m['accuracy'] for m in metrics]) if len(metrics) > 1 else None,
                    'precision': np.mean([m['precision'] for m in metrics]),
                    'recall': np.mean([m['recall'] for m in metrics]),
                    'f1': np.mean([m['f1'] for m in metrics]),
                    'n_households': metrics[0]['n_households']
                }
            
            # Timing
            timing = [e['timing_evaluation'].get(decision_name)
                     for e in all_final_evals if decision_name in e['timing_evaluation']]
            
            if timing:
                correct_rates = np.array([t.get('correct_timing_rate', np.nan) for t in timing], dtype=float)
                avg_errors = np.array([t.get('average_timing_error', np.nan) for t in timing], dtype=float)
                early_rates = np.array([t.get('early_prediction_rate', np.nan) for t in timing], dtype=float)
                late_rates = np.array([t.get('late_prediction_rate', np.nan) for t in timing], dtype=float)
                
                aggregated['timing_evaluation'][decision_name] = {
                    'correct_timing_rate': float(np.nanmean(correct_rates)) if not np.all(np.isnan(correct_rates)) else np.nan,
                    'average_timing_error': float(np.nanmean(avg_errors)) if not np.all(np.isnan(avg_errors)) else np.nan,
                    'early_prediction_rate': float(np.nanmean(early_rates)) if not np.all(np.isnan(early_rates)) else np.nan,
                    'late_prediction_rate': float(np.nanmean(late_rates)) if not np.all(np.isnan(late_rates)) else np.nan
                }
        
        return aggregated
    
    def _identify_target_households(self, states, train_end_time):
        """Identify target households (inactive at train_end_time)"""
        target = {}
        for decision_k in range(3):
            decision_name = ['vacant', 'repair', 'sell'][decision_k]
            inactive_mask = states[:, train_end_time, decision_k] == 0
            target[decision_name] = torch.where(inactive_mask)[0].tolist()
        return target
    
    def _create_summary(self, train_results, test_results):
        """Create summary comparing train and test"""
        train_overall = train_results['prediction_metrics'].get('overall', {})
        test_overall = test_results['prediction_metrics'].get('overall', {})
        train_struct = train_results.get('structure_metrics', {})
        test_struct = test_results.get('structure_metrics', {})
        
        return {
            'prediction_comparison': {
                'train_auc': train_overall.get('auc'),
                'test_auc': test_overall.get('auc')
            },
            'structure_comparison': {
                'train_link_auc': train_struct.get('link_existence', {}).get('auc'),
                'test_link_auc': test_struct.get('link_existence', {}).get('auc')
            }
        }


# ============================================================================
# Reporting Functions
# ============================================================================

def print_evaluation_results(results, ground_truth_network, trainer):
    """Print comprehensive evaluation results"""
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL EVALUATION RESULTS")
    print("="*80)
    
    # Train Period
    print("\n" + "="*80)
    print("📊 TRAIN PERIOD EVALUATION")
    print("="*80)
    
    train_eval = results['train_evaluation']
    _print_prediction_metrics("TRAIN", train_eval['prediction_metrics'], 
                             include_confusion=True)
    _print_step_by_step("TRAIN", train_eval['step_by_step_metrics'], 
                       train_eval['n_samples'])
    _print_structure_metrics("TRAIN", train_eval['structure_metrics'])
    
    # Test Period
    print("\n" + "="*80)
    print("📊 TEST PERIOD EVALUATION")
    print("="*80)
    
    test_eval = results['test_evaluation']
    _print_prediction_metrics("TEST", test_eval['prediction_metrics'], 
                             include_confusion=False)
    _print_step_by_step("TEST", test_eval['step_by_step_metrics'], 
                       test_eval['n_samples'])
    _print_structure_metrics("TEST", test_eval['structure_metrics'])
    
    if 'final_and_timing_evaluation' in test_eval:
        _print_final_and_timing(test_eval['final_and_timing_evaluation'])
    
    # Parameters
    print("\n" + "="*80)
    print("⚙️ ESTIMATED PARAMETERS")
    print("="*80)
    print(f"ρ₁ (bonding miss rate): {trainer.elbo_computer.rho_1.item():.4f}")
    print(f"ρ₂ (bridging miss rate): {trainer.elbo_computer.rho_2.item():.4f}")
    print(f"α₀ (initial bonding): {trainer.elbo_computer.network_evolution.alpha_0.item():.4f}")
    
    print("\n" + "="*80)


def _print_prediction_metrics(period, metrics, include_confusion=False):
    """Print prediction metrics"""
    print(f"\n📈 {period} - STATE PREDICTION METRICS:")
    print("-" * 80)
    
    overall = metrics['overall']
    n_samples = metrics.get('n_samples', 1)
    
    # Overall
    print(f"\nOverall (averaged across decisions):")
    if n_samples > 1:
        print(f"  (Aggregated across {n_samples} samples)")
    print(f"  ⚙️  Aggregation method: Temporal-weighted average")
    print(f"     (Each timestep computed separately, then weighted by n_switches)")
    
    auc = overall.get('auc')
    ap = overall.get('average_precision')
    print(f"  AUC: {auc:.4f}" if auc is not None else "  AUC: N/A")
    print(f"  Average Precision: {ap:.4f}" if ap is not None else "  Average Precision: N/A")
    
    # By decision
    print(f"\nBy Decision Type:")
    for decision_name in ['vacant', 'repair', 'sell']:
        if decision_name not in metrics['by_decision']:
            continue
        
        d = metrics['by_decision'][decision_name]
        print(f"\n  {decision_name.capitalize()}:")
        
        # AUC
        auc = d.get('auc')
        auc_std = d.get('auc_std')
        if auc is not None:
            if auc_std is not None and n_samples > 1:
                print(f"    AUC: {auc:.4f}±{auc_std:.4f}")
            else:
                print(f"    AUC: {auc:.4f}")
        else:
            print(f"    AUC: N/A")
        
        # AP
        ap = d.get('average_precision')
        ap_std = d.get('ap_std')
        if ap is not None:
            if ap_std is not None and n_samples > 1:
                print(f"    AP: {ap:.4f}±{ap_std:.4f}")
            else:
                print(f"    AP: {ap:.4f}")
        else:
            print(f"    AP: N/A")

        # AP Baseline
        ap_baseline = d.get('ap_baseline')
        ap_baseline_std = d.get('ap_baseline_std')
        if ap_baseline is not None:
            if ap_baseline_std is not None and n_samples > 1:
                print(f"    AP Baseline: {ap_baseline:.4f}±{ap_baseline_std:.4f}")
            else:
                print(f"    AP Baseline: {ap_baseline:.4f}")
        
        # AP/Baseline (Lift)
        ap_lift = d.get('ap_lift')
        ap_lift_std = d.get('ap_lift_std')
        if ap_lift is not None:
            if ap_lift_std is not None and n_samples > 1:
                print(f"    AP/Baseline: {ap_lift:.4f}±{ap_lift_std:.4f}")
            else:
                print(f"    AP/Baseline: {ap_lift:.4f}")
        
        # Confusion matrix (train only)
        if include_confusion and 'confusion_matrix' in d:
            conf = np.array(d['confusion_matrix'])
            if conf.shape == (2, 2):
                print(f"    Confusion: TN={conf[0,0]}, FP={conf[0,1]}, FN={conf[1,0]}, TP={conf[1,1]}")
    
    # Top-K table
    _print_top_k_table(period, metrics)


def _print_top_k_table(period, metrics):
    """Print Top-K metrics table"""
    print(f"\n{period} - TOP-K METRICS:")
    print("-" * 80)
    
    k_values = [5, 10, 20, 50]
    n_samples = metrics.get('n_samples', 1)
    
    # Header
    header = f"{'Decision':<12}{'Metric':<15}"
    for k in k_values:
        header += f"{'K='+str(k):<12}"
    print(header)
    print("-" * len(header))
    
    # Data
    for decision_name in ['vacant', 'repair', 'sell']:
        if decision_name not in metrics['by_decision']:
            continue
        
        d = metrics['by_decision'][decision_name]
        
        # Recall@K
        row = f"{decision_name.capitalize():<12}{'Recall@K':<15}"
        for k in k_values:
            val = d.get('recall_at_k', {}).get(k)
            std_val = d.get('recall_at_k_std', {}).get(k) if d.get('recall_at_k_std') else None
            
            if val is not None:
                if std_val is not None and n_samples > 1:
                    row += f"{val:.3f}±{std_val:.2f} "
                else:
                    row += f"{val:.4f}    "
            else:
                row += f"{'N/A':<12}"
        print(row)
        
        # Precision@K
        row = f"{'':<12}{'Precision@K':<15}"
        for k in k_values:
            val = d.get('precision_at_k', {}).get(k)
            std_val = d.get('precision_at_k_std', {}).get(k) if d.get('precision_at_k_std') else None
            
            if val is not None:
                if std_val is not None and n_samples > 1:
                    row += f"{val:.3f}±{std_val:.2f} "
                else:
                    row += f"{val:.4f}    "
            else:
                row += f"{'N/A':<12}"
        print(row)

                # 🆕 Precision@K
        row = f"{'':<12}{'Precision@K':<15}"
        for k in k_values:
            val = d.get('precision_at_k', {}).get(k)
            std_val = d.get('precision_at_k_std', {}).get(k) if d.get('precision_at_k_std') else None
            
            if val is not None:
                if std_val is not None and n_samples > 1:
                    row += f"{val:.3f}±{std_val:.2f} "
                else:
                    row += f"{val:.4f}    "
            else:
                row += f"{'N/A':<12}"
        print(row)
        
        # 🆕 Lift@K
        row = f"{'':<12}{'Lift@K':<15}"
        for k in k_values:
            val = d.get('lift_at_k', {}).get(k)
            std_val = d.get('lift_at_k_std', {}).get(k) if d.get('lift_at_k_std') else None
            
            if val is not None:
                if std_val is not None and n_samples > 1:
                    row += f"{val:.3f}±{std_val:.2f} "
                else:
                    row += f"{val:.4f}    "
            else:
                row += f"{'N/A':<12}"
        print(row)
        
        print()  
        
def _print_step_by_step(period, step_by_step_stats, n_samples):
    """Print step-by-step statistics"""
    print(f"\n📈 {period} STEP-BY-STEP STATISTICS:")
    print("-" * 80)
    if n_samples > 1:
        print(f"(Averaged across {n_samples} samples)\n")
    
    for step_stat in step_by_step_stats:
        t_from = step_stat['timestep_from']
        t_to = step_stat['timestep_to']

        has_data = any(
            step_stat['by_decision'].get(d, {}).get('auc_mean') is not None
            for d in ['vacant', 'repair', 'sell']
        )
        
        if not has_data:
            # Diagnose WHY there's no data
            # Check if it's because:
            # 1. No inactive households (all activated) OR
            # 2. No switches occurred (all negatives)
            
            total_inactive = sum(
                step_stat['by_decision'].get(d, {}).get('n_households', 0)
                for d in ['vacant', 'repair', 'sell']
            )
            total_switches = sum(
                step_stat['by_decision'].get(d, {}).get('n_switched', 0)
                for d in ['vacant', 'repair', 'sell']
            )
            
            print(f"\n⏱️  Step {t_from}→{t_to}:")
            
            if total_inactive == 0:
                # Case 1: All households already activated
                print(f"  (All target households activated - maintaining active states)")
            else:
                # Case 2: Have inactive households but no switches
                print(f"  (No state switches occurred: {int(total_inactive)} inactive households, 0 switches)")
                print(f"  ⚠️  All negative samples - unable to compute AUC/AP metrics")

            if 'structure_metrics' in step_stat:
                struct = step_stat['structure_metrics']
                
                print(f"\n  📊 Network Structure:")
                
                # Link existence
                if 'link_existence' in struct:
                    le = struct['link_existence']
                    auc = le.get('auc_mean')
                    ap = le.get('ap_mean')
                    
                    if n_samples == 1:
                        auc_str = f"AUC={auc:.3f}" if auc is not None else "AUC=N/A"
                        ap_str = f"AP={ap:.3f}" if ap is not None else "AP=N/A"
                    else:
                        auc_std = le.get('auc_std')
                        ap_std = le.get('ap_std')
                        
                        if auc is not None and auc_std is not None:
                            auc_str = f"AUC={auc:.3f}±{auc_std:.3f}"
                        else:
                            auc_str = "AUC=N/A"
                        
                        if ap is not None and ap_std is not None:
                            ap_str = f"AP={ap:.3f}±{ap_std:.3f}"
                        else:
                            ap_str = "AP=N/A"
                    
                    print(f"    Link Existence: {auc_str}, {ap_str}")
                
                # Link type
                if 'link_type' in struct:
                    lt = struct['link_type']
                    acc = lt.get('accuracy_mean')
                    
                    if n_samples == 1:
                        print(f"    Link Type Accuracy: {acc:.3f}" if acc is not None else "    Link Type Accuracy: N/A")
                    else:
                        acc_std = lt.get('accuracy_std')
                        if acc is not None and acc_std is not None:
                            print(f"    Link Type Accuracy: {acc:.3f}±{acc_std:.3f}")
                        else:
                            print(f"    Link Type Accuracy: N/A")
                
                # Per-type metrics
                for link_type in ['bonding', 'bridging']:
                    if link_type in struct:
                        lt = struct[link_type]
                        prec = lt.get('precision_mean')
                        rec = lt.get('recall_mean')
                        
                        if n_samples == 1:
                            if prec is not None and rec is not None:
                                print(f"    {link_type.capitalize()}: Prec={prec:.3f}, Rec={rec:.3f}")
                        else:
                            prec_std = lt.get('precision_std')
                            rec_std = lt.get('recall_std')
                            if prec is not None and rec is not None and prec_std is not None and rec_std is not None:
                                print(f"    {link_type.capitalize()}: Prec={prec:.3f}±{prec_std:.3f}, Rec={rec:.3f}±{rec_std:.3f}")
                
                # Confusion matrix
                if 'confusion_matrix' in struct and n_samples == 1:
                    conf = np.array(struct['confusion_matrix'])
                    if conf.shape == (2, 2):
                        print(f"    Confusion: TN={conf[0,0]}, FP={conf[0,1]}, FN={conf[1,0]}, TP={conf[1,1]}")


            continue

        print(f"\n⏱️  Step {t_from}→{t_to}:")
        
        # ============================================
        # Prediction Metrics (by decision)
        # ============================================
        for decision_name in ['vacant', 'repair', 'sell']:
            if decision_name not in step_stat['by_decision']:
                continue
            
            d = step_stat['by_decision'][decision_name]
            
            auc_mean = d.get('auc_mean')
            ap_mean = d.get('ap_mean')
            recall_10 = d.get('recall_at_k_mean', {}).get(10)
            precision_10 = d.get('precision_at_k_mean', {}).get(10)
            
            # Check if this is empty data (no inactive households)
            is_empty = (auc_mean is None and ap_mean is None and 
                       recall_10 is None and precision_10 is None)
            
            if is_empty:
                # Friendly explanation for N/A values
                print(f"  {decision_name.capitalize()}: N/A (no inactive households to predict)")
                continue
            
            if n_samples == 1:
                auc_str = f"AUC={auc_mean:.3f}" if auc_mean is not None else "AUC=N/A"
                ap_str = f"AP={ap_mean:.3f}" if ap_mean is not None else "AP=N/A"
                recall_str = f"R@10={recall_10:.3f}" if recall_10 is not None else "R@10=N/A"
                prec_str = f"P@10={precision_10:.3f}" if precision_10 is not None else "P@10=N/A"
            else:
                auc_std = d.get('auc_std')
                ap_std = d.get('ap_std')
                recall_std = d.get('recall_at_k_std', {}).get(10)
                prec_std = d.get('precision_at_k_std', {}).get(10)
                
                auc_str = f"AUC={auc_mean:.3f}±{auc_std:.3f}" if auc_mean is not None and auc_std is not None else "AUC=N/A"
                ap_str = f"AP={ap_mean:.3f}±{ap_std:.3f}" if ap_mean is not None and ap_std is not None else "AP=N/A"
                recall_str = f"R@10={recall_10:.3f}±{recall_std:.3f}" if recall_10 is not None and recall_std is not None else "R@10=N/A"
                prec_str = f"P@10={precision_10:.3f}±{prec_std:.3f}" if precision_10 is not None and prec_std is not None else "P@10=N/A"
            print(f"  {decision_name.capitalize()}: {auc_str}, {ap_str}, {recall_str}, {prec_str}")
        
        # ============================================
        # ✅ Structure Metrics 
        # ============================================
        if 'structure_metrics' in step_stat:
            struct = step_stat['structure_metrics']
            
            print(f"\n  📊 Network Structure:")
            
            # Link existence
            if 'link_existence' in struct:
                le = struct['link_existence']
                auc = le.get('auc_mean')
                ap = le.get('ap_mean')
                
                if n_samples == 1:
                    auc_str = f"AUC={auc:.3f}" if auc is not None else "AUC=N/A"
                    ap_str = f"AP={ap:.3f}" if ap is not None else "AP=N/A"
                else:
                    auc_std = le.get('auc_std')
                    ap_std = le.get('ap_std')
                    
                    if auc is not None and auc_std is not None:
                        auc_str = f"AUC={auc:.3f}±{auc_std:.3f}"
                    else:
                        auc_str = "AUC=N/A"
                    
                    if ap is not None and ap_std is not None:
                        ap_str = f"AP={ap:.3f}±{ap_std:.3f}"
                    else:
                        ap_str = "AP=N/A"
                
                print(f"    Link Existence: {auc_str}, {ap_str}")
            
            # Link type
            if 'link_type' in struct:
                lt = struct['link_type']
                acc = lt.get('accuracy_mean')
                
                if n_samples == 1:
                    print(f"    Link Type Accuracy: {acc:.3f}" if acc is not None else "    Link Type Accuracy: N/A")
                else:
                    acc_std = lt.get('accuracy_std')
                    if acc is not None and acc_std is not None:
                        print(f"    Link Type Accuracy: {acc:.3f}±{acc_std:.3f}")
                    else:
                        print(f"    Link Type Accuracy: N/A")
            
            # Per-type metrics
            for link_type in ['bonding', 'bridging']:
                if link_type in struct:
                    lt = struct[link_type]
                    prec = lt.get('precision_mean')
                    rec = lt.get('recall_mean')
                    
                    if n_samples == 1:
                        if prec is not None and rec is not None:
                            print(f"    {link_type.capitalize()}: Prec={prec:.3f}, Rec={rec:.3f}")
                    else:
                        prec_std = lt.get('precision_std')
                        rec_std = lt.get('recall_std')
                        if prec is not None and rec is not None and prec_std is not None and rec_std is not None:
                            print(f"    {link_type.capitalize()}: Prec={prec:.3f}±{prec_std:.3f}, Rec={rec:.3f}±{rec_std:.3f}")
            
            # Confusion matrix
            if 'confusion_matrix' in struct and n_samples == 1:
                conf = np.array(struct['confusion_matrix'])
                if conf.shape == (2, 2):
                    print(f"    Confusion: TN={conf[0,0]}, FP={conf[0,1]}, FN={conf[1,0]}, TP={conf[1,1]}")


def _print_structure_metrics(period, metrics):
    """Print structure metrics"""
    print(f"\n🔗 {period} - STRUCTURE INFERENCE METRICS:")
    print("-" * 80)
    
    if 'link_existence' in metrics:
        le = metrics['link_existence']
        n_samples = le.get('n_samples', 1)
        
        print(f"\nLink Existence:")
        if n_samples > 1:
            print(f"  (Aggregated across {n_samples} samples)")
        
        auc = le.get('auc')
        auc_std = le.get('auc_std')
        ap = le.get('average_precision')
        ap_std = le.get('ap_std')
        
        if auc is not None:
            if auc_std is not None and n_samples > 1:
                print(f"  AUC: {auc:.4f}±{auc_std:.4f}")
            else:
                print(f"  AUC: {auc:.4f}")
        else:
            print(f"  AUC: N/A")
        
        if ap is not None:
            if ap_std is not None and n_samples > 1:
                print(f"  Average Precision: {ap:.4f}±{ap_std:.4f}")
            else:
                print(f"  Average Precision: {ap:.4f}")
        else:
            print(f"  Average Precision: N/A")
    
    if 'link_type' in metrics:
        lt = metrics['link_type']
        acc = lt.get('accuracy')
        acc_std = lt.get('accuracy_std')
        n_samples = metrics.get('link_existence', {}).get('n_samples', 1)
        
        print(f"\nLink Type (among existing links):")
        if acc is not None:
            if acc_std is not None and n_samples > 1:
                print(f"  Accuracy: {acc:.4f}±{acc_std:.4f}")
            else:
                print(f"  Accuracy: {acc:.4f}")


def _print_final_and_timing(final_eval):
    """Print final state and timing evaluation"""
    print("\n" + "="*80)
    print("🎯 FINAL STATE & TIMING EVALUATION")
    print("="*80)
    
    n_samples = final_eval.get('n_samples', 1)
    
    for decision_name in ['vacant', 'repair', 'sell']:
        if decision_name not in final_eval['final_state_evaluation']:
            continue
        
        fs = final_eval['final_state_evaluation'][decision_name]
        timing = final_eval['timing_evaluation'][decision_name]
        
        print(f"\n{decision_name.capitalize()}:")
        print(f"  Target Households: {fs['n_households']}")
        
        if n_samples > 1:
            acc_std = fs.get('accuracy_std', 0)
            print(f"  Final State Accuracy: {fs['accuracy']:.4f}±{acc_std:.4f}")
        else:
            print(f"  Final State Accuracy: {fs['accuracy']:.4f}")
        
        # Timing
        def _fmt(val):
            return "N/A" if val is None or (isinstance(val, float) and np.isnan(val)) else f"{val:.2%}"
        def _fmt_num(val):
            return "N/A" if val is None or (isinstance(val, float) and np.isnan(val)) else f"{val:.2f}"
        
        print(f"  Timing Metrics:")
        print(f"    Correct Timing Rate: {_fmt(timing.get('correct_timing_rate'))}")
        print(f"    Average Timing Error: {_fmt_num(timing.get('average_timing_error'))} steps")


# ============================================================================
# Public Interface
# ============================================================================

def evaluate_model_corrected(trainer, test_data, train_end_time=15, test_end_time=24,
                             n_train_samples=1, n_test_samples=10,
                             custom_train_thresholds=None, custom_test_thresholds=None,
                             enable_diagnostic=True):
    """
    Main evaluation function.
    
    Args:
        n_train_samples: Number of samples for train period (default=1)
        n_test_samples: Number of samples for test period (default=10)
        enable_diagnostic: Enable probability diagnostic (default=True)
    """
    # Clear tracking
    trainer.elbo_computer.state_transition.clear_influence_tracking()
    
    # Create config
    config = EvalConfig()
    if custom_train_thresholds:
        config.train_thresholds.update(custom_train_thresholds)
    if custom_test_thresholds:
        config.test_thresholds.update(custom_test_thresholds)
    
    # Create evaluator with diagnostic enabled
    evaluator = ModelEvaluator(
        mean_field_posterior=trainer.mean_field_posterior,
        state_transition=trainer.elbo_computer.state_transition,
        config=config,
        enable_diagnostic=enable_diagnostic
    )
    
    # Run evaluation
    results = evaluator.evaluate(
        features=test_data['features'],
        ground_truth_states=test_data['states'],
        distances=test_data['distances'],
        observed_network=test_data['observed_network'],
        ground_truth_network=test_data['ground_truth_network'],
        train_end_time=train_end_time,
        test_end_time=test_end_time,
        n_train_samples=n_train_samples,
        n_test_samples=n_test_samples
    )
    
    # Store evaluator for later access
    trainer.evaluator = evaluator
    
    return results


def evaluate_and_log(trainer, test_data, model_name=None, log_file=None, **eval_kwargs):
    """
    Evaluate model and log results to file.
    
    Args:
        trainer: Trained NetworkStateTrainer
        test_data: Test data dictionary
        model_name: Model name for auto-generated log file
        log_file: Explicit log file path (overrides model_name)
        **eval_kwargs: Additional arguments for evaluate_model_corrected
    
    Returns:
        results: Evaluation results dictionary
    
    Example:
        results = evaluate_and_log(
            trainer, test_data, 
            model_name='ABM_Dense_HighSeed_A',
            train_end_time=10,
            test_end_time=23,
            n_train_samples=1,
            n_test_samples=5
        )
    """
    with EvaluationLogger(log_file=log_file, model_name=model_name):
        results = evaluate_model_corrected(trainer, test_data, **eval_kwargs)
        print_evaluation_results(results, test_data['ground_truth_network'], trainer)
    
    return results