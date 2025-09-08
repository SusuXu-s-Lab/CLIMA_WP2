"""
evaluation_corrected.py - Final corrected evaluation following exact requirements

Key Logic:
1. Train period: Step-by-step prediction evaluation + structure inference evaluation
2. Test period: Only evaluate households inactive at t=15, with final state + timing evaluation
3. Proper inference history tracking for temporal dependencies
4. FIXED: Use structure at time t to predict states t -> t+1
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
from data.data_loader import NetworkData
from evaluation.probability_comparator import ProbabilityComparator
from evaluation.detailed_logging import FRSICDetailedLogger


class CorrectedModelEvaluator:
    """
    Corrected evaluation with proper train/test logic.
    """
    
    def __init__(self, mean_field_posterior, state_transition):
        self.mean_field_posterior = mean_field_posterior
        self.state_transition = state_transition
        self.inference_history = {}  # {t: {(i,j): inferred_type}}

        self.train_thresholds = {'vacant': 0.5, 'repair': 0.85, 'sell': 0.85}
        self.test_thresholds = {'vacant': 0.7, 'repair': 0.75, 'sell': 0.75}

        self.probability_comparator = ProbabilityComparator('/Users/susangao/Desktop/CLIMA/CODE 4.8- based on 4.6, adopt sparse strategy/data/syn_data6_200_node/detailed_generator_probabilities_200node.pkl')
        self.all_detailed_logs = []  # Store all detailed probability logs

        self.detailed_logger = None

    def _identify_global_bonding_pairs(self, observed_network, train_end_time):
        """
        Identify all pairs that have bonding links (type 1) observed anywhere in training period.
        
        Args:
            observed_network: The observed network data
            train_end_time: End of training period (e.g., 15)
        
        Returns:
            set: Set of (i,j) tuples that should be enforced as bonding throughout
        """
        global_bonding_pairs = set()
        
        for t in range(train_end_time + 1):  # 0 to train_end_time inclusive
            # Get all observed pairs at time t
            observed_pairs = []
            n_households = observed_network.n_households
            for i in range(n_households):
                for j in range(i + 1, n_households):
                    if observed_network.is_observed(i, j, t):
                        observed_pairs.append((i, j))
            
            # Check each observed pair for bonding links
            for (i, j) in observed_pairs:
                if observed_network.get_link_type(i, j, t) == 1:  # bonding observed
                    global_bonding_pairs.add((i, j))
        
        return global_bonding_pairs

    def _compute_auc_safe(self, y_true, y_score):
        """
        Safely compute AUC, handling edge cases where all labels are the same class.
        
        Args:
            y_true: True binary labels
            y_score: Predicted probabilities or scores
            
        Returns:
            AUC score or None if cannot be computed
        """
        try:
            # Check if we have both classes
            unique_labels = np.unique(y_true)
            if len(unique_labels) < 2:
                return None  # Cannot compute AUC with only one class
            
            # Compute AUC
            auc = roc_auc_score(y_true, y_score)
            return auc
        except Exception as e:
            return None

# In the evaluate_model method, add initialization at the beginning and cleanup at the end:

    def evaluate_model(self,
                    features: torch.Tensor,
                    ground_truth_states: torch.Tensor,
                    distances: torch.Tensor,
                    observed_network: NetworkData,
                    ground_truth_network: NetworkData,
                    train_end_time: int = 15,
                    test_end_time: int = 24) -> Dict:
        """
        Complete corrected evaluation.
        """
        
        print("=== Corrected Model Evaluation ===")
        print(f"Train period: 0 to {train_end_time}")
        print(f"Test period: {train_end_time + 1} to {test_end_time}")
        
        # ADD THIS: Initialize detailed logger
        self.detailed_logger = FRSICDetailedLogger("evaluation_logs")
        
        n_households = features.shape[0]
        all_nodes = torch.arange(n_households, dtype=torch.long)
        
        # Initialize
        self.inference_history = {}

        self.global_bonding_pairs = self._identify_global_bonding_pairs(observed_network, train_end_time)
        print(f"Found {len(self.global_bonding_pairs)} pairs with observed bonding links during training")
        if len(self.global_bonding_pairs) > 0:
            print(f"Global bonding pairs: {list(self.global_bonding_pairs)[:10]}...")  
        
        # Results storage
        results = {
            'train_evaluation': {},
            'test_evaluation': {},
            'summary': {}
        }
        
        try:
            # === TRAIN PERIOD EVALUATION ===
            print("\n=== TRAIN PERIOD EVALUATION ===")
            train_results = self._evaluate_train_period(
                features, ground_truth_states, distances, all_nodes,
                observed_network, ground_truth_network, train_end_time
            )
            results['train_evaluation'] = train_results
            
            # === TEST PERIOD EVALUATION ===
            print("\n=== TEST PERIOD EVALUATION ===")
            test_results = self._evaluate_test_period(
                features, ground_truth_states, distances, all_nodes,
                observed_network, ground_truth_network, train_end_time, test_end_time
            )
            results['test_evaluation'] = test_results
            
            # === SUMMARY ===
            results['summary'] = self._create_summary(train_results, test_results)
            
        finally:
            # ADD THIS: Ensure logger is closed even if an error occurs
            if self.detailed_logger:
                self.detailed_logger.close()
        
        return results
    
    def _evaluate_train_period(self, features, ground_truth_states, distances, all_nodes,
                              observed_network, ground_truth_network, train_end_time):
        """
        Train period evaluation:
        1. Step-by-step prediction evaluation (use truth at t to predict t+1)
        2. Structure inference evaluation at each step
        """
        
        prediction_step_results = []
        structure_step_results = []
        
        # First, infer structure at t=0
        print(f"  Initial structure inference at t=0")
        structure_t0 = self._infer_structure_step(
            features, ground_truth_states, distances, all_nodes,
            observed_network, 0, is_train=True
        )
        structure_step_results.append(structure_t0)
        self.inference_history[0] = structure_t0['inferred_structure']
        
        # Then do the temporal steps
        for t in range(train_end_time):  # 0 to 14 (predict t+1)
            print(f"  Train step {t} -> {t+1}")
            
            # 1. State Prediction t -> t+1 (using structure at t and ground truth states at t)
            current_structure = self._get_structure_at_time(observed_network, t)
            prediction_result = self._predict_states_train_step(
                features, ground_truth_states, distances, all_nodes,
                current_structure, t, is_train_period=True
            )
            prediction_step_results.append(prediction_result)
            
            # 2. Structure Inference at t+1 (after we've made the prediction)
            structure_result = self._infer_structure_step(
                features, ground_truth_states, distances, all_nodes,
                observed_network, t+1, is_train=True
            )
            subset_pairs_train = observed_network.get_hidden_pairs(t + 1)
            prev_struct = self.inference_history.get(t, {})

            corrected_struct = self._apply_bonding_persistence_on_subset(
                structure_result['inferred_structure'],
                prev_struct,
                observed_network,
                t + 1,
                subset_pairs_train
            )

            if corrected_struct != structure_result['inferred_structure']:
                if 'complete_network' in structure_result:
                    corrected_network = self._create_complete_network(observed_network, corrected_struct, t + 1)
                    structure_result['complete_network'] = corrected_network
                structure_result['inferred_structure'] = corrected_struct
                scope = structure_result.get('evaluation_scope', 'train')
                structure_result['evaluation_scope'] = scope + '+bond_persist'

            self.inference_history[t + 1] = structure_result['inferred_structure']
            structure_step_results.append(structure_result)
        
        # Aggregate train results
        train_prediction_metrics = self._aggregate_train_prediction_results(prediction_step_results)
        train_structure_metrics = self._aggregate_train_structure_results(structure_step_results, ground_truth_network,observed_network)
        
        return {
            'prediction_metrics': train_prediction_metrics,
            'structure_metrics': train_structure_metrics,
            'step_by_step_predictions': prediction_step_results,
            'step_by_step_structures': structure_step_results
        }
    
    def _get_structure_at_time(self, observed_network, t):
        """Get the complete network structure at time t."""
        
        if t == 0:
            # Use the initial structure we inferred
            inferred_at_t = self.inference_history.get(0, {})
        else:
            # Use the structure we inferred at time t
            inferred_at_t = self.inference_history.get(t, {})
        
        return self._create_complete_network(observed_network, inferred_at_t, t)
    
    def _evaluate_test_period(self, features, ground_truth_states, distances, all_nodes,
                                observed_network, ground_truth_network, train_end_time, test_end_time):
            """
            Keep training observations fixed, only evolve unobserved pairs.
            Now includes step-by-step prediction metrics like train period.
            """
            
            # Identify target households (inactive at train_end_time)
            target_households = {}
            for decision_k in range(3):
                decision_name = ['vacant', 'repair', 'sell'][decision_k]
                inactive_mask = ground_truth_states[:, train_end_time, decision_k] == 0
                target_households[decision_name] = torch.where(inactive_mask)[0].tolist()
                print(f"  Target {decision_name} households: {len(target_households[decision_name])}")

            if not hasattr(self, "eval_target_pairs") or self.eval_target_pairs is None:
                n_households = features.shape[0]
                pairs = []
                for i in range(n_households):
                    for j in range(i + 1, n_households):
                        if not observed_network.is_observed(i, j, train_end_time):
                            pairs.append((i, j))
                self.eval_target_pairs = pairs
                self.eval_target_pairs_set = set(pairs)
                    
            # Initialize model states with ground truth up to train_end_time
            model_states = ground_truth_states[:, :train_end_time+1, :].clone()
            
            # Forward simulation records
            test_prediction_records = []
            test_structure_records = []
            
            for t in range(train_end_time, test_end_time):  # 15 to 23 (predict t+1)
                print(f"  Test step {t} -> {t+1}")
                    
                # 1. State Prediction t -> t+1 (using structure at time t)
                current_structure = self._get_structure_at_time(observed_network, t)
                prediction_result = self._predict_states_test_step(
                    features, model_states, distances, all_nodes,
                    current_structure, t, target_households, ground_truth_states  # Add ground_truth_states
                )
                test_prediction_records.append(prediction_result)
                
                # 2. Update model states with our predictions
                model_states = torch.cat([model_states, prediction_result['next_states'].unsqueeze(1)], dim=1)

                # 3.Keep training observations, infer only unobserved pairs
                structure_result = self._infer_unobserved_pairs_test(
                    features, model_states, distances, all_nodes, 
                    observed_network, t+1, train_end_time
                )
                test_structure_records.append(structure_result)

                subset_pairs_test = self.eval_target_pairs
                prev_struct = self.inference_history.get(t, {})

                corrected_struct = self._apply_bonding_persistence_on_subset(
                    structure_result['inferred_structure'],
                    prev_struct,
                    observed_network,   
                    t + 1,
                    subset_pairs_test
                )

                if corrected_struct != structure_result['inferred_structure']:
                    if 'complete_network' in structure_result:
                        corrected_network = self._create_complete_network(observed_network, corrected_struct, t + 1)
                        structure_result['complete_network'] = corrected_network
                    structure_result['inferred_structure'] = corrected_struct
                    scope = structure_result.get('evaluation_scope', 'test')
                    structure_result['evaluation_scope'] = scope + '+bond_persist'

                self.inference_history[t + 1] = structure_result['inferred_structure']
                
                # Store in history for next steps
                self.inference_history[t+1] = structure_result['inferred_structure']
            
            # Evaluate test results
            test_final_evaluation = self._evaluate_test_final_and_timing(
                target_households, model_states, ground_truth_states, train_end_time, test_end_time
            )
            
            test_structure_evaluation = self._aggregate_test_structure_results(
                test_structure_records, ground_truth_network, train_end_time
            )
            
            # NEW: Aggregate test prediction results using the same function as train period
            test_prediction_metrics = self._aggregate_train_prediction_results(test_prediction_records)
            
            return {
                'target_households': target_households,
                'final_and_timing_evaluation': test_final_evaluation,
                'structure_evaluation': test_structure_evaluation,
                'prediction_metrics': test_prediction_metrics,  # Add this line
                'forward_simulation_records': {
                    'predictions': test_prediction_records,
                    'structures': test_structure_records
                }
            }
    
    def _infer_structure_step(self, features, states, distances, all_nodes,
                            observed_network, t, is_train=True):
        """
        修改后的结构推断：Train和Test都只评估未观察到的pairs
        - Train期间：评估每个时间点t的hidden pairs
        - Test期间：评估train_end_time时未观察到的pairs
        """
        
        # Get conditional and marginal probabilities
        conditional_probs, marginal_probs = self.mean_field_posterior.compute_probabilities_batch(
            features, states, distances, all_nodes, observed_network, t
        )
        
        # Infer structure
        structure_probabilities = {}
        inferred_structure = {}
        n_households = features.shape[0]
        
        # 统一逻辑：都只评估未观察到的pairs
        if is_train:
            # Train: 评估当前时间点t的hidden pairs
            pairs_to_evaluate = observed_network.get_hidden_pairs(t)
            evaluation_scope = f'hidden_pairs_at_t{t}'
        else:
            # Test: 评估train_end_time时未观察到的pairs  
            # 这个逻辑应该在调用时处理，这里保持一致
            pairs_to_evaluate = observed_network.get_hidden_pairs(t)
            evaluation_scope = f'hidden_pairs_at_t{t}'
        
        # 首先添加所有observed pairs到inferred_structure（但不评估它们）
        for i in range(n_households):
            for j in range(i + 1, n_households):
                if observed_network.is_observed(i, j, t):
                    inferred_structure[(i, j)] = observed_network.get_link_type(i, j, t)
        
        # 只对未观察到的pairs进行推断和评估
        for i, j in pairs_to_evaluate:
            pair_key = f"{i}_{j}_{t}"
            
            if pair_key in conditional_probs:
                # Infer using conditional probabilities
                conditional_matrix = conditional_probs[pair_key]  # [3, 3]
                
                if t == 0:
                    # Initial: use marginal
                    inferred_type = torch.argmax(marginal_probs[pair_key]).item()
                else:
                    # Temporal: use conditional based on previous state
                    inferred_type = torch.argmax(marginal_probs[pair_key]).item()
                
                inferred_structure[(i, j)] = inferred_type

                if pair_key in marginal_probs:
                    structure_probabilities[(i, j)] = marginal_probs[pair_key]
            else:
                # Fallback
                if t > 0:
                    prev_type = self._get_previous_state(i, j, t-1, observed_network)
                    inferred_structure[(i, j)] = prev_type
                else:
                    inferred_structure[(i, j)] = 0
        
        # Create complete network
        inferred_structure = self._enforce_global_bonding_persistence(inferred_structure, t)
        complete_network = self._create_complete_network(observed_network, inferred_structure, t)
        
        return {
            'timestep': t,
            'inferred_structure': inferred_structure,
            'structure_probabilities': structure_probabilities,
            'complete_network': complete_network,
            'evaluation_scope': evaluation_scope,
            'pairs_evaluated': pairs_to_evaluate  # 记录评估了哪些pairs
        }
    
    def _infer_unobserved_pairs_test(self, features, model_states, distances, all_nodes, 
                                observed_network, t, train_end_time):
        """
        Infer ONLY pairs that were unobserved at t=15, keeping t=15 observations fixed.
        This is Approach 2: Keep observations from t=15 fixed, only evolve unobserved pairs from t=15.
        """
        
        n_households = features.shape[0]
        
        # Create a network that preserves observations from t=15 only
        class T15ObservationsNetwork:
            def __init__(self, original_network, train_end_time):
                self.original_network = original_network
                self.train_end_time = train_end_time  # This should be 15
                self.n_households = original_network.n_households
                self.n_timesteps = original_network.n_timesteps
                self.all_pairs = original_network.all_pairs
            
            def is_observed(self, i, j, query_t):
                # Only links observed at exactly t=15 are considered "observed"
                return self.original_network.is_observed(i, j, self.train_end_time)
            
            def get_link_type(self, i, j, query_t):
                # Use observed value from t=15
                if self.original_network.is_observed(i, j, self.train_end_time):
                    return self.original_network.get_link_type(i, j, self.train_end_time)
                return 0
            
            def get_hidden_pairs(self, query_t):
                # Only pairs that were unobserved at t=15 are "hidden"
                hidden_pairs = []
                for i, j in self.all_pairs:
                    if not self.original_network.is_observed(i, j, self.train_end_time):
                        hidden_pairs.append((i, j))
                return hidden_pairs
            
            def get_observed_edges_at_time(self, query_t):
                # Return all edges that were observed at t=15
                return self.original_network.get_observed_edges_at_time(self.train_end_time)
        
        t15_network = T15ObservationsNetwork(observed_network, train_end_time)
        
        # Get conditional and marginal probabilities for ONLY pairs unobserved at t=15
        conditional_probs, marginal_probs = self.mean_field_posterior.compute_probabilities_batch(
            features, model_states, distances, all_nodes, t15_network, t
        )
        
        # Start with structure: keep all t=15 observations fixed
        inferred_structure = {}
        
        # Keep all t=15 observations fixed
        for i in range(n_households):
            for j in range(i + 1, n_households):
                if observed_network.is_observed(i, j, train_end_time):
                    # Keep the observed link from t=15
                    inferred_structure[(i, j)] = observed_network.get_link_type(i, j, train_end_time)
        
        # Infer ONLY the pairs that were unobserved at t=15
        unobserved_at_t15 = t15_network.get_hidden_pairs(t)
        
        for i, j in unobserved_at_t15:
            pair_key = f"{i}_{j}_{t}"
            
            if pair_key in conditional_probs:
                # Infer using conditional probabilities
                conditional_matrix = conditional_probs[pair_key]  # [3, 3]
                
                # Get previous state from our inference history
                prev_type = self._get_previous_state_from_history(i, j, t-1)
                conditional_given_prev = conditional_matrix[prev_type, :]  # [3]
                inferred_type = torch.argmax(conditional_given_prev).item()
                
                inferred_structure[(i, j)] = inferred_type
            else:
                # Fallback to previous state
                prev_type = self._get_previous_state_from_history(i, j, t-1)
                inferred_structure[(i, j)] = prev_type
        inferred_structure = self._enforce_global_bonding_persistence(inferred_structure, t)
        
        return {
            'timestep': t,
            'inferred_structure': inferred_structure,
            'evaluation_scope': 'unobserved_at_t15_only'
        }
        
    
    def _get_previous_state(self, i, j, t_prev, observed_network):
        """Get previous state of link (i,j) at time t_prev."""
        i, j = min(i, j), max(i, j)
        
        if observed_network.is_observed(i, j, t_prev):
            return observed_network.get_link_type(i, j, t_prev)
        elif t_prev in self.inference_history and (i, j) in self.inference_history[t_prev]:
            return self.inference_history[t_prev][(i, j)]
        else:
            return 0  # Default
        
    def _get_previous_state_from_history(self, i, j, t_prev):
        """Get previous state from inference history only."""
        i, j = min(i, j), max(i, j)
        
        if t_prev in self.inference_history and (i, j) in self.inference_history[t_prev]:
            return self.inference_history[t_prev][(i, j)]
        else:
            return 0  # Default
    
    def _predict_states_train_step(self, features, ground_truth_states, distances, all_nodes,
                                  complete_network, t, is_train_period=True):
        """
        Predict states for train period: use ground truth at t to predict t+1.
        Only predict for households inactive at t.
        """
        
        predictions_by_decision = {}
        
        for decision_k in range(3):
            decision_name = ['vacant', 'repair', 'sell'][decision_k]
            
            # Find households inactive at time t
            inactive_mask = ground_truth_states[:, t, decision_k] == 0
            inactive_households = torch.where(inactive_mask)[0]
            
            if len(inactive_households) == 0:
                predictions_by_decision[decision_name] = {
                    'inactive_households': [],
                    'predictions': [],
                    'probabilities': [],
                    'ground_truth': [],
                    'n_households': 0,
                    'n_switched': 0
                }
                continue
            
            # Make predictions
            prediction_result = self._compute_state_predictions(
                        inactive_households, decision_k, features, ground_truth_states,
                        distances, complete_network, t
                    )
            
            predictions = prediction_result['predictions']
            probabilities = prediction_result['probabilities']
            
            # Get ground truth outcomes at t+1
            ground_truth = ground_truth_states[inactive_households, t+1, decision_k]
            n_switched = torch.sum(ground_truth).item()
            
            predictions_by_decision[decision_name] = {
                'inactive_households': inactive_households.tolist(),
                'predictions': predictions.cpu().numpy(),
                'probabilities': probabilities.cpu().numpy(),
                'ground_truth': ground_truth.cpu().numpy(),
                'n_households': len(inactive_households),
                'n_switched': n_switched
            }
        
        return {
            'timestep_from': t,
            'timestep_to': t+1,
            'predictions_by_decision': predictions_by_decision
        }
    
    def _predict_states_test_step(self, features, model_states, distances, all_nodes,
                                    complete_network, t, target_households, ground_truth_states):
            """
            Predict states for test period: use model states at t to predict t+1.
            Only predict for target households that are still inactive in our model.
            Now includes ground truth comparison for metrics calculation.
            """
            
            current_states = model_states[:, -1, :].clone()  # Current model state
            next_states = current_states.clone()
            
            predictions_by_decision = {}
            
            for decision_k in range(3):
                decision_name = ['vacant', 'repair', 'sell'][decision_k]
                target_hh = target_households[decision_name]
                
                # Find target households that are still inactive in our model
                still_inactive = []
                for hh in target_hh:
                    if current_states[hh, decision_k] == 0:
                        still_inactive.append(hh)
                print(f"  ###### Predicting {len(still_inactive)} households for decision '{decision_name}' at t={t}")
                
                if len(still_inactive) == 0:
                    predictions_by_decision[decision_name] = {
                        'target_households': target_hh,
                        'still_inactive': [],
                        'predictions': [],
                        'probabilities': [],
                        'ground_truth': [],
                        'n_households': 0,
                        'n_switched': 0
                    }
                    continue
                
                still_inactive_tensor = torch.tensor(still_inactive, dtype=torch.long)
                
                # Make predictions
                prediction_result = self._compute_state_predictions(
                    still_inactive_tensor, decision_k, features, model_states,
                    distances, complete_network, t, is_train_period=False
                )
                
                predictions = prediction_result['predictions']
                probabilities = prediction_result['probabilities']

                # Get ground truth outcomes at t+1 for still inactive households
                ground_truth = ground_truth_states[still_inactive, t+1, decision_k]
                n_switched = torch.sum(ground_truth).item()
                
                # Update next_states
                next_states[still_inactive, decision_k] = predictions
                
                predictions_by_decision[decision_name] = {
                    'target_households': target_hh,
                    'still_inactive': still_inactive,
                    'predictions': predictions.cpu().numpy(),
                    'probabilities': probabilities.cpu().numpy(),
                    'ground_truth': ground_truth.cpu().numpy(),
                    'n_households': len(still_inactive),
                    'n_switched': n_switched
                }
            
            return {
                'timestep_from': t,
                'timestep_to': t+1,
                'next_states': next_states,
                'predictions_by_decision': predictions_by_decision
            }
    
    def _compute_state_predictions(self, household_indices, decision_k, features,
                                  states, distances, complete_network, t, is_train_period=True):
        """Compute state predictions using complete network."""

        print(f"_compute_state_predictions: {t} for decision {decision_k}")
        print(f"Household indices length: {len(household_indices)}")
        
        # Create deterministic samples from network structure
        deterministic_samples = {}
        n_households = features.shape[0]
        
        for i in range(n_households):
            for j in range(i + 1, n_households):
                link_type = complete_network.get_link_type(i, j, t)
                pair_key = f"{i}_{j}_{t}"
                deterministic_samples[pair_key] = F.one_hot(
                    torch.tensor(link_type), num_classes=3
                ).float()
        
        # Compute activation probabilities
        if len(household_indices) > 0:
            activation_probs, detailed_breakdown = self.state_transition.compute_detailed_activation_probability(
                household_idx=household_indices,
                decision_type=decision_k,
                features=features,
                states=states,
                distances=distances,
                network_data=complete_network,
                gumbel_samples=deterministic_samples,
                time=t
            )
            
            # Store detailed logs for later analysis
            self.all_detailed_logs.extend(detailed_breakdown)

            if self.detailed_logger:
                period = 'train' if is_train_period else 'test'
                self.detailed_logger.log_detailed_breakdown(detailed_breakdown, period)
        
            
            # Compare with generator at key timesteps
            if t in [2, 5, 10, 16, 20] and self.probability_comparator.generator_data is not None:
                self.probability_comparator.compare_probabilities(detailed_breakdown, t, decision_k)
        else:
            # Fallback for empty household_indices
            activation_probs = torch.tensor([])
        
        # Convert to binary predictions

        decision_name = ['vacant', 'repair', 'sell'][decision_k]
        if is_train_period:
            threshold = self.train_thresholds[decision_name]
        else:
            base_threshold = self.test_thresholds[decision_name]
            relative_t = max(0, t - 8)  
            threshold = max(0.05, base_threshold - relative_t * 0.08)
        
        predictions = (activation_probs > threshold).float()
        # print(f"##### Predictions for decision {decision_k} at t={t}: {torch.mean(predictions)}")
        print(f"distribution of activation_probs for decision {decision_k} at t={t}: {torch.mean(activation_probs)}, {torch.median(activation_probs)}")
        
        return {
        'predictions': predictions,
        'probabilities': activation_probs,  # 新增：保存原始概率用于AUC
        'threshold_used': threshold
        }
    
    def _create_complete_network(self, base_network, inferred_structure, t):
        """Create complete network combining observed and inferred."""
        
        class CombinedNetwork:
            def __init__(self, base_network, inferred_structure, timestep):
                self.base_network = base_network
                self.inferred_structure = inferred_structure
                self.timestep = timestep
            
            def get_link_type(self, i, j, query_t):
                if query_t != self.timestep:
                    return self.base_network.get_link_type(i, j, query_t)
                
                if self.base_network.is_observed(i, j, query_t):
                    return self.base_network.get_link_type(i, j, query_t)
                else:
                    return self.inferred_structure.get((min(i,j), max(i,j)), 0)
            
            def is_observed(self, i, j, query_t):
                return self.base_network.is_observed(i, j, query_t)
        
        return CombinedNetwork(base_network, inferred_structure, t)
    
    def _apply_bonding_persistence_on_subset(self,
                                            inferred_structure_t: dict,
                                            prev_structure: dict,
                                            observed_network,
                                            t: int,
                                            subset_pairs):
        """
        Enforce strict bonding persistence (1→1) only on `subset_pairs` at time t.
        Also enforces global bonding persistence for all pairs.

        Rule:
        - If (i,j) was type 1 at t-1 (from prev_structure or observed as 1 at t-1),
        and (i,j) is UNOBSERVED at time t, force inferred type to 1 at t.
        - Never overwrite observations at time t.
        - GLOBAL RULE: If (i,j) was observed as bonding anywhere in training, force to 1.
        """
        if t == 0:
            # Even at t=0, apply global bonding persistence
            return self._enforce_global_bonding_persistence(inferred_structure_t)

        corrected = dict(inferred_structure_t)
        
        # Original local bonding persistence logic
        for (i, j) in subset_pairs:
            prev_type = prev_structure.get((i, j), 0)

            if observed_network.is_observed(i, j, t - 1):
                if observed_network.get_link_type(i, j, t - 1) == 1:
                    prev_type = 1

            if (prev_type == 1) and (not observed_network.is_observed(i, j, t)):
                corrected[(i, j)] = 1

        # Apply global bonding persistence
        corrected = self._enforce_global_bonding_persistence(corrected, t)

        return corrected

    def _enforce_global_bonding_persistence(self, inferred_structure, t):
        """
        Enforce global bonding persistence: if any pair was observed as bonding
        during training, it must be bonding at all time steps.
        
        Args:
            inferred_structure: Dictionary of inferred link types
            t: Current time step
        
        Returns:
            dict: Corrected inferred structure
        """
        if not hasattr(self, 'global_bonding_pairs'):
            return inferred_structure
        
        corrected = dict(inferred_structure)
        corrections_made = 0
        
        for (i, j) in self.global_bonding_pairs:
            if (i, j) in corrected and corrected[(i, j)] != 1:
                corrected[(i, j)] = 1
                corrections_made += 1
            elif (i, j) not in corrected:
                # This pair should be bonding but wasn't in inferred structure
                corrected[(i, j)] = 1
                corrections_made += 1
        
        if corrections_made > 0:
            print(f"  Global bonding enforcement: corrected {corrections_made} pairs at t={t}")
        
        return corrected

    
    def _aggregate_train_prediction_results(self, prediction_step_results):
        """Aggregate train period prediction results across all steps."""
        
        aggregated = {
            'by_decision': {},
            'overall': {},
            'step_by_step': prediction_step_results
        }
        
        k_values = [5, 10, 20, 50]  # Define k values for Top-k recall
        
        for decision_name in ['vacant', 'repair', 'sell']:
            all_predictions = []
            all_probabilities = []
            all_ground_truth = []
            total_households = 0
            total_switched = 0
            
            # Collect step-by-step Top-k results for temporal aggregation
            step_top_k_results = []
            step_n_positives = []
            
            for step_result in prediction_step_results:
                step_data = step_result['predictions_by_decision'][decision_name]
                if step_data['n_households'] > 0:
                    all_predictions.extend(step_data['predictions'])
                    all_probabilities.extend(step_data['probabilities'])
                    all_ground_truth.extend(step_data['ground_truth'])
                    total_households += step_data['n_households']
                    total_switched += step_data['n_switched']
                    
                    # Compute step-level Top-k recall
                    step_top_k = self._compute_top_k_recall_safe(
                        np.array(step_data['ground_truth']), 
                        np.array(step_data['probabilities']), 
                        k_values
                    )
                    step_top_k_results.append(step_top_k)
                    step_n_positives.append(step_data['n_switched'])
                else:
                    step_top_k_results.append({k: None for k in k_values})
                    step_n_positives.append(0)
            
            if len(all_predictions) > 0:
                accuracy = accuracy_score(all_ground_truth, all_predictions)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    all_ground_truth, all_predictions, average='binary', zero_division=0
                )
                conf_matrix = confusion_matrix(all_ground_truth, all_predictions)
                
                # Compute all three AUC-style metrics
                auc = self._compute_auc_safe(all_ground_truth, all_probabilities)
                pr_auc = self._compute_pr_auc_safe(all_ground_truth, all_probabilities)
                ap = self._compute_average_precision_safe(all_ground_truth, all_probabilities)
                
                # Compute overall Top-k recall
                overall_top_k = self._compute_top_k_recall_safe(
                    np.array(all_ground_truth), np.array(all_probabilities), k_values
                )
                
                # Compute temporally aggregated Top-k recall
                temporal_top_k = self._aggregate_top_k_temporal(
                    step_top_k_results, step_n_positives, k_values
                )
                
                aggregated['by_decision'][decision_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': auc,
                    'pr_auc': pr_auc,
                    'average_precision': ap,
                    'top_k_recall_overall': overall_top_k,
                    'top_k_recall_temporal': temporal_top_k,
                    'confusion_matrix': conf_matrix,
                    'total_households_evaluated': total_households,
                    'total_switched': total_switched,
                    'switch_rate': total_switched / total_households if total_households > 0 else 0
                }
        
        # Overall metrics (average across decision types)
        decision_types = ['vacant', 'repair', 'sell']
        overall_metrics = {}
        
        # Traditional metrics
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            values = [aggregated['by_decision'][dt][metric] for dt in decision_types 
                    if dt in aggregated['by_decision']]
            overall_metrics[metric] = np.mean(values) if values else 0
        
        # AUC-style metrics
        for metric in ['auc', 'pr_auc', 'average_precision']:
            values = [aggregated['by_decision'][dt][metric] for dt in decision_types 
                    if dt in aggregated['by_decision'] and aggregated['by_decision'][dt][metric] is not None]
            overall_metrics[metric] = np.mean(values) if values else None
        
        # Top-k recall metrics
        for k in k_values:
            # Overall Top-k
            values = [aggregated['by_decision'][dt]['top_k_recall_overall'][k] 
                    for dt in decision_types if dt in aggregated['by_decision'] 
                    and aggregated['by_decision'][dt]['top_k_recall_overall'][k] is not None]
            overall_metrics[f'top_{k}_recall_overall'] = np.mean(values) if values else None
            
            # Temporal Top-k
            values = [aggregated['by_decision'][dt]['top_k_recall_temporal'][k] 
                    for dt in decision_types if dt in aggregated['by_decision'] 
                    and aggregated['by_decision'][dt]['top_k_recall_temporal'][k] is not None]
            overall_metrics[f'top_{k}_recall_temporal'] = np.mean(values) if values else None
        
        aggregated['overall'] = overall_metrics
        return aggregated
    
    def _aggregate_train_structure_results(self, structure_step_results, ground_truth_network, observed_network):
        """
        修改后的Train结构评估聚合：只统计hidden pairs的性能
        """
        
        all_existence_pred = []
        all_existence_true = []
        all_type_pred = []
        all_type_true = []
        all_connection_probs = []
        
        total_pairs_evaluated = 0
        
        for step_result in structure_step_results:
            t = step_result['timestep']
            inferred = step_result['inferred_structure']
            probabilities = step_result.get('structure_probabilities', {})
            pairs_evaluated = step_result.get('pairs_evaluated', [])
            
            # 只统计实际评估的pairs（hidden pairs）
            for (i, j) in pairs_evaluated:
                if (i, j) in inferred:  # 确保这个pair被推断了
                    predicted_type = inferred[(i, j)]
                    true_type = ground_truth_network.get_link_type(i, j, t)
                    
                    # Link existence
                    all_existence_pred.append(predicted_type > 0)
                    all_existence_true.append(true_type > 0)
                    
                    if (i, j) in probabilities:
                        probs = probabilities[(i, j)]  
                        connection_prob = (probs[1] + probs[2]).item()  
                        all_connection_probs.append(connection_prob)
                    
                    # Link type (for existing links)
                    if true_type > 0:
                        all_type_pred.append(predicted_type)
                        all_type_true.append(true_type)
                    
                    total_pairs_evaluated += 1

        results = {}
        
        if len(all_existence_pred) > 0:
            existence_accuracy = accuracy_score(all_existence_true, all_existence_pred)
            existence_precision, existence_recall, existence_f1, _ = precision_recall_fscore_support(
                all_existence_true, all_existence_pred, average='binary', zero_division=0
            )
            
            if len(all_connection_probs) > 0:
                existence_auc = self._compute_auc_safe(all_existence_true, all_connection_probs)
            else:
                existence_auc = None
            
            results['link_existence'] = {
                'accuracy': existence_accuracy,
                'precision': existence_precision,
                'recall': existence_recall,
                'f1': existence_f1,
                'auc': existence_auc,
                'n_pairs': len(all_existence_pred)
            }
            
            print(f"Train期间结构推断统计: 评估了{total_pairs_evaluated}个hidden pairs")
        
        if len(all_type_pred) > 0:
            type_accuracy = accuracy_score(all_type_true, all_type_pred)
            results['link_type'] = {
                'accuracy': type_accuracy,
                'n_links': len(all_type_pred)
            }
        
        return results
    
    def _evaluate_test_final_and_timing(self, target_households, model_states, 
                                       ground_truth_states, train_end_time, test_end_time):
        """
        Evaluate test period: final states + timing for target households only.
        """
        
        results = {
            'final_state_evaluation': {},
            'timing_evaluation': {}
        }
        
        for decision_k in range(3):
            decision_name = ['vacant', 'repair', 'sell'][decision_k]
            target_hh = target_households[decision_name]
            
            if len(target_hh) == 0:
                continue
            
            target_tensor = torch.tensor(target_hh)
            
            # Final state evaluation
            our_final = model_states[target_tensor, test_end_time, decision_k]
            true_final = ground_truth_states[target_tensor, test_end_time, decision_k]
            
            final_accuracy = (our_final == true_final).float().mean().item()
            final_precision, final_recall, final_f1, _ = precision_recall_fscore_support(
                true_final.cpu().numpy(), our_final.cpu().numpy(), 
                average='binary', zero_division=0
            )
            
            # Compute AUC for final state prediction
            # final_auc = self._compute_auc_safe(true_final.cpu().numpy(), our_final.cpu().numpy())
            final_auc = None  # AUC not meaningful for binary final states
            
            results['final_state_evaluation'][decision_name] = {
                'accuracy': final_accuracy,
                'precision': final_precision,
                'recall': final_recall,
                'f1': final_f1,
                'auc': final_auc,
                'n_households': len(target_hh),
                'n_final_active': torch.sum(true_final).item()
            }
            
            # Timing evaluation
            timing_results = []
            
            for hh in target_hh:
                # Find when household first became active in ground truth
                true_transitions = ground_truth_states[hh, train_end_time+1:test_end_time+1, decision_k]
                true_first_active = torch.where(true_transitions == 1)[0]
                true_timing = true_first_active[0].item() + train_end_time + 1 if len(true_first_active) > 0 else None
                
                # Find when household first became active in our model
                our_transitions = model_states[hh, train_end_time+1:test_end_time+1, decision_k]
                our_first_active = torch.where(our_transitions == 1)[0]
                our_timing = our_first_active[0].item() + train_end_time + 1 if len(our_first_active) > 0 else None
                
                timing_results.append({
                    'household': hh,
                    'true_timing': true_timing,
                    'our_timing': our_timing,
                    'timing_error': abs(true_timing - our_timing) if true_timing is not None and our_timing is not None else None
                })
            
            # Aggregate timing metrics
            timing_errors = [r['timing_error'] for r in timing_results if r['timing_error'] is not None]
            correct_timing = sum(1 for r in timing_results if r['timing_error'] == 0)
            
            results['timing_evaluation'][decision_name] = {
                'correct_timing_rate': correct_timing / len(target_hh) if len(target_hh) > 0 else 0,
                'average_timing_error': np.mean(timing_errors) if timing_errors else 0,
                'timing_details': timing_results
            }
        
        return results
    
    def _aggregate_test_structure_results(self, structure_step_results, ground_truth_network, train_end_time):
        """
        修改后的Test结构评估聚合：只统计train_end_time时未观察到的pairs的性能
        """
        
        all_existence_pred = []
        all_existence_true = []
        all_type_pred = []
        all_type_true = []
        all_connection_probs = []
        
        total_pairs_evaluated = 0
        
        for step_result in structure_step_results:
            t = step_result['timestep']
            if t <= train_end_time:
                continue
                
            inferred = step_result['inferred_structure']
            probabilities = step_result.get('structure_probabilities', {})
            
            # Test期间：只评估那些在train_end_time时未观察到的pairs
            # 这些pairs在_infer_unobserved_pairs_test中已经被筛选过了
            for (i, j), predicted_type in inferred.items():
                # 检查这个pair在train_end_time时是否未被观察到
                # 注意：这里需要用原始的observed_network来检查train_end_time的状态
                if not hasattr(self, 'eval_target_pairs_set') or (i, j) not in self.eval_target_pairs_set:
                    continue  # 跳过在train_end_time时已观察到的pairs
                
                true_type = ground_truth_network.get_link_type(i, j, t)
                
                all_existence_pred.append(predicted_type > 0)
                all_existence_true.append(true_type > 0)
                all_type_pred.append(predicted_type)
                all_type_true.append(true_type)

                if (i, j) in probabilities:
                    probs = probabilities[(i, j)]  # [p0, p1, p2]
                    connection_prob = (probs[1] + probs[2]).item()  
                    all_connection_probs.append(connection_prob)
                
                total_pairs_evaluated += 1
        
        results = {}
        
        if len(all_existence_pred) > 0:
            existence_accuracy = accuracy_score(all_existence_true, all_existence_pred)
            existence_precision, existence_recall, existence_f1, _ = precision_recall_fscore_support(
                all_existence_true, all_existence_pred, average='binary', zero_division=0
            )
            
            # Compute AUC for test period structure inference
            if len(all_connection_probs) > 0 and len(all_connection_probs) == len(all_existence_true):
                existence_auc = self._compute_auc_safe(all_existence_true, all_connection_probs)
            else:
                existence_auc = self._compute_auc_safe(all_existence_true, all_existence_pred)
            
            results['link_existence'] = {
                'accuracy': existence_accuracy,
                'precision': existence_precision,
                'recall': existence_recall,
                'f1': existence_f1,
                'auc': existence_auc,
                'n_pairs': len(all_existence_pred)
            }
            
        
        if len(all_type_pred) > 0:
            type_accuracy = accuracy_score(all_type_true, all_type_pred)
            results['link_type'] = {
                'accuracy': type_accuracy,
                'n_links': len(all_type_pred)
            }
        
        return results
    
    def _create_summary(self, train_results, test_results):
            """Create evaluation summary."""
            
            summary = {
                'train_vs_test_prediction': {},
                'structure_inference': {},
                'key_insights': []
            }
            
            # Compare train vs test prediction performance
            train_pred = train_results['prediction_metrics']['overall']
            test_pred = test_results['prediction_metrics']['overall']  # Now available
            test_final = {}
            
            for decision_name in ['vacant', 'repair', 'sell']:
                if decision_name in test_results['final_and_timing_evaluation']['final_state_evaluation']:
                    test_final[decision_name] = test_results['final_and_timing_evaluation']['final_state_evaluation'][decision_name]['f1']
            
            summary['train_vs_test_prediction'] = {
                'train_overall_f1': train_pred['f1'],
                'train_overall_auc': train_pred.get('auc'),
                'test_overall_f1': test_pred['f1'],  # Now available
                'test_overall_auc': test_pred.get('auc'),  # Now available
                'test_final_f1_by_decision': test_final
            }
            
            # Structure inference comparison
            train_struct = train_results['structure_metrics']
            test_struct = test_results['structure_evaluation']
            
            summary['structure_inference'] = {
                'train_link_existence_accuracy': train_struct.get('link_existence', {}).get('accuracy', 0),
                'train_link_existence_auc': train_struct.get('link_existence', {}).get('auc'),
                'test_link_existence_accuracy': test_struct.get('link_existence', {}).get('accuracy', 0),
                'test_link_existence_auc': test_struct.get('link_existence', {}).get('auc')
            }
            
            return summary

    def finalize_probability_analysis(self):
        """Call this at the end of evaluation to analyze overall patterns"""
        if hasattr(self, 'probability_comparator') and hasattr(self, 'all_detailed_logs'):
            self.probability_comparator.analyze_overall_patterns(self.all_detailed_logs)

    def _compute_pr_auc_safe(self, y_true, y_score):
        """
        Safely compute PR-AUC, handling edge cases where all labels are the same class.
        """
        try:
            from sklearn.metrics import precision_recall_curve, auc
            unique_labels = np.unique(y_true)
            if len(unique_labels) < 2:
                return None
            
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            pr_auc = auc(recall, precision)
            return pr_auc
        except Exception as e:
            return None

    def _compute_average_precision_safe(self, y_true, y_score):
        """
        Safely compute Average Precision, handling edge cases.
        """
        try:
            from sklearn.metrics import average_precision_score
            unique_labels = np.unique(y_true)
            if len(unique_labels) < 2:
                return None
            
            ap = average_precision_score(y_true, y_score)
            return ap
        except Exception as e:
            return None

    def _compute_top_k_recall_safe(self, y_true, y_score, k_values=[5, 10, 20, 50]):
        """
        Compute Top-k recall for multiple k values with proper handling of edge cases.
        
        Args:
            y_true: True binary labels
            y_score: Predicted probabilities
            k_values: List of k values to compute recall for
        
        Returns:
            Dict with k as keys and recall values (or None) as values
        """
        results = {}
        n_positives = np.sum(y_true)
        
        if n_positives == 0:
            return {k: None for k in k_values}
        
        n_samples = len(y_true)
        
        for k in k_values:
            if k > n_samples:
                results[k] = None
                continue
                
            # Get top k predictions
            top_k_indices = np.argsort(y_score)[-k:]
            top_k_true = y_true[top_k_indices]
            
            # Calculate recall
            recall = np.sum(top_k_true) / n_positives
            results[k] = recall
        
        return results

    def _aggregate_top_k_temporal(self, step_results, step_n_positives, k_values=[5, 10, 20, 50]):
        """
        Aggregate Top-k recall across timesteps using weighted average.
        
        Args:
            step_results: List of dicts containing top-k results for each step
            step_n_positives: List of number of positive cases for each step
            k_values: List of k values
        
        Returns:
            Dict with k as keys and aggregated recall values as values
        """
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
            
            if valid_steps > 0:
                aggregated[k] = weighted_sum / valid_steps
            else:
                aggregated[k] = None
        
        return aggregated


def print_evaluation_results(results, ground_truth_network, trainer):
    """Print comprehensive evaluation results with step-by-step details."""
    
    print("\n" + "="*80)
    print("CORRECTED MODEL EVALUATION RESULTS")
    print("="*80)
    
    # STEP-BY-STEP STATE PREDICTION SUMMARY
    print("\n📈 STEP-BY-STEP STATE PREDICTION SUMMARY:")
    print("="*80)
    
    # Train period step-by-step (t=0→1 to t=14→15)
    print("\n🔹 TRAIN PERIOD (t=0→1 to t=14→15):")
    train_steps = results['train_evaluation']['step_by_step_predictions']
    for step_result in train_steps:
        t_from = step_result['timestep_from']
        t_to = step_result['timestep_to']
        print(f"\n  Step {t_from}→{t_to}:")
        
        # Show separate results for each decision type
        for decision_name in ['vacant', 'repair', 'sell']:
            step_data = step_result['predictions_by_decision'][decision_name]
            n_hh = step_data['n_households']
            n_switched = step_data['n_switched']
            
            if n_hh > 0:
                pred = step_data['predictions']
                prob = step_data['probabilities']
                true = step_data['ground_truth']
                
                accuracy = accuracy_score(true, pred)
                precision, recall, f1, _ = precision_recall_fscore_support(true, pred, average='binary', zero_division=0)
                conf_matrix = confusion_matrix(true, pred)
                
                # Compute AUC for this step
                auc = None
                try:
                    unique_labels = np.unique(true)
                    if len(unique_labels) >= 2:
                        auc = roc_auc_score(true, prob)  # 使用概率
                    else:
                        auc = None
                except:
                    auc = None
                
                print(f"    {decision_name.capitalize()}: {n_hh} households, {n_switched} switched")
                auc_str = f", AUC: {auc:.3f}" if auc is not None else ", AUC: N/A"
                # 计算PR-AUC和AP
                step_pr_auc = None
                step_ap = None
                try:
                    unique_labels = np.unique(true)
                    if len(unique_labels) >= 2:
                        from sklearn.metrics import precision_recall_curve, auc as sk_auc, average_precision_score
                        precision_curve, recall_curve, _ = precision_recall_curve(true, prob)
                        step_pr_auc = sk_auc(recall_curve, precision_curve)
                        step_ap = average_precision_score(true, prob)
                except:
                    pass

                pr_auc_str = f", PR-AUC: {step_pr_auc:.3f}" if step_pr_auc is not None else ", PR-AUC: N/A"
                ap_str = f", AP: {step_ap:.3f}" if step_ap is not None else ", AP: N/A"

                print(f"      Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}{auc_str}{pr_auc_str}{ap_str}")
                
                # Handle confusion matrix corner cases
                if conf_matrix.shape == (1, 1):
                    if true[0] == 0:
                        print(f"      Confusion Matrix: [[{conf_matrix[0,0]}, 0], [0, 0]] (all negative)")
                    else:
                        print(f"      Confusion Matrix: [[0, 0], [0, {conf_matrix[0,0]}]] (all positive)")
                elif conf_matrix.shape == (2, 2):
                    print(f"      Confusion Matrix: [[{conf_matrix[0,0]}, {conf_matrix[0,1]}], [{conf_matrix[1,0]}, {conf_matrix[1,1]}]]")
                else:
                    print(f"      Confusion Matrix: {conf_matrix.tolist()}")
            else:
                print(f"    {decision_name.capitalize()}: 0 households to predict")
        
        # Aggregate across all decision types for this timestep
        all_pred = []
        all_prob = []
        all_true = []
        total_households = 0
        
        for decision_name in ['vacant', 'repair', 'sell']:
            step_data = step_result['predictions_by_decision'][decision_name]
            if step_data['n_households'] > 0:
                all_pred.extend(step_data['predictions'])
                all_prob.extend(step_data['probabilities'])
                all_true.extend(step_data['ground_truth'])
                total_households += step_data['n_households']
        
        if total_households > 0:
            accuracy = accuracy_score(all_true, all_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(all_true, all_pred, average='binary', zero_division=0)
            conf_matrix = confusion_matrix(all_true, all_pred)
            
            # Compute combined AUC
            combined_auc = None
            try:
                unique_labels = np.unique(all_true)
                if len(unique_labels) >= 2:
                    combined_auc = roc_auc_score(all_true, all_prob)  
                else:
                    combined_auc = None
            except:
                combined_auc = None
            
            print(f"    COMBINED: {total_households} households predicted")
            auc_str = f", AUC: {combined_auc:.3f}" if combined_auc is not None else ", AUC: N/A"
            # 计算PR-AUC和AP
            step_pr_auc = None
            step_ap = None
            try:
                unique_labels = np.unique(true)
                if len(unique_labels) >= 2:
                    from sklearn.metrics import precision_recall_curve, auc as sk_auc, average_precision_score
                    precision_curve, recall_curve, _ = precision_recall_curve(true, prob)
                    step_pr_auc = sk_auc(recall_curve, precision_curve)
                    step_ap = average_precision_score(true, prob)
            except:
                pass

            pr_auc_str = f", PR-AUC: {step_pr_auc:.3f}" if step_pr_auc is not None else ", PR-AUC: N/A"
            ap_str = f", AP: {step_ap:.3f}" if step_ap is not None else ", AP: N/A"

            print(f"      Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}{auc_str}{pr_auc_str}{ap_str}")
            
            # Handle confusion matrix corner cases
            if conf_matrix.shape == (1, 1):
                if all_true[0] == 0:
                    print(f"      Confusion Matrix: [[{conf_matrix[0,0]}, 0], [0, 0]] (all negative)")
                else:
                    print(f"      Confusion Matrix: [[0, 0], [0, {conf_matrix[0,0]}]] (all positive)")
            elif conf_matrix.shape == (2, 2):
                print(f"      Confusion Matrix: [[{conf_matrix[0,0]}, {conf_matrix[0,1]}], [{conf_matrix[1,0]}, {conf_matrix[1,1]}]]")
            else:
                print(f"      Confusion Matrix: {conf_matrix.tolist()}")
        else:
            print(f"    COMBINED: 0 households to predict")

    # Test period step-by-step (t=15→16 to t=23→24) - NOW WITH DETAILED METRICS
    print("\n🔹 TEST PERIOD (t=15→16 to t=23→24):")
    test_steps = results['test_evaluation']['forward_simulation_records']['predictions']
    
    for step_result in test_steps:
        t_from = step_result['timestep_from']
        t_to = step_result['timestep_to']
        print(f"\n  Step {t_from}→{t_to}:")
        
        # 为每个决策类型单独计算指标
        decision_metrics = {}  # 存储每个决策类型的指标
        
        for decision_name in ['vacant', 'repair', 'sell']:
            step_data = step_result['predictions_by_decision'][decision_name]
            n_hh = step_data['n_households']
            n_switched = step_data['n_switched']
            
            if n_hh > 0:
                pred = step_data['predictions']
                # 关键修复：确保probabilities存在
                prob = step_data.get('probabilities', pred)
                true = step_data['ground_truth']
                
                # 调试信息：检查数据长度
                print(f"    DEBUG {decision_name}: pred_len={len(pred)}, prob_len={len(prob)}, true_len={len(true)}")
                
                # 确保数组长度一致
                if len(pred) != len(prob) or len(pred) != len(true):
                    print(f"    WARNING: {decision_name}数据长度不匹配，使用predictions作为probabilities")
                    prob = pred
                
                accuracy = accuracy_score(true, pred)
                precision, recall, f1, _ = precision_recall_fscore_support(true, pred, average='binary', zero_division=0)
                conf_matrix = confusion_matrix(true, pred)
                
                # 为当前决策类型计算所有指标
                auc = None
                pr_auc = None
                ap = None
                
                try:
                    unique_labels = np.unique(true)
                    if len(unique_labels) >= 2:
                        auc = roc_auc_score(true, prob)
                        
                        # 关键修复：在正确的作用域内计算PR-AUC
                        from sklearn.metrics import precision_recall_curve, auc as sk_auc, average_precision_score
                        precision_curve, recall_curve, _ = precision_recall_curve(true, prob)
                        pr_auc = sk_auc(recall_curve, precision_curve)
                        ap = average_precision_score(true, prob)
                    else:
                        print(f"    {decision_name}: 只有一个类别，无法计算AUC相关指标")
                except Exception as e:
                    print(f"    {decision_name}指标计算错误: {e}")
                
                # 存储指标供后续使用
                decision_metrics[decision_name] = {
                    'pred': pred, 'prob': prob, 'true': true,
                    'auc': auc, 'pr_auc': pr_auc, 'ap': ap
                }
                
                print(f"    {decision_name.capitalize()}: {n_hh} households, {n_switched} switched")
                auc_str = f", AUC: {auc:.3f}" if auc is not None else ", AUC: N/A"
                pr_auc_str = f", PR-AUC: {pr_auc:.3f}" if pr_auc is not None else ", PR-AUC: N/A"
                ap_str = f", AP: {ap:.3f}" if ap is not None else ", AP: N/A"
                
                print(f"      Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}{auc_str}{pr_auc_str}{ap_str}")
                
                # 混淆矩阵处理
                if conf_matrix.shape == (1, 1):
                    if len(true) > 0 and np.all(true == 0):
                        print(f"      Confusion Matrix: [[{conf_matrix[0,0]}, 0], [0, 0]] (all negative)")
                    else:
                        print(f"      Confusion Matrix: [[0, 0], [0, {conf_matrix[0,0]}]] (all positive)")
                elif conf_matrix.shape == (2, 2):
                    print(f"      Confusion Matrix: [[{conf_matrix[0,0]}, {conf_matrix[0,1]}], [{conf_matrix[1,0]}, {conf_matrix[1,1]}]]")
            else:
                print(f"    {decision_name.capitalize()}: 0 households to predict")
        
        # 关键修复：重新初始化合并数组
        all_pred = []
        all_prob = []
        all_true = []
        total_households = 0
        
        # 合并所有决策类型的数据
        for decision_name in ['vacant', 'repair', 'sell']:
            if decision_name in decision_metrics:
                metrics = decision_metrics[decision_name]
                all_pred.extend(metrics['pred'])
                all_prob.extend(metrics['prob'])
                all_true.extend(metrics['true'])
                step_data = step_result['predictions_by_decision'][decision_name]
                total_households += step_data['n_households']
        
        if total_households > 0:
            # 验证数据完整性
            print(f"    data check: pred={len(all_pred)}, prob={len(all_prob)}, true={len(all_true)}")
            
            # 确保所有数组长度一致
            min_length = min(len(all_pred), len(all_prob), len(all_true))
            if min_length < max(len(all_pred), len(all_prob), len(all_true)):
                print(f"    警告：数组长度不一致，截断到{min_length}")
                all_pred = all_pred[:min_length]
                all_prob = all_prob[:min_length]
                all_true = all_true[:min_length]
            
            # 计算合并指标
            accuracy = accuracy_score(all_true, all_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(all_true, all_pred, average='binary', zero_division=0)
            conf_matrix = confusion_matrix(all_true, all_pred)
            
            # 计算合并的AUC相关指标
            combined_auc = None
            combined_pr_auc = None
            combined_ap = None
            
            try:
                unique_labels = np.unique(all_true)
                if len(unique_labels) >= 2:
                    combined_auc = roc_auc_score(all_true, all_prob)
                    
                    from sklearn.metrics import precision_recall_curve, auc as sk_auc, average_precision_score
                    precision_curve, recall_curve, _ = precision_recall_curve(all_true, all_prob)
                    combined_pr_auc = sk_auc(recall_curve, precision_curve)
                    combined_ap = average_precision_score(all_true, all_prob)
                else:
                    print(f"    合并数据只有一个类别，无法计算AUC相关指标")
            except Exception as e:
                print(f"    合并指标计算错误: {e}")
            
            print(f"    COMBINED: {total_households} households predicted")
            auc_str = f", AUC: {combined_auc:.3f}" if combined_auc is not None else ", AUC: N/A"
            pr_auc_str = f", PR-AUC: {combined_pr_auc:.3f}" if combined_pr_auc is not None else ", PR-AUC: N/A"
            ap_str = f", AP: {combined_ap:.3f}" if combined_ap is not None else ", AP: N/A"
            
            print(f"      Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}{auc_str}{pr_auc_str}{ap_str}")
            
            # 处理混淆矩阵显示
            if conf_matrix.shape == (1, 1):
                if len(all_true) > 0 and np.all(np.array(all_true) == 0):
                    print(f"      Confusion Matrix: [[{conf_matrix[0,0]}, 0], [0, 0]] (all negative)")
                else:
                    print(f"      Confusion Matrix: [[0, 0], [0, {conf_matrix[0,0]}]] (all positive)")
            elif conf_matrix.shape == (2, 2):
                print(f"      Confusion Matrix: [[{conf_matrix[0,0]}, {conf_matrix[0,1]}], [{conf_matrix[1,0]}, {conf_matrix[1,1]}]]")
        else:
            print(f"    COMBINED: 0 households to predict")
    
    # STEP-BY-STEP STRUCTURE INFERENCE SUMMARY - COMPLETE METRICS
    print("\n🔹 TRAIN PERIOD - Hidden Pairs Only (Each timestep's unobserved pairs):")
    train_struct_steps = results['train_evaluation']['step_by_step_structures']
    print(f"Total steps evaluated: {len(train_struct_steps)}")

    for step_result in train_struct_steps:
        t = step_result['timestep']
        inferred = step_result['inferred_structure']
        probabilities = step_result.get('structure_probabilities', {})
        pairs_evaluated = step_result.get('pairs_evaluated', [])
        evaluation_scope = step_result.get('evaluation_scope', 'unknown')
        
        print(f"\n  Structure at t={t} (Scope: {evaluation_scope}):")
        
        # 只统计实际评估的pairs (hidden pairs)
        existence_pred = []
        existence_true = []
        type_pred = []
        type_true = []
        
        for (i, j) in pairs_evaluated:
            if (i, j) in inferred:
                predicted_type = inferred[(i, j)]
                true_type = ground_truth_network.get_link_type(i, j, t)
                
                existence_pred.append(predicted_type > 0)
                existence_true.append(true_type > 0)
                
                if true_type > 0:  # Only for existing links in ground truth
                    type_pred.append(predicted_type)
                    type_true.append(true_type)
        
        n_pairs_evaluated = len(pairs_evaluated)
        n_existing_in_gt = sum(existence_true) if existence_true else 0
        
        print(f"    Hidden pairs evaluated: {n_pairs_evaluated}")
        
        if len(existence_pred) > 0:
            existence_acc = accuracy_score(existence_true, existence_pred)
            existence_prec, existence_rec, existence_f1, _ = precision_recall_fscore_support(
                existence_true, existence_pred, average='binary', zero_division=0
            )
            existence_conf = confusion_matrix(existence_true, existence_pred)
            
            # Compute AUC for link existence (only on evaluated pairs)
            existence_auc = None
            try:
                unique_labels = np.unique(existence_true)
                if len(unique_labels) >= 2:
                    # 为当前timestep收集connection probabilities (只对evaluated pairs)
                    current_connection_probs = []
                    for (i, j) in pairs_evaluated:
                        if (i, j) in probabilities:
                            probs = probabilities[(i, j)]
                            if hasattr(probs, '__len__') and len(probs) >= 3:
                                prob_val = probs[1] + probs[2]  # connection probability
                                if hasattr(prob_val, 'item'):
                                    current_connection_probs.append(prob_val.item())
                                else:
                                    current_connection_probs.append(float(prob_val))
                    
                    if len(current_connection_probs) == len(existence_true) and len(current_connection_probs) > 0:
                        existence_auc = roc_auc_score(existence_true, current_connection_probs)
                    elif len(existence_true) > 0:
                        existence_auc = roc_auc_score(existence_true, [float(p) for p in existence_pred])
            except Exception as e:
                print(f"    AUC calculation failed at t={t}: {e}")
                existence_auc = None
            
            auc_str = f", AUC: {existence_auc:.3f}" if existence_auc is not None else ", AUC: N/A"
            print(f"    Link Existence (Hidden pairs only) - Acc: {existence_acc:.3f}, Prec: {existence_prec:.3f}, Rec: {existence_rec:.3f}, F1: {existence_f1:.3f}{auc_str}")
            
            # Handle confusion matrix shape
            if existence_conf.shape == (2, 2):
                print(f"    Existence Confusion: [[{existence_conf[0,0]}, {existence_conf[0,1]}], [{existence_conf[1,0]}, {existence_conf[1,1]}]]")
            elif existence_conf.shape == (1, 1):
                if len(existence_true) > 0 and np.all(np.array(existence_true) == 0):
                    print(f"    Existence Confusion: [[{existence_conf[0,0]}, 0], [0, 0]] (all negative in hidden pairs)")
                else:
                    print(f"    Existence Confusion: [[0, 0], [0, {existence_conf[0,0]}]] (all positive in hidden pairs)")
            else:
                print(f"    Existence Confusion: {existence_conf.tolist()}")
        else:
            print(f"    No hidden pairs to evaluate at t={t}")
        
        # print(f"    Existing links in ground truth (among hidden pairs): {n_existing_in_gt}")
        
        if len(type_pred) > 0:
            type_acc = accuracy_score(type_true, type_pred)
            print(f"    Type inference accuracy (hidden pairs): {type_acc:.3f}")
        else:
            print(f"    Type inference accuracy: N/A (no existing links in hidden pairs)")

    # Test period structure step-by-step 
    print("\n🔹 TEST PERIOD - Pairs Unobserved at Train End (t=15):")
    test_struct_steps = results['test_evaluation']['forward_simulation_records']['structures']
    
    if not test_struct_steps:
        print("  No test structure results available")
    else:
    
        evaluator = trainer.evaluator if hasattr(trainer, 'evaluator') else None
        total_unobserved_pairs = len(evaluator.eval_target_pairs) if evaluator and hasattr(evaluator, 'eval_target_pairs') else "Unknown"
        print(f"  Total pairs unobserved at t=15: {total_unobserved_pairs}")

    for step_result in test_struct_steps:
        t = step_result['timestep']
        inferred = step_result['inferred_structure']
        probabilities = step_result.get('structure_probabilities', {})
        evaluation_scope = step_result.get('evaluation_scope', 'unknown')
        
        print(f"\n  Structure at t={t} (Scope: {evaluation_scope}):")
        
        
        existence_pred = []
        existence_true = []
        type_pred = []
        type_true = []
        
        pairs_evaluated_count = 0
        for (i, j), predicted_type in inferred.items():
            
            if evaluator and hasattr(evaluator, 'eval_target_pairs_set'):
                if (i, j) not in evaluator.eval_target_pairs_set:
                    continue  
            
            true_type = ground_truth_network.get_link_type(i, j, t)
            
            existence_pred.append(predicted_type > 0)
            existence_true.append(true_type > 0)
            
            if true_type > 0:
                type_pred.append(predicted_type)
                type_true.append(true_type)
            
            pairs_evaluated_count += 1
        
        n_existing_in_gt = sum(existence_true) if existence_true else 0
        
        print(f"    Unobserved pairs evaluated: {pairs_evaluated_count}")
        
        if len(existence_pred) > 0:
            existence_acc = accuracy_score(existence_true, existence_pred)
            existence_prec, existence_rec, existence_f1, _ = precision_recall_fscore_support(
                existence_true, existence_pred, average='binary', zero_division=0
            )
            existence_conf = confusion_matrix(existence_true, existence_pred)
            
            # Compute AUC for test period structure
            existence_auc = None
            try:
                unique_labels = np.unique(existence_true)
                if len(unique_labels) >= 2:
                    
                    current_connection_probs = []
                    for (i, j), predicted_type in inferred.items():
                        if evaluator and hasattr(evaluator, 'eval_target_pairs_set'):
                            if (i, j) not in evaluator.eval_target_pairs_set:
                                continue
                        
                        if (i, j) in probabilities:
                            probs = probabilities[(i, j)]
                            if hasattr(probs, '__len__') and len(probs) >= 3:
                                prob_val = probs[1] + probs[2]  # connection probability
                                if hasattr(prob_val, 'item'):
                                    current_connection_probs.append(prob_val.item())
                                else:
                                    current_connection_probs.append(float(prob_val))
                    
                    if len(current_connection_probs) == len(existence_true) and len(current_connection_probs) > 0:
                        existence_auc = roc_auc_score(existence_true, current_connection_probs)
                    elif len(existence_true) > 0:
                        existence_auc = roc_auc_score(existence_true, [float(p) for p in existence_pred])
            except Exception as e:
                print(f"    AUC calculation failed at t={t}: {e}")
                existence_auc = None
            
            auc_str = f", AUC: {existence_auc:.3f}" if existence_auc is not None else ", AUC: N/A"
            print(f"    Link Existence (Unobserved pairs only) - Acc: {existence_acc:.3f}, Prec: {existence_prec:.3f}, Rec: {existence_rec:.3f}, F1: {existence_f1:.3f}{auc_str}")
            
            if existence_conf.shape == (2, 2):
                print(f"    Existence Confusion: [[{existence_conf[0,0]}, {existence_conf[0,1]}], [{existence_conf[1,0]}, {existence_conf[1,1]}]]")
            elif existence_conf.shape == (1, 1):
                if len(existence_true) > 0 and np.all(np.array(existence_true) == 0):
                    print(f"    Existence Confusion: [[{existence_conf[0,0]}, 0], [0, 0]] (all negative in unobserved pairs)")
                else:
                    print(f"    Existence Confusion: [[0, 0], [0, {existence_conf[0,0]}]] (all positive in unobserved pairs)")
            else:
                print(f"    Existence Confusion: {existence_conf.tolist()}")
        else:
            print(f"    No unobserved pairs to evaluate at t={t}")
        
        print(f"    Existing links in ground truth (among unobserved pairs): {n_existing_in_gt}")
        
        if len(type_pred) > 0:
            type_acc = accuracy_score(type_true, type_pred)
            print(f"    Type inference accuracy (unobserved pairs): {type_acc:.3f}")
        else:
            print(f"    Type inference accuracy: N/A (no existing links in unobserved pairs)")

    # ESTIMATED PARAMETERS
    print("\n⚙️ ESTIMATED PARAMETERS:")
    print("="*80)

    print(f"Observation Model Parameters:")
    print(f"  ρ₁ (bonding miss rate): {trainer.elbo_computer.rho_1.item():.4f}")
    print(f"  ρ₂ (bridging miss rate): {trainer.elbo_computer.rho_2.item():.4f}")

    print(f"\nNetwork Evolution Parameters:")
    print(f"  α₀ (initial bonding): {trainer.elbo_computer.network_evolution.alpha_0.item():.4f}")

    print(f"\nNormalization Factors:")
    print(f"  σ_demo²: {trainer.elbo_computer.network_evolution.sigma_demo_sq:.4f}")
    print(f"  σ_geo²: {trainer.elbo_computer.network_evolution.sigma_geo_sq:.4f}")

    # OVERALL SUMMARY (keeping all existing code)
    print("\n📊 OVERALL SUMMARY:")
    print("="*80)
    
    # Train period results
    print("\n📊 TRAIN PERIOD - OVERALL PREDICTION:")
    print("-" * 60)
    
    train_pred = results['train_evaluation']['prediction_metrics']
    print(f"Overall Performance:")
    for metric, value in train_pred['overall'].items():
        if metric in ['auc', 'pr_auc', 'average_precision'] and value is not None:
            print(f"  {metric.upper().replace('_', '-')}: {value:.3f}")
        elif metric.startswith('top_') and value is not None:
            # Format Top-k metrics nicely
            metric_display = metric.replace('_', '-').upper()
            print(f"  {metric_display}: {value:.3f}")
        elif not metric.startswith('top_') and metric not in ['auc', 'pr_auc', 'average_precision']:
            print(f"  {metric.capitalize()}: {value:.3f}" if isinstance(value, float) else f"  {metric.capitalize()}: {value}")
        
    print(f"\nBy Decision Type:")
    for decision_name in ['vacant', 'repair', 'sell']:
        if decision_name in train_pred['by_decision']:
            metrics = train_pred['by_decision'][decision_name]
            print(f"  {decision_name.capitalize()}:")
            for k, v in metrics.items():
                if k == 'auc':
                    if v is not None:
                        print(f"    {k}: {v:.3f}")
                    else:
                        print(f"    {k}: N/A")
                elif isinstance(v, float):
                    print(f"    {k}: {v:.3f}")
                elif isinstance(v, (int, np.integer)):
                    print(f"    {k}: {v}")
                elif isinstance(v, np.ndarray):
                    print(f"    {k}: {v.tolist()}")
    
    # Train structure results
    print(f"\n🔗 TRAIN PERIOD - STRUCTURE INFERENCE (Hidden Pairs Only):")
    print("-" * 60)
    
    train_struct = results['train_evaluation']['structure_metrics']
    for key, metrics in train_struct.items():
        print(f"{key.replace('_', ' ').capitalize()} (Hidden pairs only):")
        if isinstance(metrics, dict):
            for mk, mv in metrics.items():
                if mk == 'auc':
                    if mv is not None:
                        print(f"  {mk}: {mv:.3f}")
                    else:
                        print(f"  {mk}: N/A")
                elif isinstance(mv, float):
                    print(f"  {mk}: {mv:.3f}")
                elif isinstance(mv, (int, np.integer)):
                    print(f"  {mk}: {mv}")

    
def print_evaluation_results(results, ground_truth_network, trainer):
    """Print comprehensive evaluation results with updated evaluation scope."""
    
    print("\n" + "="*80)
    print("CORRECTED MODEL EVALUATION RESULTS")
    print("="*80)
    
    # STEP-BY-STEP STATE PREDICTION SUMMARY
    print("\n📈 STEP-BY-STEP STATE PREDICTION SUMMARY:")
    print("="*80)
    
    # Train period step-by-step (t=0→1 to t=14→15)
    print("\n🔹 TRAIN PERIOD (t=0→1 to t=14→15):")
    train_steps = results['train_evaluation']['step_by_step_predictions']
    for step_result in train_steps:
        t_from = step_result['timestep_from']
        t_to = step_result['timestep_to']
        print(f"\n  Step {t_from}→{t_to}:")
        
        # Show separate results for each decision type
        for decision_name in ['vacant', 'repair', 'sell']:
            step_data = step_result['predictions_by_decision'][decision_name]
            n_hh = step_data['n_households']
            n_switched = step_data['n_switched']
            
            if n_hh > 0:
                pred = step_data['predictions']
                prob = step_data['probabilities']
                true = step_data['ground_truth']
                
                accuracy = accuracy_score(true, pred)
                precision, recall, f1, _ = precision_recall_fscore_support(true, pred, average='binary', zero_division=0)
                conf_matrix = confusion_matrix(true, pred)
                
                # Compute AUC for this step
                auc = None
                try:
                    unique_labels = np.unique(true)
                    if len(unique_labels) >= 2:
                        auc = roc_auc_score(true, prob)  # 使用概率
                    else:
                        auc = None
                except:
                    auc = None
                
                print(f"    {decision_name.capitalize()}: {n_hh} households, {n_switched} switched")
                auc_str = f", AUC: {auc:.3f}" if auc is not None else ", AUC: N/A"
                print(f"      Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}{auc_str}")
                
                # Handle confusion matrix corner cases
                if conf_matrix.shape == (1, 1):
                    if true[0] == 0:
                        print(f"      Confusion Matrix: [[{conf_matrix[0,0]}, 0], [0, 0]] (all negative)")
                    else:
                        print(f"      Confusion Matrix: [[0, 0], [0, {conf_matrix[0,0]}]] (all positive)")
                elif conf_matrix.shape == (2, 2):
                    print(f"      Confusion Matrix: [[{conf_matrix[0,0]}, {conf_matrix[0,1]}], [{conf_matrix[1,0]}, {conf_matrix[1,1]}]]")
                else:
                    print(f"      Confusion Matrix: {conf_matrix.tolist()}")
            else:
                print(f"    {decision_name.capitalize()}: 0 households to predict")
        
        # Aggregate across all decision types for this timestep
        all_pred = []
        all_prob = []
        all_true = []
        total_households = 0
        
        for decision_name in ['vacant', 'repair', 'sell']:
            step_data = step_result['predictions_by_decision'][decision_name]
            if step_data['n_households'] > 0:
                all_pred.extend(step_data['predictions'])
                all_prob.extend(step_data['probabilities'])
                all_true.extend(step_data['ground_truth'])
                total_households += step_data['n_households']
        
        if total_households > 0:
            accuracy = accuracy_score(all_true, all_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(all_true, all_pred, average='binary', zero_division=0)
            conf_matrix = confusion_matrix(all_true, all_pred)
            
            # Compute combined AUC
            combined_auc = None
            try:
                unique_labels = np.unique(all_true)
                if len(unique_labels) >= 2:
                    combined_auc = roc_auc_score(all_true, all_prob)  
                else:
                    combined_auc = None
            except:
                combined_auc = None
            
            print(f"    COMBINED: {total_households} households predicted")
            auc_str = f", AUC: {combined_auc:.3f}" if combined_auc is not None else ", AUC: N/A"
            print(f"      Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}{auc_str}")
            
            # Handle confusion matrix corner cases
            if conf_matrix.shape == (1, 1):
                if all_true[0] == 0:
                    print(f"      Confusion Matrix: [[{conf_matrix[0,0]}, 0], [0, 0]] (all negative)")
                else:
                    print(f"      Confusion Matrix: [[0, 0], [0, {conf_matrix[0,0]}]] (all positive)")
            elif conf_matrix.shape == (2, 2):
                print(f"      Confusion Matrix: [[{conf_matrix[0,0]}, {conf_matrix[0,1]}], [{conf_matrix[1,0]}, {conf_matrix[1,1]}]]")
            else:
                print(f"      Confusion Matrix: {conf_matrix.tolist()}")
        else:
            print(f"    COMBINED: 0 households to predict")

    # Test period step-by-step (t=15→16 to t=23→24) - NOW WITH DETAILED METRICS
    print("\n🔹 TEST PERIOD (t=15→16 to t=23→24):")
    test_steps = results['test_evaluation']['forward_simulation_records']['predictions']
    
    for step_result in test_steps:
        t_from = step_result['timestep_from']
        t_to = step_result['timestep_to']
        print(f"\n  Step {t_from}→{t_to}:")
        
        # Show separate results for each decision type (same format as train period)
        for decision_name in ['vacant', 'repair', 'sell']:
            step_data = step_result['predictions_by_decision'][decision_name]
            n_hh = step_data['n_households']
            n_switched = step_data['n_switched']
            
            if n_hh > 0:
                pred = step_data['predictions']
                true = step_data['ground_truth']
                
                accuracy = accuracy_score(true, pred)
                precision, recall, f1, _ = precision_recall_fscore_support(true, pred, average='binary', zero_division=0)
                conf_matrix = confusion_matrix(true, pred)
                
                # Compute AUC for this step
                auc = None
                try:
                    unique_labels = np.unique(true)
                    if len(unique_labels) >= 2:
                        auc = roc_auc_score(true, pred)
                except:
                    pass
                
                print(f"    {decision_name.capitalize()}: {n_hh} households, {n_switched} switched")
                auc_str = f", AUC: {auc:.3f}" if auc is not None else ", AUC: N/A"
                print(f"      Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}{auc_str}")
                
                # Handle confusion matrix corner cases
                if conf_matrix.shape == (1, 1):
                    if true[0] == 0:
                        print(f"      Confusion Matrix: [[{conf_matrix[0,0]}, 0], [0, 0]] (all negative)")
                    else:
                        print(f"      Confusion Matrix: [[0, 0], [0, {conf_matrix[0,0]}]] (all positive)")
                elif conf_matrix.shape == (2, 2):
                    print(f"      Confusion Matrix: [[{conf_matrix[0,0]}, {conf_matrix[0,1]}], [{conf_matrix[1,0]}, {conf_matrix[1,1]}]]")
                else:
                    print(f"      Confusion Matrix: {conf_matrix.tolist()}")
            else:
                print(f"    {decision_name.capitalize()}: 0 households to predict")
        
        # Aggregate across all decision types for this timestep
        all_pred = []
        all_true = []
        total_households = 0
        
        for decision_name in ['vacant', 'repair', 'sell']:
            step_data = step_result['predictions_by_decision'][decision_name]
            if step_data['n_households'] > 0:
                all_pred.extend(step_data['predictions'])
                all_true.extend(step_data['ground_truth'])
                total_households += step_data['n_households']
        
        if total_households > 0:
            accuracy = accuracy_score(all_true, all_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(all_true, all_pred, average='binary', zero_division=0)
            conf_matrix = confusion_matrix(all_true, all_pred)
            
            # Compute combined AUC
            combined_auc = None
            try:
                unique_labels = np.unique(all_true)
                if len(unique_labels) >= 2:
                    combined_auc = roc_auc_score(all_true, all_pred)
            except:
                pass
            
            print(f"    COMBINED: {total_households} households predicted")
            auc_str = f", AUC: {combined_auc:.3f}" if combined_auc is not None else ", AUC: N/A"
            print(f"      Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}{auc_str}")
            
            # Handle confusion matrix corner cases
            if conf_matrix.shape == (1, 1):
                if all_true[0] == 0:
                    print(f"      Confusion Matrix: [[{conf_matrix[0,0]}, 0], [0, 0]] (all negative)")
                else:
                    print(f"      Confusion Matrix: [[0, 0], [0, {conf_matrix[0,0]}]] (all positive)")
            elif conf_matrix.shape == (2, 2):
                print(f"      Confusion Matrix: [[{conf_matrix[0,0]}, {conf_matrix[0,1]}], [{conf_matrix[1,0]}, {conf_matrix[1,1]}]]")
            else:
                print(f"      Confusion Matrix: {conf_matrix.tolist()}")
        else:
            print(f"    COMBINED: 0 households to predict")
    
    # STEP-BY-STEP STRUCTURE INFERENCE SUMMARY - 更新后的评估范围
    print("\n🔗 STEP-BY-STEP STRUCTURE INFERENCE SUMMARY:")
    print("="*80)

    # Train period structure step-by-step - 只评估hidden pairs
    print("\n🔹 TRAIN PERIOD - Hidden Pairs Only (Each timestep's unobserved pairs):")
    train_struct_steps = results['train_evaluation']['step_by_step_structures']
    print(f"Total steps evaluated: {len(train_struct_steps)}")

    for step_result in train_struct_steps:
        t = step_result['timestep']
        inferred = step_result['inferred_structure']
        probabilities = step_result.get('structure_probabilities', {})
        pairs_evaluated = step_result.get('pairs_evaluated', [])
        evaluation_scope = step_result.get('evaluation_scope', 'unknown')
        
        print(f"\n  Structure at t={t} (Scope: {evaluation_scope}):")
        
        # 只统计实际评估的pairs (hidden pairs)
        existence_pred = []
        existence_true = []
        type_pred = []
        type_true = []
        
        for (i, j) in pairs_evaluated:
            if (i, j) in inferred:
                predicted_type = inferred[(i, j)]
                true_type = ground_truth_network.get_link_type(i, j, t)
                
                existence_pred.append(predicted_type > 0)
                existence_true.append(true_type > 0)
                
                if true_type > 0:  # Only for existing links in ground truth
                    type_pred.append(predicted_type)
                    type_true.append(true_type)
        
        n_pairs_evaluated = len(pairs_evaluated)
        n_existing_in_gt = sum(existence_true) if existence_true else 0
        
        print(f"    Hidden pairs evaluated: {n_pairs_evaluated}")
        
        if len(existence_pred) > 0:
            existence_acc = accuracy_score(existence_true, existence_pred)
            existence_prec, existence_rec, existence_f1, _ = precision_recall_fscore_support(
                existence_true, existence_pred, average='binary', zero_division=0
            )
            existence_conf = confusion_matrix(existence_true, existence_pred)
            
            # Compute AUC for link existence (only on evaluated pairs)
            existence_auc = None
            try:
                unique_labels = np.unique(existence_true)
                if len(unique_labels) >= 2:
                    # 为当前timestep收集connection probabilities (只对evaluated pairs)
                    current_connection_probs = []
                    for (i, j) in pairs_evaluated:
                        if (i, j) in probabilities:
                            probs = probabilities[(i, j)]
                            if hasattr(probs, '__len__') and len(probs) >= 3:
                                prob_val = probs[1] + probs[2]  # connection probability
                                if hasattr(prob_val, 'item'):
                                    current_connection_probs.append(prob_val.item())
                                else:
                                    current_connection_probs.append(float(prob_val))
                    
                    if len(current_connection_probs) == len(existence_true) and len(current_connection_probs) > 0:
                        existence_auc = roc_auc_score(existence_true, current_connection_probs)
                    elif len(existence_true) > 0:
                        existence_auc = roc_auc_score(existence_true, [float(p) for p in existence_pred])
            except Exception as e:
                print(f"    AUC calculation failed at t={t}: {e}")
                existence_auc = None
            
            auc_str = f", AUC: {existence_auc:.3f}" if existence_auc is not None else ", AUC: N/A"
            print(f"    Link Existence (Hidden pairs only) - Acc: {existence_acc:.3f}, Prec: {existence_prec:.3f}, Rec: {existence_rec:.3f}, F1: {existence_f1:.3f}{auc_str}")
            
            # Handle confusion matrix shape
            if existence_conf.shape == (2, 2):
                print(f"    Existence Confusion: [[{existence_conf[0,0]}, {existence_conf[0,1]}], [{existence_conf[1,0]}, {existence_conf[1,1]}]]")
            elif existence_conf.shape == (1, 1):
                if len(existence_true) > 0 and np.all(np.array(existence_true) == 0):
                    print(f"    Existence Confusion: [[{existence_conf[0,0]}, 0], [0, 0]] (all negative in hidden pairs)")
                else:
                    print(f"    Existence Confusion: [[0, 0], [0, {existence_conf[0,0]}]] (all positive in hidden pairs)")
            else:
                print(f"    Existence Confusion: {existence_conf.tolist()}")
        else:
            print(f"    No hidden pairs to evaluate at t={t}")
        
        print(f"    Existing links in ground truth (among hidden pairs): {n_existing_in_gt}")
        
        if len(type_pred) > 0:
            type_acc = accuracy_score(type_true, type_pred)
            print(f"    Type inference accuracy (hidden pairs): {type_acc:.3f}")
        else:
            print(f"    Type inference accuracy: N/A (no existing links in hidden pairs)")

    # Test period structure step-by-step - 只评估在train_end_time时未观察到的pairs
    print("\n🔹 TEST PERIOD - Pairs Unobserved at Train End (t=15):")
    test_struct_steps = results['test_evaluation']['forward_simulation_records']['structures']
    
    if not test_struct_steps:
        print("  No test structure results available")
    else:
        # 获取在train_end_time时未观察到的pairs总数
        evaluator = trainer.evaluator if hasattr(trainer, 'evaluator') else None
        total_unobserved_pairs = len(evaluator.eval_target_pairs) if evaluator and hasattr(evaluator, 'eval_target_pairs') else "Unknown"
        print(f"  Total pairs unobserved at t=15: {total_unobserved_pairs}")

    for step_result in test_struct_steps:
        t = step_result['timestep']
        inferred = step_result['inferred_structure']
        probabilities = step_result.get('structure_probabilities', {})
        evaluation_scope = step_result.get('evaluation_scope', 'unknown')
        
        print(f"\n  Structure at t={t} (Scope: {evaluation_scope}):")
        
        
        existence_pred = []
        existence_true = []
        type_pred = []
        type_true = []
        
        pairs_evaluated_count = 0
        for (i, j), predicted_type in inferred.items():
            
            if evaluator and hasattr(evaluator, 'eval_target_pairs_set'):
                if (i, j) not in evaluator.eval_target_pairs_set:
                    continue  
            
            true_type = ground_truth_network.get_link_type(i, j, t)
            
            existence_pred.append(predicted_type > 0)
            existence_true.append(true_type > 0)
            
            if true_type > 0:
                type_pred.append(predicted_type)
                type_true.append(true_type)
            
            pairs_evaluated_count += 1
        
        n_existing_in_gt = sum(existence_true) if existence_true else 0
        
        print(f"    Unobserved pairs evaluated: {pairs_evaluated_count}")
        
        if len(existence_pred) > 0:
            existence_acc = accuracy_score(existence_true, existence_pred)
            existence_prec, existence_rec, existence_f1, _ = precision_recall_fscore_support(
                existence_true, existence_pred, average='binary', zero_division=0
            )
            existence_conf = confusion_matrix(existence_true, existence_pred)
            
            # Compute AUC for test period structure
            existence_auc = None
            try:
                unique_labels = np.unique(existence_true)
                if len(unique_labels) >= 2:
                    # 收集这些pairs的connection probabilities
                    current_connection_probs = []
                    for (i, j), predicted_type in inferred.items():
                        if evaluator and hasattr(evaluator, 'eval_target_pairs_set'):
                            if (i, j) not in evaluator.eval_target_pairs_set:
                                continue
                        
                        if (i, j) in probabilities:
                            probs = probabilities[(i, j)]
                            if hasattr(probs, '__len__') and len(probs) >= 3:
                                prob_val = probs[1] + probs[2]  # connection probability
                                if hasattr(prob_val, 'item'):
                                    current_connection_probs.append(prob_val.item())
                                else:
                                    current_connection_probs.append(float(prob_val))
                    
                    if len(current_connection_probs) == len(existence_true) and len(current_connection_probs) > 0:
                        existence_auc = roc_auc_score(existence_true, current_connection_probs)
                    elif len(existence_true) > 0:
                        existence_auc = roc_auc_score(existence_true, [float(p) for p in existence_pred])
            except Exception as e:
                print(f"    AUC calculation failed at t={t}: {e}")
                existence_auc = None
            
            auc_str = f", AUC: {existence_auc:.3f}" if existence_auc is not None else ", AUC: N/A"
            print(f"    Link Existence (Unobserved pairs only) - Acc: {existence_acc:.3f}, Prec: {existence_prec:.3f}, Rec: {existence_rec:.3f}, F1: {existence_f1:.3f}{auc_str}")
            
            if existence_conf.shape == (2, 2):
                print(f"    Existence Confusion: [[{existence_conf[0,0]}, {existence_conf[0,1]}], [{existence_conf[1,0]}, {existence_conf[1,1]}]]")
            elif existence_conf.shape == (1, 1):
                if len(existence_true) > 0 and np.all(np.array(existence_true) == 0):
                    print(f"    Existence Confusion: [[{existence_conf[0,0]}, 0], [0, 0]] (all negative in unobserved pairs)")
                else:
                    print(f"    Existence Confusion: [[0, 0], [0, {existence_conf[0,0]}]] (all positive in unobserved pairs)")
            else:
                print(f"    Existence Confusion: {existence_conf.tolist()}")
        else:
            print(f"    No unobserved pairs to evaluate at t={t}")
        
        print(f"    Existing links in ground truth (among unobserved pairs): {n_existing_in_gt}")
        
        if len(type_pred) > 0:
            type_acc = accuracy_score(type_true, type_pred)
            print(f"    Type inference accuracy (unobserved pairs): {type_acc:.3f}")
        else:
            print(f"    Type inference accuracy: N/A (no existing links in unobserved pairs)")


    print(f"\n🔗 TRAIN PERIOD - STRUCTURE INFERENCE (Hidden Pairs Only):")
    print("-" * 60)
    
    train_struct = results['train_evaluation']['structure_metrics']
    for key, metrics in train_struct.items():
        print(f"{key.replace('_', ' ').capitalize()} (Hidden pairs only):")
        if isinstance(metrics, dict):
            for mk, mv in metrics.items():
                if mk == 'auc':
                    if mv is not None:
                        print(f"  {mk}: {mv:.3f}")
                    else:
                        print(f"  {mk}: N/A")
                elif isinstance(mv, float):
                    print(f"  {mk}: {mv:.3f}")
                elif isinstance(mv, (int, np.integer)):
                    print(f"  {mk}: {mv}")
    
    print(f"\n🔗 TEST PERIOD - STRUCTURE INFERENCE (Pairs Unobserved at Train End):")
    print("-" * 60)
    
    test_struct = results['test_evaluation']['structure_evaluation']
    for key, metrics in test_struct.items():
        print(f"{key.replace('_', ' ').capitalize()} (Unobserved at train end only):")
        if isinstance(metrics, dict):
            for mk, mv in metrics.items():
                if mk == 'auc':
                    if mv is not None:
                        print(f"  {mk}: {mv:.3f}")
                    else:
                        print(f"  {mk}: N/A")
                elif isinstance(mv, float):
                    print(f"  {mk}: {mv:.3f}")
                elif isinstance(mv, (int, np.integer)):
                    print(f"  {mk}: {mv}")

    print("\n" + "="*80)
    
    # NEW: Test period overall prediction results
    print(f"\n📊 TEST PERIOD - OVERALL PREDICTION:")
    print("-" * 60)
    
    test_pred = results['test_evaluation']['prediction_metrics']
    print(f"Overall Performance:")
    for metric, value in test_pred['overall'].items():
        if metric == 'auc' and value is not None:
            print(f"  {metric.upper()}: {value:.3f}")
        elif metric != 'auc':
            print(f"  {metric.capitalize()}: {value:.3f}" if isinstance(value, float) else f"  {metric.capitalize()}: {value}")
    
    print(f"\nBy Decision Type:")
    for decision_name in ['vacant', 'repair', 'sell']:
        if decision_name in test_pred['by_decision']:
            metrics = test_pred['by_decision'][decision_name]
            print(f"  {decision_name.capitalize()}:")
            for k, v in metrics.items():
                if k == 'auc':
                    if v is not None:
                        print(f"    {k}: {v:.3f}")
                    else:
                        print(f"    {k}: N/A")
                elif isinstance(v, float):
                    print(f"    {k}: {v:.3f}")
                elif isinstance(v, (int, np.integer)):
                    print(f"    {k}: {v}")
                elif isinstance(v, np.ndarray):
                    print(f"    {k}: {v.tolist()}")
    
    # Test period results
    print(f"\n📊 TEST PERIOD - FINAL STATE & TIMING:")
    print("-" * 60)
    
    test_eval = results['test_evaluation']
    target_hh = test_eval['target_households']
    
    print(f"Target Households (inactive at t=15):")
    for decision_name in ['vacant', 'repair', 'sell']:
        print(f"  {decision_name.capitalize()}: {len(target_hh[decision_name])} households")
    
    print(f"\nFinal State Metrics:")
    final_eval = test_eval['final_and_timing_evaluation']['final_state_evaluation']
    for decision_name in ['vacant', 'repair', 'sell']:
        if decision_name in final_eval:
            metrics = final_eval[decision_name]
            print(f"  {decision_name.capitalize()}:")
            for k, v in metrics.items():
                if k == 'auc':
                    if v is not None:
                        print(f"    {k}: {v:.3f}")
                    else:
                        print(f"    {k}: N/A")
                elif isinstance(v, float):
                    print(f"    {k}: {v:.3f}")
                elif isinstance(v, (int, np.integer)):
                    print(f"    {k}: {v}")
    
    print(f"\nTiming Metrics:")
    timing_eval = test_eval['final_and_timing_evaluation']['timing_evaluation']
    for decision_name in ['vacant', 'repair', 'sell']:
        if decision_name in timing_eval:
            metrics = timing_eval[decision_name]
            print(f"  {decision_name.capitalize()}:")
            for k, v in metrics.items():
                if k == 'timing_details':
                    print(f"    {k}: [list of {len(v)} entries]")
                elif isinstance(v, float):
                    print(f"    {k}: {v:.3f}")
                elif isinstance(v, (int, np.integer)):
                    print(f"    {k}: {v}")
    
    # Test structure results
    print(f"\n🔗 TEST PERIOD - OVERALL STRUCTURE INFERENCE:")
    print("-" * 60)
    
    test_struct = test_eval['structure_evaluation']
    for key, metrics in test_struct.items():
        print(f"{key.replace('_', ' ').capitalize()}:")
        if isinstance(metrics, dict):
            for mk, mv in metrics.items():
                if mk == 'auc':
                    if mv is not None:
                        print(f"  {mk}: {mv:.3f}")
                    else:
                        print(f"  {mk}: N/A")
                elif isinstance(mv, float):
                    print(f"  {mk}: {mv:.3f}")
                elif isinstance(mv, (int, np.integer)):
                    print(f"  {mk}: {mv}")


    # TOP-K RECALL SUMMARY
    print("\n📊 TOP-K RECALL SUMMARY:")
    print("="*80)

    print("\n🔹 TRAIN PERIOD - Top-k Recall:")
    train_pred = results['train_evaluation']['prediction_metrics']
    k_values = [5, 10, 20, 50]

    for decision_name in ['vacant', 'repair', 'sell']:
        if decision_name in train_pred['by_decision']:
            metrics = train_pred['by_decision'][decision_name]
            print(f"\n  {decision_name.capitalize()}:")
            print(f"    Total households evaluated: {metrics['total_households_evaluated']}")
            print(f"    Total switched: {metrics['total_switched']}")
            
            if 'top_k_recall_overall' in metrics:
                print("    Top-k Recall (Overall):")
                for k in k_values:
                    recall_val = metrics['top_k_recall_overall'].get(k)
                    if recall_val is not None:
                        print(f"      Top-{k}: {recall_val:.3f}")
                    else:
                        print(f"      Top-{k}: N/A")
            
            if 'top_k_recall_temporal' in metrics:
                print("    Top-k Recall (Temporal Weighted):")
                for k in k_values:
                    recall_val = metrics['top_k_recall_temporal'].get(k)
                    if recall_val is not None:
                        print(f"      Top-{k}: {recall_val:.3f}")
                    else:
                        print(f"      Top-{k}: N/A")

    print("\n🔹 TEST PERIOD - Top-k Recall:")
    test_pred = results['test_evaluation']['prediction_metrics']

    for decision_name in ['vacant', 'repair', 'sell']:
        if decision_name in test_pred['by_decision']:
            metrics = test_pred['by_decision'][decision_name]
            print(f"\n  {decision_name.capitalize()}:")
            print(f"    Total households evaluated: {metrics['total_households_evaluated']}")
            print(f"    Total switched: {metrics['total_switched']}")
            
            if 'top_k_recall_overall' in metrics:
                print("    Top-k Recall (Overall):")
                for k in k_values:
                    recall_val = metrics['top_k_recall_overall'].get(k)
                    if recall_val is not None:
                        print(f"      Top-{k}: {recall_val:.3f}")
                    else:
                        print(f"      Top-{k}: N/A")
    
    print("\n" + "="*80)


# Simple interface function
# At the end of evaluation_corrected.py, modify the evaluate_model_corrected function:
# Around line 1200+, replace with this:

def evaluate_model_corrected(trainer, test_data, train_end_time=15, test_end_time=24,
                             custom_train_thresholds=None, custom_test_thresholds=None):
    """
    Corrected evaluation interface with detailed FR-SIC logging.
    
    Usage:
        results = evaluate_model_corrected(trainer, test_data)
        print_evaluation_results(results, ground_truth_network, trainer)
    """
    
    evaluator = CorrectedModelEvaluator(
        trainer.mean_field_posterior,
        trainer.elbo_computer.state_transition
    )

    if custom_train_thresholds:
        evaluator.train_thresholds.update(custom_train_thresholds)
    if custom_test_thresholds:
        evaluator.test_thresholds.update(custom_test_thresholds)
    
    try:
        results = evaluator.evaluate_model(
            features=test_data['features'],
            ground_truth_states=test_data['states'],
            distances=test_data['distances'],
            observed_network=test_data['observed_network'],
            ground_truth_network=test_data['ground_truth_network'],
            train_end_time=train_end_time,
            test_end_time=test_end_time
        )
        
        # Store evaluator reference for later access
        trainer.evaluator = evaluator
        
        return results
        
    except Exception as e:
        # Ensure cleanup happens even if evaluation fails
        if hasattr(evaluator, 'detailed_logger') and evaluator.detailed_logger:
            evaluator.detailed_logger.close()
        raise e
    
