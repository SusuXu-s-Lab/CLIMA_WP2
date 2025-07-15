"""
evaluation_corrected.py - Final corrected evaluation following exact requirements

Key Logic:
1. Train period: Step-by-step prediction evaluation + structure inference evaluation
2. Test period: Only evaluate households inactive at t=15, with final state + timing evaluation
3. Proper inference history tracking for temporal dependencies
4. FIXED: Use structure at time t to predict states t -> t+1
"""
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from data.data_loader import NetworkData
from evaluation.probability_comparator import ProbabilityComparator
import pandas as pd

class CorrectedModelEvaluator:
    """
    Corrected evaluation with proper train/test logic.
    """
    
    def __init__(self, mean_field_posterior, state_transition):
        self.mean_field_posterior = mean_field_posterior
        self.state_transition = state_transition
        self.inference_history = {}  # {t: {(i,j): inferred_type}}

        self.probability_comparator = ProbabilityComparator('/Users/susangao/Desktop/CLIMA/CODE 4.6/data/syn_data_ruxiao_v2/detailed_generator_probabilities.pkl')
        self.all_detailed_logs = []  # Store all detailed probability logs

    
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
        
        n_households = features.shape[0]
        all_nodes = torch.arange(n_households, dtype=torch.long)
        
        # Initialize
        self.inference_history = {}
        
        # Results storage
        results = {
            'train_evaluation': {},
            'test_evaluation': {},
            'summary': {}
        }
        
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
                current_structure, t
            )
            prediction_step_results.append(prediction_result)
            
            # 2. Structure Inference at t+1 (after we've made the prediction)
            structure_result = self._infer_structure_step(
                features, ground_truth_states, distances, all_nodes,
                observed_network, t+1, is_train=True
            )
            structure_step_results.append(structure_result)
            
            # Store in history for next steps
            self.inference_history[t+1] = structure_result['inferred_structure']
        
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
        """
        
        # Identify target households (inactive at train_end_time)
        target_households = {}
        for decision_k in range(3):
            decision_name = ['vacant', 'repair', 'sell'][decision_k]
            inactive_mask = ground_truth_states[:, train_end_time, decision_k] == 0
            target_households[decision_name] = torch.where(inactive_mask)[0].tolist()
            print(f"  Target {decision_name} households: {len(target_households[decision_name])}")
        
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
                current_structure, t, target_households
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
            
            # Store in history for next steps
            self.inference_history[t+1] = structure_result['inferred_structure']
        
        # Evaluate test results
        test_final_evaluation = self._evaluate_test_final_and_timing(
            target_households, model_states, ground_truth_states, train_end_time, test_end_time
        )
        
        test_structure_evaluation = self._aggregate_test_structure_results(
            test_structure_records, ground_truth_network, train_end_time
        )
        
        return {
            'target_households': target_households,
            'final_and_timing_evaluation': test_final_evaluation,
            'structure_evaluation': test_structure_evaluation,
            'forward_simulation_records': {
                'predictions': test_prediction_records,
                'structures': test_structure_records
            }
        }

    # def _infer_entire_network_test(self, features, model_states, distances, all_nodes, t):
    #     """
    #     Infer ENTIRE network structure at timestep t during test period.
        
    #     Unlike training period where we only infer hidden pairs, during test we:
    #     1. Have NO observational constraints (no test-time observations)
    #     2. Infer ALL pairs based purely on model dynamics
    #     3. Use predicted states from previous timesteps
        
    #     """
        
    #     n_households = features.shape[0]
        
    #     # Create a "no observations" network for clean inference
    #     class NoObservationsNetwork:
    #         def __init__(self, n_households, n_timesteps):
    #             self.n_households = n_households
    #             self.n_timesteps = n_timesteps
    #             self.all_pairs = [(i, j) for i in range(n_households) for j in range(i + 1, n_households)]
            
    #         def is_observed(self, i, j, query_t):
    #             # No observations in test period - everything is hidden
    #             return False
            
    #         def get_link_type(self, i, j, query_t):
    #             # No observed links
    #             return 0
            
    #         def get_hidden_pairs(self, query_t):
    #             # All pairs are hidden
    #             return self.all_pairs
            
    #         def get_observed_edges_at_time(self, query_t):
    #             # No observed edges
    #             return []
        
    #     # Create network with no observations
    #     no_obs_network = NoObservationsNetwork(n_households, model_states.shape[1])
        
    #     # Get conditional and marginal probabilities for ALL pairs
    #     conditional_probs, marginal_probs = self.mean_field_posterior.compute_probabilities_batch(
    #         features, model_states, distances, all_nodes, no_obs_network, t
    #     )
        
    #     # Infer structure for ALL pairs
    #     inferred_structure = {}
        
    #     for i in range(n_households):
    #         for j in range(i + 1, n_households):
    #             pair_key = f"{i}_{j}_{t}"
                
    #             if pair_key in conditional_probs:
    #                 # Use conditional probabilities based on previous inferred state
    #                 conditional_matrix = conditional_probs[pair_key]  # [3, 3]
                    
    #                 if t == 0:
    #                     # Initial timestep: use marginal probabilities
    #                     inferred_type = torch.argmax(marginal_probs[pair_key]).item()
    #                 else:
    #                     # Temporal inference: use conditional based on previous state
    #                     prev_type = self._get_previous_state_from_history(i, j, t-1)
    #                     conditional_given_prev = conditional_matrix[prev_type, :]  # [3]
    #                     inferred_type = torch.argmax(conditional_given_prev).item()
                    
    #                 inferred_structure[(i, j)] = inferred_type
    #             else:
    #                 # Fallback: use previous state or default
    #                 if t > 0:
    #                     prev_type = self._get_previous_state_from_history(i, j, t-1)
    #                     inferred_structure[(i, j)] = prev_type
    #                 else:
    #                     inferred_structure[(i, j)] = 0  # Default: no connection
        
    #     return {
    #         'timestep': t,
    #         'inferred_structure': inferred_structure,
    #         'evaluation_scope': 'entire_network'
    #     }

    
    def _infer_structure_step(self, features, states, distances, all_nodes,
                             observed_network, t, is_train=True):
        """
        Infer structure at timestep t using proper conditional probabilities.
        """
        
        # Get conditional and marginal probabilities
        conditional_probs, marginal_probs = self.mean_field_posterior.compute_probabilities_batch(
            features, states, distances, all_nodes, observed_network, t
        )

        # if t == 0:
        #     print(f"  Inference at t={t}: Using conditional probabilities")
        #     print(f"  Conditional prob: {conditional_probs}")
        #     print(f"  Marginal prob: {marginal_probs}")
        
        # Infer structure
        inferred_structure = {}
        n_households = features.shape[0]
        
        # Get pairs to evaluate
        if is_train:
            # Train: only hidden pairs
            pairs_to_infer = observed_network.get_hidden_pairs(t)
        else:
            # Test: all pairs (this shouldn't be called for test period in fixed version)
            pairs_to_infer = [(i, j) for i in range(n_households) for j in range(i+1, n_households)]
        
        for i, j in pairs_to_infer:
            pair_key = f"{i}_{j}_{t}"
            
            if (observed_network.is_observed(i, j, t)) & (is_train):
                # Use observed value
                inferred_structure[(i, j)] = observed_network.get_link_type(i, j, t)
            elif pair_key in conditional_probs:
                # Infer using conditional probabilities
                conditional_matrix = conditional_probs[pair_key]  # [3, 3]
                
                if t == 0:
                    # Initial: use marginal
                    inferred_type = torch.argmax(marginal_probs[pair_key]).item()
                else:
                    # Temporal: use conditional based on previous state
                    inferred_type = torch.argmax(marginal_probs[pair_key]).item()
                
                inferred_structure[(i, j)] = inferred_type
            else:
                # Fallback
                if t > 0:
                    prev_type = self._get_previous_state(i, j, t-1, observed_network)
                    inferred_structure[(i, j)] = prev_type
                else:
                    inferred_structure[(i, j)] = 0
        
        # Create complete network
        complete_network = self._create_complete_network(observed_network, inferred_structure, t)
        
        return {
            'timestep': t,
            'inferred_structure': inferred_structure,
            'complete_network': complete_network,
            'evaluation_scope': 'hidden_pairs' if is_train else 'all_pairs'
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
                                  complete_network, t):
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
                    'ground_truth': [],
                    'n_households': 0,
                    'n_switched': 0
                }
                continue
            
            # Make predictions
            predictions = self._compute_state_predictions(
                inactive_households, decision_k, features, ground_truth_states,
                distances, complete_network, t
            )
            
            # Get ground truth outcomes at t+1
            ground_truth = ground_truth_states[inactive_households, t+1, decision_k]
            n_switched = torch.sum(ground_truth).item()
            
            predictions_by_decision[decision_name] = {
                'inactive_households': inactive_households.tolist(),
                'predictions': predictions.cpu().numpy(),
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
                                 complete_network, t, target_households):
        """
        Predict states for test period: use model states at t to predict t+1.
        Only predict for target households that are still inactive in our model.
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
            
            if len(still_inactive) == 0:
                predictions_by_decision[decision_name] = {
                    'target_households': target_hh,
                    'still_inactive': [],
                    'predictions': [],
                    'n_predicted': 0
                }
                continue
            
            still_inactive_tensor = torch.tensor(still_inactive, dtype=torch.long)
            
            # Make predictions
            predictions = self._compute_state_predictions(
                still_inactive_tensor, decision_k, features, model_states,
                distances, complete_network, t
            )
            
            # Update next_states
            next_states[still_inactive, decision_k] = predictions
            
            predictions_by_decision[decision_name] = {
                'target_households': target_hh,
                'still_inactive': still_inactive,
                'predictions': predictions.cpu().numpy(),
                'n_predicted': len(still_inactive)
            }
        
        return {
            'timestep_from': t,
            'timestep_to': t+1,
            'next_states': next_states,
            'predictions_by_decision': predictions_by_decision
        }
    
    def _compute_state_predictions(self, household_indices, decision_k, features,
                                  states, distances, complete_network, t):
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
            
            # Compare with generator at key timesteps
            if t in [2, 5, 10, 16, 20] and self.probability_comparator.generator_data is not None:
                self.probability_comparator.compare_probabilities(detailed_breakdown, t, decision_k)
        else:
            # Fallback for empty household_indices
            activation_probs = torch.tensor([])

        # if t>=15:
        #     print(f"Activation probabilities at t={t} for decision {decision_k}: {max(activation_probs)}")
        # if (t>3) & (t<10):
        #     print(f"Max Activation probabilities at t={t} for decision {decision_k}: {max(activation_probs)}")
        
        # Convert to binary predictions
        threshold = 0.9
        predictions = (activation_probs > threshold).float()
        return predictions
    
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
    
    def _aggregate_train_prediction_results(self, prediction_step_results):
        """Aggregate train period prediction results across all steps."""
        
        aggregated = {
            'by_decision': {},
            'overall': {},
            'step_by_step': prediction_step_results
        }
        
        for decision_name in ['vacant', 'repair', 'sell']:
            all_predictions = []
            all_ground_truth = []
            total_households = 0
            total_switched = 0
            
            for step_result in prediction_step_results:
                step_data = step_result['predictions_by_decision'][decision_name]
                if step_data['n_households'] > 0:
                    all_predictions.extend(step_data['predictions'])
                    all_ground_truth.extend(step_data['ground_truth'])
                    total_households += step_data['n_households']
                    total_switched += step_data['n_switched']
            
            if len(all_predictions) > 0:
                accuracy = accuracy_score(all_ground_truth, all_predictions)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    all_ground_truth, all_predictions, average='binary', zero_division=0
                )
                conf_matrix = confusion_matrix(all_ground_truth, all_predictions)
                
                aggregated['by_decision'][decision_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'confusion_matrix': conf_matrix,
                    'total_households_evaluated': total_households,
                    'total_switched': total_switched,
                    'switch_rate': total_switched / total_households if total_households > 0 else 0
                }
        
        # Overall metrics
        decision_types = ['vacant', 'repair', 'sell']
        overall_metrics = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            values = [aggregated['by_decision'][dt][metric] for dt in decision_types 
                     if dt in aggregated['by_decision']]
            overall_metrics[metric] = np.mean(values) if values else 0
        
        aggregated['overall'] = overall_metrics
        return aggregated
    
    def _aggregate_train_structure_results(self, structure_step_results, ground_truth_network, observed_network):
        """Aggregate train period structure inference results."""
        
        all_existence_pred = []
        all_existence_true = []
        all_type_pred = []
        all_type_true = []
        
        for step_result in structure_step_results:
            t = step_result['timestep']
            inferred = step_result['inferred_structure']
            
            for (i, j), predicted_type in inferred.items():
                if not observed_network.is_observed(i, j, t):
                    true_type = ground_truth_network.get_link_type(i, j, t)
                    
                    # Link existence
                    all_existence_pred.append(predicted_type > 0)
                    all_existence_true.append(true_type > 0)
                    
                    # Link type (for existing links)
                    # if true_type > 0 or predicted_type > 0:
                    if true_type > 0:
                        all_type_pred.append(predicted_type)
                        all_type_true.append(true_type)

        results = {}
        
        if len(all_existence_pred) > 0:
            # print(f"all existence true: {sum(all_existence_true)}")
            # print(f"all existence pred: {sum(all_existence_pred)}")
            existence_accuracy = accuracy_score(all_existence_true, all_existence_pred)
            existence_precision, existence_recall, existence_f1, _ = precision_recall_fscore_support(
                all_existence_true, all_existence_pred, average='binary', zero_division=0
            )
            
            results['link_existence'] = {
                'accuracy': existence_accuracy,  # ------high because of many 0s
                'precision': existence_precision,
                'recall': existence_recall,
                'f1': existence_f1,
                'n_pairs': len(all_existence_pred)
            }
        
        if len(all_type_pred) > 0:
            # print(f"all type true: {all_type_true}")
            # print(f"all type pred: {all_type_pred}")
            type_accuracy = accuracy_score(all_type_true, all_type_pred)
            # print(f"Type accuracy: {type_accuracy}")
            results['link_type'] = {
                'accuracy': type_accuracy,
                'n_links': len(all_type_pred)
            }
        
        return results

    # def _aggregate_train_structure_results(self, structure_step_results, ground_truth_network, observed_network):
    #     """Aggregate train period structure inference results."""
        
    #     all_existence_pred = []
    #     all_existence_true = []
    #     all_type_pred = []
    #     all_type_true = []
        
    #     for step_result in structure_step_results:
    #         t = step_result['timestep']
    #         inferred = step_result['inferred_structure']
            
    #         # Get ALL hidden pairs at this timestep
    #         hidden_pairs_t = ground_truth_network.get_hidden_pairs(t)
            
    #         for (i, j) in hidden_pairs_t:
    #             # Get model's prediction (or default if missing)
    #             if (i, j) in inferred:
    #                 predicted_type = inferred[(i, j)]
    #             else:
    #                 predicted_type = 0  # Default: no connection
                
    #             # Get ground truth
    #             true_type = ground_truth_network.get_link_type(i, j, t)
                
    #             # Link existence
    #             all_existence_pred.append(predicted_type > 0)
    #             all_existence_true.append(true_type > 0)
                
    #             # Link type (for existing links)
    #             #if true_type > 0 or predicted_type > 0:
    #             if true_type > 0:
    #                 all_type_pred.append(predicted_type)
    #                 all_type_true.append(true_type)
        
    #     results = {}
        
    #     if len(all_existence_pred) > 0:
    #         existence_accuracy = accuracy_score(all_existence_true, all_existence_pred)
    #         existence_precision, existence_recall, existence_f1, _ = precision_recall_fscore_support(
    #             all_existence_true, all_existence_pred, average='binary', zero_division=0
    #         )
            
    #         results['link_existence'] = {
    #             'accuracy': existence_accuracy,
    #             'precision': existence_precision,
    #             'recall': existence_recall,
    #             'f1': existence_f1,
    #             'n_pairs': len(all_existence_pred)
    #         }
        
    #     if len(all_type_pred) > 0:
    #         type_accuracy = accuracy_score(all_type_true, all_type_pred)
    #         results['link_type'] = {
    #             'accuracy': type_accuracy,
    #             'n_links': len(all_type_pred)
    #         }
        
    #     return results
    
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
            
            results['final_state_evaluation'][decision_name] = {
                'accuracy': final_accuracy,
                'precision': final_precision,
                'recall': final_recall,
                'f1': final_f1,
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
        """Aggregate test period structure results (all pairs)."""
        
        all_existence_pred = []
        all_existence_true = []
        all_type_pred = []
        all_type_true = []
        
        for step_result in structure_step_results:
            t = step_result['timestep']
            if t <= train_end_time:
                continue
                
            inferred = step_result['inferred_structure']
            
            for (i, j), predicted_type in inferred.items():
                true_type = ground_truth_network.get_link_type(i, j, t)
                
                all_existence_pred.append(predicted_type > 0)
                all_existence_true.append(true_type > 0)
                all_type_pred.append(predicted_type)
                all_type_true.append(true_type)
        
        results = {}
        
        if len(all_existence_pred) > 0:
            existence_accuracy = accuracy_score(all_existence_true, all_existence_pred)
            existence_precision, existence_recall, existence_f1, _ = precision_recall_fscore_support(
                all_existence_true, all_existence_pred, average='binary', zero_division=0
            )
            
            results['link_existence'] = {
                'accuracy': existence_accuracy,
                'precision': existence_precision,
                'recall': existence_recall,
                'f1': existence_f1,
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
        test_final = {}
        
        for decision_name in ['vacant', 'repair', 'sell']:
            if decision_name in test_results['final_and_timing_evaluation']['final_state_evaluation']:
                test_final[decision_name] = test_results['final_and_timing_evaluation']['final_state_evaluation'][decision_name]['f1']
        
        summary['train_vs_test_prediction'] = {
            'train_overall_f1': train_pred['f1'],
            'test_final_f1_by_decision': test_final
        }
        
        # Structure inference comparison
        train_struct = train_results['structure_metrics']
        test_struct = test_results['structure_evaluation']
        
        summary['structure_inference'] = {
            'train_link_existence_accuracy': train_struct.get('link_existence', {}).get('accuracy', 0),
            'test_link_existence_accuracy': test_struct.get('link_existence', {}).get('accuracy', 0)
        }
        
        return summary




    def finalize_probability_analysis(self):
        """Call this at the end of evaluation to analyze overall patterns"""
        if hasattr(self, 'probability_comparator') and hasattr(self, 'all_detailed_logs'):
            self.probability_comparator.analyze_overall_patterns(self.all_detailed_logs)


def print_evaluation_results(results, csv_path, trainer, test_data):
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
                true = step_data['ground_truth']
                
                accuracy = accuracy_score(true, pred)
                precision, recall, f1, _ = precision_recall_fscore_support(true, pred, average='binary', zero_division=0)
                conf_matrix = confusion_matrix(true, pred)
                
                print(f"    {decision_name.capitalize()}: {n_hh} households, {n_switched} switched")
                print(f"      Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
                
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
            
            print(f"    COMBINED: {total_households} households predicted")
            print(f"      Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
            
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

    # Test period step-by-step (t=15→16 to t=23→24)
    print("\n🔹 TEST PERIOD (t=15→16 to t=23→24):")
    test_steps = results['test_evaluation']['forward_simulation_records']['predictions']
    for step_result in test_steps:
        t_from = step_result['timestep_from']
        t_to = step_result['timestep_to']
        print(f"\n  Step {t_from}→{t_to}:")
        
        # Show detailed results for each decision type with confusion matrices
        for decision_name in ['vacant', 'repair', 'sell']:
            step_data = step_result['predictions_by_decision'][decision_name]
            n_target = len(step_data['target_households'])
            n_still_inactive = len(step_data['still_inactive'])
            n_predicted = step_data['n_predicted']
            
            print(f"    {decision_name.capitalize()}: {n_target} target, {n_still_inactive} still inactive, {n_predicted} predicted")
            
            # Calculate confusion matrix and metrics if we have predictions
            if n_predicted > 0:
                still_inactive = step_data['still_inactive']
                predictions = step_data['predictions']
                
                # Get ground truth for these households at timestep t_to
                decision_k = ['vacant', 'repair', 'sell'].index(decision_name)
                ground_truth = []
                for hh in still_inactive:
                    # Get ground truth state from the original test data
                    ground_truth.append(test_data['states'][hh, t_to, decision_k].item())
                
                # Calculate metrics
                if len(predictions) > 0 and len(ground_truth) > 0:
                    accuracy = accuracy_score(ground_truth, predictions)
                    precision, recall, f1, _ = precision_recall_fscore_support(ground_truth, predictions, average='binary', zero_division=0)
                    conf_matrix = confusion_matrix(ground_truth, predictions)
                    
                    print(f"      Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
                    
                    # Handle confusion matrix corner cases
                    if conf_matrix.shape == (1, 1):
                        if ground_truth[0] == 0:
                            print(f"      Confusion Matrix: [[{conf_matrix[0,0]}, 0], [0, 0]] (all negative)")
                        else:
                            print(f"      Confusion Matrix: [[0, 0], [0, {conf_matrix[0,0]}]] (all positive)")
                    elif conf_matrix.shape == (2, 2):
                        print(f"      Confusion Matrix: [[{conf_matrix[0,0]}, {conf_matrix[0,1]}], [{conf_matrix[1,0]}, {conf_matrix[1,1]}]]")
                    else:
                        print(f"      Confusion Matrix: {conf_matrix.tolist()}")
        
        # Aggregate across all decision types for this timestep
        all_pred = []
        all_true = []
        total_predicted = 0
        
        for decision_name in ['vacant', 'repair', 'sell']:
            step_data = step_result['predictions_by_decision'][decision_name]
            n_predicted = step_data['n_predicted']
            
            if n_predicted > 0:
                still_inactive = step_data['still_inactive']
                predictions = step_data['predictions']
                
                # Get ground truth for these households
                decision_k = ['vacant', 'repair', 'sell'].index(decision_name)
                ground_truth = []
                for hh in still_inactive:
                    # Get ground truth state from the original test data  
                    ground_truth.append(test_data['states'][hh, t_to, decision_k].item())
                
                all_pred.extend(predictions)
                all_true.extend(ground_truth)
                total_predicted += n_predicted
        
        if total_predicted > 0:
            accuracy = accuracy_score(all_true, all_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(all_true, all_pred, average='binary', zero_division=0)
            conf_matrix = confusion_matrix(all_true, all_pred)
            
            print(f"    COMBINED: {total_predicted} households predicted")
            print(f"      Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
            
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
            print(f"    COMBINED: 0 households predicted")



    # STEP-BY-STEP STRUCTURE INFERENCE SUMMARY - COMPLETE METRICS
    print("\n🔗 STEP-BY-STEP STRUCTURE INFERENCE SUMMARY:")
    print("="*80)

    # ESTIMATED PARAMETERS
    print("\n⚙️ ESTIMATED PARAMETERS:")
    print("="*80)

    # You need to pass the trainer to the print function
    # Modify function signature: def print_evaluation_results(results, ground_truth_network, trainer):

    print(f"Observation Model Parameters:")
    print(f"  ρ₁ (bonding miss rate): {trainer.elbo_computer.rho_1.item():.4f}")
    print(f"  ρ₂ (bridging miss rate): {trainer.elbo_computer.rho_2.item():.4f}")

    print(f"\nNetwork Evolution Parameters:")
    print(f"  α₀ (initial bonding): {trainer.elbo_computer.network_evolution.alpha_0.item():.4f}")

    print(f"\nNormalization Factors:")

    print(f"  σ_demo²: {trainer.elbo_computer.network_evolution.sigma_demo_sq:.4f}")
    print(f"  σ_geo²: {trainer.elbo_computer.network_evolution.sigma_geo_sq:.4f}")

    # ORIGINAL OVERALL SUMMARY (keeping all existing code)
    print("\n📊 OVERALL SUMMARY:")
    print("="*80)
    
    # Train period results
    print("\n📊 TRAIN PERIOD - OVERALL PREDICTION:")
    print("-" * 60)
    
    train_pred = results['train_evaluation']['prediction_metrics']
    print(f"Overall Performance:")
    for metric, value in train_pred['overall'].items():
        print(f"  {metric.capitalize()}: {value:.3f}" if isinstance(value, float) else f"  {metric.capitalize()}: {value}")
    
    print(f"\nBy Decision Type:")
    for decision_name in ['vacant', 'repair', 'sell']:
        if decision_name in train_pred['by_decision']:
            metrics = train_pred['by_decision'][decision_name]
            print(f"  {decision_name.capitalize()}:")
            for k, v in metrics.items():
                if isinstance(v, float):
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
                if isinstance(v, float):
                    print(f"    {k}: {v:.3f}")
                elif isinstance(v, (int, np.integer)):
                    print(f"    {k}: {v}")
    
    # Add visualizations at the end of evaluation results
    print("\n🎨 GENERATING VISUALIZATIONS...")
    print("="*80)
    
    # 1. States prediction visualization
    print("📊 Creating states prediction visualization...")
    visualize_states_prediction(results, test_data, train_end_time=15, test_end_time=24)
    
    # 2. Links prediction visualization
    print("🔗 Creating links prediction visualization...")
    visualize_links_prediction(results, test_data['ground_truth_network'], test_data['observed_network'], train_end_time=15)


def visualize_states_prediction(results, test_data, train_end_time=15, test_end_time=24):
    """
    Visualize states prediction results with 9 curves:
    - For each state (vacant, repair, sell): 3 curves each
      1. Cumulative predicted count (houses don't flip back once they change state)
      2. Cumulative correct predicted count  
      3. Ground truth count (actual state distribution at each time point)
    """
    
    # Initialize data structures
    decision_types = ['vacant', 'repair', 'sell']
    
    # Get actual time dimension size from test_data
    n_households, n_time_points, n_states = test_data['states'].shape
    time_points = list(range(1, n_time_points))  # t=1 to t=n_time_points-1 (since we use 0-indexed)
    
    # Data containers for each state
    predicted_counts = {state: [] for state in decision_types}
    correct_counts = {state: [] for state in decision_types}
    ground_truth_counts = {state: [] for state in decision_types}
    
    # Extract ground truth counts from test_data for all time points
    for t in time_points:
        for decision_k, decision_name in enumerate(decision_types):
            # Ground truth count at time t (using 0-indexed for tensor access)
            gt_count = int(test_data['states'][:, t, decision_k].sum().item())
            ground_truth_counts[decision_name].append(gt_count)
    
    # Process training period results (t=0→1 to t=14→15)
    train_steps = results['train_evaluation']['step_by_step_predictions']
    
    # Initialize prediction tracking
    for t in time_points:
        for decision_name in decision_types:
            predicted_counts[decision_name].append(0)
            correct_counts[decision_name].append(0)
    
    # Process training period
    for step_result in train_steps:
        t_to = step_result['timestep_to']
        
        # Check if t_to is within valid range for our arrays
        if t_to - 1 < len(predicted_counts['vacant']):
            for decision_name in decision_types:
                step_data = step_result['predictions_by_decision'][decision_name]
                
                if step_data['n_households'] > 0:
                    predictions = step_data['predictions']
                    ground_truth = step_data['ground_truth']
                    
                    # Count predictions and correct predictions
                    predicted_count = sum(predictions)
                    correct_count = sum(1 for p, g in zip(predictions, ground_truth) if p == 1 and g == 1)
                    
                    predicted_counts[decision_name][t_to - 1] += predicted_count
                    correct_counts[decision_name][t_to - 1] += correct_count
    
    # Process test period results (t=15→16 to t=23→24)
    test_steps = results['test_evaluation']['forward_simulation_records']['predictions']
    
    for step_result in test_steps:
        t_to = step_result['timestep_to']
        
        # Check if t_to is within valid range for both our arrays and test_data
        if t_to - 1 < len(predicted_counts['vacant']) and t_to < n_time_points:
            for decision_name in decision_types:
                step_data = step_result['predictions_by_decision'][decision_name]
                
                if step_data['n_predicted'] > 0:
                    still_inactive = step_data['still_inactive']
                    predictions = step_data['predictions']
                    
                    # Get ground truth for these households
                    decision_k = decision_types.index(decision_name)
                    ground_truth = []
                    for hh in still_inactive:
                        ground_truth.append(test_data['states'][hh, t_to, decision_k].item())
                    
                    # Count predictions and correct predictions
                    predicted_count = sum(predictions)
                    correct_count = sum(1 for p, g in zip(predictions, ground_truth) if p == 1 and g == 1)
                    
                    predicted_counts[decision_name][t_to - 1] += predicted_count
                    correct_counts[decision_name][t_to - 1] += correct_count
    
    # Convert to cumulative counts (since houses don't flip back once they change state)
    print("📊 Converting to cumulative counts...")
    for decision_name in decision_types:
        # Convert to cumulative sums
        predicted_counts[decision_name] = np.cumsum(predicted_counts[decision_name]).tolist()
        correct_counts[decision_name] = np.cumsum(correct_counts[decision_name]).tolist()
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Cumulative States Prediction Analysis Over Time', fontsize=16, fontweight='bold')
    
    colors = {
        'predicted': '#1f77b4',    # blue
        'correct': '#2ca02c',      # green  
        'ground_truth': '#ff7f0e'  # orange
    }
    
    for i, decision_name in enumerate(decision_types):
        ax = axes[i]
        
        # Plot the three curves for this state
        ax.plot(time_points, predicted_counts[decision_name], 
                color=colors['predicted'], linewidth=2, marker='o', markersize=4,
                label=f'Cumulative Predicted {decision_name.capitalize()}')
        
        ax.plot(time_points, correct_counts[decision_name], 
                color=colors['correct'], linewidth=2, marker='s', markersize=4,
                label=f'Cumulative Correct {decision_name.capitalize()}')
        
        ax.plot(time_points, ground_truth_counts[decision_name], 
                color=colors['ground_truth'], linewidth=2, marker='^', markersize=4,
                label=f'Ground Truth {decision_name.capitalize()}')
        
        # Customize subplot
        ax.set_title(f'{decision_name.capitalize()} States', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Point', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Add vertical line to separate train/test periods
        ax.axvline(x=train_end_time, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax.text(train_end_time-1, ax.get_ylim()[1]*0.9, 'Train', ha='right', fontsize=10, color='red')
        ax.text(train_end_time+1, ax.get_ylim()[1]*0.9, 'Test', ha='left', fontsize=10, color='red')
        
        # Set reasonable y-axis limits
        y_max = max(max(predicted_counts[decision_name]), 
                   max(correct_counts[decision_name]), 
                   max(ground_truth_counts[decision_name]))
        ax.set_ylim(0, y_max * 1.1)
        
        # Set x-axis ticks
        ax.set_xticks(range(1, n_time_points, 2))
    
    plt.tight_layout()
    plt.savefig('cumulative_states_prediction_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n📊 CUMULATIVE VISUALIZATION SUMMARY:")
    print("="*50)
    for decision_name in decision_types:
        # Use final values since these are cumulative counts
        final_predicted = predicted_counts[decision_name][-1] if predicted_counts[decision_name] else 0
        final_correct = correct_counts[decision_name][-1] if correct_counts[decision_name] else 0
        final_ground_truth = ground_truth_counts[decision_name][-1] if ground_truth_counts[decision_name] else 0
        
        accuracy = final_correct / final_predicted if final_predicted > 0 else 0
        recall = final_correct / final_ground_truth if final_ground_truth > 0 else 0
        
        print(f"\n{decision_name.capitalize()} State (Final Cumulative):")
        print(f"  Final Predicted: {final_predicted}")
        print(f"  Final Correct: {final_correct}")
        print(f"  Final Ground Truth: {final_ground_truth}")
        print(f"  Overall Accuracy: {accuracy:.3f}")
        print(f"  Overall Recall: {recall:.3f}")
    
    print(f"\nVisualization saved as 'cumulative_states_prediction_analysis.png'")
    print("="*50)


def visualize_links_prediction(results, ground_truth_network, observed_network, train_end_time=15):
    """
    Visualize links prediction results for training period only (t=0 to t=15):
    - Only show predictions for UNOBSERVED links at each timestep
    - Show bonding and bridging links in the same plot
    - For each link type (bonding, bridging): 3 curves each
      1. Ground truth count (unobserved links only)
      2. Predicted count (unobserved links only)
      3. Correct predicted count (unobserved links only)
    """
    
    # Initialize data structures
    link_types = ['bonding', 'bridging']
    link_type_mapping = {'bonding': 1, 'bridging': 2}
    
    # Get training steps only
    train_steps = results['train_evaluation']['step_by_step_structures']
    
    # Get timesteps from training period only (t=0 to t=train_end_time)
    time_points = []
    for step_result in train_steps:
        t = step_result['timestep']
        if t <= train_end_time:
            time_points.append(t)
    
    # Sort time points
    time_points = sorted(list(set(time_points)))
    
    # Data containers for each link type
    ground_truth_counts = {link_type: [] for link_type in link_types}
    predicted_counts = {link_type: [] for link_type in link_types}
    correct_counts = {link_type: [] for link_type in link_types}
    
    # Debug: Track unobserved pairs count
    unobserved_pairs_count = []
    
    # Process each timestep
    for t in time_points:
        # Find the corresponding structure result
        structure_result = None
        for step_result in train_steps:
            if step_result['timestep'] == t:
                structure_result = step_result
                break
        
        # Initialize counts for this timestep
        for link_type in link_types:
            ground_truth_counts[link_type].append(0)
            predicted_counts[link_type].append(0)
            correct_counts[link_type].append(0)
        
        # Count unobserved pairs
        unobserved_count = 0
        
        if structure_result is not None:
            inferred_structure = structure_result['inferred_structure']
            
            # Count ground truth, predicted, and correct predictions
            # ONLY for links that were UNOBSERVED at this timestep
            for (i, j), predicted_type in inferred_structure.items():
                # Skip if this link was observed at this timestep
                if observed_network.is_observed(i, j, t):
                    continue
                
                unobserved_count += 1
                
                # Get ground truth for this pair
                true_type = ground_truth_network.get_link_type(i, j, t)
                
                # Count ground truth (unobserved links only)
                if true_type == 1:  # bonding
                    ground_truth_counts['bonding'][-1] += 1
                elif true_type == 2:  # bridging
                    ground_truth_counts['bridging'][-1] += 1
                
                # Count predictions (unobserved links only)
                if predicted_type == 1:  # bonding
                    predicted_counts['bonding'][-1] += 1
                elif predicted_type == 2:  # bridging
                    predicted_counts['bridging'][-1] += 1
                
                # Count correct predictions (unobserved links only)
                if predicted_type == true_type and predicted_type > 0:
                    if predicted_type == 1:  # bonding
                        correct_counts['bonding'][-1] += 1
                    elif predicted_type == 2:  # bridging
                        correct_counts['bridging'][-1] += 1
        
        unobserved_pairs_count.append(unobserved_count)
    
    # Debug output
    print(f"\n🔍 DEBUG INFO - UNOBSERVED LINKS ANALYSIS:")
    print("="*60)
    for i, t in enumerate(time_points):
        print(f"t={t:2d}: Unobserved pairs={unobserved_pairs_count[i]:3d}, "
              f"GT_bonding={ground_truth_counts['bonding'][i]:2d}, "
              f"GT_bridging={ground_truth_counts['bridging'][i]:2d}, "
              f"Pred_bonding={predicted_counts['bonding'][i]:2d}, "
              f"Pred_bridging={predicted_counts['bridging'][i]:2d}")
    
    # Create single plot visualization
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    fig.suptitle('Unobserved Links Prediction Analysis (Training Period Only)', fontsize=16, fontweight='bold')
    
    # Color scheme for bonding and bridging
    colors = {
        'bonding': {
            'ground_truth': '#ff4500',  # Red-orange
            'predicted': '#ff8c00',     # Dark orange
            'correct': '#ffa500'        # Orange
        },
        'bridging': {
            'ground_truth': '#1f77b4',  # Blue
            'predicted': '#4169e1',     # Royal blue
            'correct': '#87ceeb'        # Sky blue
        }
    }
    
    # Line styles for different metrics
    line_styles = {
        'ground_truth': '-',
        'predicted': '--',
        'correct': '-.'
    }
    
    # Markers for different metrics
    markers = {
        'ground_truth': 'o',
        'predicted': 's',
        'correct': '^'
    }
    
    # Plot all lines on the same axes
    for link_type in link_types:
        # Ground truth line
        ax.plot(time_points, ground_truth_counts[link_type], 
                color=colors[link_type]['ground_truth'], 
                linewidth=3, linestyle=line_styles['ground_truth'],
                marker=markers['ground_truth'], markersize=7, 
                markerfacecolor=colors[link_type]['ground_truth'], 
                markeredgecolor='white', markeredgewidth=1,
                label=f'GT {link_type.capitalize()}', zorder=6)
        
        # Predicted line
        ax.plot(time_points, predicted_counts[link_type], 
                color=colors[link_type]['predicted'], 
                linewidth=2.5, linestyle=line_styles['predicted'],
                marker=markers['predicted'], markersize=6, 
                markerfacecolor='white', 
                markeredgecolor=colors[link_type]['predicted'], markeredgewidth=2,
                label=f'Pred {link_type.capitalize()}', zorder=4)
        
        # Correct line
        ax.plot(time_points, correct_counts[link_type], 
                color=colors[link_type]['correct'], 
                linewidth=2.5, linestyle=line_styles['correct'],
                marker=markers['correct'], markersize=6, 
                markerfacecolor='white', 
                markeredgecolor=colors[link_type]['correct'], markeredgewidth=2,
                label=f'Correct {link_type.capitalize()}', zorder=2)
    
    # Customize plot
    ax.set_title('Bonding and Bridging Links Prediction (Unobserved Only)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time Point', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Organize legend in two columns
    ax.legend(fontsize=10, loc='upper left', ncol=2, frameon=True, fancybox=True, shadow=True)
    
    # Set reasonable y-axis limits with padding
    all_counts = []
    for link_type in link_types:
        all_counts.extend(ground_truth_counts[link_type])
        all_counts.extend(predicted_counts[link_type])
        all_counts.extend(correct_counts[link_type])
    
    y_max = max(all_counts) if all_counts else 0
    y_min = min(all_counts) if all_counts else 0
    
    # Set y-axis limits with padding
    if y_max > 0:
        ax.set_ylim(max(0, y_min - 2), y_max + 5)
    else:
        ax.set_ylim(0, 10)  # Default range when no data
    
    # Set x-axis ticks
    ax.set_xticks(time_points)
    ax.set_xticklabels(time_points, rotation=45)
    
    plt.tight_layout()
    plt.savefig('unobserved_links_prediction_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n📊 UNOBSERVED LINKS PREDICTION VISUALIZATION SUMMARY:")
    print("="*60)
    print("Training Period Only (t=0 to t=15) - Unobserved Links Only")
    print("="*60)
    
    for link_type in link_types:
        total_ground_truth = sum(ground_truth_counts[link_type])
        total_predicted = sum(predicted_counts[link_type])
        total_correct = sum(correct_counts[link_type])
        
        precision = total_correct / total_predicted if total_predicted > 0 else 0
        recall = total_correct / total_ground_truth if total_ground_truth > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n{link_type.capitalize()} Links:")
        print(f"  Total Ground Truth: {total_ground_truth}")
        print(f"  Total Predicted: {total_predicted}")
        print(f"  Total Correct: {total_correct}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1-Score: {f1:.3f}")
        
        # Print timestep-by-timestep breakdown
        print(f"  Time Series Data:")
        for t, gt, pred, correct in zip(time_points, ground_truth_counts[link_type], 
                                        predicted_counts[link_type], correct_counts[link_type]):
            print(f"    t={t:2d}: GT={gt:2d}, Pred={pred:2d}, Correct={correct:2d}")
    
    print(f"\nVisualization saved as 'unobserved_links_prediction_analysis.png'")
    print("="*60)


# Simple interface function
def evaluate_model_corrected(trainer, test_data, train_end_time=15, test_end_time=24):
    """
    Corrected evaluation interface.
    
    Usage:
        results = evaluate_model_corrected(trainer, test_data)
        print_evaluation_results(results)
    """
    
    evaluator = CorrectedModelEvaluator(
        trainer.mean_field_posterior,
        trainer.elbo_computer.state_transition
    )
    
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

