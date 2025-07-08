"""
probability_comparator.py - Compare model probabilities with generator ground truth
"""
import torch
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional

class ProbabilityComparator:
    """
    Compare model probabilities with generator ground truth
    """
    
    def __init__(self, generator_data_path: Optional[str] = None):
        self.generator_data = None
        if generator_data_path and Path(generator_data_path).exists():
            with open(generator_data_path, 'rb') as f:
                self.generator_data = pickle.load(f)
            print(f"Loaded generator data with {len(self.generator_data['detailed_log'])} entries")
        else:
            print("No generator data available for comparison")
    
    def compare_probabilities(self, model_detailed_breakdown: List[Dict], t: int, decision_k: int):
        """Fixed version of probability comparison"""
        if self.generator_data is None:
            print("No generator data available for comparison")
            return
        
        # FIXED: Correct decision type mapping
        # Generator: vacancy=0, repair=1, sales=2  
        # Model: vacant=0, repair=1, sell=2
        model_to_generator = {0: 0, 1: 1, 2: 2}  # Direct mapping
        generator_decision_k = model_to_generator[decision_k]
        
        decision_names = ['vacant', 'repair', 'sell']
        decision_name = decision_names[decision_k]
        
        print(f"\n{'='*80}")
        print(f"GENERATOR vs MODEL COMPARISON t={t}, decision={decision_name}")
        print(f"Model decision_k={decision_k} → Generator decision_type={generator_decision_k}")
        print(f"{'='*80}")
        
        # Build generator lookup using correct index
        generator_lookup = {}
        for entry in self.generator_data['detailed_log']:
            if entry['timestep'] == t and entry['decision_type'] == generator_decision_k:
                key = entry['household_index']  # This should now be correct 0-based index
                generator_lookup[key] = entry
        
        print(f"Found {len(generator_lookup)} generator entries")
        if len(generator_lookup) > 0:
            print(f"Generator household indices: {sorted(generator_lookup.keys())[:10]}...")
        
        # Model results by household_id
        model_lookup = {}
        for model_result in model_detailed_breakdown:
            hh_idx = model_result['household_id']
            model_lookup[hh_idx] = model_result
        
        print(f"Found {len(model_lookup)} model entries")
        if len(model_lookup) > 0:
            print(f"Model household indices: {sorted(model_lookup.keys())[:10]}...")
        
        # Find common households
        common_households = set(generator_lookup.keys()) & set(model_lookup.keys())
        print(f"Common households: {len(common_households)}")
        
        if len(common_households) == 0:
            print("❌ NO COMMON HOUSEHOLDS FOUND!")
            print("Check if index mapping is correct...")
            
            # Debug info
            print(f"Generator indices range: {min(generator_lookup.keys()) if generator_lookup else 'N/A'} - {max(generator_lookup.keys()) if generator_lookup else 'N/A'}")
            print(f"Model indices range: {min(model_lookup.keys()) if model_lookup else 'N/A'} - {max(model_lookup.keys()) if model_lookup else 'N/A'}")
            return
        
        # Compare common households
        for hh_idx in sorted(list(common_households)[:5]):  # Only compare first 5
            gen_result = generator_lookup[hh_idx]
            model_result = model_lookup[hh_idx]
            self._compare_single_household(model_result, gen_result, hh_idx)
    
    def _compare_single_household(self, model_result: Dict, gen_result: Dict, hh_idx: int):
        """Compare probabilities for a single household"""
        
        print(f"\n{'='*50}")
        print(f"Household {hh_idx}")
        print(f"{'='*50}")
        
        # Compare self-activation
        self._compare_self_activation(model_result, gen_result)
        
        # Compare network structure
        self._compare_network_structure(model_result, gen_result)
        
        # Compare neighbor influences
        self._compare_neighbor_influences(model_result, gen_result)
        
        # Compare final probabilities
        self._compare_final_probabilities(model_result, gen_result)
    
    def _compare_self_activation(self, model_result: Dict, gen_result: Dict):
        """Compare self-activation probabilities"""
        gen_self = gen_result['self_activation_prob']
        model_self = model_result['self_activation_prob']
        
        ratio = model_self / (gen_self + 1e-10)
        abs_diff = abs(model_self - gen_self)
        
        print(f"\n📊 Self-activation:")
        print(f"  Generator: {gen_self:.8f}")
        print(f"  Model:     {model_self:.8f}")
        print(f"  Ratio:     {ratio:.3f}")
        print(f"  Abs diff:  {abs_diff:.8f}")
        
        if ratio < 0.1:
            print(f"  ⚠️  Model self-activation much lower than generator")
        elif ratio > 10:
            print(f"  ⚠️  Model self-activation much higher than generator")
        elif 0.8 <= ratio <= 1.2:
            print(f"  ✅ Self-activation probabilities are close")
        else:
            print(f"  ⚠️  Moderate difference in self-activation")
    
    def _compare_network_structure(self, model_result: Dict, gen_result: Dict):
        """Compare network structure (which neighbors are active)"""
        gen_neighbors = set(ni['neighbor_index'] for ni in gen_result['neighbor_influences'])
        model_neighbors = set(ni['neighbor_id'] for ni in model_result['neighbor_influences'])
        
        print(f"\n🔗 Network structure:")
        print(f"  Generator active neighbors: {sorted(gen_neighbors)}")
        print(f"  Model active neighbors:     {sorted(model_neighbors)}")
        
        missing_neighbors = gen_neighbors - model_neighbors
        extra_neighbors = model_neighbors - gen_neighbors
        common_neighbors = gen_neighbors & model_neighbors
        
        print(f"  Common neighbors:  {sorted(common_neighbors)} ✅")
        if missing_neighbors:
            print(f"  Missing neighbors: {sorted(missing_neighbors)} ❌")
        if extra_neighbors:
            print(f"  Extra neighbors:   {sorted(extra_neighbors)} ❌")
        
        if len(gen_neighbors) == 0:
            accuracy = 1.0 if len(model_neighbors) == 0 else 0.0
        else:
            accuracy = len(common_neighbors) / len(gen_neighbors | model_neighbors)
        print(f"  Network accuracy:  {accuracy:.3f}")
    
    def _compare_neighbor_influences(self, model_result: Dict, gen_result: Dict):
        """Compare neighbor-by-neighbor influences"""
        gen_influences = {ni['neighbor_index']: ni for ni in gen_result['neighbor_influences']}
        model_influences = {ni['neighbor_id']: ni for ni in model_result['neighbor_influences']}
        
        all_neighbors = set(gen_influences.keys()) | set(model_influences.keys())
        
        print(f"\n👥 Neighbor influences:")
        
        for neighbor_idx in sorted(all_neighbors):
            print(f"  Neighbor {neighbor_idx}:")
            
            if neighbor_idx in gen_influences and neighbor_idx in model_influences:
                gen_ni = gen_influences[neighbor_idx]
                model_ni = model_influences[neighbor_idx]
                
                # Compare link types
                gen_link = gen_ni['link_type']
                model_link = model_ni['link_type']
                link_match = "✅" if gen_link == model_link else "❌"
                
                print(f"    Link type: Gen={gen_link}, Model={model_link} {link_match}")
                
                # Compare influence probabilities
                gen_inf = gen_ni['influence_prob']
                model_inf = model_ni['influence_prob']
                inf_ratio = model_inf / (gen_inf + 1e-10)
                
                print(f"    Influence: Gen={gen_inf:.8f}, Model={model_inf:.8f}")
                print(f"    Ratio: {inf_ratio:.3f}")
                
                # Check connection strength for model
                if 'connection_strength' in model_ni:
                    print(f"    Connection strength: {model_ni['connection_strength']:.3f}")
                
                # Validate no-link logic
                if model_link == 0 and model_inf > 1e-8:
                    print(f"    ❌ BUG: No-link neighbor has non-zero influence!")
                
                if abs(inf_ratio - 1.0) < 0.2:
                    print(f"    ✅ Influence probabilities are close")
                else:
                    print(f"    ⚠️  Large difference in influence probabilities")
                    
            elif neighbor_idx in gen_influences:
                gen_inf = gen_influences[neighbor_idx]['influence_prob']
                print(f"    ❌ MISSING in model (Generator: {gen_inf:.8f})")
                
            else:
                model_inf = model_influences[neighbor_idx]['influence_prob']
                print(f"    ❌ EXTRA in model (Model: {model_inf:.8f})")
    
    def _compare_final_probabilities(self, model_result: Dict, gen_result: Dict):
        """Compare final combined probabilities"""
        gen_social = gen_result['social_influence_term']
        model_social = model_result['social_influence_term']
        gen_final = gen_result['final_activation_prob']
        model_final = model_result['final_activation_prob']
        
        print(f"\n🎯 Final probabilities:")
        print(f"  Social influence term:")
        print(f"    Generator: {gen_social:.8f}")
        print(f"    Model:     {model_social:.8f}")
        print(f"    Ratio:     {model_social / (gen_social + 1e-10):.3f}")
        
        print(f"  Final activation probability:")
        print(f"    Generator: {gen_final:.8f}")
        print(f"    Model:     {model_final:.8f}")
        print(f"    Ratio:     {model_final / (gen_final + 1e-10):.3f}")
        
        # Overall assessment
        final_ratio = model_final / (gen_final + 1e-10)
        if 0.8 <= final_ratio <= 1.2:
            print(f"    ✅ Final probabilities are close")
        elif final_ratio < 0.1:
            print(f"    ❌ Model probability much too low")
        elif final_ratio > 10:
            print(f"    ❌ Model probability much too high")
        else:
            print(f"    ⚠️  Moderate difference in final probabilities")
    
    def analyze_overall_patterns(self, all_detailed_logs: List[Dict]):
        """Analyze overall probability patterns across all timesteps"""
        if not all_detailed_logs:
            return
        
        print(f"\n{'='*80}")
        print(f"OVERALL PROBABILITY PATTERN ANALYSIS")
        print(f"{'='*80}")
        
        # Group by timestep and decision type
        by_time_decision = {}
        for entry in all_detailed_logs:
            key = (entry['timestep'], entry['decision_type'])
            if key not in by_time_decision:
                by_time_decision[key] = []
            by_time_decision[key].append(entry)
        
        for (t, decision_k), entries in sorted(by_time_decision.items()):
            decision_name = ['repair', 'vacancy', 'sales'][decision_k]
            print(f"\n📊 t={t}, {decision_name.upper()}:")
            print("-" * 50)
            
            if len(entries) == 0:
                continue
            
            # Self-activation statistics
            self_probs = [e['self_activation_prob'] for e in entries]
            print(f"Self-activation probabilities:")
            print(f"  Range: [{min(self_probs):.6f}, {max(self_probs):.6f}]")
            print(f"  Mean: {np.mean(self_probs):.6f}")
            print(f"  Households with self_prob > 0.01: {sum(p > 0.01 for p in self_probs)}/{len(self_probs)}")
            
            # Social influence statistics
            social_terms = [e['social_influence_term'] for e in entries]
            neighbor_counts = [e['active_neighbors'] for e in entries]
            
            print(f"Social influence:")
            print(f"  Social influence range: [{min(social_terms):.6f}, {max(social_terms):.6f}]")
            print(f"  Mean social influence: {np.mean(social_terms):.6f}")
            print(f"  Active neighbors range: [{min(neighbor_counts)}, {max(neighbor_counts)}]")
            print(f"  Mean active neighbors: {np.mean(neighbor_counts):.1f}")
            
            # Final probability statistics
            final_probs = [e['final_activation_prob'] for e in entries]
            print(f"Final activation probabilities:")
            print(f"  Range: [{min(final_probs):.6f}, {max(final_probs):.6f}]")
            print(f"  Mean: {np.mean(final_probs):.6f}")
            print(f"  Would activate with threshold 0.5: {sum(p > 0.5 for p in final_probs)}/{len(final_probs)}")
            print(f"  Would activate with threshold 0.1: {sum(p > 0.1 for p in final_probs)}/{len(final_probs)}")
            print(f"  Would activate with threshold 0.01: {sum(p > 0.01 for p in final_probs)}/{len(final_probs)}")