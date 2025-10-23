"""
Diagnostic tool to analyze self-activation vs neighbor influence probabilities.
"""

import torch
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple


class ProbabilityDiagnostic:
    """Track and analyze probability distributions during evaluation."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracking data."""
        self.timestep_data = defaultdict(lambda: {
            'self_probs': [],
            'neighbor_influence_probs': [],  # 1 - current_neighbor_influence_probs
            'lingering_influence_probs': [],  # 1 - lingering_influence_probs
            'final_probs': [],
            'decision_types': [],
            'switched': [],  # Whether the household actually switched
            'num_active_neighbors': [],
            'num_lingering_links': []
        })
        
    def record_probabilities(self, timestep: int, decision_type: int,
                           p_self: torch.Tensor,
                           current_neighbor_product: torch.Tensor,
                           lingering_product: torch.Tensor,
                           final_probs: torch.Tensor,
                           switched: torch.Tensor = None,
                           num_active_neighbors: torch.Tensor = None,
                           num_lingering: torch.Tensor = None):
        """
        Record probability components for analysis.
        
        Args:
            timestep: Current timestep
            decision_type: Decision type (0=vacant, 1=repair, 2=sell)
            p_self: Self-activation probabilities [batch_size]
            current_neighbor_product: Product term (1-p_neighbor) [batch_size]
            lingering_product: Product term (1-p_lingering) [batch_size]
            final_probs: Final activation probabilities [batch_size]
            switched: Whether households actually switched [batch_size]
            num_active_neighbors: Number of active neighbors [batch_size]
            num_lingering: Number of lingering links [batch_size]
        """
        data = self.timestep_data[timestep]
        
        # Convert to numpy for easier analysis
        data['self_probs'].extend(p_self.detach().cpu().numpy().tolist())
        
        # Convert products to influence probabilities: p_influence = 1 - product
        neighbor_influence = 1 - current_neighbor_product.detach().cpu().numpy()
        lingering_influence = 1 - lingering_product.detach().cpu().numpy()
        
        data['neighbor_influence_probs'].extend(neighbor_influence.tolist())
        data['lingering_influence_probs'].extend(lingering_influence.tolist())
        data['final_probs'].extend(final_probs.detach().cpu().numpy().tolist())
        data['decision_types'].extend([decision_type] * len(p_self))
        
        if switched is not None:
            data['switched'].extend(switched.detach().cpu().numpy().tolist())
        
        if num_active_neighbors is not None:
            data['num_active_neighbors'].extend(num_active_neighbors.detach().cpu().numpy().tolist())
        
        if num_lingering is not None:
            data['num_lingering'].extend(num_lingering.detach().cpu().numpy().tolist())
    
    def get_timestep_summary(self, timestep: int) -> Dict:
        """Get summary statistics for a specific timestep."""
        if timestep not in self.timestep_data:
            return None
        
        data = self.timestep_data[timestep]
        
        summary = {}
        for decision_type in range(3):
            # Filter data for this decision type
            mask = np.array(data['decision_types']) == decision_type
            if not mask.any():
                continue
            
            self_probs = np.array(data['self_probs'])[mask]
            neighbor_probs = np.array(data['neighbor_influence_probs'])[mask]
            lingering_probs = np.array(data['lingering_influence_probs'])[mask]
            final_probs = np.array(data['final_probs'])[mask]
            
            decision_name = ['Vacant', 'Repair', 'Sell'][decision_type]
            summary[decision_name] = {
                'count': len(self_probs),
                'self': {
                    'mean': float(np.mean(self_probs)),
                    'median': float(np.median(self_probs)),
                    'std': float(np.std(self_probs)),
                    'min': float(np.min(self_probs)),
                    'max': float(np.max(self_probs)),
                },
                'neighbor_influence': {
                    'mean': float(np.mean(neighbor_probs)),
                    'median': float(np.median(neighbor_probs)),
                    'std': float(np.std(neighbor_probs)),
                    'min': float(np.min(neighbor_probs)),
                    'max': float(np.max(neighbor_probs)),
                },
                'lingering_influence': {
                    'mean': float(np.mean(lingering_probs)),
                    'median': float(np.median(lingering_probs)),
                    'std': float(np.std(lingering_probs)),
                    'min': float(np.min(lingering_probs)),
                    'max': float(np.max(lingering_probs)),
                },
                'final': {
                    'mean': float(np.mean(final_probs)),
                    'median': float(np.median(final_probs)),
                    'std': float(np.std(final_probs)),
                },
            }
            
            # Add contribution analysis
            # Which component contributes more to final probability?
            # Use a simple heuristic: relative magnitude
            summary[decision_name]['contribution_ratio'] = {
                'self_vs_neighbor': float(np.mean(self_probs) / (np.mean(neighbor_probs) + 1e-8)),
                'self_vs_lingering': float(np.mean(self_probs) / (np.mean(lingering_probs) + 1e-8)),
            }
            
            # Add neighbor and lingering counts if available
            if data.get('num_active_neighbors') and len(data['num_active_neighbors']) > 0:
                num_neighbors = np.array(data['num_active_neighbors'])[mask]
                summary[decision_name]['avg_active_neighbors'] = float(np.mean(num_neighbors))
            
            if data.get('num_lingering') and len(data['num_lingering']) > 0:
                num_linger = np.array(data['num_lingering'])[mask]
                summary[decision_name]['avg_lingering_links'] = float(np.mean(num_linger))
        
        return summary
    
    def print_timestep_summary(self, timestep: int, period: str = ""):
        """Print formatted summary for a timestep."""
        summary = self.get_timestep_summary(timestep)
        if summary is None:
            print(f"No data for timestep {timestep}")
            return
        
        print(f"\n{'='*80}")
        print(f"📊 PROBABILITY DIAGNOSTIC - {period} Step {timestep}")
        print(f"{'='*80}")
        
        for decision_name, stats in summary.items():
            print(f"\n{decision_name} (n={stats['count']}):")
            print(f"  Self-activation:")
            print(f"    Mean: {stats['self']['mean']:.4f}, Median: {stats['self']['median']:.4f}")
            print(f"    Range: [{stats['self']['min']:.4f}, {stats['self']['max']:.4f}]")
            
            print(f"  Neighbor Influence:")
            print(f"    Mean: {stats['neighbor_influence']['mean']:.4f}, Median: {stats['neighbor_influence']['median']:.4f}")
            print(f"    Range: [{stats['neighbor_influence']['min']:.4f}, {stats['neighbor_influence']['max']:.4f}]")
            
            if 'avg_active_neighbors' in stats:
                print(f"    Avg active neighbors: {stats['avg_active_neighbors']:.2f}")
            
            print(f"  Lingering Influence:")
            print(f"    Mean: {stats['lingering_influence']['mean']:.4f}, Median: {stats['lingering_influence']['median']:.4f}")
            
            if 'avg_lingering_links' in stats:
                print(f"    Avg lingering links: {stats['avg_lingering_links']:.2f}")
            
            print(f"  Final Probability:")
            print(f"    Mean: {stats['final']['mean']:.4f}, Median: {stats['final']['median']:.4f}")
            
            print(f"  Contribution Analysis:")
            print(f"    Self/Neighbor ratio: {stats['contribution_ratio']['self_vs_neighbor']:.2f}x")
            print(f"    Self/Lingering ratio: {stats['contribution_ratio']['self_vs_lingering']:.2f}x")
            
            # Interpretation
            if stats['contribution_ratio']['self_vs_neighbor'] > 5:
                print(f"    ⚠️  Self-activation dominates (>{stats['contribution_ratio']['self_vs_neighbor']:.1f}x stronger than neighbors)")
            elif stats['contribution_ratio']['self_vs_neighbor'] < 0.2:
                print(f"    ℹ️  Neighbor influence dominates")
            else:
                print(f"    ✓ Balanced contribution")
    
    def get_period_summary(self, start_time: int, end_time: int) -> Dict:
        """Get aggregated summary across a time period."""
        period_data = {
            'Vacant': {'self': [], 'neighbor': [], 'lingering': [], 'final': []},
            'Repair': {'self': [], 'neighbor': [], 'lingering': [], 'final': []},
            'Sell': {'self': [], 'neighbor': [], 'lingering': [], 'final': []},
        }
        
        for t in range(start_time, end_time + 1):
            if t not in self.timestep_data:
                continue
            
            data = self.timestep_data[t]
            for decision_type in range(3):
                mask = np.array(data['decision_types']) == decision_type
                if not mask.any():
                    continue
                
                decision_name = ['Vacant', 'Repair', 'Sell'][decision_type]
                period_data[decision_name]['self'].extend(np.array(data['self_probs'])[mask].tolist())
                period_data[decision_name]['neighbor'].extend(np.array(data['neighbor_influence_probs'])[mask].tolist())
                period_data[decision_name]['lingering'].extend(np.array(data['lingering_influence_probs'])[mask].tolist())
                period_data[decision_name]['final'].extend(np.array(data['final_probs'])[mask].tolist())
        
        # Compute statistics
        summary = {}
        for decision_name, data in period_data.items():
            if len(data['self']) == 0:
                continue
            
            summary[decision_name] = {
                'count': len(data['self']),
                'self_mean': float(np.mean(data['self'])),
                'neighbor_mean': float(np.mean(data['neighbor'])),
                'lingering_mean': float(np.mean(data['lingering'])),
                'final_mean': float(np.mean(data['final'])),
                'self_vs_neighbor_ratio': float(np.mean(data['self']) / (np.mean(data['neighbor']) + 1e-8)),
            }
        
        return summary
    
    def print_period_summary(self, start_time: int, end_time: int, period_name: str = ""):
        """Print aggregated summary for a period."""
        summary = self.get_period_summary(start_time, end_time)
        
        print(f"\n{'='*80}")
        print(f"📈 PERIOD SUMMARY: {period_name} (t={start_time} to {end_time})")
        print(f"{'='*80}")
        
        for decision_name, stats in summary.items():
            print(f"\n{decision_name} (total samples: {stats['count']}):")
            print(f"  Average Self-activation:      {stats['self_mean']:.4f}")
            print(f"  Average Neighbor Influence:   {stats['neighbor_mean']:.4f}")
            print(f"  Average Lingering Influence:  {stats['lingering_mean']:.4f}")
            print(f"  Average Final Probability:    {stats['final_mean']:.4f}")
            print(f"  Self/Neighbor Ratio:          {stats['self_vs_neighbor_ratio']:.2f}x")
            
            if stats['self_vs_neighbor_ratio'] > 5:
                print(f"  ⚠️  WARNING: Self-activation dominates ({stats['self_vs_neighbor_ratio']:.1f}x)")
            elif stats['self_vs_neighbor_ratio'] > 2:
                print(f"  ⚠️  Self-activation is stronger ({stats['self_vs_neighbor_ratio']:.1f}x)")
            else:
                print(f"  ✓ Reasonable balance")
