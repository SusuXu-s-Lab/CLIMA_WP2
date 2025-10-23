import pdb
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
from collections import defaultdict
from matplotlib.animation import FuncAnimation
import warnings
import json
from config import ALL_SCENARIOS, ScenarioConfig
from typing import List, Dict, Any

warnings.filterwarnings('ignore')

class ScenarioVisualizer:
    """Visualizer for different scenario datasets"""
    
    def __init__(self):
        # Get the parent directory of current script, then go to 'datasets' folder
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        self.base_data_path = os.path.join(parent_dir, 'dataset')
        
        # Create base datasets directory if it doesn't exist
        os.makedirs(self.base_data_path, exist_ok=True)
    
    def get_scenario_data_path(self, scenario_name: str) -> str:
        """Get the data path for a specific scenario"""
        return os.path.join(self.base_data_path, scenario_name)
    
    def get_scenario_output_dir(self, scenario_name: str) -> str:
        """Get the output directory for a specific scenario (same as data path)"""
        scenario_path = self.get_scenario_data_path(scenario_name)
        return scenario_path
    
    def load_scenario_data(self, scenario_name: str) -> Dict[str, pd.DataFrame]:
        """Load all data files for a scenario"""
        data_path = self.get_scenario_data_path(scenario_name)
        
        try:
            data = {
                'states': pd.read_csv(os.path.join(data_path, 'household_states_raw.csv')),
                'ground_truth_network': pd.read_csv(os.path.join(data_path, 'ground_truth_network_raw.csv')),
                'observed_network': pd.read_csv(os.path.join(data_path, 'observed_network_raw.csv')),
                'locations': pd.read_csv(os.path.join(data_path, 'household_locations_raw.csv')),
                'features': pd.read_csv(os.path.join(data_path, 'household_features_raw.csv')),
                'similarity': pd.read_csv(os.path.join(data_path, 'similarity_df_raw.csv'), index_col=0)
            }
            
            # Load statistics if available
            stats_path = os.path.join(data_path, 'statistics.json')
            if os.path.exists(stats_path):
                with open(stats_path, 'r') as f:
                    data['statistics'] = json.load(f)
            
            return data
        except FileNotFoundError as e:
            print(f"Error loading data for {scenario_name}: {e}")
            return None
    
    def visualize_states_single_scenario(self, scenario_name: str):
        """Create state visualization for a single scenario"""
        data = self.load_scenario_data(scenario_name)
        if data is None:
            return
        
        df = data['states']
        df['time'] = df['time'].astype(int)
        
        # Aggregate: count households in state=1 at each timestep
        agg = df.groupby('time')[['repair_state', 'vacancy_state', 'sales_state']].sum().reset_index()
        
        # Normalize to percentage
        total_households = df['home'].nunique()
        agg_percent = agg.copy()
        agg_percent[['repair_state', 'vacancy_state', 'sales_state']] = agg_percent[
            ['repair_state', 'vacancy_state', 'sales_state']] / total_households * 100
        
        # Melt to long format for plotting
        agg_melted = agg_percent.melt(id_vars='time', var_name='State', value_name='Percentage')
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.lineplot(data=agg_melted, x='time', y='Percentage', hue='State', marker="o", linewidth=2)
        plt.title(f"Activation Percentage Over Time - {scenario_name}", fontsize=14, fontweight='bold')
        plt.ylabel("Percentage of Active Households (%)")
        plt.xlabel("Time Step")
        plt.grid(True, alpha=0.3)
        plt.legend(title='State Type')
        plt.tight_layout()
        
        # Save plot in scenario's own folder
        output_dir = self.get_scenario_output_dir(scenario_name)
        output_path = os.path.join(output_dir, 'states_evolution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        print(f"States visualization saved: {output_path}")
    
    def visualize_links_single_scenario(self, scenario_name: str):
        """Create network links visualization for a single scenario"""
        data = self.load_scenario_data(scenario_name)
        if data is None:
            return
        
        df = data['ground_truth_network']
        
        plt.figure(figsize=(12, 8))
        
        # Count links over time by type
        link_counts_time = df.groupby(['time_step', 'link_type']).size().unstack(fill_value=0)
        
        for col in link_counts_time.columns:
            link_type_name = 'Bonding' if col == 1 else 'Bridging' if col == 2 else f'Type {col}'
            plt.plot(link_counts_time.index, link_counts_time[col], 
                    marker='o', label=link_type_name, linewidth=2, markersize=6)
        
        plt.title(f'Number of Links Over Time - {scenario_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Time Step')
        plt.ylabel('Number of Links')
        plt.legend(title='Link Type')
        plt.grid(True, alpha=0.3)
        
        # Save plot in scenario's own folder
        output_dir = self.get_scenario_output_dir(scenario_name)
        output_path = os.path.join(output_dir, 'links_evolution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        print(f"Links visualization saved: {output_path}")
    
    def visualize_similarity_single_scenario(self, scenario_name: str):
        """Create similarity distribution visualization for a single scenario"""
        data = self.load_scenario_data(scenario_name)
        if data is None:
            return
        
        similarity_df = data['similarity']
        similarity_matrix = similarity_df.values
        
        # Remove diagonal
        np.fill_diagonal(similarity_matrix, np.nan)
        values = similarity_matrix.flatten()
        values = values[~np.isnan(values)]
        
        plt.figure(figsize=(10, 6))
        plt.hist(values, bins=40, color='salmon', edgecolor='black', alpha=0.7)
        plt.title(f"Similarity Distribution - {scenario_name}", fontsize=14, fontweight='bold')
        plt.xlabel("Similarity Value")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_sim = np.mean(values)
        std_sim = np.std(values)
        plt.axvline(mean_sim, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_sim:.3f}')
        plt.legend()
        
        # Save plot in scenario's own folder
        output_dir = self.get_scenario_output_dir(scenario_name)
        output_path = os.path.join(output_dir, 'similarity_distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        print(f"Similarity visualization saved: {output_path}")
    
    def visualize_new_adopters_single_scenario(self, scenario_name: str):
        """Create visualization showing new decision adopters at each timestep"""
        data = self.load_scenario_data(scenario_name)
        if data is None:
            return
        
        df = data['states']
        df['time'] = df['time'].astype(int)
        
        # Calculate new adopters at each timestep for each decision type
        new_adopters = {'time': [], 'repair': [], 'vacancy': [], 'sales': []}
        
        # Get unique households and times
        households = df['home'].unique()
        times = sorted(df['time'].unique())
        
        for t in times:
            if t == 0:
                # At t=0, all active states are "new adopters"
                df_t0 = df[df['time'] == 0]
                new_adopters['time'].append(t)
                new_adopters['repair'].append(int(df_t0['repair_state'].sum()))
                new_adopters['vacancy'].append(int(df_t0['vacancy_state'].sum()))
                new_adopters['sales'].append(int(df_t0['sales_state'].sum()))
            else:
                # For t>0, count households that switched from 0 to 1
                df_prev = df[df['time'] == t-1].set_index('home')
                df_curr = df[df['time'] == t].set_index('home')
                
                # New adopters = currently active AND was inactive at t-1
                repair_new = int(((df_curr['repair_state'] == 1) & (df_prev['repair_state'] == 0)).sum())
                vacancy_new = int(((df_curr['vacancy_state'] == 1) & (df_prev['vacancy_state'] == 0)).sum())
                sales_new = int(((df_curr['sales_state'] == 1) & (df_prev['sales_state'] == 0)).sum())
                
                new_adopters['time'].append(t)
                new_adopters['repair'].append(repair_new)
                new_adopters['vacancy'].append(vacancy_new)
                new_adopters['sales'].append(sales_new)
        
        # Create DataFrame for plotting
        adopters_df = pd.DataFrame(new_adopters)
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.plot(adopters_df['time'], adopters_df['repair'], 'o-', label='Repair', linewidth=2, markersize=6)
        plt.plot(adopters_df['time'], adopters_df['vacancy'], 's-', label='Vacancy', linewidth=2, markersize=6)
        plt.plot(adopters_df['time'], adopters_df['sales'], '^-', label='Sales', linewidth=2, markersize=6)
        
        plt.title(f"New Decision Adopters per Timestep - {scenario_name}", fontsize=14, fontweight='bold')
        plt.xlabel("Time Step")
        plt.ylabel("Number of New Adopters")
        plt.legend(title='Decision Type')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot in scenario's own folder
        output_dir = self.get_scenario_output_dir(scenario_name)
        output_path = os.path.join(output_dir, 'new_adopters_evolution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        print(f"New adopters visualization saved: {output_path}")
    
    def compare_scenarios_states(self, scenario_names: List[str] = None):
        """Compare state evolution across multiple scenarios"""
        if scenario_names is None:
            scenario_names = [s.name for s in ALL_SCENARIOS]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, scenario_name in enumerate(scenario_names):
            if idx >= 6:  # Only plot first 6 scenarios
                break
                
            data = self.load_scenario_data(scenario_name)
            if data is None:
                continue
            
            df = data['states']
            df['time'] = df['time'].astype(int)
            
            # Calculate percentages
            agg = df.groupby('time')[['repair_state', 'vacancy_state', 'sales_state']].sum().reset_index()
            total_households = df['home'].nunique()
            agg_percent = agg.copy()
            agg_percent[['repair_state', 'vacancy_state', 'sales_state']] = agg_percent[
                ['repair_state', 'vacancy_state', 'sales_state']] / total_households * 100
            
            # Plot on subplot
            ax = axes[idx]
            ax.plot(agg_percent['time'], agg_percent['repair_state'], 'o-', label='Repair', linewidth=2)
            ax.plot(agg_percent['time'], agg_percent['vacancy_state'], 's-', label='Vacancy', linewidth=2)
            ax.plot(agg_percent['time'], agg_percent['sales_state'], '^-', label='Sales', linewidth=2)
            
            ax.set_title(scenario_name, fontweight='bold')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Percentage (%)')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.suptitle('State Evolution Comparison Across Scenarios', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save comparison plot in base datasets folder
        output_path = os.path.join(self.base_data_path, 'states_comparison_all_scenarios.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        print(f"States comparison saved: {output_path}")
    
    def compare_scenarios_networks(self, scenario_names: List[str] = None):
        """Compare network evolution across multiple scenarios"""
        if scenario_names is None:
            scenario_names = [s.name for s in ALL_SCENARIOS]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, scenario_name in enumerate(scenario_names):
            if idx >= 6:
                break
                
            data = self.load_scenario_data(scenario_name)
            if data is None:
                continue
            
            df = data['ground_truth_network']
            
            # Calculate total links over time
            link_counts = df.groupby('time_step').size()
            bonding_counts = df[df['link_type'] == 1].groupby('time_step').size()
            bridging_counts = df[df['link_type'] == 2].groupby('time_step').size()
            
            # Plot on subplot
            ax = axes[idx]
            ax.plot(link_counts.index, link_counts.values, 'o-', label='Total Links', linewidth=2)
            ax.plot(bonding_counts.index, bonding_counts.values, 's-', label='Bonding', linewidth=2)
            ax.plot(bridging_counts.index, bridging_counts.values, '^-', label='Bridging', linewidth=2)
            
            ax.set_title(scenario_name, fontweight='bold')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Number of Links')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.suptitle('Network Evolution Comparison Across Scenarios', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save comparison plot in base datasets folder
        output_path = os.path.join(self.base_data_path, 'networks_comparison_all_scenarios.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        print(f"Networks comparison saved: {output_path}")
    
    def generate_all_visualizations(self, scenario_names: List[str] = None):
        """Generate all visualizations for specified scenarios"""
        if scenario_names is None:
            scenario_names = [s.name for s in ALL_SCENARIOS]
        
        print("Generating visualizations for all scenarios...")
        
        # Generate individual scenario visualizations
        for scenario_name in scenario_names:
            print(f"\nGenerating visualizations for {scenario_name}...")
            self.visualize_states_single_scenario(scenario_name)
            self.visualize_links_single_scenario(scenario_name)
            self.visualize_similarity_single_scenario(scenario_name)
            self.visualize_new_adopters_single_scenario(scenario_name)
        
        print(f"\nAll visualizations completed!")

# Usage functions
def visualize_single_scenario(scenario_name: str):
    """Convenience function to visualize a single scenario"""
    visualizer = ScenarioVisualizer()
    visualizer.visualize_states_single_scenario(scenario_name)
    visualizer.visualize_links_single_scenario(scenario_name)
    visualizer.visualize_similarity_single_scenario(scenario_name)
    visualizer.visualize_new_adopters_single_scenario(scenario_name)

def visualize_all_scenarios():
    """Convenience function to generate all visualizations"""
    visualizer = ScenarioVisualizer()
    visualizer.generate_all_visualizations()

def compare_specific_scenarios(scenario_names: List[str]):
    """Convenience function to compare specific scenarios"""
    visualizer = ScenarioVisualizer()
    visualizer.compare_scenarios_states(scenario_names)
    visualizer.compare_scenarios_networks(scenario_names)

if __name__ == "__main__":
    # Example usage:
    
    # Option 1: Generate all visualizations for all scenarios
    visualize_all_scenarios()
    
    # Option 2: Visualize specific scenarios
    # visualize_single_scenario("G1_Sparse_LowSeed")
    # compare_specific_scenarios(["G1_Sparse_LowSeed", "G2_Sparse_HighSeed", "G3_Medium_LowSeed"])