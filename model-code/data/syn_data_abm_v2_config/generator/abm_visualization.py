# abm_visualization.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import warnings
from typing import List, Dict, Any
from abm_config import ALL_ABM_SCENARIOS
import pickle

warnings.filterwarnings('ignore')

class ABMScenarioVisualizer:
    """Visualizer for ABM scenario datasets"""
    
    def __init__(self):
        # Get parent directory for datasets
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        self.base_data_path = os.path.join(parent_dir, 'dataset')
        
        os.makedirs(self.base_data_path, exist_ok=True)
    
    def get_scenario_data_path(self, scenario_name: str) -> str:
        """Get data path for a specific scenario"""
        return os.path.join(self.base_data_path, scenario_name)
    
    def load_scenario_data(self, scenario_name: str) -> Dict[str, pd.DataFrame]:
        """Load all data files for a scenario"""
        data_path = self.get_scenario_data_path(scenario_name)
        
        try:
            data = {
                'states': pd.read_csv(os.path.join(data_path, 'household_states.csv')),
                'ground_truth_network': pd.read_csv(os.path.join(data_path, 'ground_truth_network.csv')),
                'observed_network': pd.read_csv(os.path.join(data_path, 'observed_network.csv')),
                'locations': pd.read_csv(os.path.join(data_path, 'household_locations.csv')),
                'features': pd.read_csv(os.path.join(data_path, 'household_features.csv')),
                'similarity': pd.read_csv(os.path.join(data_path, 'similarity_matrix.csv'), index_col=0)
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
        
        # Aggregate: count households in each state at each timestep
        agg = df.groupby('timestep').apply(
            lambda x: pd.Series({
                'repair': (x['repair'] == 1).sum(),
                'vacant': (x['vacant'] == 1).sum(),
                'sell': (x['sell'] == 1).sum()
            })
        ).reset_index()
        
        # Normalize to percentage
        total_households = df['household_id'].nunique()
        agg_percent = agg.copy()
        agg_percent[['repair', 'vacant', 'sell']] = (
            agg_percent[['repair', 'vacant', 'sell']] / total_households * 100
        )
        
        # Melt to long format
        agg_melted = agg_percent.melt(id_vars='timestep', var_name='Decision', value_name='Percentage')
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.lineplot(data=agg_melted, x='timestep', y='Percentage', 
                    hue='Decision', marker="o", linewidth=2)
        plt.title(f"Decision Activation Over Time - {scenario_name}", 
                 fontsize=14, fontweight='bold')
        plt.ylabel("Percentage of Active Households (%)")
        plt.xlabel("Time Step")
        plt.grid(True, alpha=0.3)
        plt.legend(title='Decision Type')
        plt.tight_layout()
        
        # Save to scenario folder
        output_dir = self.get_scenario_data_path(scenario_name)
        output_path = os.path.join(output_dir, 'states_evolution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
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
        link_counts_time = df.groupby(['timestep', 'link_type']).size().unstack(fill_value=0)
        
        for col in link_counts_time.columns:
            link_type_name = 'Bonding' if col == 1 else 'Bridging' if col == 2 else f'Type {col}'
            plt.plot(link_counts_time.index, link_counts_time[col], 
                    marker='o', label=link_type_name, linewidth=2, markersize=4)
        
        plt.title(f'Number of Links Over Time - {scenario_name}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Time Step')
        plt.ylabel('Number of Links')
        plt.legend(title='Link Type')
        plt.grid(True, alpha=0.3)
        
        # Save to scenario folder
        output_dir = self.get_scenario_data_path(scenario_name)
        output_path = os.path.join(output_dir, 'links_evolution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Links visualization saved: {output_path}")
    
    def visualize_new_adopters_single_scenario(self, scenario_name: str):
        """Create visualization showing new decision adopters at each timestep for ABM data"""
        data = self.load_scenario_data(scenario_name)
        if data is None:
            return
        
        df = data['states']  # ABM format: columns are 'household_id', 'timestep', 'repair', 'vacant', 'sell'
        
        # Sort to ensure proper ordering
        df = df.sort_values(['household_id', 'timestep']).reset_index(drop=True)
        
        # Get all timesteps
        timesteps = sorted(df['timestep'].unique())
        
        # Initialize results
        new_adopters = {
            'timestep': [],
            'repair': [],
            'vacant': [],
            'sell': []
        }
        
        for t in timesteps:
            new_adopters['timestep'].append(t)
            
            if t == 0:
                # At t=0, all active households are "new adopters"
                df_t0 = df[df['timestep'] == 0]
                new_adopters['repair'].append(int(df_t0['repair'].sum()))
                new_adopters['vacant'].append(int(df_t0['vacant'].sum()))
                new_adopters['sell'].append(int(df_t0['sell'].sum()))
            else:
                # For t>0, count transitions from 0 to 1
                df_prev = df[df['timestep'] == t-1].set_index('household_id')
                df_curr = df[df['timestep'] == t].set_index('household_id')
                
                # Ensure both dataframes have same households
                common_ids = df_prev.index.intersection(df_curr.index)
                df_prev = df_prev.loc[common_ids]
                df_curr = df_curr.loc[common_ids]
                
                # Count new adopters: currently 1 AND previously 0
                repair_new = int(((df_curr['repair'] == 1) & (df_prev['repair'] == 0)).sum())
                vacant_new = int(((df_curr['vacant'] == 1) & (df_prev['vacant'] == 0)).sum())
                sell_new = int(((df_curr['sell'] == 1) & (df_prev['sell'] == 0)).sum())
                
                new_adopters['repair'].append(repair_new)
                new_adopters['vacant'].append(vacant_new)
                new_adopters['sell'].append(sell_new)
        
        # Create DataFrame for plotting
        adopters_df = pd.DataFrame(new_adopters)
        
        # Plot with smaller markers
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(adopters_df['timestep'], adopters_df['repair'], 'o-', 
               label='Repair', linewidth=2, markersize=4, color='#1f77b4')
        ax.plot(adopters_df['timestep'], adopters_df['vacant'], 's-', 
               label='Vacant', linewidth=2, markersize=4, color='#ff7f0e')
        ax.plot(adopters_df['timestep'], adopters_df['sell'], '^-', 
               label='Sell', linewidth=2, markersize=4, color='#2ca02c')
        
        ax.set_title(f"New Decision Adopters per Timestep - {scenario_name}", 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel("Time Step", fontsize=12)
        ax.set_ylabel("Number of New Adopters", fontsize=12)
        ax.legend(title='Decision Type', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Set integer ticks for both axes
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        plt.tight_layout()
        
        # Save to scenario folder
        output_dir = self.get_scenario_data_path(scenario_name)
        output_path = os.path.join(output_dir, 'new_adopters_evolution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"New adopters visualization saved: {output_path}")
    
    def compare_scenarios_states(self, scenario_names: List[str] = None):
        """Compare state evolution across multiple scenarios"""
        if scenario_names is None:
            scenario_names = [s.name for s in ALL_ABM_SCENARIOS]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, scenario_name in enumerate(scenario_names):
            if idx >= 6:
                break
                
            data = self.load_scenario_data(scenario_name)
            if data is None:
                continue
            
            df = data['states']
            
            # Calculate percentages
            agg = df.groupby('timestep')[['repair', 'vacant', 'sell']].sum().reset_index()
            total_households = df['household_id'].nunique()
            agg_percent = agg.copy()
            agg_percent[['repair', 'vacant', 'sell']] = (
                agg_percent[['repair', 'vacant', 'sell']] / total_households * 100
            )
            
            # Plot on subplot
            ax = axes[idx]
            ax.plot(agg_percent['timestep'], agg_percent['repair'], 
                   'o-', label='Repair', linewidth=2, markersize=4)
            ax.plot(agg_percent['timestep'], agg_percent['vacant'], 
                   's-', label='Vacant', linewidth=2, markersize=4)
            ax.plot(agg_percent['timestep'], agg_percent['sell'], 
                   '^-', label='Sell', linewidth=2, markersize=4)
            
            ax.set_title(scenario_name, fontweight='bold')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Percentage (%)')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.suptitle('State Evolution Comparison Across ABM Scenarios', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save comparison plot
        output_path = os.path.join(self.base_data_path, 
                                   'states_comparison_all_abm_scenarios.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"States comparison saved: {output_path}")
    
    def compare_scenarios_networks(self, scenario_names: List[str] = None):
        """Compare network evolution across multiple scenarios"""
        if scenario_names is None:
            scenario_names = [s.name for s in ALL_ABM_SCENARIOS]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, scenario_name in enumerate(scenario_names):
            if idx >= 6:
                break
                
            data = self.load_scenario_data(scenario_name)
            if data is None:
                continue
            
            df = data['ground_truth_network']
            
            # Calculate link counts
            link_counts = df.groupby('timestep').size()
            bonding_counts = df[df['link_type'] == 1].groupby('timestep').size()
            bridging_counts = df[df['link_type'] == 2].groupby('timestep').size()
            
            # Plot on subplot
            ax = axes[idx]
            ax.plot(link_counts.index, link_counts.values, 
                   'o-', label='Total Links', linewidth=2, markersize=4)
            ax.plot(bonding_counts.index, bonding_counts.values, 
                   's-', label='Bonding', linewidth=2, markersize=4)
            ax.plot(bridging_counts.index, bridging_counts.values, 
                   '^-', label='Bridging', linewidth=2, markersize=4)
            
            ax.set_title(scenario_name, fontweight='bold')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Number of Links')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.suptitle('Network Evolution Comparison Across ABM Scenarios', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save comparison plot
        output_path = os.path.join(self.base_data_path, 
                                   'networks_comparison_all_abm_scenarios.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Networks comparison saved: {output_path}")
    
    def create_statistics_summary_plot(self):
        """Create visualization of key statistics across all scenarios"""
        summary_path = os.path.join(self.base_data_path, 'dataset_summary.xlsx')
        if not os.path.exists(summary_path):
            print(f"Summary file not found: {summary_path}")
            return
        
        summary_df = pd.read_excel(summary_path)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        x = range(len(summary_df))
        width = 0.25
        
        # Plot 1: Initial Decision Percentages
        ax1 = axes[0, 0]
        ax1.bar([i - width for i in x], summary_df['repair_pct_t0'], 
               width, label='Repair', alpha=0.7)
        ax1.bar(x, summary_df['vacant_pct_t0'], width, label='Vacant', alpha=0.7)
        ax1.bar([i + width for i in x], summary_df['sell_pct_t0'], 
               width, label='Sell', alpha=0.7)
        ax1.set_title('Initial Decision Percentages (t=0)')
        ax1.set_ylabel('Percentage (%)')
        ax1.set_xticks(x)
        ax1.set_xticklabels([s.split('_')[1] for s in summary_df['scenario_name']], 
                           rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Final Decision Percentages
        ax2 = axes[0, 1]
        ax2.bar([i - width for i in x], summary_df['repair_pct_tT'], 
               width, label='Repair', alpha=0.7)
        ax2.bar(x, summary_df['vacant_pct_tT'], width, label='Vacant', alpha=0.7)
        ax2.bar([i + width for i in x], summary_df['sell_pct_tT'], 
               width, label='Sell', alpha=0.7)
        ax2.set_title('Final Decision Percentages (t=T)')
        ax2.set_ylabel('Percentage (%)')
        ax2.set_xticks(x)
        ax2.set_xticklabels([s.split('_')[1] for s in summary_df['scenario_name']], 
                           rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Network Density (Average Degree)
        ax3 = axes[0, 2]
        ax3.bar([i - width/2 for i in x], summary_df['avg_degree_t0'], 
               width, label='t=0', alpha=0.7)
        ax3.bar([i + width/2 for i in x], summary_df['avg_degree_tT'], 
               width, label='t=T', alpha=0.7)
        ax3.set_title('Average Network Degree')
        ax3.set_ylabel('Average Degree')
        ax3.set_xticks(x)
        ax3.set_xticklabels([s.split('_')[1] for s in summary_df['scenario_name']], 
                           rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Bonding vs Bridging Links (t=0)
        ax4 = axes[1, 0]
        ax4.bar([i - width/2 for i in x], summary_df['bonding_links_t0'], 
               width, label='Bonding', alpha=0.7)
        ax4.bar([i + width/2 for i in x], summary_df['bridging_links_t0'], 
               width, label='Bridging', alpha=0.7)
        ax4.set_title('Link Types at t=0')
        ax4.set_ylabel('Number of Links')
        ax4.set_xticks(x)
        ax4.set_xticklabels([s.split('_')[1] for s in summary_df['scenario_name']], 
                           rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Link Evolution
        ax5 = axes[1, 1]
        bonding_change = summary_df['bonding_links_tT'] - summary_df['bonding_links_t0']
        bridging_change = summary_df['bridging_links_tT'] - summary_df['bridging_links_t0']
        ax5.bar([i - width/2 for i in x], bonding_change, 
               width, label='Bonding Change', alpha=0.7)
        ax5.bar([i + width/2 for i in x], bridging_change, 
               width, label='Bridging Change', alpha=0.7)
        ax5.set_title('Link Changes (t=T - t=0)')
        ax5.set_ylabel('Change in Number of Links')
        ax5.set_xticks(x)
        ax5.set_xticklabels([s.split('_')[1] for s in summary_df['scenario_name']], 
                           rotation=45)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Plot 6: Decision Growth
        ax6 = axes[1, 2]
        repair_growth = summary_df['repair_pct_tT'] - summary_df['repair_pct_t0']
        vacant_growth = summary_df['vacant_pct_tT'] - summary_df['vacant_pct_t0']
        sell_growth = summary_df['sell_pct_tT'] - summary_df['sell_pct_t0']
        ax6.bar([i - width for i in x], repair_growth, width, label='Repair', alpha=0.7)
        ax6.bar(x, vacant_growth, width, label='Vacant', alpha=0.7)
        ax6.bar([i + width for i in x], sell_growth, width, label='Sell', alpha=0.7)
        ax6.set_title('Decision Percentage Growth')
        ax6.set_ylabel('Percentage Point Change')
        ax6.set_xticks(x)
        ax6.set_xticklabels([s.split('_')[1] for s in summary_df['scenario_name']], 
                           rotation=45)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.suptitle('Summary Statistics Across All ABM Scenarios', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save summary plot
        output_path = os.path.join(self.base_data_path, 
                                   'summary_statistics_all_abm_scenarios.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Summary statistics plot saved: {output_path}")
    
    def generate_all_visualizations(self, scenario_names: List[str] = None):
        """Generate all visualizations for specified scenarios"""
        if scenario_names is None:
            scenario_names = [s.name for s in ALL_ABM_SCENARIOS]
        
        print("Generating visualizations for all ABM scenarios...")
        
        # Generate individual scenario visualizations
        for scenario_name in scenario_names:
            print(f"\nGenerating visualizations for {scenario_name}...")
            self.visualize_states_single_scenario(scenario_name)
            self.visualize_links_single_scenario(scenario_name)
            self.visualize_new_adopters_single_scenario(scenario_name)
        
        # Generate comparison visualizations
        print("\nGenerating comparison visualizations...")
        self.compare_scenarios_states(scenario_names)
        self.compare_scenarios_networks(scenario_names)
        
        # Generate summary statistics plot
        print("\nGenerating summary statistics plot...")
        self.create_statistics_summary_plot()
        
        print(f"\nAll visualizations completed! Check {self.base_data_path} for outputs.")

    def visualize_new_adopters_multiseed_average(self, scenario_name: str, n_runs: int = 10):
        """
        Create visualization showing AVERAGE new decision adopters across multiple runs
        with confidence intervals (mean ± 1 std)
        
        Args:
            scenario_name: Name of the scenario
            n_runs: Number of runs to average over (default: 10)
        """
        scenario_path = self.get_scenario_data_path(scenario_name)
        
        # Try loading from cache first (faster)
        cache_path = os.path.join(scenario_path, 'multiseed_states_cache.pkl')
        
        if os.path.exists(cache_path):
            print(f"Loading from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                all_states = pickle.load(f)
        else:
            # Fall back to loading from run directories
            print(f"Cache not found, loading from run directories...")
            all_states = []
            for i in range(n_runs):
                run_path = os.path.join(scenario_path, f'run_{i}', 'household_states.csv')
                if os.path.exists(run_path):
                    df = pd.read_csv(run_path)
                    all_states.append(df)
                else:
                    print(f"Warning: Run {i} data not found at {run_path}")
            
            if len(all_states) == 0:
                print(f"ERROR: No data found for scenario {scenario_name}")
                return
        
        print(f"Processing {len(all_states)} runs for {scenario_name}...")
        
        # Calculate new adopters for each run
        all_runs_data = {
            'repair': [],
            'vacant': [],
            'sell': []
        }
        
        for run_idx, df in enumerate(all_states):
            df = df.sort_values(['household_id', 'timestep']).reset_index(drop=True)
            timesteps = sorted(df['timestep'].unique())
            
            run_data = {
                'timestep': [],
                'repair': [],
                'vacant': [],
                'sell': []
            }
            
            for t in timesteps:
                run_data['timestep'].append(t)
                
                if t == 0:
                    # At t=0, all active households are "new adopters"
                    df_t0 = df[df['timestep'] == 0]
                    run_data['repair'].append(int(df_t0['repair'].sum()))
                    run_data['vacant'].append(int(df_t0['vacant'].sum()))
                    run_data['sell'].append(int(df_t0['sell'].sum()))
                else:
                    # For t>0, count transitions from 0 to 1
                    df_prev = df[df['timestep'] == t-1].set_index('household_id')
                    df_curr = df[df['timestep'] == t].set_index('household_id')
                    
                    # Ensure both dataframes have same households
                    common_ids = df_prev.index.intersection(df_curr.index)
                    df_prev = df_prev.loc[common_ids]
                    df_curr = df_curr.loc[common_ids]
                    
                    # Count new adopters
                    repair_new = int(((df_curr['repair'] == 1) & (df_prev['repair'] == 0)).sum())
                    vacant_new = int(((df_curr['vacant'] == 1) & (df_prev['vacant'] == 0)).sum())
                    sell_new = int(((df_curr['sell'] == 1) & (df_prev['sell'] == 0)).sum())
                    
                    run_data['repair'].append(repair_new)
                    run_data['vacant'].append(vacant_new)
                    run_data['sell'].append(sell_new)
            
            # Store this run's data
            all_runs_data['repair'].append(run_data['repair'])
            all_runs_data['vacant'].append(run_data['vacant'])
            all_runs_data['sell'].append(run_data['sell'])
        
        # Convert to numpy arrays for easier calculation
        timesteps = run_data['timestep']  # All runs have same timesteps
        
        repair_array = np.array(all_runs_data['repair'])  # shape: (n_runs, n_timesteps)
        vacant_array = np.array(all_runs_data['vacant'])
        sell_array = np.array(all_runs_data['sell'])
        
        # Calculate mean and std
        repair_mean = np.mean(repair_array, axis=0)
        repair_std = np.std(repair_array, axis=0)
        
        vacant_mean = np.mean(vacant_array, axis=0)
        vacant_std = np.std(vacant_array, axis=0)
        
        sell_mean = np.mean(sell_array, axis=0)
        sell_std = np.std(sell_array, axis=0)
        
        # Plot with confidence intervals
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Repair
        ax.plot(timesteps, repair_mean, 'o-', label='Repair (mean)', 
            linewidth=2, markersize=4, color='#1f77b4')
        ax.fill_between(timesteps, 
                        repair_mean - repair_std, 
                        repair_mean + repair_std,
                        alpha=0.2, color='#1f77b4', label='Repair (±1 std)')
        
        # Vacant
        ax.plot(timesteps, vacant_mean, 's-', label='Vacant (mean)', 
            linewidth=2, markersize=4, color='#ff7f0e')
        ax.fill_between(timesteps, 
                        vacant_mean - vacant_std, 
                        vacant_mean + vacant_std,
                        alpha=0.2, color='#ff7f0e', label='Vacant (±1 std)')
        
        # Sell
        ax.plot(timesteps, sell_mean, '^-', label='Sell (mean)', 
            linewidth=2, markersize=4, color='#2ca02c')
        ax.fill_between(timesteps, 
                        sell_mean - sell_std, 
                        sell_mean + sell_std,
                        alpha=0.2, color='#2ca02c', label='Sell (±1 std)')
        
        ax.set_title(f"Average New Decision Adopters per Timestep - {scenario_name}\n({len(all_states)} runs)", 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel("Time Step", fontsize=12)
        ax.set_ylabel("Number of New Adopters", fontsize=12)
        ax.legend(title='Decision Type', fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Set integer ticks for both axes
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        plt.tight_layout()
        
        # Save to scenario folder
        output_path = os.path.join(scenario_path, 'new_adopters_multiseed_average.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Multi-seed average visualization saved: {output_path}")

    def generate_all_multiseed_visualizations(self, scenario_names: list = None, n_runs: int = 10):
        """
        Generate multi-seed averaged visualizations for all scenarios
        
        Args:
            scenario_names: List of scenario names (default: all from config)
            n_runs: Number of runs per scenario
        """
        if scenario_names is None:
            from abm_config import ALL_ABM_SCENARIOS
            scenario_names = [s.name for s in ALL_ABM_SCENARIOS]
        
        print(f"\nGenerating multi-seed averaged visualizations for {len(scenario_names)} scenarios...")
        
        for scenario_name in scenario_names:
            print(f"\nProcessing {scenario_name}...")
            try:
                self.visualize_new_adopters_multiseed_average(scenario_name, n_runs=n_runs)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\nAll multi-seed visualizations completed!")


# Convenience functions
def visualize_single_abm_scenario(scenario_name: str):
    """Convenience function to visualize a single ABM scenario"""
    visualizer = ABMScenarioVisualizer()
    visualizer.visualize_states_single_scenario(scenario_name)
    visualizer.visualize_links_single_scenario(scenario_name)
    visualizer.visualize_new_adopters_single_scenario(scenario_name)

def visualize_all_abm_scenarios():
    """Convenience function to generate all ABM visualizations"""
    visualizer = ABMScenarioVisualizer()
    visualizer.generate_all_visualizations()

def compare_specific_abm_scenarios(scenario_names: List[str]):
    """Convenience function to compare specific ABM scenarios"""
    visualizer = ABMScenarioVisualizer()
    visualizer.compare_scenarios_states(scenario_names)
    visualizer.compare_scenarios_networks(scenario_names)

def visualize_multiseed_average_single_scenario(scenario_name: str, n_runs: int = 10):
    """Convenience function to visualize multi-seed average for a single scenario"""
    visualizer = ABMScenarioVisualizer()
    visualizer.visualize_new_adopters_multiseed_average(scenario_name, n_runs=n_runs)

def visualize_multiseed_average_all_scenarios(n_runs: int = 10):
    """Convenience function to generate multi-seed averaged visualizations for all scenarios"""
    visualizer = ABMScenarioVisualizer()
    visualizer.generate_all_multiseed_visualizations(n_runs=n_runs)


if __name__ == "__main__":
    # Option 1: Generate all visualizations for all scenarios
    visualize_all_abm_scenarios()
    # visualize_multiseed_average_single_scenario()
    # visualize_multiseed_average_all_scenarios()
    
    # Option 2: Visualize specific scenario (uncomment to use)
    # visualize_single_abm_scenario("ABM_Sparse_LowSeed_B")
    
    # Option 3: Compare specific scenarios (uncomment to use)
    # compare_specific_abm_scenarios([
    #     "ABM_Sparse_LowSeed_B", 
    #     "ABM_Medium_LowSeed_B", 
    #     "ABM_Dense_LowSeed_B"
    # ])