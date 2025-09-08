import pdb
import imageio
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
from collections import defaultdict
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings('ignore')

def visulize_state():
    # Load the household state dataframe
    df = pd.read_csv("/Users/susangao/Desktop/CLIMA/CODE 4.13- new syn_ruxiao_3/data/syn_data_ruxiao_v3/syn_data1_200node/household_states_raw.csv")

    # Convert time to integer if needed
    df['time'] = df['time'].astype(int)

    # Aggregate: count how many households are in state=1 at each timestep for each state type
    agg = df.groupby('time')[['repair_state', 'vacancy_state', 'sales_state']].sum().reset_index()

    # Normalize if needed (e.g., to percentage), depending on total households
    total_households = df['home'].nunique()
    agg_percent = agg.copy()
    agg_percent[['repair_state', 'vacancy_state', 'sales_state']] = agg_percent[
        ['repair_state', 'vacancy_state', 'sales_state']] / total_households * 100

    # Melt to long format for easier plotting
    agg_melted = agg_percent.melt(id_vars='time', var_name='State', value_name='Percentage')

    # Plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=agg_melted, x='time', y='Percentage', hue='State', marker="o")
    plt.title("Activation Percentage Over Time by State")
    plt.ylabel("Percentage of Active Households (%)")
    plt.xlabel("Time Step")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("images/states.png")
    plt.show()

def visulize_links():
    df = pd.read_csv("/Users/susangao/Desktop/CLIMA/CODE 4.13- new syn_ruxiao_3/data/syn_data_ruxiao_v3/syn_data1_200node/ground_truth_network_raw.csv")
    print("Data shape:", df.shape)
    print("Time steps range:", df['time_step'].min(), "to", df['time_step'].max())
    print("Link types:", df['link_type'].unique())
    print("Number of unique households:", len(set(df['household_id_1'].tolist() + df['household_id_2'].tolist())))

    # Create comprehensive visualizations on a single figure
    # plt.style.use('seaborn-v0_8')
    plt.figure(figsize=(10, 6))

    link_counts_time = df.groupby(['time_step', 'link_type']).size().unstack(fill_value=0)
    for col in link_counts_time.columns:
        plt.plot(link_counts_time.index, link_counts_time[col], marker='o', label=f'Type {col}')

    plt.title('Number of Links Over Time by Type')
    plt.xlabel('Time Step')
    plt.ylabel('Number of Links')
    plt.legend(title='Link Type')
    plt.grid(True, alpha=0.3)
    plt.savefig("images/links.png")
    plt.tight_layout()
    plt.show()


    # Summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Total number of links: {len(df)}")
    print(f"Number of unique households: {len(set(df['household_id_1'].tolist() + df['household_id_2'].tolist()))}")
    print(f"Time range: {df['time_step'].min()} to {df['time_step'].max()}")
    print(f"Link types: {sorted(df['link_type'].unique())}")
    print(f"Average links per time step: {len(df) / len(df['time_step'].unique()):.2f}")

    # Time-based statistics
    print("\n=== TIME-BASED STATISTICS ===")
    time_stats = df.groupby('time_step').agg({
        'household_id_1': 'count',
        'link_type': lambda x: len(x.unique())
    }).rename(columns={'household_id_1': 'total_links', 'link_type': 'unique_link_types'})

    print(time_stats.head())

def visulize_interpot_gif():
    inter_t_all = np.load("/Users/susangao/Desktop/CLIMA/CODE 4.13- new syn_ruxiao_3/data/syn_data_ruxiao_v3/syn_data1_200node/inter_t_all.npy")
    gif_frames = []
    temp_dir = "tmp_frames"
    os.makedirs(temp_dir, exist_ok=True)

    for t, inter_t in enumerate(inter_t_all):
        values = inter_t.flatten()
        values = values[~np.isnan(values)]  # 去掉 NaN 的对角线

        plt.figure(figsize=(6, 4))
        plt.hist(values, bins=30, color='skyblue', edgecolor='black')
        plt.title(f'Interaction Potential Distribution at t={t}')
        plt.xlabel('Interaction Value')
        plt.ylabel('Frequency')
        plt.tight_layout()

        frame_path = os.path.join(temp_dir, f"frame_{t:03d}.png")
        plt.savefig(frame_path)
        gif_frames.append(imageio.imread(frame_path))
        plt.close()

    imageio.mimsave("images/interpot_animation.gif", gif_frames, duration=0.5, loop=0)

    for frame in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, frame))
    os.rmdir(temp_dir)


def visulize_sim():
    similarity_df_loaded = pd.read_csv("/Users/susangao/Desktop/CLIMA/CODE 4.13- new syn_ruxiao_3/data/syn_data_ruxiao_v3/syn_data1_200node/similarity_df_raw.csv", index_col=0)
    similarity_matrix = similarity_df_loaded.values
    np.fill_diagonal(similarity_matrix, np.nan)
    values = similarity_matrix.flatten()
    values = values[~np.isnan(values)]

    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=30, color='salmon', edgecolor='black')
    plt.title("Similarity Matrix Distribution (excluding diagonal)")
    plt.xlabel("Similarity Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("images/imilarity_distribution.png")
    plt.show()

    plt.close()
def visulize_prob():
    df=pd.read_csv('/Users/susangao/Desktop/CLIMA/CODE 4.13- new syn_ruxiao_3/data/syn_data_ruxiao_v3/syn_data1_200node/link_probs_raw.csv')
    # Plot distributions
    plt.figure(figsize=(8, 6))
    plt.hist(df['P_no_link'], bins=30, alpha=0.6, label='P_no_link', density=True)
    plt.hist(df['P_bonding'], bins=30, alpha=0.6, label='P_bonding', density=True)
    plt.hist(df['P_bridging'], bins=30, alpha=0.6, label='P_bridging', density=True)

    plt.xlabel('Probability Value')
    plt.ylabel('Density')
    plt.title('Distribution of Link Probabilities')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("images/prob.png")
    plt.show()

def visualize_p_values_gif():
    """Create animated GIFs showing p_self and p_ji distribution changes over time"""
    # Load p_self and p_ji data
    p_self_df = pd.read_csv("/Users/susangao/Desktop/CLIMA/CODE 4.13- new syn_ruxiao_3/data/syn_data_ruxiao_v3/syn_data1_200node/p_self_all_values.csv")
    p_ji_df = pd.read_csv("/Users/susangao/Desktop/CLIMA/CODE 4.13- new syn_ruxiao_3/data/syn_data_ruxiao_v3/syn_data1_200node/p_ji_all_values.csv")
    
    # Create temporary directory for frames
    temp_dir = "tmp_frames"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Get unique time steps and dimensions
    time_steps = sorted(p_self_df['time_step'].unique())
    dimensions = sorted(p_self_df['dimension'].unique())
    dim_names = p_self_df['dimension_name'].unique()
    
    # Create p_self distribution GIF
    gif_frames_pself = []
    
    for t in time_steps:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f'P_self Distribution at Time Step {t}', fontsize=16)
        
        for i, dim in enumerate(dimensions):
            data = p_self_df[(p_self_df['time_step'] == t) & (p_self_df['dimension'] == dim)]
            if not data.empty:
                values = data['p_self_value'].values
                axes[i].hist(values, bins=20, color='lightblue', edgecolor='black', alpha=0.7)
                axes[i].set_title(f'{data["dimension_name"].iloc[0]}')
                axes[i].set_xlabel('P_self Value')
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        frame_path = os.path.join(temp_dir, f"pself_frame_{t:03d}.png")
        plt.savefig(frame_path)
        gif_frames_pself.append(imageio.imread(frame_path))
        plt.close()
    
    # Save p_self GIF
    imageio.mimsave("images/p_self_distribution.gif", gif_frames_pself, duration=0.5, loop=0)
    
    # Create p_ji distribution GIF
    gif_frames_pji = []
    
    for t in time_steps:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f'P_ji Distribution at Time Step {t}', fontsize=16)
        
        for i, dim in enumerate(dimensions):
            data = p_ji_df[(p_ji_df['time_step'] == t) & (p_ji_df['dimension'] == dim)]
            if not data.empty:
                values = data['p_ji_value'].values
                axes[i].hist(values, bins=20, color='lightcoral', edgecolor='black', alpha=0.7)
                axes[i].set_title(f'{data["dimension_name"].iloc[0]}')
                axes[i].set_xlabel('P_ji Value')
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        frame_path = os.path.join(temp_dir, f"pji_frame_{t:03d}.png")
        plt.savefig(frame_path)
        gif_frames_pji.append(imageio.imread(frame_path))
        plt.close()
    
    # Save p_ji GIF
    imageio.mimsave("images/p_ji_distribution.gif", gif_frames_pji, duration=0.5, loop=0)
    
    # Clean up temporary files
    for frame in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, frame))
    os.rmdir(temp_dir)
    
    print("Generated p_self_distribution.gif and p_ji_distribution.gif in images/ folder")

def create_activation_distribution_gifs(csv_file_path='/Users/susangao/Desktop/CLIMA/CODE 4.13- new syn_ruxiao_3/data/syn_data_ruxiao_v3/syn_data1_200node/activation_records.csv', output_dir='images'):
    """
    Create GIF animations showing the distribution of prod_term and activate_prob over time.
    
    Parameters:
    -----------
    csv_file_path : str
        Path to the activation_records.csv file
    output_dir : str
        Directory to save the GIF files
    """
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read the activation records
    print("Loading activation records...")
    df = pd.read_csv(csv_file_path)
    
    # Get unique time steps and dimensions
    time_steps = sorted(df['time_step'].unique())
    dimensions = sorted(df['dimension'].unique())
    dim_names = ['vacancy_state', 'repair_state', 'sales_state']
    
    print(f"Found {len(time_steps)} time steps and {len(dimensions)} dimensions")
    
    # Create GIFs for each dimension
    for dim in dimensions:
        dim_name = dim_names[dim]
        dim_data = df[df['dimension'] == dim]
        
        if len(dim_data) == 0:
            print(f"No data found for dimension {dim} ({dim_name})")
            continue
            
        print(f"Creating GIFs for dimension {dim} ({dim_name})...")
        
        # Create prod_term distribution GIF
        create_single_activation_gif(
            dim_data, 
            'prod_term', 
            f'{output_dir}/prod_term_distribution_{dim_name}.gif',
            f'Product Term Distribution Over Time - {dim_name.replace("_", " ").title()}',
            time_steps
        )
        
        # Create activate_prob distribution GIF
        create_single_activation_gif(
            dim_data, 
            'activate_prob', 
            f'{output_dir}/activate_prob_distribution_{dim_name}.gif',
            f'Activation Probability Distribution Over Time - {dim_name.replace("_", " ").title()}',
            time_steps
        )
    
    print("All activation GIFs created successfully!")

def create_single_activation_gif(data, column, output_path, title, time_steps):
    """
    Create a single GIF showing distribution over time for a specific column.
    """
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    def animate(frame):
        ax.clear()
        
        t = time_steps[frame]
        t_data = data[data['time_step'] == t][column].dropna()
        
        if len(t_data) == 0:
            ax.text(0.5, 0.5, f'No data for time step {t}', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'{title}\nTime Step: {t}')
            return
        
        # Create histogram
        ax.hist(t_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Add statistics
        mean_val = t_data.mean()
        std_val = t_data.std()
        median_val = t_data.median()
        
        # Set consistent y-axis limits based on all data
        all_data = data[column].dropna()
        if len(all_data) > 0:
            hist_counts, _ = np.histogram(all_data, bins=30)
            ax.set_ylim(0, hist_counts.max() * 1.1)
        
        # Set x-axis limits
        ax.set_xlim(all_data.min() - 0.01, all_data.max() + 0.01)
        
        ax.set_title(f'{title}\nTime Step: {t}')
        ax.set_xlabel(column.replace('_', ' ').title())
        ax.set_ylabel('Frequency')
        
        # Add statistics text
        stats_text = f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}\nMedian: {median_val:.4f}\nCount: {len(t_data)}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=len(time_steps), interval=500, repeat=True)
    
    # Save as GIF
    print(f"Saving {output_path}...")
    anim.save(output_path, writer='pillow', fps=2)
    plt.close(fig)

def create_combined_activation_gif(csv_file_path='/Users/susangao/Desktop/CLIMA/CODE 4.13- new syn_ruxiao_3/data/syn_data_ruxiao_v3/data_ruxiao_process.py/activation_records.csv', output_dir='images'):
    """
    Create a combined GIF showing both prod_term and activate_prob distributions side by side.
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read the activation records
    print("Loading activation records for combined GIF...")
    df = pd.read_csv(csv_file_path)
    
    time_steps = sorted(df['time_step'].unique())
    dimensions = sorted(df['dimension'].unique())
    dim_names = ['vacancy_state', 'repair_state', 'sales_state']
    
    for dim in dimensions:
        dim_name = dim_names[dim]
        dim_data = df[df['dimension'] == dim]
        
        if len(dim_data) == 0:
            continue
            
        print(f"Creating combined GIF for dimension {dim} ({dim_name})...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        def animate(frame):
            ax1.clear()
            ax2.clear()
            
            t = time_steps[frame]
            t_data = dim_data[dim_data['time_step'] == t]
            
            if len(t_data) == 0:
                for ax in [ax1, ax2]:
                    ax.text(0.5, 0.5, f'No data for time step {t}', 
                           transform=ax.transAxes, ha='center', va='center')
                ax1.set_title(f'Product Term Distribution\nTime Step: {t}')
                ax2.set_title(f'Activation Probability Distribution\nTime Step: {t}')
                return
            
            # Plot prod_term
            prod_data = t_data['prod_term'].dropna()
            if len(prod_data) > 0:
                ax1.hist(prod_data, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
                mean_prod = prod_data.mean()
                ax1.axvline(mean_prod, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_prod:.4f}')
                ax1.legend()
            
            # Plot activate_prob
            prob_data = t_data['activate_prob'].dropna()
            if len(prob_data) > 0:
                ax2.hist(prob_data, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
                mean_prob = prob_data.mean()
                ax2.axvline(mean_prob, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_prob:.4f}')
                ax2.legend()
            
            ax1.set_title(f'Product Term Distribution\nTime Step: {t}')
            ax1.set_xlabel('Product Term')
            ax1.set_ylabel('Frequency')
            
            ax2.set_title(f'Activation Probability Distribution\nTime Step: {t}')
            ax2.set_xlabel('Activation Probability')
            ax2.set_ylabel('Frequency')
            
            # Set consistent axis limits
            all_prod = dim_data['prod_term'].dropna()
            all_prob = dim_data['activate_prob'].dropna()
            
            if len(all_prod) > 0:
                ax1.set_xlim(all_prod.min() - 0.01, all_prod.max() + 0.01)
            if len(all_prob) > 0:
                ax2.set_xlim(all_prob.min() - 0.01, all_prob.max() + 0.01)
            
            plt.suptitle(f'{dim_name.replace("_", " ").title()} - Activation Analysis', fontsize=14)
            plt.tight_layout()
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(time_steps), interval=500, repeat=True)
        
        # Save as GIF
        output_path = f'{output_dir}/combined_activation_distribution_{dim_name}.gif'
        print(f"Saving {output_path}...")
        anim.save(output_path, writer='pillow', fps=2)
        plt.close(fig)

# Uncomment the lines below to run specific visualizations
visulize_sim()
# visulize_interpot_gif()
visulize_links()
visulize_state()
visulize_prob()
# visualize_p_values_gif()

# Uncomment to create activation distribution GIFs
create_activation_distribution_gifs()
create_combined_activation_gif()