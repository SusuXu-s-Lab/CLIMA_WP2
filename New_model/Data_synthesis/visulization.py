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
    df = pd.read_csv("sysnthetic_data/household_states.csv")

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
    plt.show()

def visulize_links():
    df = pd.read_csv("sysnthetic_data/ground_truth_network.csv")
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

    plt.tight_layout()
    plt.show()

    # # Additional: Network visualization for first few time steps
    # fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    # axes = axes.flatten()
    #
    # for i, t in enumerate(sorted(df['time_step'].unique())[:4]):
    #     subset = df[df['time_step'] == t]
    #     G = nx.from_pandas_edgelist(subset, source='household_id_1', target='household_id_2',
    #                                 edge_attr='link_type')
    #
    #     pos = nx.spring_layout(G, k=1, iterations=50)
    #
    #     # Draw edges with different colors for different link types
    #     edge_colors = ['red' if G[u][v]['link_type'] == 1.0 else 'blue' for u, v in G.edges()]
    #
    #     nx.draw(G, pos, ax=axes[i], node_color='lightblue', node_size=300,
    #             edge_color=edge_colors, with_labels=False, font_size=8, alpha=0.7)
    #
    #     axes[i].set_title(f'Network at Time Step {t}')
    #     axes[i].set_aspect('equal')
    #
    # # Add legend for edge colors
    # from matplotlib.lines import Line2D
    # legend_elements = [Line2D([0], [0], color='red', lw=2, label='Link Type 1.0'),
    #                    Line2D([0], [0], color='blue', lw=2, label='Link Type 2.0')]
    # fig.legend(handles=legend_elements, loc='upper right')
    #
    # plt.suptitle('Network Evolution - First 4 Time Steps', fontsize=16)
    # plt.tight_layout()
    # plt.show()

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
    inter_t_all = np.load("inter_t_all.npy")
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

    imageio.mimsave("interpot_animation.gif", gif_frames, duration=0.5, loop=0)

    for frame in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, frame))
    os.rmdir(temp_dir)


def visulize_sim():
    similarity_df_loaded = pd.read_csv("similarity_df.csv", index_col=0)
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
    plt.show()
    # plt.savefig("similarity_distribution/similarity_distribution.png")
    plt.close()
def visulize_prob():
    df=pd.read_csv('link_probs.csv')
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
    plt.show()

visulize_sim()
visulize_interpot_gif()
visulize_links()
visulize_state()
visulize_prob()
