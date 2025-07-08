
import geopandas as gpd      # only needed if you later reload .gpkg
import networkx as nx
import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# 1.  Load population slice (already spatially filtered to your bbox)
# ------------------------------------------------------------------
pop_pop = pd.read_csv("fl/fl/pop_subset_updated.csv")          # id, lat, long …

# (optional) confirm size
print(f"Population slice contains {len(pop_pop):,} persons")

# ------------------------------------------------------------------
# 2.  Build combined social graph from multiple CSV files
#     File format: ego, neighbour1, neighbour2, …
# ------------------------------------------------------------------
G = nx.Graph()

# Create a set of valid person IDs from the population data for faster lookup
valid_ids = set(pop_pop["id"].astype(str))  # Convert to string for consistency
print(f"Valid person IDs: {len(valid_ids):,}")

# List of all social network files
network_files = [
    "fl/fl/social_networks/fl_daycare_network.csv",
    "fl/fl/social_networks/fl_household_network.csv", 
    "fl/fl/social_networks/fl_school_network.csv",
    "fl/fl/social_networks/fl_work_network.csv"
]

# Load each network and add edges to the graph (only for valid IDs)
for network_file in network_files:
    network_type = network_file.split('/')[-1].replace('fl_', '').replace('_network.csv', '')
    edges_count = 0
    skipped_edges = 0
    
    with open(network_file, "r") as f:
        for line in f:
            nodes = line.strip().split(",")
            if len(nodes) < 2:
                continue
            ego, neighbours = nodes[0], nodes[1:]
            
            # Only process if ego is in our valid population
            if ego in valid_ids:
                for nbr in neighbours:
                    # Only add edge if both nodes are in our valid population
                    if nbr in valid_ids:
                        G.add_edge(ego, nbr)
                        edges_count += 1
                    else:
                        skipped_edges += 1
            else:
                skipped_edges += len(neighbours)
    
    print(f"{network_type.capitalize()} network: {edges_count:,} edges added, {skipped_edges:,} edges skipped")

print(f"\nCombined graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

# ------------------------------------------------------------------
# 3.  Build a coordinate lookup for every person in the slice
# ------------------------------------------------------------------
person_coords = {
    str(row["id"]): (row["lat"], row["long"])   # (lat, lon) - ensure string consistency
    for _, row in pop_pop.iterrows()
}

# Add isolated nodes (people without social network connections) to the graph
for pid in person_coords.keys():
    if pid not in G:
        G.add_node(pid)

print(f"Graph after adding isolates: {G.number_of_nodes():,} nodes")
