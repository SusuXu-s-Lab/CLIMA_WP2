import geopandas as gpd
import networkx as nx
import pandas as pd
import csv
from collections import defaultdict
from datetime import datetime, date
import calendar

def generate_timesteps():
    """Generate all months from 2022-07 to 2023-09"""
    timesteps = []
    
    # 2022: July to December
    for month in range(7, 13):
        timesteps.append(f"2022-{month:02d}")
    
    # 2023: January to September
    for month in range(1, 10):
        timesteps.append(f"2023-{month:02d}")
    
    return timesteps

def build_household_social_network():
    """
    Build a household-based social network
    Convert individual-level social connections to household-level connections
    Add link_type to distinguish connection sources: 1=household, 2=other
    Add timestep column to replicate the network for each month
    """
    
    # ------------------------------------------------------------------
    # 1. Load population data and create a mapping from individual ID to household ID
    # ------------------------------------------------------------------
    print("Loading population data...")
    pop_data = pd.read_csv("fl/fl/pop_subset_updated.csv")
    print(f"Population slice contains {len(pop_data):,} persons")
    
    # Create a dictionary mapping individual ID to household ID
    person_to_household = {}
    for _, row in pop_data.iterrows():
        person_id = str(row["id"])
        household_id = str(row["hhold"])
        person_to_household[person_id] = household_id
    
    print(f"Created mapping for {len(person_to_household):,} individuals to households")
    
    # Count the number of unique households
    unique_households = set(person_to_household.values())
    print(f"Total number of unique households: {len(unique_households):,}")
    
    # ------------------------------------------------------------------
    # 2. Load individual-level social network and track connection types
    # ------------------------------------------------------------------
    print("\nBuilding individual-level social network...")
    G_individual = nx.Graph()
    
    # Create a set of valid individual IDs
    valid_person_ids = set(person_to_household.keys())
    
    # Used to store the type information of each edge
    edge_types = {}  # (person1, person2) -> link_type
    
    # List of social network files
    network_files = [
        ("fl/fl/social_networks/fl_daycare_network.csv", 2),
        ("fl/fl/social_networks/fl_household_network.csv", 1), 
        ("fl/fl/social_networks/fl_school_network.csv", 2),
        ("fl/fl/social_networks/fl_work_network.csv", 2)
    ]
    
    # Load each network file and add edges
    total_individual_edges = 0
    for network_file, link_type in network_files:
        network_type = network_file.split('/')[-1].replace('fl_', '').replace('_network.csv', '')
        edges_count = 0
        skipped_edges = 0
        
        print(f"Processing {network_type} network (type {link_type})...")
        
        with open(network_file, "r") as f:
            for line in f:
                nodes = line.strip().split(",")
                if len(nodes) < 2:
                    continue
                ego, neighbours = nodes[0], nodes[1:]
                
                # Only process valid individual IDs
                if ego in valid_person_ids:
                    for nbr in neighbours:
                        if nbr in valid_person_ids:
                            # Ensure the order of the edge is consistent
                            edge = (ego, nbr) if ego < nbr else (nbr, ego)
                            
                            # Add edge to the graph
                            G_individual.add_edge(ego, nbr)
                            
                            # Record the type of the edge (if the same edge has multiple types, priority: household(1) > other(2))
                            if edge not in edge_types or (edge_types[edge] == 2 and link_type == 1):
                                edge_types[edge] = link_type
                            
                            edges_count += 1
                        else:
                            skipped_edges += 1
                else:
                    skipped_edges += len(neighbours)
        
        total_individual_edges += edges_count
        print(f"  {network_type.capitalize()}: {edges_count:,} edges added, {skipped_edges:,} skipped")
    
    print(f"\nIndividual-level graph: {G_individual.number_of_nodes():,} nodes, {G_individual.number_of_edges():,} edges")
    
    # ------------------------------------------------------------------
    # 3. Convert individual-level network to household-level network
    # ------------------------------------------------------------------
    print("\nConverting to household-level network...")
    
    # Use a dictionary to store household-level connections and their types
    household_connections = {}  # (household1, household2) -> link_type
    
    # Iterate over each edge at the individual level
    for person1, person2 in G_individual.edges():
        household1 = person_to_household[person1]
        household2 = person_to_household[person2]
        
        # Only create connections at the household level if two people belong to different households
        if household1 != household2:
            # Ensure the order of the connection is consistent (lexicographical order) to avoid duplication
            household_pair = (household1, household2) if household1 < household2 else (household2, household1)
            
            # Get the type of individual connection
            person_pair = (person1, person2) if person1 < person2 else (person2, person1)
            individual_link_type = edge_types.get(person_pair, 2)
            
            # If the household connection already exists, maintain the higher priority type (1 > 2)
            if household_pair not in household_connections or (household_connections[household_pair] == 2 and individual_link_type == 1):
                household_connections[household_pair] = individual_link_type
    
    print(f"Household-level connections: {len(household_connections):,} unique household pairs")
    
    # Count the number of connections of different types
    type_1_count = sum(1 for link_type in household_connections.values() if link_type == 1)
    type_2_count = sum(1 for link_type in household_connections.values() if link_type == 2)
    print(f"  - Type 1 (household) connections: {type_1_count}")
    print(f"  - Type 2 (other) connections: {type_2_count}")
    
    # ------------------------------------------------------------------
    # 4. Generate timesteps
    # ------------------------------------------------------------------
    timesteps = generate_timesteps()
    print(f"\nGenerating network for {len(timesteps)} timesteps: {timesteps[0]} to {timesteps[-1]}")
    
    # ------------------------------------------------------------------
    # 5. Create a household-level graph for statistical analysis
    # ------------------------------------------------------------------
    G_household = nx.Graph()
    for (h1, h2), link_type in household_connections.items():
        G_household.add_edge(h1, h2, link_type=link_type)
    
    # Add isolated households without external connections
    for household_id in unique_households:
        if household_id not in G_household:
            G_household.add_node(household_id)
    
    print(f"Household-level graph: {G_household.number_of_nodes():,} nodes, {G_household.number_of_edges():,} edges")
    
    # ------------------------------------------------------------------
    # 6. Save the household-level network to a CSV file (including timesteps)
    # ------------------------------------------------------------------
    output_file = "fl/fl/household_social_network.csv"
    print(f"\nSaving household network to {output_file}...")
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write the header
        writer.writerow(['household_id_1', 'household_id_2', 'link_type', 'timestep'])
        
        # Write all household-level connections for each timestep
        total_rows = 0
        for timestep in timesteps:
            for (h1, h2), link_type in sorted(household_connections.items()):
                writer.writerow([h1, h2, link_type, timestep])
                total_rows += 1
        
        print(f"Successfully saved {total_rows:,} household connections ({len(household_connections):,} unique pairs × {len(timesteps)} timesteps) to {output_file}")
    
    # ------------------------------------------------------------------
    # 7. Output network statistics
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("NETWORK STATISTICS SUMMARY")
    print("="*60)
    
    print(f"Individual level:")
    print(f"  - Total individuals: {len(person_to_household):,}")
    print(f"  - Individuals with connections: {G_individual.number_of_nodes():,}")
    print(f"  - Individual connections: {G_individual.number_of_edges():,}")
    if G_individual.number_of_nodes() > 0:
        avg_degree_individual = 2 * G_individual.number_of_edges() / G_individual.number_of_nodes()
        print(f"  - Average degree: {avg_degree_individual:.2f}")
    
    print(f"\nHousehold level (per timestep):")
    print(f"  - Total households: {len(unique_households):,}")
    connected_households = sum(1 for node in G_household.nodes() if G_household.degree(node) > 0)
    print(f"  - Households with external connections: {connected_households:,}")
    print(f"  - Household connections: {G_household.number_of_edges():,}")
    print(f"    * Type 1 (household): {type_1_count}")
    print(f"    * Type 2 (other): {type_2_count}")
    
    if G_household.number_of_nodes() > 0:
        avg_degree_household = 2 * G_household.number_of_edges() / G_household.number_of_nodes()
        print(f"  - Average degree: {avg_degree_household:.2f}")
        print(f"  - Network density: {nx.density(G_household):.6f}")
    
    # Calculate connectivity
    if G_household.number_of_edges() > 0:
        if nx.is_connected(G_household):
            print(f"  - Network is connected")
            print(f"  - Diameter: {nx.diameter(G_household)}")
        else:
            components = list(nx.connected_components(G_household))
            print(f"  - Connected components: {len(components)}")
            if components:
                largest_component_size = max(len(c) for c in components)
                print(f"  - Largest component size: {largest_component_size}")
    
    print(f"\nTemporal network:")
    print(f"  - Timesteps: {len(timesteps)} ({timesteps[0]} to {timesteps[-1]})")
    print(f"  - Total rows in CSV: {len(household_connections) * len(timesteps):,}")
    
    # Household size statistics
    household_sizes = defaultdict(int)
    for household_id in unique_households:
        size = sum(1 for pid, hid in person_to_household.items() if hid == household_id)
        household_sizes[size] += 1
    
    print(f"\nHousehold size distribution:")
    for size in sorted(household_sizes.keys()):
        print(f"  - Size {size}: {household_sizes[size]:,} households")
    
    return household_connections, G_household, timesteps

if __name__ == "__main__":
    household_connections, G_household, timesteps = build_household_social_network() 
    