
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

# ------------------------------------------------------------------
# 4.  Compute map centre
# ------------------------------------------------------------------
lats = [lat for lat, _ in person_coords.values()]
lons = [lon for _, lon in person_coords.values()]
center_lat, center_lon = np.mean(lats), np.mean(lons)

# ------------------------------------------------------------------
# 5.  Folium interactive map
# ------------------------------------------------------------------
m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="OpenStreetMap")

# add every person as a blue dot
for pid, (lat, lon) in person_coords.items():
    folium.CircleMarker(
        location=[lat, lon],
        radius=2,
        color="blue",
        fill=True,
        fill_color="blue",
        fill_opacity=0.6,
        tooltip=pid,
    ).add_to(m)

# add social network edges (skip if any endpoint lacks coords)
edge_cnt = 0
for u, v in G.edges():
    if u in person_coords and v in person_coords:
        lat1, lon1 = person_coords[u]
        lat2, lon2 = person_coords[v]
        folium.PolyLine(
            locations=[[lat1, lon1], [lat2, lon2]],
            color="red",
            weight=1,
            opacity=0.3,
        ).add_to(m)
        edge_cnt += 1

print(f"Drawn {edge_cnt:,} social network edges on map")
m.save("fl_social_network_map.html")
print("Saved interactive map → fl_social_network_map.html")

# ------------------------------------------------------------------
# 6.  Static Matplotlib snapshot (limit edges for clarity)
# ------------------------------------------------------------------
plt.figure(figsize=(15, 10))
plt.scatter(lons, lats, s=10, c="blue", alpha=0.6, label="People")

for u, v in list(G.edges())[:1000]:  # draw first 1 000 edges
    if u in person_coords and v in person_coords:
        lon1, lat1 = person_coords[u][1], person_coords[u][0]
        lon2, lat2 = person_coords[v][1], person_coords[v][0]
        plt.plot([lon1, lon2], [lat1, lat2], "r-", alpha=0.2, linewidth=0.5)

plt.title(
    "FL Social Network – Combined Networks\n"
    "(daycare + household + school + work; first 1 000 edges for clarity)"
)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.axis("equal")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fl_social_network_static.png", dpi=300)
plt.show()
print("Saved static snapshot → fl_social_network_static.png")

# ------------------------------------------------------------------
# 7.  Basic network statistics
# ------------------------------------------------------------------
avg_deg = np.mean([deg for _, deg in G.degree()])
print("\n=== Network statistics ===")
print(f"nodes          : {G.number_of_nodes():,}")
print(f"edges          : {G.number_of_edges():,}")
print(f"average degree : {avg_deg:.2f}")
print(f"density        : {nx.density(G):.6f}")
if nx.is_connected(G):
    print("graph is connected")
    print(f"diameter       : {nx.diameter(G)}")
else:
    print(f"connected comps: {nx.number_connected_components(G)}")
