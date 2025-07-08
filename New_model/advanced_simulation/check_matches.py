import pandas as pd
import numpy as np

# Check approximate match details
bbox = (-81.97581623909956, 26.504952873345545, -81.95459566656734, 26.52311972089953)
minx, miny, maxx, maxy = bbox

sales_df = pd.read_csv('fl/fl/sales_data_with_recent_sale.csv')
repair_df = pd.read_csv('fl/fl/merged_with_coordinates.csv')

sales_df_filtered = sales_df[
    (sales_df['lon'] >= minx) & (sales_df['lon'] <= maxx) &
    (sales_df['lat'] >= miny) & (sales_df['lat'] <= maxy)
]

repair_df_filtered = repair_df[
    (repair_df['lon'] >= minx) & (repair_df['lon'] <= maxx) &
    (repair_df['lat'] >= miny) & (repair_df['lat'] <= maxy)
]

print(f'Filtered Repair data: {len(repair_df_filtered)} rows')

sales_coords = sales_df_filtered[['lon', 'lat']].drop_duplicates()
repair_coords = repair_df_filtered[['lon', 'lat']].drop_duplicates()

print(f'Unique Sales coordinates: {len(sales_coords)}')
print(f'Unique Repair coordinates: {len(repair_coords)}')

print(f'\nApproximate matching details (tolerance 0.0001 degrees):')
tolerance = 0.0001
matches_found = []

for sales_idx, sales_row in sales_coords.iterrows():
    sales_lon, sales_lat = sales_row['lon'], sales_row['lat']
    
    distances = np.sqrt(
        (repair_coords['lon'] - sales_lon)**2 + 
        (repair_coords['lat'] - sales_lat)**2
    )
    
    min_distance = distances.min()
    if min_distance <= tolerance:
        closest_repair_idx = distances.idxmin()
        repair_row = repair_coords.loc[closest_repair_idx]
        matches_found.append({
            'sales_lon': sales_lon,
            'sales_lat': sales_lat, 
            'repair_lon': repair_row['lon'],
            'repair_lat': repair_row['lat'],
            'distance': min_distance
        })

print(f'Found {len(matches_found)} approximate matches:')
for i, match in enumerate(matches_found):
    print(f'{i+1}. Sales: ({match["sales_lon"]:.6f}, {match["sales_lat"]:.6f})')
    print(f'   Repair: ({match["repair_lon"]:.6f}, {match["repair_lat"]:.6f})')
    print(f'   Distance: {match["distance"]:.6f} degrees (~{match["distance"]*111320:.1f} meters)')
    print()

# Check spatial extent
print("=== Geographic Range of Data ===")
print(f"Sales coordinate range:")
print(f"  Longitude: {sales_coords['lon'].min():.6f} to {sales_coords['lon'].max():.6f}")
print(f"  Latitude: {sales_coords['lat'].min():.6f} to {sales_coords['lat'].max():.6f}")

print(f"Repair coordinate range:")
print(f"  Longitude: {repair_coords['lon'].min():.6f} to {repair_coords['lon'].max():.6f}")
print(f"  Latitude: {repair_coords['lat'].min():.6f} to {repair_coords['lat'].max():.6f}")
