import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import itertools


bbox = (-81.97581623909956, 26.504952873345545, -81.95459566656734, 26.52311972089953)   # (minx, miny, maxx, maxy)
minx, miny, maxx, maxy = bbox

# Read the filtered datasets
sales_df = pd.read_csv("fl/fl/sales_data_with_recent_sale.csv")
repair_df = pd.read_csv("fl/fl/repair_coords_mapped_to_sales.csv")  # This contains repair data
pop_data = pd.read_csv("fl/fl/pop_subset_small_updated.csv")  # This contains household data

sales_df = sales_df[
    (sales_df['lon'] >= minx) & (sales_df['lon'] <= maxx) &
    (sales_df['lat'] >= miny) & (sales_df['lat'] <= maxy)
]
repair_df = repair_df[
    (repair_df['lon'] >= minx) & (repair_df['lon'] <= maxx) &
    (repair_df['lat'] >= miny) & (repair_df['lat'] <= maxy)
]
pop_data = pop_data[
    (pop_data['long'] >= minx) & (pop_data['long'] <= maxx) &
    (pop_data['lat'] >= miny) & (pop_data['lat'] <= maxy)
]

# Define time range: 2022-07 to 2023-10
start_date = datetime(2022, 7, 1)
end_date = datetime(2023, 10, 1)

# Generate all months in the time range
months = []
current = start_date
while current < end_date:
    months.append(current.strftime("%Y-%m"))
    # Move to next month
    if current.month == 12:
        current = current.replace(year=current.year + 1, month=1)
    else:
        current = current.replace(month=current.month + 1)

print(f"Generated {len(months)} time steps: {months[:3]}...{months[-3:]}")

# Get unique coordinates from all datasets
sales_coords = sales_df[['lon', 'lat']].drop_duplicates()
repair_coords = repair_df[['lon', 'lat']].drop_duplicates()
# Rename 'long' to 'lon' for consistency
pop_coords = pop_data[['long', 'lat']].rename(columns={'long': 'lon'}).drop_duplicates()
all_coords = pd.concat([sales_coords, repair_coords, pop_coords]).drop_duplicates().reset_index(drop=True)

print(f"Total unique coordinates: {len(all_coords)}")

# Create the base dataframe with all coordinate-timestep combinations
coord_month_combinations = list(itertools.product(range(len(all_coords)), months))
merged_df = pd.DataFrame(coord_month_combinations, columns=['coord_idx', 'timestep'])
merged_df = merged_df.merge(all_coords.reset_index().rename(columns={'index': 'coord_idx'}), on='coord_idx')
merged_df = merged_df[['lon', 'lat', 'timestep']].copy()

# Initialize repair, sales and household_id columns
merged_df['repair'] = 0
merged_df['sales'] = 0
merged_df['household_id'] = ''

print("Processing repair data...")
# Process repair data - convert record_date to datetime and extract month
repair_df['record_date'] = pd.to_datetime(repair_df['record_date'])
repair_df['month'] = repair_df['record_date'].dt.strftime("%Y-%m")

# Mark repair occurrences
for _, row in repair_df.iterrows():
    if row['month'] in months:
        mask = (merged_df['lon'] == row['lon']) & \
               (merged_df['lat'] == row['lat']) & \
               (merged_df['timestep'] == row['month'])
        merged_df.loc[mask, 'repair'] = 1

print("Processing sales data...")
# Process sales data
for _, row in sales_df.iterrows():
    if row['all_sales_in_period'] != '0':
        # Parse multiple dates separated by semicolons
        sales_dates = row['all_sales_in_period'].split(';')
        for date_str in sales_dates:
            try:
                sales_date = datetime.strptime(date_str, "%Y-%m-%d")
                sales_month = sales_date.strftime("%Y-%m")
                
                if sales_month in months:
                    mask = (merged_df['lon'] == row['lon']) & \
                           (merged_df['lat'] == row['lat']) & \
                           (merged_df['timestep'] == sales_month)
                    merged_df.loc[mask, 'sales'] = 1
            except ValueError:
                continue

print("Adding household IDs...")
# Create household mapping from population data
household_mapping = pop_data[['long', 'lat', 'hhold']].rename(columns={'long': 'lon'}).drop_duplicates()

# Merge household IDs based on coordinates
merged_df = merged_df.merge(household_mapping, on=['lon', 'lat'], how='left')
merged_df['household_id'] = merged_df['hhold'].fillna('')
merged_df = merged_df.drop('hhold', axis=1)

# Sort by coordinates and timestep
merged_df = merged_df.sort_values(['lon', 'lat', 'timestep']).reset_index(drop=True)

print("Applying cumulative logic to sales and repair status...")
# Apply cumulative logic: once sales/repair becomes 1, it stays 1 for all subsequent months
for coord_group in merged_df.groupby(['lon', 'lat']):
    coord_data = coord_group[1]
    indices = coord_data.index
    
    # Apply cumulative effect for sales
    sales_cumsum = coord_data['sales'].cumsum()
    merged_df.loc[indices, 'sales'] = (sales_cumsum > 0).astype(int)
    
    # Apply cumulative effect for repair
    repair_cumsum = coord_data['repair'].cumsum()
    merged_df.loc[indices, 'repair'] = (repair_cumsum > 0).astype(int)

# Save the merged dataset
merged_df.to_csv("fl/fl/merged_sales_repair_data.csv", index=False)

print("Merge completed!")
print(f"Final dataset shape: {merged_df.shape}")
print("\nSample of merged data:")
print(merged_df.head(10))
print(f"\nSummary statistics:")
print(f"Total records: {len(merged_df)}")
print(f"Records with repairs: {merged_df['repair'].sum()}")
print(f"Records with sales: {merged_df['sales'].sum()}")
print(f"Records with both: {((merged_df['repair'] == 1) & (merged_df['sales'] == 1)).sum()}")
print(f"Records with household IDs: {(merged_df['household_id'] != '').sum()}")
print(f"Unique household IDs: {merged_df[merged_df['household_id'] != '']['household_id'].nunique()}") 
