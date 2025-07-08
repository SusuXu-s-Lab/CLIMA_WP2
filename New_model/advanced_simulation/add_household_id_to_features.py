import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import pdb




def add_household_id_to_features():
    """Add household_id column to small_house_features.csv using optimized vectorized operations"""
    bbox = (-81.97581623909956, 26.504952873345545, -81.95459566656734, 26.52311972089953)   # (minx, miny, maxx, maxy)

    features_df = pd.read_csv('fl/fl/house_features.csv')

    minx, miny, maxx, maxy = bbox

    features_df = features_df[
        (features_df['lon'] >= minx) & (features_df['lon'] <= maxx) &
        (features_df['lat'] >= miny) & (features_df['lat'] <= maxy)
    ]
    
    try:
        # Read data
        print("Loading data...")
        pop_df = pd.read_csv("fl/fl/pop_subset_small_updated.csv")
        
        print(f"House features data: {len(features_df)} rows, {len(features_df.columns)} columns")
        print(f"Population data: {len(pop_df)} rows, {len(pop_df.columns)} columns")
        
        # Get unique household coordinates
        household_coords = pop_df.groupby('hhold')[['long', 'lat']].first().reset_index()
        print(f"Unique households: {len(household_coords)}")
        
        # Prepare coordinate arrays for vectorized distance calculation
        print("Preparing coordinate matrices for optimized calculation...")
        feature_points = features_df[['lon', 'lat']].values
        household_points = household_coords[['long', 'lat']].values
        
        # Calculate distance matrix using vectorized operations (much faster)
        print("Calculating distance matrix using vectorized operations...")
        # Use numpy broadcasting for efficient distance calculation
        feature_points_expanded = feature_points[:, np.newaxis, :]  # Shape: (n_features, 1, 2)
        household_points_expanded = household_points[np.newaxis, :, :]  # Shape: (1, n_households, 2)
        distance_matrix = np.sqrt(np.sum((feature_points_expanded - household_points_expanded)**2, axis=2))
        
        print(f"Distance matrix calculated: {distance_matrix.shape}")
        
        # Find nearest household for each feature (vectorized)
        print("Finding nearest households...")
        nearest_household_indices = np.argmin(distance_matrix, axis=1)
        nearest_distances = np.min(distance_matrix, axis=1)
        
        # Create result dataframe efficiently
        print("Creating result dataframe...")
        result_df = features_df.copy()
        
        # Add household information
        result_df['household_id'] = household_coords.iloc[nearest_household_indices]['hhold'].values
        # Store distances temporarily for analysis but don't save to file
        distance_to_household_degrees = nearest_distances
        distance_to_household_meters = nearest_distances * 111320
        
        # Display first 10 matches
        print("\nFirst 10 matches:")
        for i in range(min(10, len(result_df))):
            feature_row = features_df.iloc[i]
            household_id = result_df.iloc[i]['household_id']
            distance_meters = distance_to_household_meters[i]
            print(f"Feature {i}: ({feature_row['lon']:.6f}, {feature_row['lat']:.6f}) -> "
                  f"Household {household_id}, distance: {distance_meters:.1f}m")
        
        # Statistical analysis of distance distribution (vectorized for performance)
        distances_meters = distance_to_household_meters
        print(f"\nDistance statistics:")
        print(f"Minimum distance: {distances_meters.min():.1f}m")
        print(f"Maximum distance: {distances_meters.max():.1f}m")
        print(f"Average distance: {distances_meters.mean():.1f}m")
        print(f"Median distance: {np.median(distances_meters):.1f}m")
        
        # Analyze distance distribution using vectorized operations
        print(f"\nDistance distribution analysis:")
        bins = np.array([0, 10, 50, 100, 200, 500, float('inf')])
        labels = ['0-10m', '10-50m', '50-100m', '100-200m', '200-500m', '>500m']
        
        # Use numpy's digitize for faster binning
        bin_indices = np.digitize(distances_meters, bins) - 1
        unique_bins, counts = np.unique(bin_indices, return_counts=True)
        
        for i, label in enumerate(labels):
            count = counts[unique_bins == i][0] if i in unique_bins else 0
            percentage = count / len(distances_meters) * 100
            print(f"{label}: {count} points ({percentage:.1f}%)")
        
        # Save results
        result_df.to_csv("fl/fl/small_house_features_with_household_id.csv", index=False)
        
        # Display result sample
        print(f"\nResult sample (first 3 rows):")
        print(result_df[['lon', 'lat', 'household_id']].head(3))
        
        # Statistics for household feature assignment (vectorized)
        household_counts = result_df['household_id'].value_counts()
        print(f"Total households: {len(household_coords)}")
        
        return result_df
        
    except Exception as e:
        print(f"Error: {e}")
        return None



result = add_household_id_to_features()
if result is not None:
    print(f"\n completed matching. Each house feature has been assigned the nearest household_id")
else:
    print("❌ Processing failed") 