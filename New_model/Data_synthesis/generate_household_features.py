import numpy as np
import pandas as pd
import pdb

def generate_household_features(house_df):
    # Assume 'house_df' contains a column 'home' with 8-digit geohash codes
    # Extract the first 6 characters as community tags
    house_df['community'] = house_df['home'].str[:6]

    # -------------------------------------------
    # Load real hurricane Ian data
    # -------------------------------------------
    hf_df = pd.read_csv('hurricane_ian_real_features.csv')
    
    # Get the number of households we need
    n_houses = len(house_df)
    
    # Truncate real data to match house_df length
    if len(hf_df) > n_houses:
        hf_df = hf_df.head(n_houses).copy()
    elif len(hf_df) < n_houses:
        # If real data is shorter, repeat it to fill the length
        repeat_times = (n_houses // len(hf_df)) + 1
        hf_df = pd.concat([hf_df] * repeat_times, ignore_index=True).head(n_houses).copy()
    
    # -------------------------------------------
    # Map DamageLevel strings to numerical values
    # -------------------------------------------
    damage_level_mapping = {
        'Inaccessible': 0.0,
        'Affected': 0.25, 
        'Minor': 0.5,
        'Major': 1.0
    }
    
    hf_df['damage_level_numeric'] = hf_df['DamageLevel'].map(damage_level_mapping)
    
    # Handle any missing values in the mapping
    hf_df['damage_level_numeric'] = hf_df['damage_level_numeric'].fillna(0.0)
    
    # -------------------------------------------
    # Replace features with real data and normalize
    # -------------------------------------------
    
    # Extract raw values and handle missing data
    building_values = hf_df['BldgValue'].copy()
    est_losses = hf_df['EstLoss'].copy()
    damage_levels = hf_df['damage_level_numeric'].copy()
    
    # Normalize to 0-1 range (handling NaN values properly)
    def normalize_to_01(series):
        # Remove NaN values for min/max calculation
        valid_values = series.dropna()
        if len(valid_values) == 0:
            # All NaN - return zeros
            return pd.Series(np.zeros(len(series)), index=series.index)
        
        min_val = valid_values.min()
        max_val = valid_values.max()
        
        if max_val == min_val:
            # All same values -> all zeros, but preserve NaN positions
            result = pd.Series(np.zeros(len(series)), index=series.index)
            result[series.isna()] = 0.0  # Set NaN positions to 0
            return result
        
        # Normalize valid values
        normalized = (series - min_val) / (max_val - min_val)
        # Fill NaN with 0 (could also use mean: normalized.fillna(0.5))
        normalized = normalized.fillna(0.0)
        return normalized
    
    # Apply normalization and reset indices to ensure proper alignment
    normalized_building = normalize_to_01(building_values).values  # Use .values to get array
    normalized_income = normalize_to_01(est_losses).values  # Use .values to get array
    normalized_damage = damage_levels.fillna(0.0).values  # Use .values to get array
    
    # Assign using .iloc to avoid index mismatch issues
    house_df = house_df.reset_index(drop=True)  # Reset index to be consecutive
    house_df['building_value'] = normalized_building
    house_df['income'] = normalized_income
    house_df['damage_level'] = normalized_damage

    # -------------------------------------------
    # Generate remaining features (using original logic)
    # -------------------------------------------
    
    # List of unique communities (needed for age and race generation)
    communities = house_df['community'].unique()
    
    # Population count (normalized to 0~1), assume 1~7 people household scaled
    house_df['population_scaled'] = np.random.randint(1, 8, size=len(house_df)) / 7.0

    # Randomly assign age-type labels to communities
    age_types = np.random.choice(['old', 'young', 'mixed'], size=len(communities), p=[0.3, 0.3, 0.4])
    community_age_map = dict(zip(communities, age_types))

    # Define age center for each community type with reduced variance
    age_centers = {
        'old': lambda: np.random.normal(loc=0.8, scale=0.15),
        'young': lambda: np.random.normal(loc=0.2, scale=0.15),
        'mixed': lambda: np.random.normal(loc=0.5, scale=0.15)
    }

    # Assign household age
    house_df['age'] = house_df['community'].apply(
        lambda c: np.clip(age_centers[community_age_map[c]](), 0, 1)
    )

    # Define race levels and sampling weights
    race_levels = [0.0, 0.25, 0.5, 1.0]
    race_weights = {
        r: [0.55 if i == j else 0.15 for j in range(4)]
        for i, r in enumerate(race_levels)
    }

    # Assign dominant race to each community (80% White, 20% others)
    shuffled = np.random.permutation(communities)
    dominant_race_by_community = {
        c: 0.0 if i < int(0.8 * len(communities)) else np.random.choice(race_levels[1:])
        for i, c in enumerate(shuffled)
    }

    # Sample household race based on community dominant race
    house_df['race'] = house_df['community'].apply(
        lambda c: np.random.choice(race_levels, p=race_weights[dominant_race_by_community[c]])
    )
    return house_df

