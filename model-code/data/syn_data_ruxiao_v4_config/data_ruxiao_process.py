import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
filename = ["G1_Sparse_LowSeed", "G1_Sparse_LowSeed_Structural",
            "G2_Sparse_HighSeed", "G2_Sparse_HighSeed_Structural",
            "G3_Medium_LowSeed", "G3_Medium_LowSeed_Structural",
            "G4_Medium_HighSeed", "G4_Medium_HighSeed_Structural",
            "G5_Dense_LowSeed", "G5_Dense_LowSeed_Structural",
            "G6_Dense_HighSeed", "G6_Dense_HighSeed_Structural"]

filename = ["G1_Sparse_LowSeed", 
            "G2_Sparse_HighSeed", 
            "G3_Medium_LowSeed", 
            "G4_Medium_HighSeed", 
            "G5_Dense_LowSeed", 
            "G6_Dense_HighSeed"]

filename = ["G1_Sparse_LowSeed_n50",  
            "G5_Dense_LowSeed_n50"]
for f in filename:
    data_dir = os.path.join(current_dir, "dataset", f)
    # Read all CSV files
    print("Reading CSV files...")
    # ground_truth_network = pd.read_csv(os.path.join(data_dir, "ground_truth_network_raw_rho_20%.csv"))
    # household_features = pd.read_csv(os.path.join(data_dir, "household_features_raw_rho_20%.csv"))
    # household_locations = pd.read_csv(os.path.join(data_dir, "household_locations_raw_rho_20%.csv"))
    # household_states = pd.read_csv(os.path.join(data_dir, "household_states_raw_rho_20%.csv"))
    # observed_network = pd.read_csv(os.path.join(data_dir, "observed_network_raw_rho_20%.csv"))

    # ground_truth_network = pd.read_csv(os.path.join(data_dir, "ground_truth_network_raw_with_log2_200node.csv"))
    # household_features = pd.read_csv(os.path.join(data_dir, "household_features_raw_with_log2_200node.csv"))
    # household_locations = pd.read_csv(os.path.join(data_dir, "household_locations_raw_with_log2_200node.csv"))
    # household_states = pd.read_csv(os.path.join(data_dir, "household_states_raw_with_log2_200node.csv"))
    # observed_network = pd.read_csv(os.path.join(data_dir, "observed_network_raw_with_log2_200node.csv"))

    ground_truth_network = pd.read_csv(os.path.join(data_dir, "ground_truth_network_raw.csv"))
    household_features = pd.read_csv(os.path.join(data_dir, "household_features_raw.csv"))
    household_locations = pd.read_csv(os.path.join(data_dir, "household_locations_raw.csv"))
    household_states = pd.read_csv(os.path.join(data_dir, "household_states_raw.csv"))
    observed_network = pd.read_csv(os.path.join(data_dir, "observed_network_raw.csv"))


    print("Data loaded successfully!")

    # Step 1: Create household ID mapping (string to integer)
    print("\nCreating household ID mapping...")

    # Collect all unique household IDs from all files
    all_household_ids = set()

    # From network files
    all_household_ids.update(ground_truth_network['household_id_1'].unique())
    all_household_ids.update(ground_truth_network['household_id_2'].unique())
    all_household_ids.update(observed_network['household_id_1'].unique())
    all_household_ids.update(observed_network['household_id_2'].unique())

    # From feature and location files
    all_household_ids.update(household_features['household_id'].unique())
    all_household_ids.update(household_locations['household_id'].unique())

    # From states file (assuming 'home' is household_id)
    all_household_ids.update(household_states['home'].unique())

    # Create mapping dictionary
    household_ids_sorted = sorted(list(all_household_ids))
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(household_ids_sorted, 1)}

    print(f"Total unique households: {len(household_ids_sorted)}")
    print(f"ID mapping created: {household_ids_sorted[:5]} -> [1, 2, 3, 4, 5]...")

    # Step 2: Process each file according to requirements

    # Process ground_truth_network_raw.csv
    print("\nProcessing ground_truth_network_raw.csv...")
    ground_truth_processed = ground_truth_network.copy()
    ground_truth_processed['household_id_1'] = ground_truth_processed['household_id_1'].map(id_mapping)
    ground_truth_processed['household_id_2'] = ground_truth_processed['household_id_2'].map(id_mapping)
    # Rename time_step if needed (it's already named correctly)
    ground_truth_processed.rename(columns={'time_step': 'timestep'}, inplace=True)

    # Process observed_network_raw.csv
    print("Processing observed_network_raw.csv...")
    observed_processed = observed_network.copy()
    observed_processed['household_id_1'] = observed_processed['household_id_1'].map(id_mapping)
    observed_processed['household_id_2'] = observed_processed['household_id_2'].map(id_mapping)

    observed_processed.rename(columns={'time_step': 'timestep'}, inplace=True)

    # Process household_features_raw.csv
    print("Processing household_features_raw.csv...")
    household_features_processed = household_features.copy()
    household_features_processed['household_id'] = household_features_processed['household_id'].map(id_mapping)
    # household_features_processed.drop(columns=['household_id'], inplace=True, errors='ignore')

    # One-hot encode community column
    community_dummies = pd.get_dummies(household_features_processed['community'], prefix='community')
    household_features_processed = pd.concat([household_features_processed.drop('community', axis=1), community_dummies], axis=1)
    print(f"Community one-hot encoding created: {list(community_dummies.columns)}")

    # household_features_processed.drop(columns=['community'], inplace=True, errors='ignore')

    # Process household_locations_raw.csv (note: fixing the typo in filename)
    print("Processing household_loactions_raw.csv...")
    household_locations_processed = household_locations.copy()
    household_locations_processed['household_id'] = household_locations_processed['household_id'].map(id_mapping)

    # Keep only 3 columns: household_id, latitude, longitude
    household_locations_processed = household_locations_processed[['household_id', 'latitude', 'longitude']]

    # Process household_states_raw.csv
    print("Processing household_states_raw.csv...")
    household_states_processed = household_states.copy()
    household_states_processed['home'] = household_states_processed['home'].map(id_mapping)

    # Rename 'time' column to 'timestep'
    if 'time' in household_states_processed.columns:
        household_states_processed = household_states_processed.rename(columns={'time': 'timestep'})

    # Remove "_state" suffix from column names
    column_rename_map = {}
    for col in household_states_processed.columns:
        if col.endswith('_state'):
            new_col = col.replace('_state', '')
            column_rename_map[col] = new_col

    if column_rename_map:
        household_states_processed = household_states_processed.rename(columns=column_rename_map)
        print(f"Renamed columns: {column_rename_map}")

    household_states_processed.rename(columns={'home': 'household_id','sales':'sell','vacancy':'vacant'}, inplace=True)


    # Step 3: Save processed files
    print("\nSaving processed files...")

    # Save with "_raw" suffix removed
    # ground_truth_processed.to_csv(os.path.join(data_dir, "ground_truth_network_community_one_hot_rho_20%.csv"), index=False)
    # print("Saved ground_truth_network.csv")

    # observed_processed.to_csv(os.path.join(data_dir, "observed_network_community_one_hot_rho_20%.csv"), index=False)
    # print("Saved observed_network.csv")

    # household_features_processed.to_csv(os.path.join(data_dir, "household_features_community_one_hot_rho_20%.csv"), index=False)
    # print("Saved household_features.csv")

    # household_locations_processed.to_csv(os.path.join(data_dir, "household_locations_community_one_hot_rho_20%.csv"), index=False)
    # print("Saved household_loactions.csv")

    # household_states_processed.to_csv(os.path.join(data_dir, "household_states_community_one_hot_rho_20%.csv"), index=False)
    # print("Saved household_states.csv")

    ground_truth_processed.to_csv(os.path.join(data_dir, "ground_truth_network_community_one_hot_with_log2.csv"), index=False)
    print("Saved ground_truth_network.csv")

    observed_processed.to_csv(os.path.join(data_dir, "observed_network_community_one_hot_with_log2.csv"), index=False)
    print("Saved observed_network.csv")

    household_features_processed.to_csv(os.path.join(data_dir, "household_features_community_one_hot_with_log2.csv"), index=False)
    print("Saved household_features.csv")

    household_locations_processed.to_csv(os.path.join(data_dir, "household_locations_community_one_hot_with_log2.csv"), index=False)
    print("Saved household_loactions.csv")

    household_states_processed.to_csv(os.path.join(data_dir, "household_states_community_one_hot_with_log2.csv"), index=False)
    print("Saved household_states.csv")

    # Print summary of changes
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)

    print(f"\n1. Household ID Mapping:")
    print(f"   - Total unique households: {len(household_ids_sorted)}")
    print(f"   - Mapped from strings to integers 1-{len(household_ids_sorted)}")

    # print(f"\n2. Community One-Hot Encoding:")
    # print(f"   - Original communities: {household_features['community'].unique()}")
    # print(f"   - Created columns: {list(community_dummies.columns)}")

    print(f"\n3. Household Locations:")
    print(f"   - Reduced from {len(household_locations.columns)} to 3 columns")
    print(f"   - Kept: household_id, latitude, longitude")

    print(f"\n4. Column Renaming:")
    if 'time' in household_states.columns:
        print(f"   - Renamed 'time' to 'timestep' in household_states")
    else:
        print(f"   - No 'time' column found to rename")

    # Show _state suffix removal
    state_columns = [col for col in household_states.columns if col.endswith('_state')]
    if state_columns:
        renamed_state_columns = [col.replace('_state', '') for col in state_columns]
        print(f"   - Removed '_state' suffix: {state_columns} -> {renamed_state_columns}")
    else:
        print(f"   - No '_state' columns found to rename")

    print(f"\n5. Files Saved:")
    for filename in ["ground_truth_network.csv", "observed_network.csv", 
                    "household_features.csv", "household_loactions.csv", "household_states.csv"]:
        print(f"   - {filename}")

    print("\nProcessing completed successfully!")