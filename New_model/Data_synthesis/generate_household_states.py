import pandas as pd
import numpy as np

def generate_T0_states(house_df_with_features, T):
    # Initialize state dataframe: one row per household per time step
    # T = 24
    states = []

    # Compute community-level average damage
    community_damage = house_df_with_features.groupby('community')['damage_level'].mean().to_dict()

    for idx, row in house_df_with_features.iterrows():
        home_id = row['home']
        community = row['community']
        damage = row['damage_level']

        # At t = 0:
        # 1. Assign repair state: only possible if damage > 0
        if damage > 0:
            p_repair = min(0.2 + 0.6 * damage, 1.0)  # Higher damage → higher chance
            repair_0 = int(np.random.rand() < p_repair)
        else:
            repair_0 = 0

        # 2. Assign vacancy state based on community damage level
        p_vacant = min(0.03 + 0.2 * community_damage.get(community, 0), 0.3)
        vacant_0 = int(np.random.rand() < p_vacant)

        # 3. Assign sales state similarly
        p_sales = min(0.05 + 0.3 * community_damage.get(community, 0), 0.3)
        sales_0 = int(np.random.rand() < p_sales)

        for t in range(T):
            states.append({
                'home': home_id,
                'time': t,
                'repair_state': repair_0 if t == 0 else 0,
                'vacancy_state': vacant_0 if t == 0 else 0,
                'sales_state': sales_0 if t == 0 else 0
            })

    # Convert to DataFrame
    states_df = pd.DataFrame(states)
    return states_df