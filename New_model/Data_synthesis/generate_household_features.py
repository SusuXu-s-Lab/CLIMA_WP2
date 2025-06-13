import numpy as np

def generate_household_features(house_df):
    # Assume 'house_df' contains a column 'home' with 8-digit geohash codes
    # Extract the first 6 characters as community tags
    house_df['community'] = house_df['home'].str[:6]

    # List of unique communities
    communities = house_df['community'].unique()

    # -------------------------------------------
    # 1. Generate community-level baseline values
    # -------------------------------------------

    # For damage: select a few communities to have higher damage
    high_damage_communities = np.random.choice(communities, size=int(len(communities) * 0.15), replace=False)

    # Randomly select a subset of communities to have higher building values
    high_value_communities = np.random.choice(communities, size=int(len(communities) * 0.15), replace=False)
    # Assign building value center for each community, with high-value communities having higher mean
    building_value_centers = {
        comm: np.clip(np.random.normal(loc=0.7 if comm in high_value_communities else 0.5, scale=0.15), 0, 1)
        for comm in communities
    }

    # For income: simulate spatial disparity using a spatial gradient
    income_centers = {comm: np.clip(np.random.beta(a=2 if i < len(communities) // 2 else 0.5, b=2), 0, 1)
                      for i, comm in enumerate(communities)}

    # -------------------------------------------
    # 2. Assign household-level features
    # -------------------------------------------

    # Building value: small noise around community mean
    house_df['building_value'] = house_df['community'].apply(lambda c:
                                                             np.clip(np.random.normal(loc=building_value_centers[c],
                                                                                      scale=0.05), 0, 1))

    # Income: larger variance allowed
    house_df['income'] = house_df['community'].apply(lambda c:
                                                     np.clip(np.random.normal(loc=income_centers[c], scale=0.1), 0, 1))

    # Damage level: categorical with geographic bias
    damage_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    damage_probs_high = [0.1, 0.1, 0.2, 0.3, 0.3]  # 90% of being damaged
    damage_probs_normal = [0.7, 0.1, 0.1, 0.05, 0.05]  # 70% of being undamaged

    def sample_damage(c):
        probs = damage_probs_high if c in high_damage_communities else damage_probs_normal
        return np.random.choice(damage_levels, p=probs)

    house_df['damage_level'] = house_df['community'].apply(sample_damage)

    # Population count (normalized to 0~1), assume 1~7 people household scaled
    house_df['population_scaled'] = np.random.randint(1, 8, size=len(house_df)) / 7.0

    # Randomly assign age-type labels to communities
    age_types = np.random.choice(['old', 'young', 'mixed'], size=len(communities), p=[0.3, 0.3, 0.4])
    community_age_map = dict(zip(communities, age_types))

    # Define age center for each community type
    age_centers = {
        'old': lambda: np.random.normal(loc=0.8, scale=0.05),
        'young': lambda: np.random.normal(loc=0.2, scale=0.05),
        'mixed': lambda: np.random.normal(loc=0.5, scale=0.1)
    }

    # Assign household age
    house_df['age'] = house_df['community'].apply(
        lambda c: np.clip(age_centers[community_age_map[c]](), 0, 1)
    )

    # Define race levels and sampling weights
    race_levels = [0.0, 0.25, 0.5, 1.0]
    race_weights = {
        r: [0.7 if i == j else 0.1 for j in range(4)]
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

