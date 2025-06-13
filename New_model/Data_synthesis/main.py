import pdb
from generate_household_features import generate_household_features
from generate_household_states import generate_T0_states
import pandas as pd
from household_features_function import compute_similarity, compute_interaction_potential

import math
import json, itertools

from scipy.spatial.distance import pdist, squareform
from numpy.random import default_rng
import numba as nb

'''
Read real Household Features
'''
## Read T=0 partially observed social links
df_ori = pd.read_csv('household_swinMaryland_20190101.csv')
df_ori = df_ori[['home_1', 'home_2', 'home_1_number', 'home_2_number']]

# Extract home and their locations
home1 = df_ori[['home_1']].rename(columns={'home_1': 'home'})
home2 = df_ori[['home_2']].rename(columns={'home_2': 'home'})

# Merge and remove duplicates
house_df = pd.concat([home1, home2], ignore_index=True).drop_duplicates(subset='home')
house_df = house_df.dropna()


'''
Household Feautres Generation
'''
house_df_with_features = generate_household_features(house_df)

'''
T=0 Household States Generation
'''
house_states=generate_T0_states(house_df_with_features,24)

'''
T=0 Household Similarity and Intercation Potential
'''
# Run both functions on provided example data
similarity_df = compute_similarity(house_df_with_features)
interaction_df = compute_interaction_potential(house_df_with_features, house_states, t=0)


# Optional: drop community column if not needed
# house_df.drop(columns=['community'], inplace=True)