import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import pdb

bbox = (-81.97581623909956, 26.504952873345545, -81.95459566656734, 26.52311972089953)   # (minx, miny, maxx, maxy)

feaures_df = pd.read_csv('fl/fl/house_features.csv')

minx, miny, maxx, maxy = bbox


feaures_df = feaures_df[
    (feaures_df['lon'] >= minx) & (feaures_df['lon'] <= maxx) &
    (feaures_df['lat'] >= miny) & (feaures_df['lat'] <= maxy)
]

print('sales_df_filtered:\n',feaures_df.head())
print('repair_df_filtered:\n',feaures_df.head())
feaures_df.to_csv('fl/fl/small_house_features.csv')