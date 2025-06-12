import pandas as pd
import numpy as np
import math
import json, itertools
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from numpy.random import default_rng
import numba as nb

## Read T=0 partially observed social links
df_ori = pd.read_csv('Household_swinMaryland_2019-01-01.csv')
df_ori = df_ori[['home_1', 'home_2', 'home_1_number', 'home_2_number']]


