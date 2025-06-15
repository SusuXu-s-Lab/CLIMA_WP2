import numpy as np

weights = np.array([-2.0, -1.0, -1.0,  # f_ij part
                    -1.0, -1.0, -1.0,  # s_i part
                    -1.0, -1.0, -1.0,  # s_j part
                    -10.0])  # dist_ij
print(weights/10)