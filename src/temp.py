import numpy as np

from sklearn.model_selection import KFold


deep_size = [100,100,100]
for d in range(len(deep_size)):
    deep_size = deep_size[1:]
    print(deep_size)