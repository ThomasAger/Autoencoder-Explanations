import data as dt
import numpy as np

class_all = "../data/movies/classify/keywords/class-all"

real_class = np.asarray(dt.import2dArray(class_all)).transpose()

for i in real_class:
    count = 0
    for j in i:
        if j == 1:
            count+=1
    print(count)