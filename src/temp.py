from scipy.sparse import csr_matrix
import numpy as np
from scipy import sparse
from numpy.random import RandomState
import timeit
import time
A = sparse.rand(1,90000, 0.1, random_state = RandomState(1))
v = sparse.rand(1,90000, 0.1, random_state = RandomState(1))
current_time = time.time()
for i in range(18000):
    A.dot(v.T)[0,0]
new_time = time.time()
print(new_time - current_time)

for i in range(18000):
    np.dot(A, v.T)[0,0]
print(time.time() - new_time)