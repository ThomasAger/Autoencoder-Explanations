
import timeit
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
import skcuda.linalg as linalg
from scipy.sparse import random
import skcuda.misc as misc
import scipy.sparse as sp
import cProfile

def cpuDot(a_S, b_S):
    for i in range(len(a_S)):
        np.dot(a_S[i], b_S[i])

def sparseDot(a_S, b_S):
    for i in range(a_S.shape[0]):
        a_S.dot(b_S)

def gpuDot(a_gpu, b_gpu):
    for i in range(len(a_gpu)):
        linalg.dot(a_gpu[i], b_gpu[i])

if __name__ == '__main__':
    a = np.asarray(np.random.rand(1500, 1500), np.float32)
    b = np.asarray(np.random.rand(1500, 1500), np.float32)
    a_S = sp.csr_matrix(a)
    b_S = sp.csr_matrix(b)
    linalg.init()
    a_gpu = gpuarray.to_gpu(a)
    b_gpu = gpuarray.to_gpu(b)
    print(cProfile.run("cpuDot(a, b)"))
    print(cProfile.run("gpuDot(a_gpu, b_gpu)"))
    print(cProfile.run("sparseDot(a_S, b_S)"))

"""
d = np.asarray(np.random.rand(5), np.float32)
e = np.asarray(np.random.rand(5), np.float32)
d_gpu = gpuarray.to_gpu(d)
e_gpu = gpuarray.to_gpu(e)
f = linalg.dot(d_gpu, e_gpu)

np.allclose(np.dot(d, e), f)
"""