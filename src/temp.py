import multiprocessing
from os import getpid

def worker(procnum):
    print('I am number %d in process %d' % (procnum, getpid()))
    return [getpid(), getpid()*2, getpid()*3]

if __name__ == '__main__':
    pool = multiprocessing.Pool(processes = 1)
    kappa = pool.map(worker, range(3))
    print(kappa)
