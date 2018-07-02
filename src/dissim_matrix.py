import numpy as np
import scipy.sp as sp
from math import pi

# Just a function to import a 2d array. specify the file_type as f = float, i = integer. In this case its a float.
def import2dArray(file_name, file_type="f", return_sparse=False):
    if file_name[-4:] == ".npz":
        print("Loading sparse array")
        array = sp.load_npz(file_name)
        if return_sparse is False:
            array = array.toarray()
    elif file_name[-4:] == ".npy":
        print("Loading numpy array")
        array = np.load(file_name)#
    else:
        with open(file_name, "r") as infile:
            if file_type == "i":
                array = [list(map(int, line.strip().split())) for line in infile]
            elif file_type == "f":
                array = [list(map(float, line.strip().split())) for line in infile]
            elif file_type == "discrete":
                array = [list(line.strip().split()) for line in infile]
                for dv in array:
                    for v in range(len(dv)):
                        dv[v] = int(dv[v][:-1])
            else:
                array = np.asarray([list(line.strip().split()) for line in infile])
        array = np.asarray(array)
    print("successful import", file_name)
    return array

# Just a function to write a 2d array to a text file. Also produces a numpy file
def write2dArray(array, name):
    try:
        file = open(name, "w")
        print("starting array")
        for i in range(len(array)):
            for n in range(len(array[i])):
                file.write(str(array[i][n]) + " ")
            file.write("\n")
        file.close()
    except FileNotFoundError:
        print("FAILURE")
    try:
        if name[-4:] == ".txt":
            name = name[:-4]
        array = np.asarray(array)
        np.save(name, array)
    except:
        print("failed")

    print("successful write", name)

# All this stuff is what we actually wnt to do
def calcAngSparse(e1, e2, e2_transposed, norm_1, norm_2):
    dp = 0
    # Get the dot product
    s_dp = e1.dot(e2_transposed)
    if s_dp.nnz != 0:
        dp = s_dp.data[0]
    # Multiply the norms
    norm_dp = norm_1 * norm_2
    # Get the dissimilarity
    return (2 / pi) * np.arccos(dp / norm_dp)

def getDsimMatrix(tf):
    # Convert to sparse matrix
    tf_transposed = sp.csc_matrix(tf)
    tf = tf.transpose()
    # Convert to float32 to save space
    tf = sp.csr_matrix(tf).astype("float32")
    docs_len = tf.shape[0]
    print(docs_len, "If this is not your amount of documents, then it needs to be transposed")
    # Create empty dissim/norm matrix
    dm = np.zeros([docs_len, docs_len], dtype="float32")
    norms = np.zeros(docs_len, dtype="float32")
    # Get the norms
    for ei in range(docs_len):
        norms[ei] = sp.linalg.norm(tf[ei])
        if ei %100 == 0:
            print("norms", ei)
    # Get the dissimilarities
    for i in range(docs_len):
        for j in range(i+1):
            dm[i][j] = calcAngSparse(tf[i], tf[j], tf_transposed[:,j], norms[i], norms[j])
            if j %10000 == 0:
                print("j", j)
        print("i", i)
    return dm

if __name__ == '__main__':
    term_frequency_fn = "bow fn" # This is your bag of words
    dm_fn = "output dissimilarity matrix fn" # This is where you want the space to output. Make sure it exists

    tf = import2dArray(term_frequency_fn)
    dm = getDsimMatrix(tf)
    write2dArray(dm, dm_fn)