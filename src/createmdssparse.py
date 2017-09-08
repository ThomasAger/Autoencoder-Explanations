from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import FunctionTransformer
import data as dt
import numpy as np
import MovieTasks as mt
import scipy.sparse as sp
import data as dt
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA
from math import pi

def createMDS(dm, depth):
    dm = np.asarray(np.nan_to_num(dm), dtype="float64")
    mds = manifold.MDS(n_components=depth, max_iter=1000, eps=1e-9,
                   dissimilarity="precomputed", n_jobs=1)
    pos = mds.fit(dm).embedding_

    nmds = manifold.MDS(n_components=depth, metric=False, max_iter=1000, eps=1e-12,
                        dissimilarity="precomputed", n_jobs=1,
                        n_init=1)
    npos = nmds.fit_transform(dm.astype(np.float64), init=pos)

    return npos



def convertPPMISparse(mat):
    """
     Compute the PPMI values for the raw co-occurrence matrix.
     PPMI values will be written to mat and it will get overwritten.
     """
    (nrows, ncols) = mat.shape
    print("no. of rows =", nrows)
    print("no. of cols =", ncols)
    colTotals = mat.sum(axis=0)
    rowTotals = mat.sum(axis=1).T
    N = np.sum(rowTotals)

    P = mat.multiply(N)
    mat = None
    rowMat = np.ones((nrows, ncols), dtype=np.float)
    for i in range(nrows):
        if rowTotals[0, i] == 0: #
            rowMat[i, :] = 0
        else:
            rowMat[i, :] = rowMat[i, :] * (1.0 / rowTotals[0,i])
        print(i)
    P = P.multiply(rowMat)
    rowMat = None
    colMat = np.ones((nrows, ncols), dtype=np.float)
    for j in range(ncols):
        if colTotals[0,j] == 0:
            colMat[:,j] = 0
        else:
            colMat[:,j] = (1.0 / colTotals[0,j])
        print(j)
    P = P.multiply(colMat)
    colMat = None
    P = P.toarray()
    P = np.fmax(np.zeros((nrows,ncols), dtype=np.float64), np.log(P))
    return P

def convertPPMI(mat):
    """
     Compute the PPMI values for the raw co-occurrence matrix.
     PPMI values will be written to mat and it will get overwritten.
     """
    (nrows, ncols) = mat.shape
    print("no. of rows =", nrows)
    print("no. of cols =", ncols)
    colTotals = mat.sum(axis=0)
    rowTotals = mat.sum(axis=1).T
    N = np.sum(rowTotals)
    rowMat = np.ones((nrows, ncols), dtype=np.float)
    for i in range(nrows):
        rowMat[i, :] = 0 \
            if rowTotals[0,i] == 0 \
            else rowMat[i, :] * (1.0 / rowTotals[0,i])
        print(i)
    colMat = np.ones((nrows, ncols), dtype=np.float)
    for j in range(ncols):
        colMat[:,j] = 0 if colTotals[0,j] == 0 else (1.0 / colTotals[0,j])
        print(j)
    mat = mat.toarray()
    P = N * mat * rowMat * colMat
    P = np.fmax(np.zeros((nrows,ncols), dtype=np.float64), np.log(P))
    return P

def getDissimilarityMatrixSparse(tf):
    tflen = tf.shape[0]
    dm = np.empty([tflen, tflen], dtype="float64")
    pithing = 2/pi
    norms = np.empty(tflen, dtype="float64")

    #Calculate norms
    for ei in range(tflen):
        norms[ei] = sp.linalg.norm(tf[ei])
        print("norm", ei)
    dot_product = np.empty([tflen, tflen], dtype="float64")

    #Calculate dot products
    for ei in range(tflen):
        for ej in range(tflen):
            dot_product[ei][ej] = tf[ei].dot(tf[ej])
        print("dp", ei)

    norm_multiplied = np.empty([tflen, tflen], dtype="float64")

    # Calculate dot products
    for ei in range(len(tf)):
        for ej in range(len(tf)):
            norm_multiplied[ei][ej] = norms[ei] * norms[ej]
        print("dp", ei)

    norm_multiplied = dt.shortenFloatsNoFn(norm_multiplied)
    dot_product = dt.shortenFloatsNoFn(dot_product)

    #Get angular differences
    for ei in range(len(tf)):
        for ej in range(len(tf)):
            ang = pithing * np.arccos(dot_product[ei][ej] / norm_multiplied[ei][ej])
            dm[ei][ej] = ang
        print(ei)
    return dm


def getDissimilarityMatrix(tf):
    dm = np.empty([len(tf), len(tf)], dtype="float64")
    pithing = 2/pi
    norms = np.empty(len(tf), dtype="float64")

    #Calculate norms
    for ei in range(len(tf)):
        norms[ei] = np.linalg.norm(tf[ei])
        print("norm", ei)
    dot_product = np.empty([len(tf), len(tf)], dtype="float64")

    #Calculate dot products
    for ei in range(len(tf)):
        for ej in range(len(tf)):
            dot_product[ei][ej] = np.dot(tf[ei], tf[ej])
        print("dp", ei)

    norm_multiplied = np.empty([len(tf), len(tf)], dtype="float64")

    # Calculate dot products
    for ei in range(len(tf)):
        for ej in range(len(tf)):
            norm_multiplied[ei][ej] = norms[ei] * norms[ej]
        print("dp", ei)

    norm_multiplied = dt.shortenFloatsNoFn(norm_multiplied)
    dot_product = dt.shortenFloatsNoFn(dot_product)

    #Get angular differences
    for ei in range(len(tf)):
        for ej in range(len(tf)):
            ang = pithing * np.arccos(dot_product[ei][ej] / norm_multiplied[ei][ej])
            dm[ei][ej] = ang
        print(ei)
    return dm

def main(data_type, clf, highest_amt, lowest_amt, depth, rewrite_files):

    min = lowest_amt
    max = highest_amt
    dm_fn = "../data/" + data_type + "/mds/class-all-" + str(min) + "-" + str(max) \
                    + "-" + clf  + "dm"
    dm_shorten_fn = "../data/" + data_type + "/mds/class-all-" + str(min) + "-" + str(max) \
                    + "-" + clf  + "dmround"
    mds_fn = "../data/"+data_type+"/mds/class-all-" + str(min) + "-" + str(max) \
                                           + "-" + clf+ "d" + str(depth)
    svd_fn = "../data/"+data_type+"/svd/class-all-" + str(min) + "-" + str(max) \
                                           + "-" + clf + "d" + str(depth)
    pca_fn = "../data/"+data_type+"/pca/class-all-" + str(min) + "-" + str(max) \
                                           + "-" + clf + "d" + str(depth)
    shorten_fn = "../data/" + data_type + "/bow/ppmi/class-all-" + str(min) + "-" + str(max) \
                                           + "-" + clf+ "round"

    term_frequency_fn = init_vector_path = "../data/" + data_type + "/bow/ppmi/class-all-" + str(min) + "-" + str(max) \
                                           + "-" + clf
    if dt.allFnsAlreadyExist([dm_fn, mds_fn, svd_fn, shorten_fn]):
        print("all files exist")
        exit()
    newsgroups_train = fetch_20newsgroups(subset='train', shuffle=False)
    newsgroups_test = fetch_20newsgroups(subset='test', shuffle=False)


    vectors = newsgroups_test.data#np.concatenate((newsgroups_train.data, newsgroups_test.data), axis=0)
    newsgroups_test = None
    newsgroups_train = None
    # Get sparse tf rep
    tf_vectorizer = CountVectorizer(max_df=highest_amt, min_df=lowest_amt, stop_words='english')
    print("completed vectorizer")
    tf = tf_vectorizer.fit_transform(vectors)
    vectors = None
    # Get sparse PPMI rep from sparse tf rep
    print("done ppmisaprse")
    sparse_ppmi = convertPPMISparse(tf)
    # Get sparse Dsim matrix from sparse PPMI rep
    dm = getDissimilarityMatrix(sparse_ppmi)
    # Use as input to mds
    mds = createMDS(dm, depth)
    # save MDS
    dt.write2dArray(mds, mds_fn)

    #dt.write2dArray(dm, dm_fn)
    #print("wrote dm")





data_type = "newsgroups"
clf = "all"

highest_amt = 0.95
lowest_amt = 50
depth = 100

rewrite_files = True


if  __name__ =='__main__':main(data_type, clf, highest_amt, lowest_amt, depth, rewrite_files)