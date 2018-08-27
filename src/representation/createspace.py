
from sklearn.decomposition import TruncatedSVD
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA
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

def createSVD(tf, depth):
    svd = TruncatedSVD(n_components=depth, algorithm="arpack") # use the scipy algorithm "arpack"
    pos = svd.fit_transform(tf)
    return pos

def createPCA(tf, depth):
    pca = PCA(n_components=depth)
    pos = pca.fit_transform(tf)
    return pos



def getDissimilarityMatrix(tf):
    tf = tf.transpose().astype(np.float32).toarray()
    docs_len = tf.shape[0]

    dm = np.empty([docs_len, docs_len], dtype="float32")
    pithing = 2/pi
    norms = np.empty(docs_len, dtype=np.float32)

    #Calculate norms
    for ei in range(docs_len):
        norms[ei] = np.linalg.norm(tf[ei])
        print("norm", ei)
    dot_product = np.empty([docs_len, docs_len], dtype="float32")

    #Calculate dot products
    for ei in range(docs_len):
        for ej in range(docs_len):
            dot_product[ei][ej] = np.dot(tf[ei], tf[ej])
        print("dp", ei)

    norm_multiplied = np.empty([docs_len, docs_len], dtype="float32")

    # Calculate dot products
    for ei in range(docs_len):
        for ej in range(docs_len):
            norm_multiplied[ei][ej] = norms[ei] * norms[ej]
        print("dp", ei)

    norm_multiplied = dt.shortenFloatsNoFn(norm_multiplied)
    dot_product = dt.shortenFloatsNoFn(dot_product)

    #Get angular differences
    for ei in range(docs_len):
        for ej in range(docs_len):
            ang = pithing * np.arccos(dot_product[ei][ej] / norm_multiplied[ei][ej])
            dm[ei][ej] = ang
        print(ei)
        dm[ei] = np.around(dm[ei], 4)
    return dm
import scipy.sparse as sp
import scipy.sparse.linalg
def getDissimilarityMatrixSparse(tf_transposed):
    tf_transposed = sp.csr_matrix(tf_transposed)
    tf = sp.csr_matrix.transpose(tf_transposed)
    tf = sp.csr_matrix(tf)
    docs_len = tf.shape[0]

    dm = np.zeros([docs_len, docs_len], dtype="float64")
    pithing = 2/pi
    #norms = np.zeros(docs_len, dtype="float64")
    s_norms = np.zeros(docs_len, dtype="float64")

    #Calculate norms
    for ei in range(docs_len):
        s_norms[ei] = sp.linalg.norm(tf[ei])
        if ei %100 == 0:
            print(ei)

    s_dot_product = np.zeros([docs_len, docs_len], dtype="float64")

    #Calculate dot products
    for ei in range(docs_len):
        for ej in range(docs_len):
            s_dp = tf[ei].dot(tf_transposed[:, ej])
            if len(s_dp.data) != 0:
                s_dot_product[ei][ej] = s_dp.data[0]
            print("dp", ej)
        print("dp", ei)

    norm_multiplied = np.zeros([docs_len, docs_len], dtype="float64")

    for ei in range(docs_len):
        for ej in range(docs_len):
            norm_multiplied[ei][ej] = s_norms[ei] * s_norms[ej]
        print("norms", ei)

    norm_multiplied = dt.shortenFloatsNoFn(norm_multiplied)
    s_dot_product = dt.shortenFloatsNoFn(s_dot_product)

    #Get angular differences
    for ei in range(docs_len):
        for ej in range(docs_len):
            ang = pithing * np.arccos(s_dot_product[ei][ej] / norm_multiplied[ei][ej])
            dm[ei][ej] = ang
        print(ei)
    return dm



def calcAngChunk(e1, e2,  norm_1, norm_2):
    dp = 0
    dp = np.dot(e1, e2)
    norm_dp = norm_1 * norm_2
    return (2 / pi) * np.arccos(dp / norm_dp)

def calcAngSparse(e1, e2, e2_transposed, norm_1, norm_2):
    dp = 0
    s_dp = e1.dot(e2_transposed)
    if s_dp.nnz != 0:
        dp = s_dp.data[0]
    norm_dp = norm_1 * norm_2
    return (2 / pi) * np.arccos(dp / norm_dp)

def getDsimMatrix(tf):
    tf = sp.csc_matrix(tf)
    tf_transposed = tf.transpose()
    tf = sp.csr_matrix(tf).astype("float32")
    docs_len = tf.shape[0]
    print(tf.shape)
    dm = np.zeros([docs_len, docs_len], dtype="float32")
    norms = np.zeros(docs_len, dtype="float32")

    #Calculate norms
    for ei in range(docs_len):
        norms[ei] = sp.linalg.norm(tf[ei])
        if ei %100 == 0:
            print("norms", ei)
    for i in range(docs_len):
        for j in range(i+1):
            dm[i][j] = calcAngSparse(tf[i], tf[j], tf_transposed[:,j], norms[i], norms[j])
            print(dm[i][j])
        print("i", i, "/", docs_len)
    return dm

def calcAng(e1, e2, norm1, norm2):
    return (2 / pi) * np.arccos(np.dot(e1, e2) / (norm1 * norm2))
import math

def getDsimMatrixDense(tf):
    #tf = np.asarray(tf.astype(np.float32).transpose().toarray())
    tf = np.asarray(tf.todense(), dtype=np.float32)
    docs_len = tf.shape[0]
    if tf.shape[0] > tf.shape[1]:
        print(tf.shape, "DOCS:", docs_len)
        raise ValueError("Probably wrong")
    dm = np.zeros([docs_len, docs_len], dtype=np.float32)
    dm2 = np.zeros([docs_len, docs_len], dtype=np.float32)
    norms = np.zeros(docs_len, dtype=np.float32)
    # Calculate norms
    for ei in range(docs_len):
        norms[ei] = np.linalg.norm(tf[ei])
        if ei % 1000 == 0:
            print("norms", ei)
    for i in range(docs_len):
        for j in range(i+1):

            dm[i][j] = calcAng(tf[i], tf[j], norms[i], norms[j])
            if math.isnan(dm[i][j]):
                dm[i][j] = 0.0
            #if j %1000 == 0:
            #    print("j", j)
        print("i", i)

    # Fill in the values of the mirrored array
    cr = 0
    for c in range(docs_len):
        for r in range(cr+1, docs_len):
            if math.isnan(dm[cr][r]):
                dm[cr][r] = 0
            dm[cr][r] = dm[r][c]
        cr += 1

    return dm

pithing = 2/pi
ang = pithing * np.arccos(0.1 / 0.1)
"""z
dt.write2dArray(getDissimilarityMatrix(dt.import2dArray("../data/sentiment/bow/ppmi/simple_numeric_stopwords_ppmi 5-all.npz", return_sparse=True)), "../data/sentiment/mds/simple_numeric_stopwords_ppmi 5-all")
"""

def main(data_type, clf, min, max, depth, rewrite_files):
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

    term_frequency_fn = init_vector_path = "../data/" + data_type + "/bow/ppmi/simple_numeric_stopwords_ppmi 2-all.npz"
    if dt.allFnsAlreadyExist([dm_fn, mds_fn, svd_fn, shorten_fn]):
        print("all files exist")
        exit()

    #Get MDS
    """
    tf = dt.import2dArray(term_frequency_fn).transpose()
    pca = sparseSVD(tf, depth)
    dt.write2dArray(pca, pca_fn)
    """

    # REMINDER: np.dot is WAY faster!
    tf = dt.import2dArray(term_frequency_fn, return_sparse=True)

    dm = getDsimMatrixDense(tf)
    dt.write2dArray(dm, dm_fn)
    print("wrote dm")

    """ Pretty sure none of this works
    if dt.allFnsAlreadyExist([mds_fn]) and not rewrite_files:
        mds = dt.import2dArray(mds_fn)
    else:
        print("starting mds")
        dm = np.asarray(dt.import2dArray(dm_shorten_fn)).transpose()
        mds = createMDS(dm, depth)
        dt.write2dArray(mds, mds_fn)
        print("wrote mds")

    # Create SVD
    if dt.allFnsAlreadyExist([shorten_fn]) and not rewrite_files:
        short = dt.import2dArray(shorten_fn)
        short = np.asarray(short).transpose()
    else:   
        print("starting svd")
        short = dt.shorten2dFloats(term_frequency_fn)
        dt.write2dArray(short, shorten_fn)
        tf = np.asarray(short).transpose()
        print("wrote shorten")

    if dt.allFnsAlreadyExist([svd_fn]) and not rewrite_files:
        svd = dt.import2dArray(svd_fn)
    else:
        print("begin svd")
        svd = createSVD(short, depth)
        dt.write2dArray(svd, svd_fn)
        print("wrote svd")

    if dt.allFnsAlreadyExist([pca_fn]) and not rewrite_files:
        pca = dt.import2dArray(pca_fn)
    else:
        print("begin pca")
        pca = createPCA(short, depth)
        dt.write2dArray(pca, pca_fn)
        print("wrote pca")
    """

data_type = "reuters"
clf = "all"

min=0
max=None
depth = 100

rewrite_files = True


if  __name__ =='__main__':main(data_type, clf, min, max, depth, rewrite_files)