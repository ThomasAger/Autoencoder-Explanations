
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
    svd = TruncatedSVD(n_components=depth)
    pos = svd.fit_transform(tf)
    return pos

def createPCA(tf, depth):
    pca = PCA(n_components=depth)
    pos = pca.fit_transform(tf)
    return pos

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

    term_frequency_fn = init_vector_path = "../data/" + data_type + "/bow/ppmi/class-all-" + str(min) + "-" + str(max) \
                                           + "-" + clf
    if dt.allFnsAlreadyExist([dm_fn, mds_fn, svd_fn, shorten_fn]):
        print("all files exist")
        exit()

    tf = None

    #Get MDS

    if dt.allFnsAlreadyExist([dm_shorten_fn]) is False:
        if dt.allFnsAlreadyExist([shorten_fn]) and not rewrite_files:
            tf = dt.import2dArray(shorten_fn)
        else:
            short = dt.shorten2dFloats(term_frequency_fn)
            dt.write2dArray(short, shorten_fn)
            tf = np.asarray(short).transpose()
            print("wrote shorten")

        if dt.allFnsAlreadyExist([dm_fn]) and not rewrite_files:
            dm = dt.import2dArray(dm_fn)
            print("read dm")
        else:
            dm = getDissimilarityMatrix(tf)
            dt.write2dArray(dm, dm_fn)
            print("wrote dm")

        dt.write2dArray(dt.shorten2dFloats(dm_fn), dm_shorten_fn)
        dm = dt.import2dArray(dm_shorten_fn)
        print("wrote shorten")

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

data_type = "wines"
clf = "all"

min=50
max=10
depth = 100

rewrite_files = True


if  __name__ =='__main__':main(data_type, clf, min, max, depth, rewrite_files)