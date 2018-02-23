from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
import numpy as np

# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Mathieu Blondel <mathieu@mblondel.org>
#         Lars Buitinck
# License: BSD 3 clause


import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
import scipy.sparse as sp
import MovieTasks as mt
import data as dt
import tensorflow
np.set_printoptions(suppress=True)
import math
# mt.printIndividualFromAll("sentim
from math import pi
import csr_csc_dot as ccd

def convertPPMISparse(mat):
    """
     Converted from code from svdmi
     https://github.com/Bollegala/svdmi/blob/master/src/svdmi.py
     """
    (nrows, ncols) = mat.shape
    print("no. of rows =", nrows)
    print("no. of cols =", ncols)
    colTotals = mat.sum(axis=0)
    rowTotals = mat.sum(axis=1).T
    N = np.sum(rowTotals)
    rowMatSparse = np.zeros(nrows, dtype=np.float)
    for i in range(nrows):
        if rowTotals[0, i] != 0:
            rowMatSparse[i] = 1.0 / rowTotals[0, i]
    colMatSparse = np.zeros(ncols, dtype=np.float)
    for j in range(ncols):
        if colTotals[0, j] != 0:
            colMatSparse[j] = 1.0 / colTotals[0, j]
    P = N * mat
    P = P.astype(np.float64)
    for i in range(len(rowMatSparse)):
        P[i] *= rowMatSparse[i]
    for i in range(len(colMatSparse)):
        P[:,i] *= colMatSparse[i]
    cx = sp.coo_matrix(P)
    for i, j, v in zip(cx.row, cx.col, cx.data):
        P[i,j] = max(math.log(v), 0)
    return P

def getDissimilarityMatrix(tf):

    tf = tf.toarray()
    tf = tf.transpose()
    docs_len = tf.shape[0]

    dm = np.empty([docs_len, docs_len], dtype="float64")
    pithing = 2/pi
    norms = np.empty(docs_len, dtype="float64")

    #Calculate norms
    for ei in range(docs_len):
        norms[ei] = np.linalg.norm(tf[ei])
        print("norm", ei)
    dot_product = np.empty([docs_len, docs_len], dtype="float64")

    #Calculate dot products
    for ei in range(docs_len):
        for ej in range(docs_len):
            dot_product[ei][ej] = np.dot(tf[ei], tf[ej])
        print("dp", ei)

    norm_multiplied = np.empty([docs_len, docs_len], dtype="float64")

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
    return dm
import scipy.sparse.linalg
import sklearn.metrics.pairwise as smp
def getDissimilarityMatrixSparse(tf_transposed):
    tf_transposed = sp.csr_matrix(tf_transposed)
    tf = sp.csr_matrix.transpose(tf_transposed)
    tf = sp.csr_matrix(tf)
    flat_tf = tf.toarray()
    docs_len = tf.shape[0]

    dm = np.zeros([docs_len, docs_len], dtype="float64")
    pithing = 2/pi
    #norms = np.zeros(docs_len, dtype="float64")
    s_norms = np.zeros(docs_len, dtype="float64")

    #Calculate norms
    for ei in range(docs_len):
        #norms[ei] = np.linalg.norm(flat_tf[ei])
        s_norms[ei] = sp.linalg.norm(tf[ei])
        #print("norm", ei, norms[ei])
        #print("s_norm", ei, s_norms[ei]
        if ei %100 == 0:
            print(ei)

    #dot_product = np.zeros([docs_len, docs_len], dtype="float64")
    s_dot_product = np.zeros([docs_len, docs_len], dtype="float64")

    #Calculate dot products
    for ei in range(docs_len):
        for ej in range(docs_len):
            s_dp = tf[ei].dot(tf_transposed[:, ej])
            if len(s_dp.data) != 0:
                s_dot_product[ei][ej] = s_dp.data[0]
            print("dp", ej)
            #print("dp", ei, docs_len, ej, dot_product[ei][ej])
            #print("s_dp", ei, docs_len, ej, s_dot_product[ei][ej])
        print("dp", ei)

    norm_multiplied = np.zeros([docs_len, docs_len], dtype="float64")

    # Calculate dot products
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

    for ei in range(docs_len):
        for ej in range(docs_len):
            dm[ei][ej] = calcAngSparse(tf[ei], tf[ej])
        print(ei)
    return dm


def calcAng(e1, e2):
    return (2 / pi) * np.arccos(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))


def calcAngSparse(e1, e2, e2_transposed, norm_1, norm_2):
    dp = 0
    s_dp = e1.dot(e2_transposed)
    if s_dp.nnz != 0:
        dp = s_dp.data[0]
    norm_dp = norm_1 * norm_2
    return (2 / pi) * np.arccos(dp / norm_dp)

import timeit

def calcAngChunk(e1, e2,  norm_1, norm_2):
    dp = 0
    dp = np.dot(e1, e2)
    norm_dp = norm_1 * norm_2
    return (2 / pi) * np.arccos(dp / norm_dp)

def getDsimMatrixChunk(tf, chunk):
    tf_transposed = sp.csc_matrix(tf)
    tf = tf.transpose()
    tf = sp.csr_matrix(tf)
    docs_len = tf.shape[0]
    dm = np.zeros([docs_len, docs_len], dtype="float64")
    norms = np.zeros(docs_len, dtype="float64")

    #Calculate norms
    for ei in range(docs_len):
        norms[ei] = sp.linalg.norm(tf[ei])
        if ei %100 == 0:
            print("norms", ei)



    for c in range(int(docs_len/chunk)):
        chunked_tf = tf[c*chunk: c+1*chunk].toarray()
        for i in range(c*chunk, c+1*chunk):
            for i in range(c * chunk, c + 1 * chunk):
                dm[i][j] = calcAngChunk(chunked_tf[i], chunked_tf[j],  norms[i], norms[j])
                if j %10000 == 0:
                    print("j", j)
            print("i", i)
    return dm

def getDsimMatrix(tf):

    tf_transposed = sp.csc_matrix(tf)
    tf = tf.transpose()
    tf = sp.csr_matrix(tf).astype(np.float64)
    docs_len = tf.shape[0]
    dm = np.zeros([docs_len, docs_len], dtype=np.float64)
    norms = np.zeros(docs_len, dtype=np.float64)

    #Calculate norms
    for ei in range(docs_len):
        norms[ei] = sp.linalg.norm(tf[ei])
        if ei %100 == 0:
            print("norms", ei)
    for i in range(docs_len):
        for j in range(docs_len):
            dm[i][j] = calcAngSparse(tf[i], tf[j], tf_transposed[:,j], norms[i], norms[j])
            if j %10000 == 0:
                print("j", j)
        print("i", i)
        break
    return dm

def getDsimMatrixDense(tf):
    tf = tf.transpose().toarray()
    docs_len = tf.shape[0]
    dm = np.zeros([docs_len, docs_len], dtype="float32")

    for i in range(docs_len):
        for j in range(docs_len):
            dm[i][j] = calcAng(tf[i], tf[j])
            if j %10000 == 0:
                print("j", j)
        print("i", i)
    return dm

import cProfile
import re
"""
tf = np.random.uniform(low=0.0, high=8.0, size=(1000, 1000))
for i in range(len(tf)):
    for j in range(len(tf[i])):
        if np.random.randint(low=0, high=2, size=1) == 0:
            tf[i][j] = 0

ppmi = sp.csr_matrix(tf)
print("saving")
sp.save_npz("../data/temp/big_matrix", ppmi)
print("saved")
"""
#ppmi = sp.load_npz("../data/temp/big_matrix.npz")

#cProfile.run('getDsimMatrix(ppmi)')

"""
orig_dm = getDsimMatrixDense(ppmi)
sparse_dm = getDissimilarityMatrix(ppmi)

for i in range(len(orig_dm)):
    broke = False
    for j in range(len(sparse_dm[i])):
        if orig_dm[i][j] != sparse_dm[i][j]:
            print(i, j, "sparse", sparse_dm[i][j], "non-sparse", orig_dm[i][j])
            broke = True
    if broke is False:
        print("Clear")
print("done")
"""

def calcAngSparse(e1, e2, e2_transposed, norm_1, norm_2):
    dp = 0
    s_dp = e1.dot(e2_transposed)
    if s_dp.nnz != 0:
        dp = s_dp.data[0]
    norm_dp = norm_1 * norm_2
    return (2 / pi) * np.arccos(dp / norm_dp)


def getDsimMatrix(tf, chunk):

    tf_transposed = sp.csc_matrix(tf)
    tf = tf.transpose()
    tf = sp.csr_matrix(tf).astype("float32")
    docs_len = tf.shape[0]
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
            if j %10000 == 0:
                print("j", j)
        print("i", i)
    return dm


#ppmi = sp.load_npz("../data/temp/big_matrix.npz")

tf = np.random.uniform(low=0.0, high=8.0, size=(10, 10))
for i in range(len(tf)):
    for j in range(len(tf[i])):
        if np.random.randint(low=0, high=2, size=1) == 0:
            tf[i][j] = 0

ppmi = sp.csr_matrix(tf)

sparse_dm = getDsimMatrix(ppmi)

print("dun")

#cProfile.run('getDsimMatrix(ppmi)')
"""
#Get some test matrices
m1=np.random.rand(5,500)
m2=np.random.rand(10,500)
m1_csr=csr_matrix(m1,dtype=np.float32)  #Sparse matrix 1
m2_csc=csc_matrix(m2,dtype=np.float32)  #Sparse_matrix 2
out=np.zeros((m1.shape[0],m2.shape[0]),np.float32)
# fast m1.dot(m2.T)

print("Sparse dot result")
print(out)


predicted_classes = [[0,1,1,0,1,1,1,0,1,0,1,1,1,0,0,0,0,0,0,0,1,1,1,1],
                     [0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1],
                     [0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1]]

real_classes = [[0,0,1,0,0,1,1,1,1,1,1,0,1,1,0,0,0,0,1,0,1,0,1,0],
                     [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0],
                     [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1]]

clf = tree.DecisionTreeClassifier(max_depth=3, criterion="entropy", class_weight="balanced")
clf.fit(predicted_classes, real_classes)


"""

""" test f1 score
#Multi-label fake data
predicted_classes = [[0,1,1,0,1,1,1,0,1,0,1,1,1,0,0,0,0,0,0,0,1,1,1,1],
                     [0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1],
                     [0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1]]

real_classes = [[0,0,1,0,0,1,1,1,1,1,1,0,1,1,0,0,0,0,1,0,1,0,1,0],
                     [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0],
                     [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1]]

# Calculating each class individually

precisions = np.zeros(3)
recalls = np.zeros(3)
f1_betas = np.zeros(3)
f1s = np.zeros(3)
calced_f1s = np.zeros(3)

for i in range(len(predicted_classes)):
    prec, recall, fbeta, score = precision_recall_fscore_support(real_classes[i], predicted_classes[i], average="binary")
    precisions[i] = prec
    recalls[i] = recall
    f1_betas[i] = fbeta
    calced_f1s[i] = 2 * ((prec * recall) / (prec + recall))
    f1s[i] = f1_score(real_classes[i], predicted_classes[i], average="binary")

# Averaging F1 scores of every class

print("If these two are the same, it means that f1_beta = f1_score")
print("stupid boi mean f1 beta", np.average(f1_betas))
print("stupid boi mean f1 score", np.average(f1s))

# Calculating Macro average with multi-label input
# Micro average with multi-label input
print("macro f1 score", f1_score(real_classes, predicted_classes, average="macro"))
print("micro f1 score", f1_score(real_classes, predicted_classes, average="micro"))

# Averaging precision recall and then calculating F1 score for every class
average_prec = np.average(precisions)
average_recall = np.average(recalls)
f1 = 2 * ((average_prec * average_recall) / (average_prec + average_recall))
print("")
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11")
print("f1 calced from average prec + average recall", f1)
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11")
print("")
# Same calculation but averaged afterwards

print("stupid boi mean f1 but calced using my formula", np.average(calced_f1s))

real_classes = np.asarray(real_classes).transpose()
predicted_classes = np.asarray(predicted_classes).transpose()

prec, recall, fbeta, score = precision_recall_fscore_support(real_classes, predicted_classes, average="macro")

f1 = 2 * ((prec * recall) / (prec + recall))
print("")
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11")
print("f1 from formula, multi-label, macro average prec recall fscore support", f1)
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11")
print("")
print("f1 from fbeta, multi-label, macro average prec recall fscore support", fbeta)

prec, recall, fbeta, score = precision_recall_fscore_support(real_classes, predicted_classes, average="micro")

f1 = 2 * ((prec * recall) / (prec + recall))

print("f1 from formula, multi-label, micro average prec recall fscore support", f1)
print("f1 from fbeta, multi-label, micro average prec recall fscore support", fbeta)

"""




