import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import helper.data as dt
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import check_random_state
from collections import defaultdict
import copy
def plot(x, y, y_):
    segments = [[[i, y[i]], [i, y_[i]]] for i in range(len(x))]
    lc = LineCollection(segments, zorder=0)
    lc.set_array(np.ones(len(y)))
    lc.set_linewidths(0.5 * np.ones(len(x)))
    fig = plt.figure()
    plt.plot(x, y, 'r.', markersize=2)
    plt.plot(x, y_, 'g.', markersize=12)
    plt.legend(('Data', 'Isotonic Fit'), loc='lower right')
    plt.title('Isotonic regression')
    plt.show()

def readPPMI(name):
    name = name.split()[0]
    file = open("../data/movies/bow/ppmi/" + "class-" + name)
    lines = file.readlines()
    frq_a = []
    for line in lines:
        frq_a.append(float(line))
    return frq_a

def pavPPMI(cluster_names_fn, ranking_fn, file_name, do_p=False, data_type="movies"):
    ranking = dt.import2dArray(ranking_fn)
    names = dt.import1dArray(cluster_names_fn)
    frq = []
    counter = 0

    for name in names:
        frq.append(readPPMI(name))

    pav_classes = []

    for f in range(len(frq)):
        print(names[f])
        x = np.asarray(frq[f])
        y = ranking[f]

        ir = IsotonicRegression()
        y_ = ir.fit_transform(x, y)
        pav_classes.append(y_)
        if do_p:
            plot(x, y, y_)
        print(f)

    dt.write2dArray(pav_classes, "../data/" + data_type + "/finetune/" + file_name + "pavPPMI.txt")
    return pav_classes

def readFreq(name):
    file = open("../data/movies/bow/frequency/phrases/" + "class-" + name)
    lines = file.readlines()
    frq_a = []
    for line in lines:
        frq_a.append(float(line))
    return frq_a

# OUTPUT: Matrix of Cluster X Movies where a property is 1 for a movie if that movie contains any cluster terms
# pavTermFrequency: The Isotonic regression between the ranks and the term frequency
# pavPPMI: The Isotonic regression between the ranks and the ppmi
# termFrequency: The term frequencies for the clusters
# normalizedTermFrequency: The term frequencies normalized for the clusters
# binaryClusterTerm: 0 if the cluster name is in the reviews, 1 if it isn't
# binaryInCluster: 0 if any names the cluster is composed of is in the reviews, 1 if it isn't
from random import randint
def maxNonZero(cluster_names_fn, fn):
    all_cluster_output = []
    cluster_names = dt.import1dArray(cluster_names_fn)
    for cn in cluster_names:
        binary = np.asarray(dt.import1dArray("../data/movies/bow/frequency/phrases/class-" + cn, "f"))
        random_binary = []
        for b in binary:
            if b > 0:
                random_binary.append(np.amax(binary))
            else:
                random_binary.append(0)
        all_cluster_output.append(random_binary)
    dt.write2dArray(all_cluster_output, "../data/movies/finetune/" + fn + "MaxNonZero.txt")

def randomNonZero(cluster_names_fn, fn):
    all_cluster_output = []
    cluster_names = dt.import1dArray(cluster_names_fn)
    for cn in cluster_names:
        binary = np.asarray(dt.import1dArray("../data/movies/bow/frequency/phrases/class-" + cn, "f"))
        random_binary = []
        for b in binary:
            if b > 0:
                random_binary.append(randint(1, np.amax(binary)))
            else:
                random_binary.append(0)
        all_cluster_output.append(random_binary)
    dt.write2dArray(all_cluster_output, "../data/movies/finetune/" + fn + "RandomNonZero.txt")

def maxAll(cluster_names_fn, fn):
    all_cluster_output = []
    cluster_names = dt.import1dArray(cluster_names_fn)
    for cn in cluster_names:
        binary = np.asarray(dt.import1dArray("../data/movies/bow/frequency/phrases/class-" + cn, "f"))
        random_binary = []
        for b in binary:
            random_binary.append(np.amax(binary))
        all_cluster_output.append(random_binary)
    dt.write2dArray(all_cluster_output, "../data/movies/finetune/" + fn + "MaxAll.txt")

def randomAll(cluster_names_fn, fn):
    all_cluster_output = []
    cluster_names = dt.import1dArray(cluster_names_fn)
    for cn in cluster_names:
        binary = np.asarray(dt.import1dArray("../data/movies/bow/frequency/phrases/class-" + cn, "f"))
        random_binary = []
        for b in binary:
            random_binary.append(randint(0, np.amax(binary)))
        all_cluster_output.append(random_binary)
    dt.write2dArray(all_cluster_output, "../data/movies/finetune/" + fn + "RandomAll.txt")

def pavTermFrequency(ranking_fn, cluster_names_fn, fn, plot):
    ranking = dt.import2dArray(ranking_fn)
    names = dt.import1dArray(cluster_names_fn)
    frq = []
    counter = 0

    for name in names:
        frq.append(readFreq(name))

    pav_classes = []

    for f in range(len(frq)):
        print(names[f])
        x = np.asarray(frq[f])
        y = ranking[f]

        ir = IsotonicRegression()
        y_ = ir.fit_transform(x, y)
        pav_classes.append(y_)
        if plot:
            plot(x, y, y_)
        print(f)

    dt.write2dArray(pav_classes, "../data/movies/finetune/" + file_name + "PavTermFrequency.txt")
    return pav_classes

def PPMI(cluster_names_fn, fn):
    all_cluster_output = []
    cluster_names = dt.import1dArray(cluster_names_fn)
    for cn in cluster_names:
        binary = dt.import1dArray("../data/movies/bow/ppmi/class-class-" + cn, "f")
        all_cluster_output.append(binary)
    dt.write2dArray(all_cluster_output, "../data/movies/finetune/" +fn + "PPMI.txt")

def termFrequency(cluster_names_fn, fn):
    all_cluster_output = []
    cluster_names = dt.import1dArray(cluster_names_fn)
    for cn in cluster_names:
        binary = dt.import1dArray("../data/movies/bow/frequency/phrases/class-" + cn, "i")
        all_cluster_output.append(binary)
    dt.write2dArray(all_cluster_output, "../data/movies/finetune/" +fn + "TermFrequency.txt")

def normalizedTermFrequency(cluster_names_fn, fn):
    all_cluster_output = []
    cluster_names = dt.import1dArray(cluster_names_fn)
    for cn in cluster_names:
        binary = dt.import1dArray("../data/movies/bow/frequency/phrases/class-" + cn, "i")
        all_cluster_output.append(binary)
    new_output = dt.scaleSpaceUnitVector(all_cluster_output, "../data/movies/finetune/" + fn + "NormalizedTermFrequency.txt")

def binaryClusterTerm(cluster_names_fn, fn):
    all_cluster_output = []
    cluster_names = dt.import1dArray(cluster_names_fn)
    for cn in cluster_names:
        binary = dt.import1dArray("../data/movies/bow/binary/phrases/class-" + cn, "i")
        all_cluster_output.append(binary)
    dt.write2dArray(all_cluster_output, "../data/movies/finetune/" +fn + "ClusterTerm.txt")

def binaryInCluster(cluster_dict_fn, fn):
    cluster = dt.readArrayDict(cluster_dict_fn)
    all_cluster_output = []
    for key, items in cluster.items():
        init_binary = dt.import1dArray("../data/movies/bow/binary/phrases/" + key, "i")
        for i in items:
            binary = dt.import1dArray("../data/movies/bow/binary/phrases/" + i, "i")
            for j in range(len(init_binary)):
                if binary[j] == 1:
                    init_binary[j] = 1
        all_cluster_output.append(init_binary)
    dt.write2dArray(all_cluster_output, "../data/movies/finetune/" +fn + "InCluster.txt")




class PAV:
    def __init__(self, property_names_fn, discrete_labels_fn, ppmi_fn, file_name, cluster_dict_fn):
        getPAVGini(cluster_dict_fn, discrete_labels_fn, ppmi_fn, file_name)

file_name = "films200L1100N0.5"
score_limit = 0.8
cluster_names_fn = "../data/movies/cluster/names/" + file_name + ".txt"
cluster_dict_fn = "../data/movies/cluster/dict/" + file_name + ".txt"
#cluster_names_fn = "../data/movies/cluster/hierarchy_names/" + file_name + str(score_limit) + ".txt"
#cluster_dict_fn = "../data/movies/cluster/hierarchy_dict/" + file_name + str(score_limit) + ".txt"
ranking_fn = "../data/movies/rank/numeric/" + file_name + ".txt"
#pavPPMI(cluster_names_fn, ranking_fn, file_name)
#pavTermFrequency(ranking_fn, cluster_names_fn, file_name, False)
#binaryClusterTerm(cluster_names_fn, file_name)
#binaryInCluster(cluster_names_fn, file_name)
PPMI(cluster_names_fn, file_name)
#maxNonZero(cluster_names_fn, file_name)
#maxAll(cluster_names_fn, file_name)
#randomAll(cluster_names_fn, file_name)
#randomNonZero(cluster_names_fn, file_name)
#pavTermFrequency(ranking_fn, cluster_names_fn, file_name, False)
#normalizedTermFrequency(cluster_names_fn, file_name)
"""
PPMI(cluster_names_fn, file_name)
binaryInCluster(cluster_dict_fn, file_name)
binaryClusterTerm(cluster_names_fn, file_name)
termFrequency(cluster_names_fn, file_name)
"""
discrete_labels_fn = "../data/movies/rank/discrete/" + file_name + "P1.txt"
#getPAV(cluster_names_fn, discrete_labels_fn, file_name, do_p=True)
