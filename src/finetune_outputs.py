import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import data as dt
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

def readPPMI(name, data_type, lowest_amt, highest_amt, classification):

    file = open("../data/"+data_type+"/bow/ppmi/" + "class-" + name + "-" + str(lowest_amt) + "-" + str(highest_amt) + "-" + classification)
    lines = file.readlines()
    frq_a = []
    for line in lines:
        frq_a.append(float(line))
    return frq_a

import MovieTasks as mt
import scipy.sparse as sp

def writeBagOfClusters(cluster_dict, data_type, lowest_amt, highest_amt, classification, fn):
    bag_of_clusters = []
    # Note, prior we used the PPMI values directly here somehow...
    loc = "../data/"+data_type+"/bow/frequency/phrases/"
    final_fn = ""
    for c in range(len(cluster_dict)):
        # Remove the colons
        for f in range(len(cluster_dict[c])):
            if ":" in cluster_dict[c][f]:
                cluster_dict[c][f] = cluster_dict[c][f][:-1]
        # Add all of the frequences together to make a bag-of-clusters
        p1 = loc + "class-" + cluster_dict[c][0]
        p2 = "-" + str(lowest_amt) + "-" + str(highest_amt) + "-" + classification
        accum_freqs = [0.0] * len(dt.import1dArray(p1 + p2 ))
        counter = 0
        # For all the cluster terms
        for f in cluster_dict[c]:
            if ":" in f:
                f = f[:-1]
            # Import the class
            class_to_add = dt.import1dArray(loc + "class-" + f + "-" + str(lowest_amt) + "-" + str(highest_amt) + "-" + classification, "f")
            # Add the current class to the older one
            accum_freqs = np.add(accum_freqs, class_to_add)
            counter += 1
        # Append this clusters frequences to the group of them
        bag_of_clusters.append(accum_freqs)
    # Obtain the PPMI values for these frequences
    ppmi_fn = "../data/"+data_type+"/bow/ppmi/" + "class-" + final_fn + str(lowest_amt) + "-" + str(highest_amt) + "-" + classification
    bag_csr = sp.csr_matrix(np.asarray(bag_of_clusters))
    ppmi_csr = mt.convertPPMI(bag_csr)
    dt.write2dArray(ppmi_csr, "../data/" + data_type + "/bow/ppmi/" + fn + ".txt")
    return ppmi_csr

def pavPPMI(cluster_names_fn, ranking_fn, file_name, do_p=False, data_type="movies", rewrite_files=False,limit_entities=False,
            classification="genres", lowest_amt=0, highest_amt=2147000000):
    pavPPMI_fn = "../data/" + data_type + "/finetune/" + file_name + ".txt"
    all_fns = [pavPPMI_fn]
    if dt.allFnsAlreadyExist(all_fns) and not rewrite_files:
        print("Skipping task", pavPPMI.__name__)
        return
    else:
        print("Running task", pavPPMI.__name__)
    print("certainly still running that old pavPPMI task, yes sir")
    if limit_entities is False:
        classification = "all"

    ranking = dt.import2dArray(ranking_fn)
    names = dt.import1dArray(cluster_names_fn)
    frq = []
    counter = 0

    for name in names:
        name = name.split()[0]
        if ":" in name:
            name = name[:-1]
        frq.append(readPPMI(name, data_type, lowest_amt, highest_amt, classification))

    pav_classes = []

    for f in range(len(frq)):
        try:
            print(names[f])
            x = np.asarray(frq[f])
            y = ranking[f]

            ir = IsotonicRegression()
            y_ = ir.fit_transform(x, y)
            pav_classes.append(y_)
            if do_p:
                plot(x, y, y_)
        except ValueError:
            print(names[f], "len ppmi", len(frq[f], "len ranking", len(ranking[f])))
            exit()
        print(f)

    dt.write2dArray(pav_classes, pavPPMI_fn)
    return pav_classes

def PPMIFT(cluster_names_fn, ranking_fn, file_name, do_p=False, data_type="movies", rewrite_files=False,limit_entities=False,
            classification="genres", lowest_amt=0, highest_amt=2147000000):
    pavPPMI_fn = "../data/" + data_type + "/finetune/" + file_name + ".txt"
    all_fns = [pavPPMI_fn]
    if dt.allFnsAlreadyExist(all_fns) and not rewrite_files:
        print("Skipping task", pavPPMI.__name__)
        return
    else:
        print("Running task", pavPPMI.__name__)
    print("certainly still running that old pavPPMI task, yes sir")
    if limit_entities is False:
        classification = "all"

    ranking = dt.import2dArray(ranking_fn)
    names = dt.import1dArray(cluster_names_fn)
    frq = []
    counter = 0

    for name in names:
        name = name.split()[0]
        if ":" in name:
            name = name[:-1]
        frq.append(readPPMI(name, data_type, lowest_amt, highest_amt, classification))

    dt.write2dArray(frq, pavPPMI_fn)
    return frq

def bagOfClustersPavPPMI(cluster_names_fn, ranking_fn, file_name, do_p=False, data_type="movies", rewrite_files=False,limit_entities=False,
            classification="genres", lowest_amt=0, highest_amt=2147000000, sparse_freqs_fn=None, bow_names_fn=None):

    pavPPMI_fn = "../data/" + data_type + "/finetune/boc/" + file_name + ".txt"
    all_fns = [pavPPMI_fn]
    if dt.allFnsAlreadyExist(all_fns) and not rewrite_files:
        print("Skipping task", bagOfClustersPavPPMI.__name__)
        return
    else:
        print("Running task", bagOfClustersPavPPMI.__name__)

    if limit_entities is False:
        classification = "all"

    bow_names = dt.import1dArray(bow_names_fn, "s")
    sparse_freqs = dt.import2dArray(sparse_freqs_fn, return_sparse=True)
    ranking = dt.import2dArray(ranking_fn)
    cluster_names = dt.import2dArray(cluster_names_fn, "s")

    frq = getLROnBag(cluster_names, data_type, lowest_amt, highest_amt, classification, file_name, bow_names, sparse_freqs)



    pav_classes = []

    for f in range(len(frq)):
        print(cluster_names[f])
        x = np.asarray(frq[f])
        y = ranking[f]

        ir = IsotonicRegression()
        y_ = ir.fit_transform(x, y)
        pav_classes.append(y_)
        if do_p:
            plot(x, y, y_)
        print(f)

    dt.write2dArray(pav_classes, pavPPMI_fn)
    return pav_classes

def bagOfClusters(cluster_names_fn, ranking_fn, file_name, do_p=False, data_type="movies", rewrite_files=False,limit_entities=False,
            classification="genres", lowest_amt=0, highest_amt=2147000000):
    pavPPMI_fn = "../data/" + data_type + "/finetune/boc/" + file_name + ".txt"
    all_fns = [pavPPMI_fn]
    if dt.allFnsAlreadyExist(all_fns) and not rewrite_files:
        print("Skipping task", bagOfClusters.__name__)
        return
    else:
        print("Running task", bagOfClusters.__name__)

    if limit_entities is False:
        classification = "all"

    ranking = dt.import2dArray(ranking_fn)
    names = dt.import2dArray(cluster_names_fn, "s")

    frq = writeBagOfClusters(names, data_type, lowest_amt, highest_amt, classification)



    dt.write2dArray(frq, pavPPMI_fn)
    return frq

def getLROnBag(cluster_dict, data_type, lowest_amt, highest_amt, classification, file_name, names, sparse_freqs):
    bag_of_clusters = []
    # Note, prior we used the PPMI values directly here somehow...
    for c in range(len(cluster_dict)):
        # Remove the colons
        for f in range(len(cluster_dict[c])):
            if ":" in cluster_dict[c][f]:
                cluster_dict[c][f] = cluster_dict[c][f][:-1]
        # Add all of the frequences together to make a bag-of-clusters
        name = cluster_dict[c][0]
        word_array = sparse_freqs[np.where(names == name)].toarray()
        accum_freqs = np.zeros(shape=len(word_array), dtype=np.int64)
        # For all the cluster terms
        for name in cluster_dict[c]:
            if ":" in name:
                name = name[:-1]
            # Import the class
            class_to_add = sparse_freqs[np.where(names == name)].toarray()
            # Add the current class to the older one
            accum_freqs = np.add(accum_freqs, class_to_add)
        # Append this clusters frequences to the group of them
        bag_of_clusters.append(accum_freqs)
    # Convert to binary
    for c in range(len(bag_of_clusters)):
        bag_of_clusters[c][bag_of_clusters[c] > 1] = 1
        bag_of_clusters[c] = bag_of_clusters[c][0] # For some reason the internal arrays are the single element of another array
    dt.write2dArray(bag_of_clusters, "../data/" + data_type + "/bow/boc/" + file_name + ".txt")
    return bag_of_clusters

def logisticRegression(cluster_names_fn, ranking_fn, file_name, do_p=False, data_type="movies", rewrite_files=False,limit_entities=False,
            classification="genres", lowest_amt=0, highest_amt=2147000000, sparse_freqs_fn=None, bow_names_fn=None):
    lr_fn = "../data/" + data_type + "/finetune/boc/" + file_name + ".txt"
    all_fns = [lr_fn]
    if dt.allFnsAlreadyExist(all_fns) and not rewrite_files:
        print("Skipping task", bagOfClusters.__name__)
        return
    else:
        print("Running task", bagOfClusters.__name__)

    if limit_entities is False:
        classification = "all"

    cluster_names = dt.import2dArray(cluster_names_fn, "s")
    bow_names = dt.import1dArray(bow_names_fn, "s")
    sparse_freqs = dt.import2dArray(sparse_freqs_fn, return_sparse=True)

    frq = getLROnBag(cluster_names, data_type, lowest_amt, highest_amt, classification, file_name, bow_names, sparse_freqs)

    dt.write2dArray(frq, lr_fn)
    return frq

def pavPPMIAverage(cluster_names_fn, ranking_fn, file_name, do_p=False, data_type="movies", rewrite_files=False,
            classification="genres", lowest_amt=0, highest_amt=2147000000, limit_entities=False, save_results_so_far=False):
    pavPPMI_fn = "../data/" + data_type + "/finetune/" + file_name + ".txt"
    all_fns = [pavPPMI_fn]
    if dt.allFnsAlreadyExist(all_fns) and not rewrite_files or save_results_so_far:
        print("Skipping task", pavPPMIAverage.__name__)
        return
    else:
        print("Running task", pavPPMIAverage.__name__)

    if limit_entities is False:
        classification = "all"

    ranking = dt.import2dArray(ranking_fn)
    names = dt.import2dArray(cluster_names_fn, "s")

    for n in range(len(names)):
        for x in range(len(names[n])):
            if ":" in names[n][x]:
                names[n][x] = names[n][x][:-1]

    frq = []
    counter = 0

    for n in range(len(names)):
        name_frq = []
        for name in names[n]:
            name_frq.append(readPPMI(name, data_type, lowest_amt, highest_amt, classification))
        avg_frq = []
        name_frq = np.asarray(name_frq).transpose()
        for name in name_frq:
            avg_frq.append(np.average(name))
        frq.append(np.asarray(avg_frq))
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

    dt.write2dArray(pav_classes, pavPPMI_fn)
    return pav_classes

def avgPPMI(cluster_names_fn, ranking_fn, file_name, do_p=False, data_type="movies", rewrite_files=False,
            classification="genres", lowest_amt=0, highest_amt=2147000000, limit_entities=False, save_results_so_far=False):
    pavPPMI_fn = "../data/" + data_type + "/finetune/" + file_name + ".txt"
    all_fns = [pavPPMI_fn]
    if dt.allFnsAlreadyExist(all_fns) and not rewrite_files or save_results_so_far:
        print("Skipping task", avgPPMI.__name__)
        return
    else:
        print("Running task", avgPPMI.__name__)

    if limit_entities is False:
        classification = "all"

    ranking = dt.import2dArray(ranking_fn)
    names = dt.import2dArray(cluster_names_fn, "s")

    for n in range(len(names)):
        for x in range(len(names[n])):
            if ":" in names[n][x]:
                names[n][x] = names[n][x][:-1]

    frq = []
    counter = 0

    for n in range(len(names)):
        name_frq = []
        for name in names[n]:
            name_frq.append(readPPMI(name, data_type, lowest_amt, highest_amt, classification))
        avg_frq = []
        name_frq = np.asarray(name_frq).transpose()
        for name in name_frq:
            avg_frq.append(np.average(name))
        frq.append(np.asarray(avg_frq))
        print(n)


    dt.write2dArray(frq, pavPPMI_fn)
    return frq

def readFreq(name, classification, lowest_amt, highest_amt):
    file = open("../data/movies/bow/frequency/phrases/" + "class-" + name + "-" + str(lowest_amt) + "-" + str(highest_amt) + "-"+classification)
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
#PPMI(cluster_names_fn, file_name)
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
