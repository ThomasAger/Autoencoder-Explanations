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
    file = open("../data/movies/bow/ppmi/" + "class-class-" + name)
    lines = file.readlines()
    frq_a = []
    for line in lines:
        frq_a.append(float(line))
    return frq_a

def getPAVNoAverage(cluster_names_fn, ranking_fn, file_name, do_p=False):
    ranking = dt.import2dArray(ranking_fn)
    names = dt.import1dArray(cluster_names_fn)
    frq = []
    counter = 0

    for name in names:
        frq.append(readPPMI(name))


    pav_classes = []

    for f in range(len(frq)):
        """
        x = copy.deepcopy(frq[f])

        y = ranking[f]

        avg_x = []
        avg_dict = defaultdict(int)
        counter_dict = defaultdict(int)
        already_done = []
        for v in range(len(x)):
            quit = False
            for a in already_done:
                if x[v] == a:
                    quit = True
                    break
            if quit:
                continue
            for n in range(len(x)):
                # If the PPMI's match
                if x[v] == x[n]:
                    avg_dict[x[v]] += y[n]
                    counter_dict[x[v]] += 1
            already_done.append(x[v])
        for key in avg_dict:
            avg_dict[key] = avg_dict[key] / counter_dict[key]
        avg_frq = []
        for v in frq[f]:
            for key in avg_dict:
                if key == v:
                    avg_frq.append(avg_dict[key])
                    break
        y = avg_frq
        """

        print(names[f])
        x = np.asarray(frq[f])
        y = ranking[f]

        ir = IsotonicRegression()
        y_ = ir.fit_transform(x, y)
        pav_classes.append(y_)
        if do_p:
            plot(x, y, y_)
        print(f)

    dt.write2dArray(pav_classes, "../data/movies/pav/" + file_name + ".txt")
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

def pavTermFrequency(cluster_names_fn):
    print("")

def pavPPMI(cluster_names_fn):
    print("")

def termFrequency(cluster_names_fn):
    print("")

def normalizedTermFrequency(cluster_names_fn):
    print("")

def binaryClusterTerm(cluster_names_fn, property_names_fn):
    print("")

def binaryInCluster(cluster_dict_fn, property_names_fn):
    print("")

def IRAverageFrequency(property_names_fn, discrete_labels_fn,file_name, do_p=False):
    ranking = dt.import2dArray(discrete_labels_fn, "s")
    property_names = dt.import1dArray(property_names_fn)
    frq = []

    for c in property_names:
        frq.append(readFreq(c))

    pav_bins = []

    discrete_label_ints = [[]]
    for r in range(len(ranking)):
        if r > 0:
            discrete_label_ints.append([])
        for i in range(len(ranking[r])):
            discrete_label_ints[r].append(101 - int(ranking[r][i][:-1]))

    for f in range(len(frq)):
        y = frq[f]
        x = discrete_label_ints[f]
        amts = [0] * 100
        totals = [0] * 100
        for i in range(len(x)):
            amts[x[i] - 1] += y[i]
            totals[x[i] - 1] += 1
        avgs = []
        for i in range(len(amts)):
            print(i, amts[i])
            avgs.append(amts[i] / totals[i])
        y = avgs
        x = range(0, 100)
        ir = IsotonicRegression()
        y_ = ir.fit_transform(x, y)
        pav_bins.append(y_)
        if do_p:
            plot(x, y, y_)
        print(f)

    pav_classes = [[]]
    for d in range(len(discrete_label_ints)):
        if d > 0:
            pav_classes.append([])
        for l in discrete_label_ints[d]:
            for b in range(len(pav_bins[d])):
                if l-1 == b:
                    label_bin = pav_bins[d][l-1]
                    pav_classes[d].append(label_bin)
        print(d)

    pav_classes = np.asarray(pav_classes)
    pav_classes = pav_classes.transpose()
    pav_classes = pav_classes.tolist()

    dt.write2dArray(pav_classes, "../data/movies/pav/" + file_name + ".txt")


class PAV:
    def __init__(self, property_names_fn, discrete_labels_fn, ppmi_fn, file_name, cluster_dict_fn):
        getPAVGini(cluster_dict_fn, discrete_labels_fn, ppmi_fn, file_name)

file_name = "films100L175N0.5"
cluster_names_fn = "../data/movies/cluster/names/" + file_name + ".txt"
ranking_fn = "../data/movies/rank/numeric/" + file_name + ".txt"


#getPAVGini(cluster_names_fn, ranking_fn, file_name, do_p=True)

discrete_labels_fn = "../data/movies/rank/discrete/" + file_name + "P1.txt"
#getPAV(cluster_names_fn, discrete_labels_fn, file_name, do_p=True)
