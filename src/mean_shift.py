import data as dt
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import normalize
from sklearn.cluster import AffinityPropagation
#import hdbscan

def gethdbscan(x, l):
    x = normalize(x)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3)
    labels = clusterer.fit_predict(x)
    unique, counts = np.unique(labels, return_counts=True)
    clusters = []
    for i in range(len(unique)):
        clusters.append([])
    for i in range(len(labels)):
        clusters[labels[i]].append(l[i])
    for i in range(len(clusters)):
        clusters[i] = np.flipud(clusters[i])
    return clusters, labels

def affinityClusters(x, l):
    model = AffinityPropagation()
    model.fit(x)
    labels = model.labels_
    cluster_centers = model.cluster_centers_
    indices = model.cluster_centers_indices_
    unique, counts = np.unique(labels, return_counts=True)
    clusters = []
    for i in range(len(unique)):
        clusters.append([])
    for i in range(len(labels)):
        clusters[labels[i]].append(l[i])
    for i in range(len(clusters)):
        clusters[i] = np.flipud(clusters[i])
    return cluster_centers, clusters

def meanShiftClusters(x, l):
    model = AffinityPropagation(preference=-5.0,damping=0.95)
    model.fit(x)
    labels = model.labels_
    cluster_centers = model.cluster_centers_
    indices = model.cluster_centers_indices_
    unique, counts = np.unique(labels, return_counts=True)
    clusters = []
    for i in range(len(unique)):
        clusters.append([])
    for i in range(len(labels)):
        clusters[labels[i]].append(l[i])
    for i in range(len(clusters)):
        clusters[i] = np.flipud(clusters[i])
    return cluster_centers, clusters

def saveClusters(directions_fn, scores_fn, names_fn,  filename, amt_of_dirs ,data_type, rewrite_files=False):

    dict_fn = "../data/" + data_type + "/cluster/dict/" + filename + ".txt"
    cluster_directions_fn = "../data/" + data_type + "/cluster/clusters/" + filename + ".txt"

    all_fns = [dict_fn]
    if dt.allFnsAlreadyExist(all_fns) and not rewrite_files:
        print("Skipping task", saveClusters.__name__)
        return
    else:
        print("Running task", saveClusters.__name__)

    p_dir = dt.import2dArray(directions_fn)
    p_names = dt.import1dArray(names_fn, "s")
    p_scores = dt.import1dArray(scores_fn, "f")

    ids = np.argsort(p_scores)

    p_dir = np.flipud(p_dir[ids])[:amt_of_dirs]
    p_names = np.flipud(p_names[ids])[:amt_of_dirs]

    c_dict, labels = gethdbscan(p_dir, p_names)

    dt.write2dArray(c_dict, dict_fn)