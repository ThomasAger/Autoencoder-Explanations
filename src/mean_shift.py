import data as dt
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans
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


def meanShift(data):
    print("Estimating bandwidth")
    bandwidth = 23
    print("Estimated bandwidth")
    ms = MeanShift(bandwidth=bandwidth,  bin_seeding=False)
    ms.fit(data)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    print(labels)
    return labels

def kMeans(data, cluster_amt):
    ms = KMeans( n_init=10, max_iter=300, verbose=1, n_clusters=cluster_amt, )
    ms.fit(data)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    print(labels)
    return labels

def saveClusters(directions_fn, scores_fn, names_fn,  filename, amt_of_dirs ,data_type, cluster_amt, rewrite_files=False, algorithm="meanshift_k"):

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
    if algorithm == "meanshift":
        labels = meanShift(p_dir)
    else:
        labels = kMeans(p_dir, cluster_amt)
    unique, counts = np.unique(labels, return_counts=True)

    clusters = []
    dir_clusters = []
    for i in range(len(unique)):
        clusters.append([])
        dir_clusters.append([])
    for i in range(len(labels)):
        clusters[labels[i]].append(p_names[i])
        dir_clusters[labels[i]].append(p_dir[i])
    cluster_directions = []
    for l in range(len(dir_clusters)):
        cluster_directions.append(dt.mean_of_array(dir_clusters[l]))


    print("------------------------")
    for c in clusters:
        print(c)
    print("------------------------")

    dt.write2dArray(clusters, dict_fn)
    dt.write2dArray(cluster_directions, cluster_directions_fn)