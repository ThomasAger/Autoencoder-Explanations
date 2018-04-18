import numpy as np
import data as dt
from sklearn.metrics import cohen_kappa_score
from numpy.linalg import norm
import webapi
import copy
# For each overall cluster direction
# For each cluster
# For each discrete bin point
# Set the components to train by equal to the entities that are before and after the point
# Train SVM to get the hyper-plane seperating the two and record the score
# Add the highest scoring bin to the list of successful bins


def perpendicular(vector):
    return norm(vector)

def getCutOff(cluster_dict_fn,  rankings_fn, file_name):

    cluster_dict = dt.readArrayDict(cluster_dict_fn)
    rankings = dt.importDiscreteVectors(rankings_fn)

    for r in rankings:
        for a in range(len(r)):
            r[a] = int(r[a][:-1])

    cutoff_clusters = []
    counter = 0
    for key, value in cluster_dict.items():
        value.insert(0, key)
        cutoffs = []
        for v in value:
            max_score = 0
            cutoff = 0
            for i in range(1, 101):
                y_pred = []
                for ve in range(len(rankings[counter])):
                    rank = rankings[counter][ve]
                    if rank > i:
                        y_pred.append(0)
                    else:
                        y_pred.append(1)
                y_test = dt.import2dArray("../data/movies/bow/frequency/phrases/class-"+v, "s")
                score = cohen_kappa_score(y_test, y_pred)
                print(v, int(i), "Score", score)
                if score > max_score:
                    max_score = score
                    cutoff = i
            cutoffs.append(cutoff)
            print("Cutoff for", v, "On", key, "Was", str(cutoff))
        cutoff_clusters.append(cutoffs)
        counter+=1
    dt.write2dArray(cutoff_clusters, "../data/movies/rules/cutoff/"+file_name+".txt")

# This method selects the highest max of word vectors for each cluster of cutoff points
def fixCutoffFormatting(cutoff_fn, file_name):
    cutoff = dt.import1dArray(cutoff_fn)
    cluster_dict = dt.readArrayDict(cluster_dict_fn)
    for c in range(len(cutoff)):
        cutoff[c] = cutoff[c].split()
        for i in range(len(cutoff[c])):
            cutoff[c][i] = int(dt.stripPunctuation(cutoff[c][i]))
    dt.write2dArray(cutoff, "../data/movies/rules/cutoff/" +file_name+ ".txt")

# This method selects the highest max of word vectors for each cluster of cutoff points
def selectCutOffByWordVector(cutoff_fn, cluster_dict_fn, file_name):
    cutoff = dt.import2dArray(cutoff_fn)
    cluster_dict = dt.readArrayDict(cluster_dict_fn)
    cutoff_words = []
    wv, wvn = dt.getWordVectors()
    cluster_boundary = 2
    cluster_dict_arrays = []
    for key, value in cluster_dict.items():
        cluster_array = []
        cluster_array.append(key)
        for v in value:
            cluster_array.append(v)
        cluster_dict_arrays.append(cluster_array)
    for c in range(len(cutoff)):
        clusters = []
        for i in range(len(cutoff[c])):
            cluster = []
            for x in range(len(cutoff[c])-1, -1, -1):
                if cutoff[c][x] is None or cutoff[c][i] is None:
                    continue
                if abs(cutoff[c][i] - cutoff[c][x]) <= cluster_boundary:
                    cluster.append(cluster_dict_arrays[c][x])
                    cutoff[c][x] = None
                    cluster_dict_arrays[c][x] = None
            if cluster is []:
                continue
            clusters.append(cluster)
        # Get the maximum similarity word vector value for each cluster, across all clusters
        for cl in range(len(clusters)):
            for wa in range(len(clusters[cl])):
                for w in range(len(clusters[cl][wa])):
                    clusters[cl[wa]]


    dt.write2dArray(cutoff_words, "../data/movies/rules/cutoff/"+file_name+"WVN.txt")

# This method selects the word vectors that correspond with the highest scoring sentences containing those words
def selectCutOffByExplanation(cutoff_fn, cluster_dict_fn, file_name):
    cutoff = dt.import2dArray(cutoff_fn)
    dupe_cutoff = copy.deepcopy(cutoff)
    cluster_dict = dt.readArrayDict(cluster_dict_fn)
    cutoff_words = []
    cluster_boundary = 2
    cluster_dict_arrays = []
    for key, value in cluster_dict.items():
        cluster_array = []
        cluster_array.append(key)
        for v in value:
            cluster_array.append(v)
        cluster_dict_arrays.append(cluster_array)
    explanations = []
    explanation_cutoffs = []
    for c in range(len(cutoff)):
        clusters = []
        for i in range(len(cutoff[c])):
            cluster = []
            for x in range(len(cutoff[c])-1, -1, -1):
                if cutoff[c][x] is None or cutoff[c][i] is None:
                    continue
                if abs(cutoff[c][i] - cutoff[c][x]) <= cluster_boundary:
                    cluster.append(cluster_dict_arrays[c][x])
                    cutoff[c][x] = None
                    cluster_dict_arrays[c][x] = None
            if cluster is []:
                continue
            clusters.append(cluster)
        # Get the m  vvcaximum similarity word vector value for each cluster, across all clusters
        # For each cluster
        explained_cutoff = []
        explained_cutoff_value = []
        for cl in range(len(clusters)):
            if len(clusters[cl]) == 0:
                print ("Skipped")
                continue
            cluster_explanation, winning_index = webapi.getHighestScore(clusters[cl])
            explained_cutoff.append(cluster_explanation+",")

            dict_index = 0
            for h in range(len(cluster_dict_arrays[cl])):
                if cluster_dict_arrays[cl][h] == clusters[cl][winning_index]:
                    dict_index = h
            explained_cutoff_value.append(dupe_cutoff[cl][dict_index])
        explanations.append(explained_cutoff)
        explanation_cutoffs.append(explained_cutoff_value)
    dt.write2dArray(explanations, "../data/movies/rules/final_names/"+file_name+"WVN.txt")
    dt.write2dArray(explanation_cutoffs, "../data/movies/rules/final_cutoff/"+file_name+".txt")
file_name = "films100L175FT"
lowest_count = 200
cluster_dict_fn = "../data/movies/cluster/dict/" +file_name+".txt"
cutoff_fn = "../data/movies/rules/cutoff/" + file_name + ".txt"
selectCutOffByExplanation(cutoff_fn, cluster_dict_fn, file_name)



"""
rankings_fn = "../data/movies/rank/discrete/"+file_name+".txt"
getCutOff(cluster_dict_fn, rankings_fn, file_name)
"""