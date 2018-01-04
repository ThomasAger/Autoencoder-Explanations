import data as dt
from sklearn.neighbors import KDTree
import os
from shutil import copyfile
import numpy as np


def kdTree(entity_names, space):
    inds_to_check = range(0,400,20)

    for i in inds_to_check:
        print(entity_names[i])
        tree = KDTree(space, leaf_size=2)
        dist, ind = tree.query([space[i]], k=5)
        ind = ind[0][:]
        for j in ind:
            print(entity_names[j])

# Top_x is the amount of top entities to show. If 0, shows all
# Cluster_ids are the clusters you want to show the top entities for. If none, then it shows all
def getTopEntitiesOnRanking(ranking, entity_names, cluster_names, cluster_length=3, top_x=-1, cluster_ids=None, output=True):
    if cluster_ids is not None:
        ranking = ranking[cluster_ids]
        cluster_names = cluster_names[cluster_ids]
    for i in range(len(cluster_names)):
        cluster_names[i] = cluster_names[i][:cluster_length]
    top_entities = []
    for c in range(len(ranking)):
        top_cluster_entities = []
        sorted_cluster = np.asarray(list(reversed(entity_names[np.argsort(ranking[c])])))
        for e in range(len(sorted_cluster)):
            top_cluster_entities.append(sorted_cluster[e])
            if e == top_x:
                break
        top_entities.append(top_cluster_entities)
        if output:
            print("Cluster:", cluster_names[c],  "Entites", top_cluster_entities)
    return top_entities


data_type = "placetypes"
file_name = "places NONNETCV5S0 SFT0 allL050kappa KMeans CA200 MC1 MS0.4 ATS2000 DS400"
cluster_names = dt.import2dArray("../data/" + data_type + "/cluster/dict/" + file_name + ".txt","s")
ranking = dt.import2dArray("../data/" + data_type + "/rank/numeric/" + file_name + ".txt")
entity_names = dt.import1dArray("../data/" + data_type + "/nnet/spaces/entitynames.txt")
top_x = 5
cluster_length = 3
cluster_ids = None
#normal_top_entities = getTopEntitiesOnRanking(ranking, entity_names, cluster_names, cluster_length, top_x, cluster_ids)

file_name = "places NONNETCV5S0 SFT0 allL050kappa KMeans CA200 MC1 MS0.4 ATS2000 DS400 foursquareFT BOCFi NTtanh1 NT1300linear"
ranking = dt.import2dArray("../data/" + data_type + "/nnet/clusters/" + file_name + ".txt")
#finetuned_top_entities = getTopEntitiesOnRanking(ranking, entity_names, cluster_names, cluster_length, top_x, cluster_ids)


def id_from_array(array, name):
    for n in range(len(array)):
        if array[n] == name:
            return n

# Must be the full top entities, with numerical values
def compareTopEntitiesOnRanking(ranking_1, ranking_2, cluster_names, cluster_length, top_x, output=True):
    all_diffs = np.zeros(shape = (len(ranking_1), len(ranking_1[0])))

    for c in range(len(ranking_1)):
        for v in range(len(ranking_1[c])):
            all_diffs[c][v] = ranking_1[c][v] - ranking_2[c][v]

    sorted_diffs = np.zeros(shape = (len(all_diffs), len(all_diffs[0])))
    sorted_names = np.empty(dtype="object", shape = (len(all_diffs), len(all_diffs[0])))
    for d in range(len(all_diffs)):
        sorted_diffs[d] = list(reversed(all_diffs[d][np.argsort(all_diffs[d])]))
        sorted_names[d] = list(reversed(entity_names[np.argsort(all_diffs[d])]))

    sorted_ranking = np.empty(dtype="object", shape = (len(all_diffs), len(all_diffs[0])))
    for c in range(len(ranking_1)):
        sorted_ranking[c] = list(reversed(entity_names[np.argsort(ranking_1[c])]))

    sorted_pos = []

    for s in range(len(sorted_names)):
        pos = []
        for n in sorted_names[s]:
            pos.append(id_from_array(sorted_ranking[s], n))
        sorted_pos.append(pos)

    if output:
        for s in range(len(sorted_diffs)):
            print("Cluster:", cluster_names[s], "Top diff entities", sorted_names[s][:top_x])
            print("Cluster:", cluster_names[s], "Top diff scores", sorted_diffs[s][:top_x])
            print("Cluster:", cluster_names[s], "Top diff scores", sorted_pos[s][:top_x])

    return all_diffs, sorted_diffs

data_type = "placetypes"
file_name = "places NONNETCV5S0 SFT0 allL050kappa KMeans CA200 MC1 MS0.4 ATS2000 DS400"
cluster_names = dt.import2dArray("../data/" + data_type + "/cluster/dict/" + file_name + ".txt","s")
ranking1 = dt.import2dArray("../data/" + data_type + "/rank/numeric/" + file_name + ".txt")
entity_names = dt.import1dArray("../data/" + data_type + "/nnet/spaces/entitynames.txt")
top_x = 5
cluster_length = 3
cluster_ids = None

file_name = "places NONNETCV5S0 SFT0 allL050kappa KMeans CA200 MC1 MS0.4 ATS2000 DS400 foursquareFT BOCFi NTtanh1 NT1300linear"
ranking2 = dt.import2dArray("../data/" + data_type + "/nnet/clusters/" + file_name + ".txt")

compareTopEntitiesOnRanking(ranking1, ranking2, cluster_names, cluster_length, top_x)
"""
data_type = "movies"
classify = "genres"
file_name = "places100"
space = dt.import2dArray("../data/"+data_type+"/nnet/spaces/"+ file_name + "-"+classify+".txt", "f")
entity_names = dt.import1dArray("../data/" + data_type + "/classify/"+classify+"/available_entities.txt", "s")
"""
def treeImages(loc, names,class_name):
    for n in names:
        copyfile(loc + class_name + " " + n + "CV0" + ".png",   output_loc + class_name + " " +  n + "CV0" + ".png")
"""
file_name = "wines100-" + classify
space = import2dArray("../data/"+data_type+"/nnet/spaces/"+ file_name + ".txt", "f")
entity_names = import1dArray("../data/" + data_type + "/classify/"+classify+"/available_entities.txt", "s")
"""
"""
data_type = "placetypes"
class_name = "TravelAndTransport"
name1 = "places NONNETCV5S4 SFT0 allL050kappa KMeans CA100 MC1 MS0.4 ATS2000 DS200 foursquare tdev3"
name2 = "places NONNETCV5S4 SFT0 allL050kappa KMeans CA100 MC1 MS0.4 ATS2000 DS200 foursquare tdev3FT BOCFi IT1300"
names = [name1, name2]
loc = "../data/" + data_type + "/rules/tree_images/"
output_loc = "../data/" + data_type + "/rules/tree_investigate/"
treeImages(loc, names, class_name)
"""

def topEntities(ranking, ens,  id=-1):
    ens = np.asarray(ens)
    ranking = np.asarray(ranking)
    sorted_entities = []
    sorted_values = []
    for r in ranking:
        sorted_entities.append(list(reversed(ens[np.argsort(r)])))
        sorted_values.append(list(reversed(r[np.argsort(r)])))
    if id > -1:
        print(sorted_entities[id])
        print(sorted_values[id])
    else:
        for s in sorted_entities:
            print(s)

data_type = "placetypes"
file_name = "places NONNETCV1S0 SFT0 allL050ndcg KMeans CA200 MC1 MS0.4 ATS2000 DS400 opencycFT BOCFi NTtanh1 NT1300linear3.txt"
ranking_fn = "../data/" + data_type+"/rules/rankings/" + file_name
ranking = dt.import2dArray(ranking_fn)
entities = dt.import1dArray("../data/" + data_type + "/nnet/spaces/entitynames.txt")

#topEntities(ranking, entities)
#print("------------------------------------------------")
compare_fn = "places NONNETCV1S0 SFT0 allL050ndcg KMeans CA200 MC1 MS0.4 ATS2000 DS400.txt"
ranking_fn = "../data/" + data_type+"/rank/numeric/" + compare_fn
ranking = dt.import2dArray(ranking_fn)
#topEntities(ranking, entities)