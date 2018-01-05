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


data_type = "movies"
file_name = "films200-genresCV1S0 SFT0 allL0100kappa KMeans CA400 MC1 MS0.4 ATS2000 DS800"
cluster_names = dt.import2dArray("../data/" + data_type + "/cluster/dict/" + file_name + ".txt","s")
ranking = dt.import2dArray("../data/" + data_type + "/rank/numeric/" + file_name + ".txt")
entity_names = dt.import1dArray("../data/" + data_type + "/nnet/spaces/entitynames.txt")
top_x = 5
cluster_length = 3
cluster_ids = None

#normal_top_entities = getTopEntitiesOnRanking(ranking, entity_names, cluster_names, cluster_length, top_x, cluster_ids)

file_name = "films200-genresCV1S0 SFT0 allL0100kappa KMeans CA400 MC1 MS0.4 ATS2000 DS800 genresFT BOCFi NTtanh1 NT1300linear"
ranking = dt.import2dArray("../data/" + data_type + "/nnet/clusters/" + file_name + ".txt")
#finetuned_top_entities = getTopEntitiesOnRanking(ranking, entity_names, cluster_names, cluster_length, top_x, cluster_ids)


def id_from_array(array, name):
    for n in range(len(array)):
        if array[n] == name:
            return n
    print("FAILED TO FIND", name)
    return None

# Must be the full top entities, with numerical values
def compareTopEntitiesOnRanking(ranking_1, ranking_2, cluster_names, cluster_length, top_x, output=True, reverse=False):
    all_diffs = np.zeros(shape = (len(ranking_1), len(ranking_1[0])))

    pos = np.zeros( shape=(len(all_diffs), len(all_diffs[0])))
    for r in range(len(ranking_1)):
        for v in range(len(ranking_1[r])):
            pos[r][v] = v

    #Convert the rankings to sorted lists and create empty 1-15,000 array
    sorted_ranking_names1 = np.empty(dtype="object",shape = (len(all_diffs), len(all_diffs[0])))
    sorted_ranking1 = np.empty(shape = (len(all_diffs), len(all_diffs[0])))
    sorted_pos = np.zeros( shape = (len(all_diffs), len(all_diffs[0])))

    for c in range(len(ranking_1)):
        sorted_ranking_names1[c] = list(reversed(entity_names[np.argsort(ranking_1[c])]))
        sorted_ranking1[c] = list(reversed(ranking[c][np.argsort(ranking_1[c])]))
        sorted_pos[c] = list(reversed(pos[c][np.argsort(ranking_1[c])]))

    sorted_ranking_names2 = np.empty(dtype="object", shape = (len(all_diffs), len(all_diffs[0])))
    sorted_ranking2 = np.zeros(shape = (len(all_diffs), len(all_diffs[0])))
    sorted_pos2 = np.zeros( shape = (len(all_diffs), len(all_diffs[0])))

    for c in range(len(ranking_1)):
        sorted_ranking_names2[c] = list(reversed(entity_names[np.argsort(ranking_2[c])]))
        sorted_ranking2[c] = list(reversed(ranking[c][np.argsort(ranking_2[c])]))
        sorted_pos2[c] = list(reversed(pos[c][np.argsort(ranking_2[c])]))

    # Get the diffs between the sorted lists
    for c in range(len(ranking_1)):
        for v in range(len(ranking_1[c])):
            if reverse:
                all_diffs[c][v] = sorted_ranking2[c][v] - sorted_ranking1[c][v]
            else:
                all_diffs[c][v] = sorted_ranking1[c][v] - sorted_ranking2[c][v]

    # Sort and include sorted pos
    sorted_diffs = np.zeros(shape = (len(all_diffs), len(all_diffs[0])))
    sorted_names = np.empty(dtype="object", shape = (len(all_diffs), len(all_diffs[0])))
    sorted_diff_pos  =np.zeros(shape = (len(all_diffs), len(all_diffs[0])))
    sorted_diff_pos2 = np.zeros(shape = (len(all_diffs), len(all_diffs[0])))
    sorted_names2 = np.empty(dtype="object", shape = (len(all_diffs), len(all_diffs[0])))


    for d in range(len(all_diffs)):
        sorted_diffs[d] = list(reversed(all_diffs[d][np.argsort(all_diffs[d])]))
        if reverse:
            sorted_names2[d] = list(reversed(sorted_ranking_names2[d][np.argsort(all_diffs[d])]))
            sorted_diff_pos2[d] = list(reversed(sorted_pos2[d][np.argsort(all_diffs[d])]))
        else:
            sorted_names[d] = list(reversed(sorted_ranking_names1[d][np.argsort(all_diffs[d])]))
            sorted_diff_pos[d] = list(reversed(sorted_pos[d][np.argsort(all_diffs[d])]))




    if output:
        for s in range(len(sorted_diffs)):
            print("Cluster:", cluster_names[s][:cluster_length], "Top diff scores", sorted_diffs[s][:top_x])
            if reverse:
                print("Cluster:", cluster_names[s][:cluster_length], "Top diff entities", sorted_names2[s][:top_x])
                print("Cluster:", cluster_names[s][:cluster_length], "Top diff scores", sorted_pos2[s][:top_x])
            else:
                print("Cluster:", cluster_names[s][:cluster_length], "Top diff entities", sorted_names[s][:top_x])
                print("Cluster:", cluster_names[s][:cluster_length], "Top diff scores", sorted_pos[s][:top_x])


    return all_diffs, sorted_diffs

data_type = "movies"
file_name = "films200-genresCV1S0 SFT0 allL0100kappa KMeans CA400 MC1 MS0.4 ATS2000 DS800"
cluster_names = dt.import2dArray("../data/" + data_type + "/cluster/dict/" + file_name + ".txt","s")
ranking1 = dt.import2dArray("../data/" + data_type + "/rank/numeric/" + file_name + ".txt")
entity_names = dt.import1dArray("../data/" + data_type + "/nnet/spaces/entitynames.txt")
top_x = 5
cluster_length = 3
cluster_ids = None
#Reverse = False: How far certain moves in A have fallen after being in B
#Reverse = True: How high certain movies have grown in A after being in B
reverse = False

file_name = "films200-genresCV1S0 SFT0 allL0100kappa KMeans CA400 MC1 MS0.4 ATS2000 DS800 genresFT BOCFi NTtanh1 NT1300linear"
ranking2 = dt.import2dArray("../data/" + data_type + "/nnet/clusters/" + file_name + ".txt")

compareTopEntitiesOnRanking(ranking1, ranking2, cluster_names, cluster_length, top_x, output=True, reverse=reverse)
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
"""
data_type = "placetypes"
file_name = "places NONNETCV1S0 SFT0 allL050ndcg KMeans CA200 MC1 MS0.4 ATS2000 DS400 opencycFT BOCFi NTtanh1 NT1300linear3.txt"
ranking_fn = "../data/" + data_type+"/rules/rankings/" + file_name
#ranking = dt.import2dArray(ranking_fn)
entities = dt.import1dArray("../data/" + data_type + "/nnet/spaces/entitynames.txt")

#topEntities(ranking, entities)
#print("------------------------------------------------")
compare_fn = "places NONNETCV1S0 SFT0 allL050ndcg KMeans CA200 MC1 MS0.4 ATS2000 DS400.txt"
ranking_fn = "../data/" + data_type+"/rank/numeric/" + compare_fn
ranking = dt.import2dArray(ranking_fn)
#topEntities(ranking, entities)
"""