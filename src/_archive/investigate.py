import data as dt
from sklearn.neighbors import KDTree
import os
from shutil import copyfile
import numpy as np
import scipy.spatial.distance

def kdTree(entity_names, space):
    inds_to_check = range(0,400,20)

    for i in inds_to_check:
        print(entity_names[i])
        tree = KDTree(space, leaf_size=2)
        dist, ind = tree.query([space[i]], k=5)
        ind = ind[0][:]
        for j in ind:
            print(entity_names[j])

def biggestEucDifference(space1, space2, entity_names):
    dists = []
    for i in range(len(space1)):
        for j in range(len(space2)):
            dists.append(scipy.spatial.distance.euclidean(space1[i], space2[j]))
    dists = np.flipud(np.sort(dists))
    print("Biggest diff entities")
    for i in range(len(dists)):
        print(dists[i], entity_names[i])
        if i == 500:
            break

space = dt.import2dArray("../data/movies/nnet/spaces/films200.npy")
ft_space = dt.import2dArray("../data/movies/nnet/spaces/mds-nodupeCV1S0 SFT0 allL010010 LR kappa KMeans CA400 MC1 MS0.4 ATS1000 DS800FT BOCFi NT[200]tanh300V1.2L0.npy")
entity_names = dt.import1dArray("../data/movies/nnet/spaces/entitynames.txt")
biggestEucDifference(space, ft_space, entity_names)

# Top_x is the amount of top entities to show. If 0, shows all
# Cluster_ids are the clusters you want to show the top entities for. If none, then it shows all
def getTopEntitiesOnRanking(ranking, entity_names, cluster_names, cluster_length=3, top_x=-1, cluster_ids=None, output=True):
    if cluster_ids is not None:
        ranking = ranking[cluster_ids]
        cluster_names = cluster_names[cluster_ids]
    for i in range(len(cluster_names)):
        cluster_names[i] = cluster_names[i][:cluster_length]
    top_entities = []
    top_rankings = []
    for c in range(len(ranking)):
        top_cluster_entities = []
        top_cluster_rankings = []
        sorted_cluster = np.asarray(list(reversed(entity_names[np.argsort(ranking[c])])))
        sorted_rankings = np.asarray(list(reversed(ranking[c][np.argsort(ranking[c])])))
        for e in range(len(sorted_cluster)):
            top_cluster_entities.append(sorted_cluster[e])
            top_cluster_rankings.append(sorted_rankings[e])
            if e == top_x:
                break
        top_entities.append(top_cluster_entities)
        top_rankings.append(top_cluster_rankings)
        if output:
            print("Cluster:", cluster_names[c],  "Entites", top_cluster_entities)
            #print("Cluster:", cluster_names[c],  "Entites", top_cluster_rankings)
    return top_entities, top_rankings


data_type = "movies"
file_name = "mds-nodupeCV1S0 SFT0 allL010010 LR kappa KMeans CA400 MC1 MS0.4 ATS1000 DS800"
cluster_names = dt.import2dArray("../data/" + data_type + "/cluster/dict/" + file_name + ".txt","s")
ranking = dt.import2dArray("../data/" + data_type + "/rank/numeric/" + file_name + ".txt")
entity_names = dt.import1dArray("../data/" + data_type + "/nnet/spaces/entitynames.txt")
top_x = 5
cluster_length = 3
cluster_ids = None

normal_top_entities = getTopEntitiesOnRanking(ranking, entity_names, cluster_names, cluster_length, top_x, cluster_ids)
print("---------------------")
file_name = "mds-nodupeCV1S0 SFT0 allL010010 LR kappa KMeans CA400 MC1 MS0.4 ATS1000 DS800FT BOCFi NT[200]tanh300V1.2"
ranking = dt.import2dArray("../data/" + data_type + "/nnet/clusters/" + file_name + ".txt")
finetuned_top_entities = getTopEntitiesOnRanking(ranking, entity_names, cluster_names, cluster_length, top_x, cluster_ids)


def id_from_array(array, name):
    for n in range(len(array)):
        if array[n] == name:
            return n
    print("FAILED TO FIND", name)
    return None

# Must be the full top entities, with numerical values
def compareTopEntitiesOnRanking(ranking_1, ranking_2, cluster_names, cluster_length, top_x, output=True, reverse=False,
                                from_top=-1):

    if from_top == -1:
        from_top = len(ranking_1[0])


    pos = np.zeros( shape=(len(ranking_1), len(ranking_1[0])))
    for r in range(len(ranking_1)):
        for v in range(len(ranking_1[r])):
            pos[r][v] = v

    #Convert the rankings to sorted lists and create empty 1-15,000 array
    sorted_ranking_names1 = np.empty(dtype="object",shape = (len(ranking_1), from_top))
    sorted_ranking1 = np.empty(shape = (len(ranking_1), from_top))
    sorted_ranking1_by2 = np.empty(shape = (len(ranking_1), from_top))
    sorted_pos = np.zeros( shape = (len(ranking_1), from_top))

    for c in range(len(ranking_1)):
        sorted_ranking_names1[c] = list(reversed(entity_names[np.argsort(ranking_1[c])]))[:from_top]
        sorted_ranking1[c] = list(reversed(ranking_1[c][np.argsort(ranking_1[c])]))[:from_top]
        sorted_ranking1_by2[c] = list(reversed(ranking_1[c][np.argsort(ranking_2[c])]))[:from_top]
        sorted_pos[c] = list(reversed(pos[c][np.argsort(ranking_1[c])]))[:from_top]

    sorted_ranking_names2 = np.empty(dtype="object", shape = (len(ranking_1), from_top))
    sorted_ranking2 = np.zeros(shape = (len(ranking_1), from_top))
    sorted_ranking2_by1 = np.empty(shape = (len(ranking_1), from_top))
    sorted_pos2 = np.zeros( shape = (len(ranking_1), from_top))

    for c in range(len(ranking_2)):
        sorted_ranking_names2[c] = list(reversed(entity_names[np.argsort(ranking_2[c])]))[:from_top]
        sorted_ranking2[c] = list(reversed(ranking_2[c][np.argsort(ranking_2[c])]))[:from_top]
        sorted_ranking2_by1[c] = list(reversed(ranking_2[c][np.argsort(ranking_1[c])]))[:from_top]
        sorted_pos2[c] = list(reversed(pos[c][np.argsort(ranking_2[c])]))[:from_top]


    all_diffs = np.zeros(shape = (len(sorted_ranking1), len(sorted_ranking2[0])))

    # Get the diffs between the sorted lists
    for c in range(len(sorted_ranking1)):
        for v in range(len(sorted_ranking1[c])):
            if reverse:
                all_diffs[c][v] = sorted_ranking2[c][v] - sorted_ranking1_by2[c][v]
            else:
                all_diffs[c][v] = sorted_ranking1[c][v] - sorted_ranking2_by1[c][v]

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


def compareEntityOnCluster(ranking1, ranking2, clusters,  entity_names, entity_name="", entity_id=-1, cluster_name="", cluster_id=-1):
    for c in range(len(clusters)):
        if cluster_name in clusters[c]:
            cluster_id = c
            break
    to_compare1 = None
    to_compare2 = None
    if cluster_id != -1:
        to_compare1 = ranking1[cluster_id]
        to_compare2 = ranking2[cluster_id]
    else:
        print("NO CLUSTER ID")
    entity_id = id_from_array(entity_names, entity_name)
    if entity_id is not None and entity_id != -1:
        print("ranking1", entity_name, to_compare1[entity_id])
        print("ranking2", entity_name, to_compare2[entity_id])
        print("difference", entity_name, to_compare1[entity_id] - to_compare2[entity_id])
    else:
        print("NO ENTITY ID")


def getSimilarClusters(cluster_dict_1, cluster_dict_2, trim_amt, file_name, data_type, threshold_for_stopping, threshold_for_stopping_1):
    matching_clusters = np.zeros(len(cluster_dict_1), dtype=np.int32)
    new_cluster_dict_2 = []
    for c in cluster_dict_2:
        new_cluster_dict_2.append(np.flipud(c))
    cluster_dict_2 = None
    cluster_dict_2 = new_cluster_dict_2
    positions = np.zeros(len(cluster_dict_1))
    for c in range(len(cluster_dict_1)):
        print(c)
        lowest_pos = 242343
        lowest_cluster = len(cluster_dict_2)-1
        for n in range(len(cluster_dict_1[c])):
            if n > threshold_for_stopping_1:
                break
            name_to_match = cluster_dict_1[c][n]
            if ":" in name_to_match:
                name_to_match = name_to_match[:-1]
            for c2 in range(len(cluster_dict_2)):
                for n2 in range(len(cluster_dict_2[c2])):
                    if n2 > threshold_for_stopping:
                        break
                    name_to_match2 = cluster_dict_2[c2][n2]
                    if ":" in name_to_match2:
                        name_to_match2 = name_to_match2[:-1]
                    if name_to_match == name_to_match2:
                        if n2 < lowest_pos:
                            lowest_cluster = c2
                            lowest_pos = n2
                            break
            matching_clusters[c] = lowest_cluster
            positions[c] = lowest_pos
    sorted_matching_indexes = matching_clusters[np.argsort(positions)]
    sorted_orig_indexes = np.asarray(list(range(len(cluster_dict_1))))[np.argsort(positions)]
    print("_--------------------------------------------------")
    print("SORTED")
    print("_--------------------------------------------------")
    lines = []
    for c in range(len(sorted_orig_indexes)):
        line_p1 = ""
        for n in cluster_dict_1[sorted_orig_indexes[c]][:trim_amt]:
            line_p1 = line_p1 + n + " "
        line_pl2 = ""
        for k in cluster_dict_2[sorted_matching_indexes[c]][:trim_amt]:
            line_pl2 = line_pl2 + k + " "
        line =  line_p1 + " |||| " + line_pl2
        lines.append(line)
        print(line)
    dt.write1dArray(lines, "../data/" + data_type + "/investigate/" + file_name + str(trim_amt) + ".txt")
"""
data_type = "movies"
file_name = "mds-nodupeCV1S0 SFT0 allL010010 LR acc KMeans CA400 MC1 MS0.4 ATS500 DS800"
cluster_names = np.asarray(dt.import2dArray("../data/" + data_type + "/cluster/dict/" + file_name + ".txt","s"))
second_cluster_name = "mds-nodupeCV1S0 SFT0 allL010010 LR kappa KMeans CA400 MC1 MS0.4 ATS1000 DS800"
second_cluster_names = np.asarray(dt.import2dArray("../data/" + data_type + "/cluster/dict/" + second_cluster_name + ".txt","s"))
topic_model_names = np.asarray(dt.import2dArray("../data/" + data_type + "/LDA/names/" + "class-all-100-10-all-nodupe.npzDTP0.001TWP0.1NT100.txt", "s"))

trim_amt = 10
threshold_for_stopping = 100
threshold_for_stopping_1 = 20
getSimilarClusters( topic_model_names, cluster_names, trim_amt, file_name, data_type, threshold_for_stopping, threshold_for_stopping_1)
"""
"""
ranking1 = dt.import2dArray("../data/" + data_type + "/rank/numeric/" + file_name + ".txt")
entity_names = dt.import1dArray("../data/" + data_type + "/nnet/spaces/entitynames.txt")
top_x = 5
cluster_length = 3
cluster_ids = None
#Reverse = False: How far certain moves in A have fallen after being in B
#Reverse = True: How high certain movies have grown in A after being in B
reverse = False
from_top = 100

file_name = "places NONNETCV5S0 SFT0 allL050kappa KMeans CA200 MC1 MS0.4 ATS2000 DS400 foursquareFT BOCFi NTtanh1 NT1300linear"
ranking2 = dt.import2dArray("../data/" + data_type + "/nnet/clusters/" + file_name + ".txt")

#compareTopEntitiesOnRanking(ranking1, ranking2, cluster_names, cluster_length, top_x, output=True, reverse=reverse, from_top=from_top)

compareEntityOnCluster(ranking1, ranking2, cluster_names,  entity_names, entity_name="house", cluster_name="classical")
"""
"""
data_type = "movies"
classify = "genres"
file_name = "places100"
space = dt.import2dArray("../data/"+data_type+"/nnet/spaces/"+ file_name + "-"+classify+".txt", "f")
entity_names = dt.import1dArray("../data/" + data_type + "/classify/"+classify+"/available_entities.txt", "s")
"""
"""
def treeImages(loc, names,class_name):
    for n in names:
        copyfile(loc + class_name + " " + n + "CV0" + ".png",   output_loc + class_name + " " +  n + "CV0" + ".png")
        """
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