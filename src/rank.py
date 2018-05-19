import numpy as np
import data as dt
from collections import OrderedDict


# Collect the rankings of movies for the given cluster directions
def getRankings(cluster_directions, vectors, cluster_names, vector_names):
    rankings = []
    for d in range(len(cluster_directions)):
        cluster_ranking = []
        cluster_ranking_names = []
        for v in range(len(vectors)):
            cluster_ranking.append(np.dot(cluster_directions[d], vectors[v]))
            cluster_ranking_names.append(vector_names[v])
        rankings.append(cluster_ranking)
        print("Cluster:", cluster_names[d])
    return rankings#, ranking_names#, sorted_rankings_a


# Create binary vectors for the top % of the rankings, 1 for if it is in that percent and 0 if not.
def createLabels(rankings, percent):
    np_rankings = np.asarray(rankings)
    labels = []
    for r in np_rankings:
        label = [0 for x in range(len(rankings[0]))]
        sorted_indices = r.argsort()
        top_indices = sorted_indices[:len(rankings[0]) * percent]
        for t in top_indices:
            label[t] = 1
        labels.append(label)
    return labels


def createDiscreteLabels(rankings, percentage_increment):
    labels = []
    for r in rankings:
        label = ["100%" for x in range(len(rankings[0]))]
        sorted_indices = r.argsort()[::-1]
        for i in range(0, 100, percentage_increment):
            top_indices = sorted_indices[len(rankings[0]) * (i * 0.01):len(rankings[0]) * ((i + percentage_increment) * 0.01)]
            for t in top_indices:
                label[t] = str(i + percentage_increment)
        labels.append(label)
    return labels

def getAllRankings(directions_fn, vectors_fn, cluster_names_fn, vector_names_fn, percent, percentage_increment, by_vector, fn, discrete=True, data_type="movies",
                 rewrite_files=False):

    #labels_fn = "../data/"+data_type+"/rank/labels/" + fn + ".txt"
    rankings_fn = "../data/"+data_type+"/rank/numeric/" + fn + ".txt"
    ranking_names_fn = "../data/"+data_type+"/rank/names/" + fn + ".txt"
    #discrete_labels_fn = "../data/"+data_type+"/rank/discrete/" + fn + ".txt"

    all_fns = [rankings_fn, ranking_names_fn]
    if dt.allFnsAlreadyExist(all_fns) and not rewrite_files:
        for f in all_fns:
            print(f, "Already exists")
        print("Skipping task", "getAllRankings")
        return
    else:
        print("Running task", "getAllRankings")

    directions = dt.import2dArray(directions_fn)
    vectors = dt.import2dArray(vectors_fn)
    cluster_names = dt.import1dArray(cluster_names_fn)
    vector_names = dt.import1dArray(vector_names_fn)
    rankings = getRankings(directions, vectors, cluster_names, vector_names)
    rankings = np.asarray(rankings)
    if discrete:
        labels = createLabels(rankings, percent)
        labels = np.asarray(labels)
        discrete_labels = createDiscreteLabels(rankings, percentage_increment)
        discrete_labels = np.asarray(discrete_labels)
    if by_vector:
        labels = labels.transpose()
        if discrete:
            discrete_labels = discrete_labels.transpose()
        rankings = rankings.transpose()
    if discrete:
        dt.write2dArray(labels, labels_fn)

    dt.write2dArray(rankings, rankings_fn)
    if discrete:
        dt.write2dArray(discrete_labels, discrete_labels_fn)
    #dt.writeTabArray(ranking_names, ranking_names_fn)


def getAllPhraseRankings(directions_fn=None, vectors_fn=None, property_names_fn=None, vector_names_fn=None, fn="no filename",
                         percentage_increment=1, scores_fn = None, top_amt=0, discrete=False, data_type="movies",
                 rewrite_files=False):
    rankings_fn_all = "../data/" + data_type + "/rank/numeric/" + fn + "ALL.txt"

    all_fns = [rankings_fn_all]
    if dt.allFnsAlreadyExist(all_fns) and not rewrite_files:
        print("Skipping task", "getAllPhraseRankings")
        return
    else:
        print("Running task", "getAllPhraseRankings")

    directions = dt.import2dArray(directions_fn)
    vectors = dt.import2dArray(vectors_fn)
    property_names = dt.import1dArray(property_names_fn)
    vector_names = dt.import1dArray(vector_names_fn)
    if top_amt != 0:
        scores = dt.import1dArray(scores_fn, "f")
        directions = dt.sortByReverseArray(directions, scores)[:top_amt]
        property_names = dt.sortByReverseArray(property_names, scores)[:top_amt]

    rankings = getRankings(directions, vectors, property_names, vector_names)
    if discrete:
        discrete_labels = createDiscreteLabels(rankings, percentage_increment)
        discrete_labels = np.asarray(discrete_labels)
    for a in range(len(rankings)):
        rankings[a] = np.around(rankings[a], decimals=4)
    #dt.write1dArray(property_names, "../data/movies/bow/names/top5kof17k.txt")

    dt.write2dArray(rankings, rankings_fn_all)
    #dt.write2dArray(discrete_labels, "../data/movies/rank/discrete/" + fn +  ".txt")

class Rankings:
    def __init__(self, directions_fn, vectors_fn, cluster_names_fn, vector_names_fn, fn, percent, percentage_increment, by_vector, data_type):
        getAllRankings(directions_fn, vectors_fn, cluster_names_fn, vector_names_fn, percent, percentage_increment,
                       by_vector, fn, data_type)
data_type = "wines"
file_name="wines100trimmed"
lowest_count = 50
class_names = "class-trimmed-all-" + str(lowest_count)
vector_path = "../data/" + data_type + "/nnet/spaces/" + file_name + ".txt"
class_path = "../data/" + data_type + "/bow/binary/phrases/" + class_names
property_names_fn = "../data/" + data_type + "/bow/names/" + str(lowest_count) + ".txt"

# Get rankings
vector_names_fn = "../data/" + data_type + "/nnet/spaces/entitynames.txt"
class_names_fn = "../data/" + data_type + "/bow/names/" + str(lowest_count) + ".txt"
directions_fn = "../data/" + data_type + "/svm/directions/" + file_name +"ppmi200.txt"
vector_names_fn =  "../data/" + data_type + "/cluster/all_names/" +file_name+"svmndcg0.9200.txt"
directions_fn = "../data/" + data_type + "/cluster/all_directions/" +file_name+"svmndcg0.9200.txt"
scores_fn = "../data/" + data_type + "/ndcg/"+file_name+"ppmi200.txt"

#getAllPhraseRankings(directions_fn, vector_path, property_names_fn, vector_names_fn, file_name, 1, scores_fn, top_amt=0, discrete=False, data_type=data_type)

"""
def main(low_threshold, high_threshold, percent, discrete_percent, cluster_fn, vector_fn, cluster_names_fn, vector_names_fn, rank_fn, by_vector):
    Rankings(cluster_fn, vector_fn, cluster_names_fn, vector_names_fn, rank_fn, percent, discrete_percent, by_vector)
"""
"""
# Get top 10 movies for a specific cluster direction
filename = "films100N0.6H75L1"
directions_fn = "Directions/films100N0.6H75L1Cut.directions"
names_fn = "SVMResults/films100N0.6H75L1Cut.names"
space_fn = "newdata/spaces/" + filename + ".mds"
movie_names_fn = "filmdata/filmNames.txt"

directions = dt.import2dArray(directions_fn)
cluster_names = dt.import1dArray(names_fn)
vectors = dt.import2dArray(space_fn)
movie_names = dt.import1dArray(movie_names_fn)

name = "class-hilarity"
directions = np.asarray(directions)
vectors = np.asarray(vectors)
top_movies = []
for c in range(len(cluster_names)):
    if cluster_names[c] == name:
        for v in range(len(vectors)):
            top_movies.append(np.dot(vectors[v], directions[c]))

indices = np.argsort(top_movies)

print indices[1644]

print "TOP"

for i in reversed(indices[-20:]):
    print movie_names[i]

print "BOTTOM"
for i in reversed(indices[:20]):
    print movie_names[i]


filename = "films100[test]"
if  __name__ =='__main__':main(0.45, 0.55,  0.02, 1,
"Clusters/films100LeastSimilarHIGH0.45,0.055.clusters",
"filmdata/films100.mds/films100.mds",
"Clusters/films100LeastSimilarHIGH0.45,0.055.names",
"filmdata/filmNames.txt", filename, False)

filename = "films100N0.6H25L3"
if  __name__ =='__main__':main(0.75, 0.67,  0.02, 1,
"Clusters/films100N0.6H25L3CutLeastSimilarHIGH0.75,0.67.clusters",
"newdata/spaces/" + filename +".mds",
"Clusters/films100N0.6H25L3CutLeastSimilarHIGH0.75,0.67.names",
"filmdata/filmNames.txt", filename, False)

filename = "films100N0.6H50L2"
main(0.82, 0.74,  0.02, 1,
"Clusters/films100N0.6H50L2CutLeastSimilarHIGH0.82,0.74.clusters",
"newdata/spaces/" + filename +".mds",
"Clusters/films100N0.6H50L2CutLeastSimilarHIGH0.82,0.74.names",
"filmdata/filmNames.txt", filename, False)

filename = "films100N0.6H75L1"
main(0.77, 0.69,  0.02, 1,
"Clusters/films100N0.6H75L1CutLeastSimilarHIGH0.77,0.69.clusters",
"newdata/spaces/" + filename +".mds",
"Clusters/films100N0.6H75L1CutLeastSimilarHIGH0.77,0.69.names",
"filmdata/filmNames.txt", filename, False)
"""