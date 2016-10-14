import numpy as np
import helper.data as dt
from collections import OrderedDict


# Collect the rankings of movies for the given cluster directions
def getRankings(cluster_directions, vectors, cluster_names, vector_names):
    rankings = []
    ranking_names = []
    for d in range(len(cluster_directions)):
        cluster_ranking = []
        cluster_ranking_names = []
        for v in range(len(vectors)):
            cluster_ranking.append(np.dot(cluster_directions[d], vectors[v]))
            cluster_ranking_names.append(vector_names[v])
        sorted_rankings = sorted(cluster_ranking)
        sorted_rankings.reverse()
        sorted_ranking_names = dt.sortByReverseArray(cluster_ranking_names, cluster_ranking)
        ranking_names.append(sorted_ranking_names)
        rankings.append(cluster_ranking)
        print("Cluster:", cluster_names[d], "Movies:", sorted_ranking_names[0], sorted_rankings[0],
              sorted_ranking_names[1], sorted_rankings[1], sorted_ranking_names[2], sorted_rankings[2])
    return rankings, ranking_names


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
                label[t] = str(i + percentage_increment) + "%"
        labels.append(label)
    return labels

def getAllRankings(directions_fn, vectors_fn, cluster_names_fn, vector_names_fn, percent, percentage_increment, by_vector, fn):
    directions = dt.import2dArray(directions_fn)
    vectors = dt.import2dArray(vectors_fn)
    cluster_names = dt.import1dArray(cluster_names_fn)
    vector_names = dt.import1dArray(vector_names_fn)
    rankings, ranking_names = getRankings(directions, vectors, cluster_names, vector_names)
    rankings = np.asarray(rankings)
    labels = createLabels(rankings, percent)
    labels = np.asarray(labels)
    discrete_labels = createDiscreteLabels(rankings, percentage_increment)
    discrete_labels = np.asarray(discrete_labels)
    if by_vector:
        labels = labels.transpose()
        discrete_labels = discrete_labels.transpose()
        rankings = rankings.transpose()
    labels_fn = "../data/movies/rank/labels/" + fn + "P" + str(percent) + ".txt"
    rankings_fn = "../data/movies/rank/numeric/" + fn + ".txt"
    discrete_labels_fn = "../data/movies/rank/discrete/" + fn + "P" + str(percentage_increment) + ".txt"
    ranking_names_fn = "../data/movies/rank/names/" + fn + ".txt"
    dt.write2dArray(labels, labels_fn)
    dt.write2dArray(rankings, rankings_fn)
    dt.write2dArray(discrete_labels, discrete_labels_fn)
    dt.writeTabArray(ranking_names, ranking_names_fn)
    return labels_fn, rankings_fn, discrete_labels_fn, ranking_names_fn


def getAllPhraseRankings(directions_fn=None, vectors_fn=None, property_names_fn=None, vector_names_fn=None, fn="no filename", percentage_increment=1, scores_fn = None, top_amt=0, discrete=False):
    directions = dt.import2dArray(directions_fn)
    vectors = dt.import2dArray(vectors_fn)
    property_names = dt.import1dArray(property_names_fn)
    vector_names = dt.import1dArray(vector_names_fn)
    if top_amt != 0:
        scores = dt.import1dArray(scores_fn, "f")
        directions = dt.sortByReverseArray(directions, scores)[:top_amt]
        property_names = dt.sortByReverseArray(property_names, scores)[:top_amt]

    rankings, ranking_names = getRankings(directions, vectors, property_names, vector_names)
    if discrete:
        discrete_labels = createDiscreteLabels(rankings, percentage_increment)
        discrete_labels = np.asarray(discrete_labels)

    dt.write1dArray(property_names, "../data/movies/bow/names/top5kof17k.txt")
    dt.write2dArray(rankings, "../data/movies/rank/numeric/" + fn + ".txt")
    #dt.write2dArray(discrete_labels, "../data/movies/rank/discrete/" + fn +  ".txt")

class Rankings:
    def __init__(self, directions_fn, vectors_fn, cluster_names_fn, vector_names_fn, fn, percent, percentage_increment, by_vector):
        getAllRankings(directions_fn, vectors_fn, cluster_names_fn, vector_names_fn, percent, percentage_increment, by_vector, fn)

file_name="films100L175N0.52"
lowest_count = 200
vector_path = "../data/movies/nnet/spaces/" + file_name + ".txt"
class_path = "../data/movies/bow/binary/phrases/class-all-200"
property_names_fn = "../data/movies/bow/names/" + str(lowest_count) + ".txt"

# Get rankings
vector_names_fn = "../data/movies/nnet/spaces/filmNames.txt"
class_names_fn = "../data/movies/bow/names/" + str(lowest_count) + ".txt"
directions_fn = "../data/movies/svm/directions/" + file_name + str(lowest_count) + ".txt"
scores_fn = "../data/movies/svm/kappa/"+file_name+"200.txt"

#getAllPhraseRankings(directions_fn, vector_path, property_names_fn, vector_names_fn, file_name, 1, scores_fn, top_amt=5000, discrete=False)

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