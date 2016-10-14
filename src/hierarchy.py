import numpy as np
import helper.data as dt
import helper.similarity as st
from sklearn.metrics import cohen_kappa_score
from sklearn import svm

class Cluster:
    kappa_scores = []
    directions = []
    cluster_direction = []
    names = []
    ranks = None

    def __init__(self, kappa_scores, directions, names):
        self.kappa_scores = np.asarray(kappa_scores)
        self.directions = np.asarray(directions)
        self.names = np.asarray(names)
        self.combineDirections()

    def combineDirections(self):
        if len(self.directions) > 1:
            direction = np.sum(self.directions, axis=0)
            direction = np.divide(direction, len(self.directions))
            self.cluster_direction = direction
        else:
            self.cluster_direction = self.directions[0]

    def getKappaScores(self):
        return self.kappa_scores

    def getDirections(self):
        return self.directions

    def getClusterDirection(self):
        return self.cluster_direction

    def getNames(self):
        return self.names

    def getRanks(self):
        return self.ranks

    def rankVectors(self, vectors):
        self.ranks = np.empty(shape=(15000, 1))
        for v in range(len(vectors)):
            rank = np.dot(self.cluster_direction, vectors[v])
            self.ranks[v][0] = rank

    def obtainKappaOnClusteredDirection(self):
        # For each discrete rank, obtain the Kappa score compared to the word occ
        kappas = np.empty(len(self.names))
        for n in range(len(self.names)):
            clf = svm.LinearSVC()
            ppmi = np.asarray(dt.import1dArray("../data/movies/bow/binary/phrases/" + names[n], "i"))
            clf.fit(self.ranks, ppmi)
            y_pred = clf.predict(self.ranks)
            score = cohen_kappa_score(ppmi, y_pred)
            kappas[n] = score
        # Return highest Kappa score
        return kappas


# Takes 2d array as input and outputs 1d array of the average of all of the arrays within that 2d array
def averageArray(array):
    average_array = np.sum(array)
    average_array = np.divide(average_array, len(array[0]))
    return average_array

from scipy.spatial.distance import cosine

def getMostSimilarCluster(cluster_index, clusters):
    highest_cluster = 0
    index = 0
    for c in range(len(clusters)):
        if clusters[c] is not None and c != cluster_index:
            s = 1 - cosine(clusters[cluster_index].getClusterDirection(), clusters[c].getClusterDirection())
            if s > highest_cluster:
                highest_cluster = s
                index = c
    return index

def getMostSimilarDirection(direction, directions):
    highest_cluster = 0
    index = 0
    for c in range(len(directions)):
        if direction is not None:
            s = 1 - cosine(direction, directions[c])
            if s > highest_cluster:
                highest_cluster = s
                index = c
    return index
""" OLD METHOD: SLOW W/AVERAGING"""

def getHierarchicalClusters(vectors, directions, scores, names, score_limit):

    clusters = []
    # Initialize a list of indexes to keep track of which directions have been combined
    for d in range(len(directions)):
        clusters.append(Cluster([scores[d]], [directions[d]], [names[d]]))

    clusters = np.asarray(clusters)

    clustersExist = True
    c = 0
    # Find the most similar direction and check if its combination has a kappa score loss larger than the score limit
    while clustersExist:
        if clusters[c] is not None:
            # Get the most similar direction to the current key
            i = getMostSimilarCluster(c, clusters)
            # Combine the most similar direction with the current direction
            new_cluster = Cluster(
                np.concatenate([clusters[c].getKappaScores(), clusters[i].getKappaScores()]),
                np.concatenate([clusters[c].getDirections(), clusters[i].getDirections()]),
                np.concatenate([clusters[c].getNames(), clusters[i].getNames()]))

            # Use the combined direction to see if the Kappa scores are not decreased an unreasonable amount
            new_cluster.rankVectors(vectors)
            cluster_scores = new_cluster.obtainKappaOnClusteredDirection()
            old_scores = new_cluster.getKappaScores()

            failed = False
            for s in range(len(old_scores)-1, -1, -1):
                #print (cluster_scores[s], old_scores[s])
                if cluster_scores[s] < old_scores[s] * score_limit:
                    failed = True
                    break

            # If the Kappa scores do not decrease that much, add the indexes of the direction that was combined
            #  with this direction to the dictionaries values and check to see if any other directions work with it
            if not failed:
                clusters[c] = None
                clusters[i] = None
                clusters = np.insert(clusters, c+1, new_cluster)
                print("Success!", new_cluster.getNames())
            else:
                print("Failure!", new_cluster.getNames())
        c += 1
        if c >= len(clusters):
            print("ended")
            for c in clusters:
                if c is not None:
                    print(c.getNames())
            break
    output_directions = []
    output_names = []
    for c in range(len(clusters)):
        if clusters[c] is not None:
            output_directions.append(clusters[c].getClusterDirection())
            output_names.append(clusters[c].getNames())
    dt.write2dArray(output_directions, "../data/movies/cluster/hierarchy_directions/"+file_name+str(score_limit)+".txt")
    dt.write2dArray(output_names, "../data/movies/cluster/hierarchy_names/" + file_name + str(score_limit)+".txt")

"""
# New method, instead of averaging, compare each individual direction. Start with one cluster and then add more.
def getHierarchicalClusters(vectors, directions, scores, names, score_limit):

    clusters = []
    # Initialize a list of indexes to keep track of which directions have been combined
    clusters.append(Cluster([scores[0]], [directions[0]], [names[0]]))
    clusters = np.asarray(clusters)

    clustersExist = True
    c = 0
    # Find the most similar direction and check if its combination has a kappa score loss larger than the score limit
    for d in range(1, len(directions)):
        highest_scoring_cluster_index = -1
        lowest_score = 5000
        lowest_cluster = None
        passed = False
        current_direction = Cluster([scores[d]], [directions[d]], [names[d]])
        for c in range(len(clusters)):
            # Get the most similar direction to the current key

            new_cluster = Cluster(
                np.concatenate([clusters[c].getKappaScores(), current_direction.getKappaScores()]),
                np.concatenate([clusters[c].getDirections(), current_direction.getDirections()]),
                np.concatenate([clusters[c].getNames(), current_direction.getNames()]))
            print(new_cluster.getNames())
            # Use the combined direction to see if the Kappa scores are not decreased an unreasonable amount
            new_cluster.rankVectors(vectors)
            cluster_scores = new_cluster.obtainKappaOnClusteredDirection()
            old_scores = new_cluster.getKappaScores()
            for s in range(len(old_scores)):
                # print (cluster_scores[s], old_scores[s])
                loss = old_scores[s] - cluster_scores[s]
                print(loss, old_scores[s] * score_limit, lowest_score)
                if loss < old_scores[s] * score_limit and loss < lowest_score:
                    highest_scoring_cluster_index = c
                    lowest_score = loss
                    lowest_cluster = new_cluster
                    passed = True
                    break
                else:
                    break
        # If the Kappa scores do not decrease that much, add the indexes of the direction that was combined
        #  with this direction to the dictionaries values and check to see if any other directions work with it
        if passed:
            # Combine the most similar direction with the current direction
            np.put(clusters, highest_scoring_cluster_index, lowest_cluster)
            print("Success!", current_direction.getNames(),clusters[c].getNames(),  d, c)
        else:
            clusters = np.append(clusters, current_direction)
            print("Failure!", current_direction.getNames(),clusters[c].getNames(),  d, c)

    output_directions = []
    output_names = []
    for c in range(len(clusters)):
        if clusters[c] is not None:
            output_directions.append(clusters[c].getClusterDirection())
            output_names.append(clusters[c].getNames())
    dt.write2dArray(output_directions, "../data/movies/cluster/hierarchy_directions/"+file_name+str(score_limit)+".txt")
    dt.write2dArray(output_names, "../data/movies/cluster/hierarchy_names/" + file_name + str(score_limit)+".txt")
"""
file_name = "films200L1100N0.5"
vector_fn = "../data/movies/nnet/spaces/" + file_name + ".txt"
directions_fn = "../data/movies/svm/directions/" +file_name+"200.txt"
scores_fn = "../data/movies/svm/kappa/"+file_name+"200.txt"
names_fn = "../data/movies/bow/names/200.txt"

amount_to_start = 1500


vectors = dt.import2dArray(vector_fn)
directions = dt.import2dArray(directions_fn)
scores = dt.import1dArray(scores_fn, "f")
names = dt.import1dArray(names_fn)

ind = np.flipud(np.argsort(scores))[:1500]

top_directions = []
top_scores = []
top_names = []

for i in ind:
    top_directions.append(directions[i])
    top_names.append(names[i])
    top_scores.append(scores[i])

score_limit = 0.5
import cProfile

cProfile.run('getHierarchicalClusters(vectors, top_directions, top_scores, top_names, score_limit)')

