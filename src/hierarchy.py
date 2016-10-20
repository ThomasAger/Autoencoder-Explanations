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
            self.ranks[v][0] = np.dot(self.cluster_direction, vectors[v])

    def obtainKappaOnClusteredDirection(self):
        # For each discrete rank, obtain the Kappa score compared to the word occ
        kappas = np.empty(len(self.names))
        for n in range(len(self.names)):
            clf = svm.LinearSVC()
            ppmi = np.asarray(dt.import1dArray("../data/movies/bow/binary/phrases/" + self.names[n], "i"))
            clf.fit(self.ranks, ppmi)
            y_pred = clf.predict(self.ranks)
            score = cohen_kappa_score(ppmi, y_pred)
            kappas[n] = score
        return kappas

# Takes 2d array as input and outputs 1d array of the average of all of the arrays within that 2d array
def averageArray(array):
    average_array = np.sum(array)
    average_array = np.divide(average_array, len(array[0]))
    return average_array

from scipy.spatial.distance import cosine

def getMostSimilarClusterByI(cluster_index, clusters):
    highest_cluster = 0
    index = 0
    for c in range(len(clusters)):
        if clusters[c] is not None and c != cluster_index:
            s = 1 - cosine(clusters[cluster_index].getClusterDirection(), clusters[c].getClusterDirection())
            if s > highest_cluster:
                highest_cluster = s
                index = c
    return index

def getMostSimilarCluster(cluster, clusters):
    highest_cluster = 0
    index = 0
    for c in range(len(clusters)):
        s = 1 - cosine(cluster.getClusterDirection(), clusters[c].getClusterDirection())
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
"""
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
            i = getMostSimilarClusterByI(c, clusters)
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
"""
# New method, instead of averaging, compare each individual direction. Start with one cluster and then add more.
# Add to the parent cluster with the highest score
def getHierarchicalClustersMaxScoring(vectors, directions, scores, names, score_limit):

    clusters = []
    # Initialize a list of indexes to keep track of which directions have been combined
    clusters.append(Cluster([scores[0]], [directions[0]], [names[0]]))
    clusters = np.asarray(clusters)

    clustersExist = True
    c = 0
    # Find the most similar direction and check if its combination has a kappa score loss larger than the score limit
    for d in range(1, len(directions)):
        print(names[d])
        highest_scoring_cluster_index = -1
        lowest_score = 5000
        lowest_cluster = None
        passed = True
        current_direction = Cluster([scores[d]], [directions[d]], [names[d]])
        for c in range(len(clusters)):
            # Get the most similar direction to the current key
            new_cluster = Cluster(
                np.concatenate([clusters[c].getKappaScores(), current_direction.getKappaScores()]),
                np.concatenate([clusters[c].getDirections(), current_direction.getDirections()]),
                np.concatenate([clusters[c].getNames(), current_direction.getNames()]))
            # Use the combined direction to see if the Kappa scores are not decreased an unreasonable amount
            new_cluster.rankVectors(vectors)
            cluster_scores = new_cluster.obtainKappaOnClusteredDirection()
            old_scores = new_cluster.getKappaScores()
            total_score_loss = 0

            for s in range(len(old_scores)):
                # print (cluster_scores[s], old_scores[s])
                loss = old_scores[s] - cluster_scores[s]
                if loss > old_scores[s] - (old_scores[s] * score_limit):
                    passed = False
                    break
                else:
                    total_score_loss += old_scores[s] - cluster_scores[s]

            if passed and total_score_loss < lowest_score:
                highest_scoring_cluster_index = c
                lowest_score = total_score_loss
                lowest_cluster = new_cluster
                print("Passed", new_cluster.getNames(),  lowest_score)
                break
        # If the Kappa scores do not decrease that much, add the indexes of the direction that was combined
        #  with this direction to the dictionaries values and check to see if any other directions work with it
        if passed:
            # Combine the most similar direction with the current direction
            np.put(clusters, highest_scoring_cluster_index, lowest_cluster)
        else:
            clusters = np.append(clusters, current_direction)

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
def getHierarchicalClusters(vectors, directions, scores, names, score_limit, similarity_threshold, max_clusters):

    clusters = []
    # Initialize a list of indexes to keep track of which directions have been combined
    clusters.append(Cluster([scores[0]], [directions[0]], [names[0]]))
    clusters = np.asarray(clusters)

    clustersExist = True
    c = 0
    # Find the most similar direction and check if its combination has a kappa score loss larger than the score limit
    for d in range(1, len(directions)):
        print(d, "/", len(directions))
        failed = True
        current_direction = Cluster([scores[d]], [directions[d]], [names[d]])
        if len(clusters) >= max_clusters:
            break
        for c in range(len(clusters)):
            passed = True
            s = 1 - cosine(clusters[c].getClusterDirection(), current_direction.getClusterDirection())
            if s < similarity_threshold:
                #print(clusters[c].getNames(), current_direction.getNames(), s)
                continue
            else:
                print(clusters[c].getNames(), current_direction.getNames(), s)
            # Get the most similar direction to the current key
            new_cluster = Cluster(
                np.concatenate([clusters[c].getKappaScores(), current_direction.getKappaScores()]),
                np.concatenate([clusters[c].getDirections(), current_direction.getDirections()]),
                np.concatenate([clusters[c].getNames(), current_direction.getNames()]))

            # Use the combined direction to see if the Kappa scores are not decreased an unreasonable amount
            new_cluster.rankVectors(vectors)
            cluster_scores = new_cluster.obtainKappaOnClusteredDirection()
            old_scores = new_cluster.getKappaScores()

            for s in range(len(old_scores)):
                lowest_score = old_scores[s] * score_limit
                print(cluster_scores[s], "must be larger than", lowest_score)
                if cluster_scores[s] < lowest_score:
                    passed = False
                    break
            if passed:
                np.put(clusters, c, new_cluster)
                print("Success", new_cluster.getNames())
                failed = False
                break
        if failed:
            clusters = np.append(clusters, current_direction)
            print("Failed", current_direction.getNames())

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
# When adding more, choose the most similar rather than checking every Kappa score
def getHierarchicalClusters(vectors, directions, scores, names, score_limit):

    clusters = []
    # Initialize a list of indexes to keep track of which directions have been combined
    clusters.append(Cluster([scores[0]], [directions[0]], [names[0]]))
    clusters = np.asarray(clusters)

    clustersExist = True
    c = 0
    # Find the most similar direction and check if its combination has a kappa score loss larger than the score limit
    for d in range(1, len(directions)):
        print(d, "/", len(directions))
        passed = True
        current_direction = Cluster([scores[d]], [directions[d]], [names[d]])
        c = getMostSimilarCluster(current_direction, clusters)
        # Get the most similar direction to the current key
        new_cluster = Cluster(
            np.concatenate([clusters[c].getKappaScores(), current_direction.getKappaScores()]),
            np.concatenate([clusters[c].getDirections(), current_direction.getDirections()]),
            np.concatenate([clusters[c].getNames(), current_direction.getNames()]))

        # Use the combined direction to see if the Kappa scores are not decreased an unreasonable amount
        new_cluster.rankVectors(vectors)
        cluster_scores = new_cluster.obtainKappaOnClusteredDirection()
        old_scores = new_cluster.getKappaScores()

        for s in range(len(old_scores)):
            lowest_score = old_scores[s] * score_limit
            print(cluster_scores[s])
            if cluster_scores[s] < lowest_score:
                passed = False
                break
        if passed:
            np.put(clusters, c, new_cluster)
            print("Success", new_cluster.getNames())
        if not passed:
            clusters = np.append(clusters, current_direction)
            print("Failed", current_direction.getNames())

    output_directions = []
    output_names = []
    output_first_names = []
    for c in range(len(clusters)):
        if clusters[c] is not None:
            output_directions.append(clusters[c].getClusterDirection())
            output_names.append(clusters[c].getNames())
            output_first_names.append(clusters[c].getNames()[0])
    dt.write2dArray(output_directions, "../data/movies/cluster/hierarchy_directions/"+file_name+str(score_limit)+".txt")
    dt.write2dArray(output_names, "../data/movies/cluster/hierarchy_dict/" + file_name + str(score_limit)+".txt")
    dt.write2dArray(output_first_names, "../data/movies/cluster/hierarchy_names/" + file_name + str(score_limit)+".txt")
"""
def initClustering(vector_fn, directions_fn, scores_fn, names_fn, amt_to_start, profiling, binary_fn, similarity_threshold, max_clusters):
    print("importing")
    vectors = dt.import2dArray(vector_fn)
    directions = dt.import2dArray(directions_fn)
    scores = dt.import1dArray(scores_fn, "f")
    names = dt.import1dArray(names_fn)
    binary = dt.import2dArray(binary_fn)

    ind = np.flipud(np.argsort(scores))[:amt_to_start]

    top_directions = []
    top_scores = []
    top_names = []
    top_binary = []

    for i in ind:
        top_directions.append(directions[i])
        top_names.append(names[i])
        top_scores.append(scores[i])
        top_binary.append(binary[i])

    dt.write2dArray(top_binary, "../data/movies/bow/binary/phrases/class-all-500")
    dt.write1dArray(top_names, "../data/movies/bow/names/500.txt")
    print("done")
    score_limit = 0.8
    if profiling:
        import cProfile
        cProfile.run('getHierarchicalClusters(vectors, top_directions, top_scores, top_names, score_limit, similarity_threshold, max_clusters)')
    else:
        getHierarchicalClusters(vectors, top_directions, top_scores, top_names, score_limit, similarity_threshold, max_clusters)


file_name = "films200L1100N0.5"
vector_fn = "../data/movies/nnet/spaces/" + file_name + ".txt"
directions_fn = "../data/movies/svm/directions/" +file_name+"200.txt"
scores_fn = "../data/movies/svm/kappa/"+file_name+"200.txt"
names_fn = "../data/movies/bow/names/200.txt"
binary_fn = "../data/movies/bow/binary/phrases/class-all-200"
similarity_threshold = 0.5
max_clusters = 400
amount_to_start = 500

initClustering(vector_fn, directions_fn, scores_fn, names_fn, amount_to_start, False, binary_fn, similarity_threshold, max_clusters)
