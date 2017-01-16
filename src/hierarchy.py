import numpy as np
import helper.data as dt
import helper.similarity as st
from sklearn.metrics import cohen_kappa_score
from sklearn import svm
import cProfile
import ndcg

class Cluster:
    kappa_scores = []
    directions = []
    cluster_direction = []
    names = []
    ranks = None
    data_type = "movies"

    def __init__(self, kappa_scores, directions, names, data_type):
        self.kappa_scores = np.asarray(kappa_scores)
        self.directions = np.asarray(directions)
        self.names = np.asarray(names)
        self.combineDirections()
        self.data_type = data_type

    def combineDirections(self):
        if len(self.directions) > 1:
            direction = np.sum(self.directions, axis=0)
            direction = np.divide(direction, len(self.directions))
            self.cluster_direction = direction
        else:
            self.cluster_direction = self.directions[0]

    def getScores(self):
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
        self.ranks = np.empty(shape=(len(vectors), 1))
        for v in range(len(vectors)):
            self.ranks[v][0] = np.dot(self.cluster_direction, vectors[v])

    def rankVectorsNDCG(self, vectors):
        self.ranks = np.empty(len(vectors))
        for v in range(len(vectors)):
            self.ranks[v] = np.dot(self.cluster_direction, vectors[v])

    # Cutoff points for the first and last direction
    def obtainKappaFirstAndLast(self):
        # For each discrete rank, obtain the Kappa score compared to the word occ
        kappas = np.empty(2)
        c = 0
        for n in [0, len(self.names) - 1]:
            clf = svm.LinearSVC()
            ppmi = np.asarray(dt.import1dArray("../data/" + self.data_type + "/bow/binary/phrases/class-" + self.names[n], "i"))
            clf.fit(self.ranks, ppmi)
            y_pred = clf.predict(self.ranks)
            score = cohen_kappa_score(ppmi, y_pred)
            kappas[c] = score
            c += 1
        return kappas


    def obtainNDCGFirstAndLast(self):
        # For each discrete rank, obtain the Kappa score compared to the word occ
        ndcgs = np.empty(2)
        c = 0
        for n in [0, len(self.names) - 1]:
            ppmi = np.asarray(dt.import1dArray("../data/" + self.data_type + "/bow/ppmi/class-" + self.names[n], "f"))
            sorted_indices = np.argsort(self.ranks)[::-1]
            score = ndcg.ndcg_from_ranking(ppmi, sorted_indices)
            ndcgs[c] = score
            #print("NDCG", self.names[n], score)
            c += 1
        return ndcgs
    def obtainNDCGFirst5(self):
        # For each discrete rank, obtain the Kappa score compared to the word occ
        ndcgs = None
        c = 0
        ndcgs = np.empty(6)
        max = len(self.names)
        if max < 5:
            index_array = range(len(self.names))
        else:
            index_array = [0,1,2,3,4,len(self.names)-1]
        for n in index_array:
            ppmi = np.asarray(dt.import1dArray("../data/" + self.data_type + "/bow/ppmi/class-" + self.names[n], "f"))
            sorted_indices = np.argsort(self.ranks)[::-1]
            score = ndcg.ndcg_from_ranking(ppmi, sorted_indices)
            ndcgs[c] = score
            #print("NDCG", self.names[n], score)
            c += 1


        return ndcgs
    def obtainNDCG(self):
        # For each discrete rank, obtain the Kappa score compared to the word occ
        ndcgs = np.empty(len(self.names))
        for n in range(len(self.names)):
            ppmi = np.asarray(dt.import1dArray("../data/" + self.data_type + "/bow/ppmi/class-" + self.names[n], "f"))
            sorted_indices = np.argsort(self.ranks)[::-1]
            score = ndcg.ndcg_from_ranking(ppmi, sorted_indices)
            ndcgs[n] = score
        return ndcgs

    # Cutoff points
    def obtainKappaOnClusteredDirection(self):
        # For each discrete rank, obtain the Kappa score compared to the word occ
        kappas = np.empty(len(self.names))
        for n in range(len(self.names)):
            clf = svm.LinearSVC()
            ppmi = np.asarray(dt.import1dArray("../data/" + self.data_type + "/bow/binary/phrases/class-" + self.names[n], "i"))
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
def getHierarchicalClusters(vectors, directions, scores, names, score_limit, dissimilarity_threshold, max_clusters,
                            file_name, kappa, similarity_threshold, data_type):

    clusters = []
    # Initialize a list of indexes to keep track of which directions have been combined
    clusters.append(Cluster([scores[0]], [directions[0]], [names[0]], data_type))
    clusters = np.asarray(clusters)
    all_subsets = []
    clustersExist = True

    c = 0
    # Find the most similar direction and check if its combination has a kappa score loss larger than the score limit
    failed_array = []
    for d in range(1, len(directions)):
        dont_add = False
        print(d, "/", len(directions))
        failed = True
        current_direction = Cluster([scores[d]], [directions[d]], [names[d]], data_type)
        if len(clusters) >= max_clusters:
            print("REACHED MAX CLUSTERS")
            break
        else:
            print(len(clusters), "/", max_clusters)

        too_similar = False
        s = 0
        for c in range(len(clusters)):

            passed = True

            s = 1 - cosine(clusters[c].getClusterDirection(), current_direction.getClusterDirection())

            if s > similarity_threshold:
                too_similar = True

            if s < dissimilarity_threshold:
                continue

            # Get the most similar direction to the current key
            new_cluster = Cluster(
                np.concatenate([clusters[c].getScores(), current_direction.getScores()]),
                np.concatenate([clusters[c].getDirections(), current_direction.getDirections()]),
                np.concatenate([clusters[c].getNames(), current_direction.getNames()]), data_type)

            # Use the combined direction to see if the Kappa scores are not decreased an unreasonable amount
            if kappa:
                new_cluster.rankVectors(vectors)
                cluster_scores = new_cluster.obtainKappaFirstAndLast()
            else:
                new_cluster.rankVectorsNDCG(vectors)
                cluster_scores = new_cluster.obtainNDCGFirstAndLast()
            old_scores = new_cluster.getScores()
            #print(cluster_scores)

            co = 0
            # Check the first and last directions
            for s in [0, len(old_scores)-1]:
                lowest_score = old_scores[s] * score_limit
                if cluster_scores[co] < lowest_score:
                    passed = False
                    break
                co += 1

            if passed:
                np.put(clusters, c, new_cluster)
                print("Success", new_cluster.getNames())
                failed = False
                all_subsets = np.append(all_subsets, new_cluster)
                break
        if failed:
            failed_array.append(s)
            if too_similar is True:
                print("Skipped", current_direction.getNames()[0], "Too similar to", clusters[c].getNames()[0])
                continue
            clusters = np.append(clusters, current_direction)
            all_subsets = np.append(all_subsets, current_direction)
            print("Failed", current_direction.getNames())
    dt.write1dArray(failed_array, "../data/temp/failed_array.txt")

    output_directions = []
    output_names = []
    for c in range(len(clusters)):
        if clusters[c] is not None:
            output_directions.append(clusters[c].getClusterDirection())
            output_names.append(clusters[c].getNames())

    all_directions = []
    all_names = []
    for c in range(len(all_subsets)):
        if all_subsets[c] is not None:
            all_directions.append(all_subsets[c].getClusterDirection())
            all_names.append(all_subsets[c].getNames())

    dt.write2dArray(output_directions, "../data/" + data_type + "/cluster/hierarchy_directions/"+file_name+str(score_limit)+str(max_clusters)+".txt")
    dt.write2dArray(output_names, "../data/" + data_type + "/cluster/hierarchy_names/" + file_name + str(score_limit)+str(max_clusters)+".txt")
    dt.write2dArray(all_directions,
                    "../data/" + data_type + "/cluster/all_directions/" + file_name + str(score_limit) + str(
                        max_clusters) + ".txt")
    dt.write2dArray(all_names, "../data/" + data_type + "/cluster/all_names/" + file_name + str(score_limit) + str(
        max_clusters) + ".txt")

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
def initClustering(vector_fn, directions_fn, scores_fn, names_fn, amt_to_start, profiling, dissimilarity_threshold,
                   max_clusters, score_limit, file_name, kappa, similarity_threshold, data_type):
    vectors = dt.import2dArray(vector_fn)
    directions = dt.import2dArray(directions_fn)
    scores = dt.import1dArray(scores_fn, "f")
    names = dt.import1dArray(names_fn)

    ind = np.flipud(np.argsort(scores))[:amt_to_start]

    top_directions = []
    top_scores = []
    top_names = []

    for i in ind:
        top_directions.append(directions[i])
        top_names.append(names[i])
        top_scores.append(scores[i])

    if profiling:
        cProfile.runctx('getHierarchicalClusters(vectors, top_directions, top_scores, top_names, score_limit, similarity_threshold, max_clusters, file_name, kappa)', globals(), locals())
    else:
        getHierarchicalClusters(vectors, top_directions, top_scores, top_names, score_limit, dissimilarity_threshold, max_clusters, file_name, kappa, similarity_threshold, data_type)


file_name = "films100"
vector_fn = "../data/movies/nnet/spaces/" + file_name + ".txt"
file_name = file_name + "ppmi"
directions_fn = "../data/movies/svm/directions/" +file_name+"200.txt"
scores_fn = "../data/movies/ndcg/"+file_name+".txt"
names_fn = "../data/movies/bow/names/200.txt"
similarity_threshold = 0.5
max_clusters = 400
amount_to_start = 1500
score_limit = 0.5

#initClustering(vector_fn, directions_fn, scores_fn, names_fn, amount_to_start, False, similarity_threshold, max_clusters, score_limit, file_name)
