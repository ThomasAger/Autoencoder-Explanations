import data as dt
import similarity as st
from collections import OrderedDict
import numpy as np
from scipy import spatial
def nameClustersMeanDirection(cluster_directions):
    # Import the word vectors from Wikipedia
    file = open("../data/wikipedia/word_vectors/glove.6B.50d.txt", encoding="utf8")
    lines = file.readlines()
    word_vectors = []
    word_vector_names = []
    for l in lines:
        l = l.split()
        word_vector_names.append(l[0])
        del l[0]
        for i in range(len(l)):
            l[i] = float(l[i])
        word_vectors.append(l)
    words = []
    for key, value in cluster_directions.items():
        for v in range(len(value)-1, -1, -1):
            if key == value[v] or key == value[v][:-1] or key[:-1] == value[v]:
                print("deleted", value[v], key)
                value[v] = "DELETE"
            for val in reversed(value):
                if val == value[v][:-1] or val[:-1] == value[v]:
                    print("deleted", value[v], val)
                    value[v] = "DELETE"
    for key, value in cluster_directions.items():
        for v in range(len(value) - 1, -1, -1):
            if value[v] == "DELETE":
                del value[v]

    for key, value in cluster_directions.items():
        cluster_word_vectors = []
        for w in range(len(word_vector_names)):
            if word_vector_names[w].strip() == key.strip():
                cluster_word_vectors.append(word_vectors[w])
                print("Success", key)
                break
            if w == len(word_vector_names) - 1:
                print("Failed", key)

        for v in range(len(value)):
            for w in range(len(word_vector_names)):
                if v > 9:
                    break
                if word_vector_names[w].strip() == value[v].strip():
                    cluster_word_vectors.append(word_vectors[w])
                    print("Success", value[v])
                    break
                if w == len(word_vector_names)-1:
                    print("Failed", value[v])
        if len(cluster_word_vectors) > 0:
            mean_vector = dt.mean_of_array(cluster_word_vectors)
            print(mean_vector)
            print(cluster_word_vectors[0])
            h_sim = 0
            closest_word = ""
            for v in range(len(word_vectors)):
                sim = st.getSimilarity(word_vectors[v], mean_vector)
                if sim > h_sim:
                    print("New highest sim", word_vector_names[v])
                    h_sim = sim
                    closest_word = word_vector_names[v]
            print("Closest Word", closest_word)
            words.append(closest_word)
        else:
            words.append(key)
    return words
# Find the medoid, remove outliers from it, and the find the mean direction
def nameClustersRemoveOutliers(cluster_directions):
    # Import the word vectors from Wikipedia
    file = open("../data/wikipedia/word_vectors/glove.6B.50d.txt", encoding="utf8")
    lines = file.readlines()
    wv = []
    wvn = []
    # Create an array of word vectors from the text file
    for l in lines:
        l = l.split()
        wvn.append(l[0])
        del l[0]
        for i in range(len(l)):
            l[i] = float(l[i])
        wv.append(l)
    words = []
    for key, value in cluster_directions.items():
        for v in range(len(value) - 1, -1, -1):
            if key == value[v] or key == value[v][:-1] or key[:-1] == value[v]:
                print("deleted", value[v], key)
                value[v] = "DELETE"
            for val in reversed(value):
                if val == value[v][:-1] or val[:-1] == value[v]:
                    print("deleted", value[v], val)
                    value[v] = "DELETE"
    for key, value in cluster_directions.items():
        for v in range(len(value) - 1, -1, -1):
            if value[v] == "DELETE":
                del value[v]

    # For every cluster (key: cluster center, value: similar terms)
    for key, value in cluster_directions.items():
        # If the center/values in the vector have a corresponding word vector, add the vectors to an array
        cluster_word_vectors = []
        cluster_word_vector_names = []
        for w in range(len(wvn)):
            if wvn[w].strip() == key.strip():
                cluster_word_vectors.append(wv[w])
                cluster_word_vector_names.append(wvn[w])
                print("Success", key)
                break
            if w == len(wvn) - 1:
                print("Failed", key)
        for v in range(len(value)):
            for w in range(len(wvn)):
                if v > 9:
                    break
                if wvn[w].strip() == value[v].strip():
                    cluster_word_vectors.append(wv[w])
                    cluster_word_vector_names.append(wvn[w])
                    print("Success", value[v])
                    break
                if w == len(wvn) - 1:
                    print("Failed", value[v])

        # If we found word vectors
        if len(cluster_word_vectors) > 1:

            # Get the angular distance between every word vector, and find the minimum angular distance point
            min_ang_dist = 214700000
            min_index = None
            ang_dists = np.zeros([len(cluster_word_vectors), len(cluster_word_vectors)])
            for i in range(len(cluster_word_vectors)):
                total_dist = 0
                for j in range(len(cluster_word_vectors)):
                    dist = spatial.distance.cosine(cluster_word_vectors[i], cluster_word_vectors[j])
                    if ang_dists[i][j] == 0:
                        ang_dists[i][j] = dist
                    total_dist += dist
                if total_dist < min_ang_dist:
                    min_ang_dist = total_dist
                    min_index = i
                    print("New min word:", cluster_word_vector_names[min_index])

            medoid_wv = []
            medoid_wvn = []
            # Delete outliers
            for i in range(len(cluster_word_vectors)):
                threshold = 0.8
                dist = spatial.distance.cosine(cluster_word_vectors[min_index], cluster_word_vectors[i])
                if dist < threshold:
                    medoid_wv.append(cluster_word_vectors[i])
                    medoid_wvn.append(cluster_word_vector_names[i])
                else:
                    print("Deleted outlier", cluster_word_vector_names[i])
            if len(medoid_wv) > 1:
                # Get the mean direction of non-outlier directions
                mean_vector = dt.mean_of_array(medoid_wv)
                # Find the most similar vector to that mean
                h_sim = 0
                closest_word = ""
                for v in range(len(wv)):
                    sim = st.getSimilarity(wv[v], mean_vector)
                    if sim > h_sim:
                        print("New highest sim", wvn[v])
                        h_sim = sim
                        closest_word = wvn[v]
                print("Closest Word", closest_word)
                words.append(closest_word)
            else:
                words.append(medoid_wvn[0])
        else:
            words.append(key)
    return words

# Find the medoid, remove outliers from it, and the find the mean direction
def nameClustersMedoid(cluster_directions):
    # Import the word vectors from Wikipedia
    file = open("../data/wikipedia/word_vectors/glove.6B.50d.txt", encoding="utf8")
    lines = file.readlines()
    wv = []
    wvn = []
    # Create an array of word vectors from the text file
    for l in lines:
        l = l.split()
        wvn.append(l[0])
        del l[0]
        for i in range(len(l)):
            l[i] = float(l[i])
        wv.append(l)
    words = []
    for key, value in cluster_directions.items():
        for v in range(len(value) - 1, -1, -1):
            if key == value[v] or key == value[v][:-1] or key[:-1] == value[v]:
                print("deleted", value[v], key)
                value[v] = "DELETE"
            for val in reversed(value):
                if val == value[v][:-1] or val[:-1] == value[v]:
                    print("deleted", value[v], val)
                    value[v] = "DELETE"
    for key, value in cluster_directions.items():
        for v in range(len(value) - 1, -1, -1):
            if value[v] == "DELETE":
                del value[v]

    # For every cluster (key: cluster center, value: similar terms)
    for key, value in cluster_directions.items():
        # If the center/values in the vector have a corresponding word vector, add the vectors to an array
        cluster_word_vectors = []
        cluster_word_vector_names = []
        for w in range(len(wvn)):
            if wvn[w].strip() == key.strip():
                cluster_word_vectors.append(wv[w])
                cluster_word_vector_names.append(wvn[w])
                print("Success", key)
                break
            if w == len(wvn) - 1:
                print("Failed", key)
        for v in range(len(value)):
            for w in range(len(wvn)):
                if v > 9:
                    break
                if wvn[w].strip() == value[v].strip():
                    cluster_word_vectors.append(wv[w])
                    cluster_word_vector_names.append(wvn[w])
                    print("Success", value[v])
                    break
                if w == len(wvn) - 1:
                    print("Failed", value[v])

        # If we found word vectors
        if len(cluster_word_vectors) > 1:

            # Get the angular distance between every word vector, and find the minimum angular distance point
            min_ang_dist = 214700000
            min_index = None
            ang_dists = np.zeros([len(cluster_word_vectors), len(cluster_word_vectors)])
            for i in range(len(cluster_word_vectors)):
                total_dist = 0
                for j in range(len(cluster_word_vectors)):
                    dist = spatial.distance.cosine(cluster_word_vectors[i], cluster_word_vectors[j])
                    if ang_dists[i][j] == 0:
                        ang_dists[i][j] = dist
                    total_dist += dist
                if total_dist < min_ang_dist:
                    min_ang_dist = total_dist
                    min_index = i
                    print("New min word:", cluster_word_vector_names[min_index])
            words.append(cluster_word_vector_names[min_index])
        else:
            words.append(key)
    return words

# Find the medoid, remove outliers from it, and the find the mean direction
def nameClustersRemoveOutliersWeight(cluster_directions, weights_fn, is_gini):
    # Import the word vectors from Wikipedia

    weights = dt.import1dArray(weights_fn)
    weights = [float(w) for w in weights]
    phrases = dt.import1dArray("../data/movies/bow/names/200.txt")
    wv, wvn = dt.getWordVectors()
    words = []
    for key, value in cluster_directions.items():
        for v in range(len(value) - 1, -1, -1):
            if key == value[v] or key == value[v][:-1] or key[:-1] == value[v]:
                print("deleted", value[v], key)
                value[v] = "DELETE"
            for val in reversed(value):
                if val == value[v][:-1] or val[:-1] == value[v]:
                    print("deleted", value[v], val)
                    value[v] = "DELETE"
    for key, value in cluster_directions.items():
        for v in range(len(value) - 1, -1, -1):
            if value[v] == "DELETE":
                del value[v]

    # For every cluster (key: cluster center, value: similar terms)
    for key, value in cluster_directions.items():
        # If the center/values in the vector have a corresponding word vector, add the vectors to an array
        cluster_word_vectors = []
        cluster_word_vector_names = []
        for w in range(len(wvn)):
            if wvn[w].strip() == key.strip():
                cluster_word_vectors.append(wv[w])
                cluster_word_vector_names.append(wvn[w])
                print("Success", key)
                break
            if w == len(wvn) - 1:
                print("Failed", key)
        for v in range(len(value)):
            for w in range(len(wvn)):
                if v > 9:
                    break
                if wvn[w].strip() == value[v].strip():
                    cluster_word_vectors.append(wv[w])
                    cluster_word_vector_names.append(wvn[w])
                    print("Success", value[v])
                    break
                if w == len(wvn) - 1:
                    print("Failed", value[v])

        # If we found word vectors
        if len(cluster_word_vectors) > 1:

            # Get the angular distance between every word vector, and find the minimum angular distance point
            min_ang_dist = 214700000
            min_index = None
            ang_dists = np.zeros([len(cluster_word_vectors), len(cluster_word_vectors)])
            for i in range(len(cluster_word_vectors)):
                total_dist = 0
                for j in range(len(cluster_word_vectors)):
                    dist = spatial.distance.cosine(cluster_word_vectors[i], cluster_word_vectors[j])
                    if ang_dists[i][j] == 0:
                        ang_dists[i][j] = dist
                    total_dist += dist
                if total_dist < min_ang_dist:
                    min_ang_dist = total_dist
                    min_index = i
                    print("New min word:", cluster_word_vector_names[min_index])

            medoid_wv = []
            medoid_wvn = []
            # Delete outliers
            for i in range(len(cluster_word_vectors)):
                threshold = 0.8
                dist = spatial.distance.cosine(cluster_word_vectors[min_index], cluster_word_vectors[i])
                if dist < threshold:
                    medoid_wv.append(cluster_word_vectors[i])
                    medoid_wvn.append(cluster_word_vector_names[i])
                else:
                    print("Deleted outlier", cluster_word_vector_names[i])
            if len(medoid_wv) > 1:
                si = []
                for wvna in medoid_wvn:
                    for w in range(len(phrases)):
                        if phrases[w][6:] == wvna:
                            si.append(w)
                a_weights = []
                for s in si:
                    a_weights.append(weights[s])
                if is_gini:
                    for s in range(len(a_weights)):
                        a_weights[s] = 1.0 - a_weights[s]
                for m in range(len(medoid_wv)):
                    for a in range(len(medoid_wv[m])):
                        medoid_wv[m][a] = medoid_wv[m][a] * a_weights[m]
                # Get the mean direction of non-outlier directions
                mean_vector = dt.mean_of_array(medoid_wv)
                # Find the most similar vector to that mean
                h_sim = 0
                closest_word = ""
                for v in range(len(wv)):
                    sim = st.getSimilarity(wv[v], mean_vector)
                    if sim > h_sim:
                        print("New highest sim", wvn[v])
                        h_sim = sim
                        closest_word = wvn[v]
                print("Closest Word", closest_word)
                words.append(closest_word)
            else:
                words.append(medoid_wvn[0])
        else:
            words.append(key)
    return words

# Find the medoid, remove outliers from it, and the find the mean direction
def nameClustersRemoveOutliersWeightDistance(cluster_directions):
    # Import the word vectors from Wikipedia
    file = open("../data/wikipedia/word_vectors/glove.6B.50d.txt", encoding="utf8")
    weights = dt.import1dArray(weights_fn)
    weights = [float(w) for w in weights]
    phrases = dt.import1dArray("../data/movies/bow/names/200.txt")
    lines = file.readlines()
    wv = []
    wvn = []
    # Create an array of word vectors from the text file
    for l in lines:
        l = l.split()
        wvn.append(l[0])
        del l[0]
        for i in range(len(l)):
            l[i] = float(l[i])
        wv.append(l)
    words = []
    for key, value in cluster_directions.items():
        for v in range(len(value) - 1, -1, -1):
            if key == value[v] or key == value[v][:-1] or key[:-1] == value[v]:
                print("deleted", value[v], key)
                value[v] = "DELETE"
            for val in reversed(value):
                if val == value[v][:-1] or val[:-1] == value[v]:
                    print("deleted", value[v], val)
                    value[v] = "DELETE"
    for key, value in cluster_directions.items():
        for v in range(len(value) - 1, -1, -1):
            if value[v] == "DELETE":
                del value[v]

    # For every cluster (key: cluster center, value: similar terms)
    for key, value in cluster_directions.items():
        # If the center/values in the vector have a corresponding word vector, add the vectors to an array
        cluster_word_vectors = []
        cluster_word_vector_names = []
        for w in range(len(wvn)):
            if wvn[w].strip() == key.strip():
                cluster_word_vectors.append(wv[w])
                cluster_word_vector_names.append(wvn[w])
                print("Success", key)
                break
            if w == len(wvn) - 1:
                print("Failed", key)
        for v in range(len(value)):
            for w in range(len(wvn)):
                if v > 9:
                    break
                if wvn[w].strip() == value[v].strip():
                    cluster_word_vectors.append(wv[w])
                    cluster_word_vector_names.append(wvn[w])
                    print("Success", value[v])
                    break
                if w == len(wvn) - 1:
                    print("Failed", value[v])

        # If we found word vectors
        if len(cluster_word_vectors) > 1:

            # Get the angular distance between every word vector, and find the minimum angular distance point
            min_ang_dist = 214700000
            min_index = None
            ang_dists = np.zeros([len(cluster_word_vectors), len(cluster_word_vectors)])
            for i in range(len(cluster_word_vectors)):
                total_dist = 0
                for j in range(len(cluster_word_vectors)):
                    dist = spatial.distance.cosine(cluster_word_vectors[i], cluster_word_vectors[j])
                    if ang_dists[i][j] == 0:
                        ang_dists[i][j] = dist
                    total_dist += dist
                if total_dist < min_ang_dist:
                    min_ang_dist = total_dist
                    min_index = i
                    print("New min word:", cluster_word_vector_names[min_index])

            medoid_wv = []
            medoid_wvn = []
            dists = []
            # Delete outliers
            for i in range(len(cluster_word_vectors)):
                threshold = 0.8
                dist = spatial.distance.cosine(cluster_word_vectors[min_index], cluster_word_vectors[i])
                dists.append(dist)
                if dist < threshold:
                    medoid_wv.append(cluster_word_vectors[i])
                    medoid_wvn.append(cluster_word_vector_names[i])
                else:
                    print("Deleted outlier", cluster_word_vector_names[i])
            if len(medoid_wv) > 1:
                for m in range(len(medoid_wv)):
                    for v in range(len(medoid_wv[m])):
                        medoid_wv[m][v] = medoid_wv[m][v] * dists[m]
                # Get the mean direction of non-outlier directions
                mean_vector = dt.mean_of_array(medoid_wv)
                # Find the most similar vector to that mean
                h_sim = 0
                closest_word = ""
                for v in range(len(wv)):
                    sim = st.getSimilarity(wv[v], mean_vector)
                    if sim > h_sim:
                        print("New highest sim", wvn[v])
                        h_sim = sim
                        closest_word = wvn[v]
                print("Closest Word", closest_word)
                words.append(closest_word)
            else:
                words.append(medoid_wvn[0])
        else:
            words.append(key)
    return words

# Splitting into high and low directions based on threshold
def splitDirections(directions_fn, scores_fn, names_fn, is_gini, amt_high_directions, amt_low_directions, high_threshold, low_threshold, half_kappa_half_ndcg):
    directions = np.asarray(dt.import2dArray(directions_fn))
    scores = np.asarray(dt.import1dArray(scores_fn, "f"))
    names = np.asarray(dt.import1dArray(names_fn))

    high_direction_names = []
    low_direction_names = []
    high_directions = []
    low_directions = []
    if len(half_kappa_half_ndcg) > 0:
        kappa_scores = dt.import1dArray(half_kappa_half_ndcg, "f")


    if amt_high_directions > 0 and amt_low_directions > 0:
        if len(half_kappa_half_ndcg) == 0:
            ids = np.flipud(np.argsort(scores))
        else:
            ind1 = np.flipud(np.argsort(scores))[:amt_low_directions/2]
            ind2 = np.zeros(len(ind1), dtype="int")
            kappa_scores = np.flipud(np.argsort(kappa_scores))
            count = 0
            added = 0
            for i in kappa_scores:
                if i not in ind1:
                    ind2[added] = i
                    added += 1
                if added >= amt_low_directions/2:
                    break
                count += 1
            shuffle_ind = np.asarray(list(range(0, len(ind1))))
            ids = np.insert(ind1, shuffle_ind, ind2)
        names = names[ids]
        directions = directions[ids]
        high_directions = directions[:amt_high_directions]
        low_directions = directions[amt_high_directions:amt_low_directions]
        high_direction_names = names[:amt_high_directions]
        low_direction_names = names[amt_high_directions:amt_low_directions]
        high_directions = high_directions.tolist()
        low_directions = low_directions.tolist()
        high_direction_names = high_direction_names.tolist()
        low_direction_names = low_direction_names.tolist()
    elif high_threshold > 0 and low_threshold > 0:
        for s in range(len(scores)):
            if scores[s] >= high_threshold:
                high_directions.append(directions[s])
                high_direction_names.append(names[s])
            elif scores[s] >= low_threshold:
                low_directions.append(directions[s])
                low_direction_names.append(names[s])
    else:
        print("no thresholds or direction amounts")
        hi = [None]
        li = [None]

    return high_direction_names, low_direction_names, high_directions, low_directions


def createTermClusters(hv_directions, lv_directions, hv_names, lv_names, amt_of_clusters):
    least_similar_clusters = []
    least_similar_cluster_ids = []
    least_similar_cluster_names = []

    print("Overall amount of HV directions: ", len(hv_directions))
    # Create high-valued clusters
    least_similar_cluster_ids.append(0)
    least_similar_clusters.append(hv_directions[0])
    least_similar_cluster_names.append(hv_names[0])
    print("Least Similar Term", hv_names[0])

    hv_to_delete = [0]
    for i in range(len(hv_directions)):
        if i >= amt_of_clusters-1:
            break
        else:
            ti = st.getNextClusterTerm(least_similar_clusters, hv_directions, least_similar_cluster_ids, 1)
            least_similar_cluster_ids.append(ti)
            least_similar_clusters.append(hv_directions[ti])
            least_similar_cluster_names.append(hv_names[ti])
            hv_to_delete.append(ti)
            print(str(i + 1) + "/" + str(amt_of_clusters), "Least Similar Term", hv_names[ti])

            # Add remaining high value directions to the low value direction list

    hv_directions = np.asarray(hv_directions)
    hv_names = np.asarray(hv_names)

    hv_directions = np.delete(hv_directions, hv_to_delete, 0)
    hv_names = np.delete(hv_names, hv_to_delete, 0)

    for i in range(len(hv_directions)):
        lv_directions.insert(0, hv_directions[i])
        lv_names.insert(0, hv_names[i])

    # Initialize dictionaries for printing / visualizing
    cluster_name_dict = OrderedDict()
    for c in least_similar_cluster_names:
        cluster_name_dict[c] = []

    # For every low value direction, find the high value direction its most similar to and append it to the directions
    every_cluster_direction = []
    for i in least_similar_clusters:
        every_cluster_direction.append([i])

    # Finding the most similar directions to each cluster_centre
    # Creating a dictionary of {cluster_centre: [cluster_direction(1), ..., cluster_direction(n)]} pairs
    for d in range(len(lv_directions)):
        i = st.getXMostSimilarIndex(lv_directions[d], least_similar_clusters, [], 1)[0]
        every_cluster_direction[i].append(lv_directions[d])
        print(str(d + 1) + "/" + str(len(lv_directions)), "Most Similar to", lv_names[d], "Is", least_similar_cluster_names[i])
        cluster_name_dict[least_similar_cluster_names[i]].append(lv_names[d])

    # Mean of all directions = cluster direction
    cluster_directions = []
    for l in range(len(least_similar_clusters)):
        cluster_directions.append(dt.mean_of_array(every_cluster_direction[l]))

    return least_similar_clusters, least_similar_cluster_names, cluster_name_dict, cluster_directions



def getClusters(directions_fn, scores_fn, names_fn, is_gini, amt_high_directions, amt_low_directions, filename,
                amt_of_clusters, high_threshold, low_threshold, data_type, rewrite_files=False, half_kappa_half_ndcg = ""):

    cluster_names_fn = "../data/" + data_type + "/cluster/names/" + filename + ".txt"
    clusters_fn = "../data/" + data_type + "/cluster/clusters/" + filename + ".txt"
    dict_fn = "../data/" + data_type + "/cluster/dict/" + filename + ".txt"
    cluster_center_fn = "../data/" + data_type + "/cluster/directions/" + filename + ".txt"

    all_fns = [cluster_names_fn, clusters_fn, dict_fn, cluster_center_fn]
    if dt.allFnsAlreadyExist(all_fns) and not rewrite_files:
        print("Skipping task", getClusters.__name__)
        return
    else:
        print("Running task", getClusters.__name__)

    hdn, ldn, hd, ld = splitDirections(directions_fn,
                                            scores_fn,
                                            names_fn, is_gini,
                                       amt_high_directions, amt_low_directions, high_threshold, low_threshold, half_kappa_half_ndcg)


    cluster_center_directions, least_similar_cluster_names, cluster_name_dict, least_similar_clusters = createTermClusters(hd, ld, hdn, ldn, amt_of_clusters)

    #word_vector_names = nameClustersMedoid(cluster_name_dict)
    additional_text = ""
    #if is_gini:
    #    additional_text = "gini"


    dt.write1dArray(least_similar_cluster_names, cluster_names_fn)
    dt.write2dArray(least_similar_clusters, clusters_fn)
    dt.writeArrayDict(cluster_name_dict, dict_fn)
    #dt.write1dArray(word_vector_names, word_vector_names_fn)
    dt.write2dArray(cluster_center_directions, cluster_center_fn)


class Cluster:
    def __init__(self,  directions_fn, scores_fn, names_fn, is_gini, low_threshold, high_threshold, filename):
        getClusters(directions_fn, scores_fn, names_fn, is_gini, low_threshold, high_threshold, filename)
#getClusters(0, 0, 0, 0, 0, 0, 0,0,0,0,0)

"""
file_name ="films100L250.txt"
cluster_directions = "../data/movies/cluster/dict/"+file_name
weights_fn = "../data/movies/gini/"+file_name
cluster_directions = dt.readArrayDict(cluster_directions)

dt.write1dArray(nameClustersRemoveOutliersWeightDistance(cluster_directions), "nameClustersRemoveOutliersWeight"+file_name)
"""
"""
def main(directions_fn, scores_fn, names_fn, is_gini, low_threshold, high_threshold, filename):
    Cluster(directions_fn, scores_fn, names_fn, is_gini, low_threshold, high_threshold, filename)
"""

#dt.write1dArray(nameClusters(dt.readArrayDict("Clusters/films100N0.6H25L3CutMostSimilarCLUSTER0.75,0.67.names")), "Clusters/Kfilms100N0.6H25L30.75,0.67.wordvectors")
"""
names_fn = "SVMResults/films100.names"
directions_fn = "Directions/films100N0.6H25L3Cut.directions"
scores_fn = "SVMResults/films100N0.6H25L3Cut.scores"
main(directions_fn, scores_fn, names_fn, False, 0.2, 0.341, "films100N0.6H25L3Cut")
"""
"""
directions_fn = "Directions/films100.directions"
scores_fn = "RuleType/ginifilms 100.txt"
names_fn = "SVMResults/films100.names"
is_gini = True

directions_fn = "Directions/films100N0.6H75L1Cut.directions"
scores_fn = "RuleType/ginifilms 100, 75 L1.txt"
names_fn = "SVMResults/films100N0.6H75L1Cut.names"

if  __name__ =='__main__':main(directions_fn, scores_fn, names_fn, is_gini, 0.31, 0.23,  "films100N0.6H75L1Cut")
directions_fn = "Directions/films100N0.6H50L2Cut.directions"
scores_fn = "RuleType/ginifilms 100, 50 L2.txt"
main(directions_fn, scores_fn, names_fn, is_gini, 0.26, 0.18, "films100N0.6H50L2Cut")

directions_fn = "Directions/films100N0.6H25L3Cut.directions"
scores_fn = "RuleType/ginifilms 100, 25 L3.txt"
main(directions_fn, scores_fn, names_fn, is_gini, 0.33, 0.25, "films100N0.6H25L3Cut")
"""