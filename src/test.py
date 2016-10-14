
import helper.data as dt
import helper.similarity as st
from scipy import spatial
import numpy as np
from sklearn.neighbors import DistanceMetric

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



def nameClustersMedoid(cluster_directions, ppmi_fn, frequency):
    # Import the word vectors from Wikipedia
    ppmi = dt.import2dArray(ppmi_fn)
    ppmi_names = dt.import1dArray("../data/movies/bow/phrase_names.txt")

    file = open("../data/wikipedia/word_vectors/glove.6B.50d.txt", encoding="utf8")
    lines = file.readlines()
    word_vectors = []
    word_vector_names = []
    # Create an array of word vectors from the text file
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

    ppmi_names, ppmi = dt.getSampledData(ppmi_names, ppmi, frequency, 21470000)

    wvn = []
    wv = []
    for w in range(len(word_vector_names)):
        for p in range(len(ppmi_names)):
            if word_vector_names[w] == ppmi_names[p]:
                wvn.append(word_vector_names[w])
                wv.append(word_vectors[w])

    # For every cluster (key: cluster center, value: similar terms)
    for key, value in cluster_directions.items():
        # If the center/values in the vector have a corresponding word vector, add the vectors to an array
        cluster_word_vectors = []
        cluster_word_vector_names = []
        for w in range(len(wvn)):
            if wvn[w].strip() == key.strip():
                cluster_word_vectors.append(wv[w])
                cluster_word_vector_names.append(wv[w])
                print("Success", key)
                break
            if w == len(wvn) - 1:
                print("Failed", key)
        for v in range(len(value)):
            for w in range(len(wvn)):
                if wvn[w].strip() == value[v].strip():
                    cluster_word_vectors.append(wv[w])
                    cluster_word_vector_names.append(wvn[w])
                    print("Success", value[v])
                    break
                if w == len(wvn)-1:
                    print("Failed", value[v])

        # If we found word vectors
        if len(cluster_word_vectors) > 0:
            # Get the angular distance between every word vector, and find the minimum angular distance point
            min_ang_dist = 214700000
            min_word = None
            for i in range(len(cluster_word_vectors)):
                ang_dist = 0
                for j in range(len(cluster_word_vectors)):
                    #ang_dist += st.getSimilarity(cluster_word_vectors[i], word_vectors[j])
                    ang_dist += spatial.distance.cosine(cluster_word_vectors[i], cluster_word_vectors[j])
                if ang_dist < min_ang_dist:
                    min_ang_dist = ang_dist
                    min_word = cluster_word_vector_names[i]
                    print ("New min word:", min_word)

            print("Min Word", min_word)
            words.append(min_word)
        else:
            words.append(key)
    return words
file_name = "films100N0.5H75L1ginikappaFTW500H500.530.4"
dt.write1dArray(nameClustersRemoveOutliers(dt.readArrayDict("../data/movies/cluster/dict/"+file_name+".txt")), "../data/clusters/word_vector_names/" +file_name+".txt")
