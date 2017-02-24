from scipy import spatial
import data as dt


def getSimilarity(vector1, vector2):
    return 1 - spatial.distance.cosine(vector1, vector2)

def getXMostSimilarIndex(term, terms_to_match, terms_to_ignore, amt):
    most_similar_term_indexes = []
    for a in range(amt):
        highest_term = 0
        term_index = 0
        for t in range(len(terms_to_match)):
            if dt.checkIfInArray(terms_to_ignore, t) is False:
                s = getSimilarity(term, terms_to_match[t])
                if s > highest_term and dt.checkIfInArray(most_similar_term_indexes, t) is False:
                    highest_term = s
                    term_index = t
        most_similar_term_indexes.append(term_index)
    return most_similar_term_indexes

def getXLeastSimilarIndex(term, terms_to_match, terms_to_ignore, amt):
    least_similar_term_indexes = []
    for a in range(amt):
        lowest_term = 99999999
        term_index = 0
        for t in range(len(terms_to_match)):
            if dt.checkIfInArray(terms_to_ignore, t) is False:
                s = getSimilarity(term, terms_to_match[t])
                if s < lowest_term and dt.checkIfInArray(least_similar_term_indexes, t) is False:
                    lowest_term = s
                    term_index = t
        least_similar_term_indexes.append(term_index)
    return least_similar_term_indexes

def getNextClusterTerm(cluster_terms, terms_to_match, terms_to_ignore, amt):
    min_value = 999999999999999
    min_index = 0
    for t in range(len(terms_to_match)):
        max_value = 0
        if dt.checkIfInArray(terms_to_ignore, t) is False:
            for c in range(len(cluster_terms)):
                s = getSimilarity(cluster_terms[c], terms_to_match[t])
                if s > max_value:
                    max_value = s
            if max_value < min_value:
                min_value = max_value
                min_index = t
    return min_index

"""
i = 1644
movie_vectors_fn = "../data/movies/nnet/spaces/films100L2100N0.8.txt"
movie_vectors = dt.import2dArray(movie_vectors_fn)
movie_names_fn = "../data/movies/nnet/spaces/filmNames.txt"
movie_names = dt.import1dArray(movie_names_fn)
print("Finding most similar directions for:", movie_names[i])
indexes = getXMostSimilarIndex(movie_vectors[i], movie_vectors, [], 20)

for ind in indexes:
    print(movie_names[ind])

file_name = "films100"
print(file_name)
movie_vectors_fn = "../data/movies/nnet/spaces/"+file_name+".txt"
movie_vectors = dt.import2dArray(movie_vectors_fn)
indexes = getXMostSimilarIndex(movie_vectors[i], movie_vectors, [], 20)

for ind in indexes:
    print(movie_names[ind])
"""