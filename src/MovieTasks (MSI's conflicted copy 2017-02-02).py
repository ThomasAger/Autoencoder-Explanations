
import data as dt
import re
import numpy as np
import string
from collections import defaultdict
import random
import theano
from theano.tensor.shared_randomstreams import RandomStreams

def getVectors(input_folder, file_names_fn, extension, output_folder, only_words_in_x_entities, words_without_x_entities, cut_first_line=False):
    file_names = dt.import1dArray(file_names_fn)
    phrase_dict = defaultdict(int)
    failed_filenames = []
    working_filenames = []

    # First, get all possible phrase names and build a dictionary of them from the files

    for f in range(len(file_names)):
        try:
            phrase_list = dt.import2dArray(input_folder + file_names[f] + extension, "s")
            if cut_first_line:
                phrase_list = phrase_list[1:]
            for p in phrase_list:
                phrase_dict[p[0]] += 1
            working_filenames.append(file_names[f])
        except FileNotFoundError:
            print("Failed to find", file_names[f])
            failed_filenames.append(file_names[f])

    # Convert to array so we can sort it
    phrase_list = []
    for key, value in phrase_dict.items():
        if value >= only_words_in_x_entities:
            phrase_list.append(key)

    phrase_list = sorted(phrase_list)

    print("Found", len(phrase_list), "Phrases")
    print(phrase_list[:20])
    print("Failed", len(failed_filenames), "Files")
    print(failed_filenames[:20])

    phrase_index_dict = defaultdict()

    # Create a dictionary to obtain the index of a phrase that's being checked

    for p in range(len(phrase_list)):
        phrase_index_dict[phrase_list[p]] = p

    # Create an empty 2d array to store a matrix of movies and phrases

    all_phrases_complete = []
    for f in working_filenames:
        all_phrases_complete.append([0]*len(phrase_list))

    print("Each entity is length", len(all_phrases_complete[0]))
    print("The overall matrix is", len(all_phrases_complete))

    # Then, populate the overall bag of words for each film (with all other phrases already set to 0

    for f in range(len(working_filenames)):
        n_phrase_list = dt.import2dArray(input_folder + file_names[f] + extension, "s")
        if cut_first_line:
            n_phrase_list = n_phrase_list[1:]
        for p in n_phrase_list:
            phrase = p[0]
            try:
                phrase_index = phrase_index_dict[phrase]
                all_phrases_complete[f][phrase_index] = int(p[1])
                #print("Kept", phrase)
            except KeyError:
                continue
                #print("Deleted phrase", phrase)

    all_phrases_complete = np.asarray(all_phrases_complete).transpose()

    indexes_to_delete = []
    for a in range(len(all_phrases_complete)):
        if np.count_nonzero(all_phrases_complete[a]) > len(all_phrases_complete[a]) - (1 + words_without_x_entities):
            print("Recorded an entity " + str(phrase_list[a]) + " with too little difference")
            indexes_to_delete.append(a)
    indexes_to_delete.sort()
    indexes_to_delete.reverse()
    for i in indexes_to_delete:
        all_phrases_complete = np.delete(all_phrases_complete, i, 0)
        print("Deleted an entity " + str(phrase_list[i]) + " with too little difference")
        phrase_list = np.delete(phrase_list, i, 0)

    dt.write1dArray(phrase_list, output_folder + "names/" + str(only_words_in_x_entities) + ".txt")

    for p in range(len(all_phrases_complete)):
        dt.write1dArray(all_phrases_complete[p], output_folder+"frequency/phrases/class-" + phrase_list[p])
        print("Wrote", phrase_list[p])


    dt.write2dArray(all_phrases_complete, output_folder+"frequency/phrases/" + "class-all-" + str(only_words_in_x_entities))
    print("Created class-all")
    all_phrases_complete = np.asarray(all_phrases_complete).transpose()
    for a in range(len(all_phrases_complete)):
        for v in range(len(all_phrases_complete[a])):
            if all_phrases_complete[a][v] > 1:
                all_phrases_complete[a][v] = 1

    all_phrases_complete = np.asarray(all_phrases_complete).transpose()

    for p in range(len(all_phrases_complete)):
        dt.write1dArray(all_phrases_complete[p], output_folder+"binary/phrases/class-" + phrase_list[p])
        print("Wrote binary", phrase_list[p])

    dt.write2dArray(all_phrases_complete, output_folder + "binary/phrases/" + "class-all-" + str(only_words_in_x_entities))
    print("Created class-all binary")

    #for p in range(len(all_phrases)):
    #    dt.write1dArray(all_phrases[p], output_folder + file_names[p] + ".txt")

#getVectors("../data/raw/previous work/placevectors/", "../data/raw/previous work/placeNames.txt", ".photos", "../data/placetypes/bow/", 50, 5, False)
#getVectors("../data/raw/previous work/winevectors/", "../data/raw/previous work/wineNames.txt", "", "../data/wines/bow/", 50, 10, True)
#getVectors("../data/raw/previous work/movieVectors/Tokens/", "../data/raw/previous work/filmIds.txt", ".film", "../data/movies/bow/", 100, 5, True)

from sklearn.feature_extraction.text import TfidfTransformer
import scipy.sparse as sp
def convertPPMI(mat):
    """
     Compute the PPMI values for the raw co-occurrence matrix.
     PPMI values will be written to mat and it will get overwritten.
     """
    (nrows, ncols) = mat.shape
    print("no. of rows =", nrows)
    print("no. of cols =", ncols)
    colTotals = mat.sum(axis=0)
    rowTotals = mat.sum(axis=1).T
    N = np.sum(rowTotals)
    rowMat = np.ones((nrows, ncols), dtype=np.float)
    for i in range(nrows):
        rowMat[i, :] = 0 \
            if rowTotals[0,i] == 0 \
            else rowMat[i, :] * (1.0 / rowTotals[0,i])
        print(i)
    colMat = np.ones((nrows, ncols), dtype=np.float)
    for j in range(ncols):
        colMat[:,j] = 0 if colTotals[0,j] == 0 else (1.0 / colTotals[0,j])
        print(j)
    P = N * mat.toarray() * rowMat * colMat
    P = np.fmax(np.zeros((nrows,ncols), dtype=np.float64), np.log(P))
    return P

#dt.write2dArray(convertPPMI( sp.csr_matrix(dt.import2dArray("../data/placetypes/bow/frequency/phrases/class-all-50"))), "../data/placetypes/bow/ppmi/class-all-50")
#write2dArray(convertPPMI( sp.csr_matrix(import2dArray("../data/wines/bow/frequency/phrases/class-all-50"))), "../data/wines/bow/ppmi/class-all-50")



def printIndividualFromAll(data_type, type, lowest_count):
    fn = "../data/" + data_type + "/bow/"
    all = np.asarray(dt.import2dArray(fn + type + "/class-all-"+str(lowest_count)))
    names = dt.import1dArray(fn + "names/"+str(lowest_count)+".txt")
    for c in range(len(all)):
        dt.write1dArray(all[c], fn+ type+"/class-"+str(names[c]))
        print("Wrote " + str(names[c]))

lowest_count = 50
#printIndividualFromAll("placetypes", "ppmi", lowest_count)
#printIndividualFromAll("wines", "ppmi", lowest_count)

def writeClassesFromNames(folder_name, file_names, output_folder):
    names = dt.getFolder(folder_name)
    all_names = defaultdict(int)
    entity_names = dt.import1dArray(file_names)
    translator = str.maketrans({key: None for key in string.punctuation})

    for type in range(len(names)):
        for n in range(len(names[type])):
            names[type][n] = dt.removeEverythingFromString(names[type][n])
            all_names[names[type][n]] += 1
    available_class_names = []
    available_indexes = []
    for n in range(len(entity_names)):
        name = entity_names[n]
        original_name = name
        name = dt.removeEverythingFromString(name)
        if all_names[name] > 0:
            available_class_names.append(original_name)
            available_indexes.append(n)
            print(name, "exists")
        else:
            print(name, "FAIL")
    dt.write1dArray(available_indexes, output_folder + "available_indexes.txt")
    dt.write1dArray(available_class_names, output_folder + "available_entities.txt")
    print("Wrote available indexes and entities")
    class_all = []
    for c in range(len(names)):
        binary_class = []
        for n in range(len(available_class_names)):
            available_class_names[n] = dt.removeEverythingFromString(available_class_names[n])
            if available_class_names[n] in names[c]:
                binary_class.append(1)
            else:
                binary_class.append(0)
        dt.write1dArray(binary_class, output_folder + "class-"+str(c)+"")
        class_all.append(binary_class)
    dt.write2dArray(np.asarray(class_all).transpose(), output_folder + "class-all")
    print("Wrote class-all")


def writeFromMultiClass():
    # Get the available indexes and entities
    print("for placetypes")
    # Initialize 2d array index with the length of the classes
    # For each line in the file
    # Split into two
    # Assign a 1 to the associated index



def trimRankings(rankings_fn, available_indexes_fn, names, folder_name):
    available_indexes = dt.import1dArray(available_indexes_fn)
    rankings = np.asarray(dt.import2dArray(rankings_fn))
    names = dt.import1dArray(names)
    trimmed_rankings = []

    for r in range(len(rankings)):
        trimmed = rankings[r].take(available_indexes)
        trimmed_rankings.append(trimmed)
    for a in range(len(trimmed_rankings)):
        print("Writing", names[a])
        dt.write1dArray(trimmed_rankings[a], folder_name + "class-" + names[a])
    print("Writing", rankings_fn[-6:])
    dt.write2dArray(trimmed_rankings, folder_name + "class-" + rankings_fn[-6:])

data_type = "wines"
output_folder = "../Data/"+data_type+"/classify/types/"
folder_name = "../data/raw/previous work/wineclasses/"
file_names = "../data/"+data_type+"/nnet/spaces/entitynames.txt"
phrase_names = "../data/"+data_type+"/bow/names/50.txt"
writeClassesFromNames(folder_name, file_names, output_folder)

folder_name = "../data/"+data_type+"/bow/binary/phrases/"

#trimRankings(folder_name + "class-all-50", "../data/"+data_type+"/classify/types/available_indexes.txt", phrase_names, folder_name)
