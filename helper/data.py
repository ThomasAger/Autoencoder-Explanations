import os
import numpy as np
from collections import OrderedDict
import unicodedata
import nltk
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

"""

DATA IMPORTING TASKS

"""
def getWordVectors():
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
    return wv, wvn

def stripPunctuation(text):
    punctutation_cats = set(['Pc', 'Pd', 'Ps', 'Pe', 'Pi', 'Pf', 'Po'])
    return ''.join(x for x in text
                   if unicodedata.category(x) not in punctutation_cats)

def importNumpyVectors(numpy_vector_path=None):
    movie_vectors = np.load(numpy_vector_path)
    movie_vectors = np.ndarray.tolist(movie_vectors)
    movie_vectors = list(reversed(zip(*movie_vectors)))
    movie_vectors = np.asarray(movie_vectors)
    return movie_vectors

def import1dArray(file_name, file_type="s"):
    with open(file_name, "r") as infile:
        if file_type == "f":
            array = [float(line.strip()) for line in infile]
        elif file_type == "i":
            array = [int(line.strip()) for line in infile]
        else:
            array = [line.strip() for line in infile]
    return array

def import2dArray(file_name, file_type="f"):
    with open(file_name, "r") as infile:
        if file_type == "i":
            array = [list(map(int, line.strip().split())) for line in infile]
        elif file_type == "f":
            array = [list(map(float, line.strip().split())) for line in infile]
        elif file_type == "discrete":
            array = [list(line.strip().split()) for line in infile]
            for dv in array:
                for v in range(len(dv)):
                    dv[v] = int(dv[v][:-1])
        else:
            array = [list(line.strip().split()) for line in infile]
    return array

def importTabArray(file_name):
    with open(file_name, "r") as infile:
        string_array = [line.split("\t")[:-1] for line in infile]
    return string_array

def writeTabArray(array, file_name):
    names_with_tabs = []
    for name_array in array:
        string_to_append = ""
        for n in name_array:
            string_to_append = string_to_append + n + "\t"
        names_with_tabs.append(string_to_append)
    write1dArray(names_with_tabs, file_name)

def getFns(folder_path):
    file_names = []
    onlyfiles = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    for i in onlyfiles:
        if i != "class-all" and i != "nonbinary" and i != "low_keywords" and i != "class-All" and i != "archive" and i != "fns" and i!="fns.txt" and i!="class-all-200":
            file_names.append(i)
    return file_names


"""

DATA EDITING TASKS

"""

def writeArrayDict(dict, name):
    file = open(name, "w")
    for key, value in dict.items():
        file.write(str(key) + ": ")
        for v in value:
            file.write(str(v) + " ")
        file.write("\n")
    file.close()

def readArrayDict(file_name):
    file = open(file_name)
    lines = file.readlines()
    dict = OrderedDict()
    for l in lines:
        l = l.split()
        name = l[0][:-1]
        del l[0]
        dict[name] = l
        print(name)
    return dict

def splitData(training_data, movie_vectors, movie_labels):
    x_train = np.asarray(movie_vectors[:training_data])
    y_train = np.asarray(movie_labels[:training_data])
    x_test = np.asarray(movie_vectors[training_data:])
    y_test = np.asarray(movie_labels[training_data:])
    return  x_train, y_train,  x_test, y_test

def convertToFloat(string_array):
    temp_floats = []
    for string in string_array:
        float_strings = string.split()
        i = 0
        for float_string in float_strings:
            float_strings[i] = float(float_string)
            i = i + 1
        temp_floats.append(float_strings)
    return temp_floats


def write2dArray(array, name):
    file = open(name, "w")
    for i in range(len(array)):
        for n in range(len(array[i])):
            file.write(str(array[i][n]) + " ")
        file.write("\n")
    file.close()

def write1dArray(array, name):
    file = open(name, "w")
    for i in range(len(array)):
        file.write(str(array[i]) + "\n")
    file.close()

def mean_of_array(array):
    total = []
    for a in array[0]:
        total.append(a)
    len_array = len(array)
    for a in range(1, len_array):
        for v in range(0, len(array[a])):
            total[v] = total[v] + array[a][v]
    for v in range(len(total)):
        divided = (total[v] / len_array)
        total[v] = divided
    return total


def checkIfInArray(array, thing):
    for t in array:
        if thing == t:
            return True
    return False

def getIndexInArray(array, thing):
    for t in range(len(array)):
        if thing == array[t]:
            return t
    return None

def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

def sortByArray(array_to_sort, array_to_sort_by):
    Y = array_to_sort_by
    X = array_to_sort
    sorted_array = [x for (y, x) in sorted(zip(Y, X))]
    return sorted_array

def sortByReverseArray(array_to_sort, array_to_sort_by):
    Y = array_to_sort_by
    X = array_to_sort
    sorted_array = [x for (y, x) in reversed(sorted(zip(Y, X)))]
    return sorted_array

def getSampledData(property_names, classes, lowest_count, largest_count):
    for yt in range(len(classes)):
        y1 = 0
        y0 = 0
        for y in range(len(classes[yt])):
            if classes[yt][y] >= 1:
                y1 += 1
            if classes[yt][y] == 0:
                y0 += 1

        if y1 < lowest_count or y1 > largest_count:
            classes[yt] = None
            property_names[yt] = None
            print("Deleted", property_names[yt])
            continue

    property_names = [x for x in property_names if x is not None]
    classes = [x for x in classes if x is not None]
    return property_names, classes

def writeClassAll(class_fn, full_phrases_fn, phrases_used_fn, file_name):
    full_phrases = import1dArray(full_phrases_fn)
    #ppmi = np.asarray(import2dArray(class_fn)).transpose()
    ppmi = import2dArray(class_fn)
    new_ppmi = []
    phrases_used = import1dArray(phrases_used_fn)
    for p in range(len(full_phrases)):
        for pi in range(len(phrases_used)):
            if full_phrases[p] == phrases_used[pi]:
                new_ppmi.append(ppmi[p])
                break
    write2dArray(new_ppmi, file_name)
"""
writeClassAll("../data/movies/bow/ppmi/class-all", "../data/movies/bow/phrase_names.txt",
              "../data/movies/bow/names/200.txt", "../data/movies/bow/ppmi/class-all-200")
"""
#writeClassAll("../data/movies/bow/frequency/phrases/class-all", "../data/movies/bow/phrase_names.txt", "../data/movies/svm/names/films100N0.6H75L1200.txt", "../data/movies/bow/frequency/phrases/class-all-200")

"""
sortAndOutput("filmdata/KeywordData/most_common_keywords.txt", "filmdata/KeywordData/most_common_keywords_values.txt",
              "filmdata/KeywordData/most_common_keywordsSORTED.txt", "filmdata/KeywordData/most_common_keyword_valuesSORTED.txt")
"""

"""
top250 = []
for s in import1dArray("filmdata/Top 250 movies.txt"):
    s = s.split()[3:]
    s[len(s)-1] = s[len(s)-1][1:-1]
    s = " ".join(s)
    top250.append(s)
write1dArray(top250, "filmdata/top250.txt")
"""

#write1dArray(getFns("../data/movies/bow/binary/phrases/"), "../data/movies/bow/phrase_names.txt")



def getScoreDifferences(name_word_file1, name_score_file1, name_score_file2, name):
    word_file1 = open(name_word_file1, "r")
    score_file1 = open(name_score_file1, "r")
    word_lines1 = word_file1.readlines()
    score_lines1 = score_file1.readlines()
    scores1 = []
    words1 = []
    for s in score_lines1:
        scores1.append(float(s.strip()))
    for w in word_lines1:
        words1.append(w.strip())
    score_file2 = open(name_score_file2, "r")
    score_lines2 = score_file2.readlines()
    scores2 = []
    words2 = []
    for s in score_lines2:
        scores2.append(float(s))
    differences_list = []
    for i in range(len(score_lines1)):
        differences_list.append(scores1[i] - scores2[i])
    most_different_words = [x for (y,x) in sorted(zip(differences_list,words1))]
    differences_list = sorted(differences_list)
    write1dArray(most_different_words, "../data/movies/SVM/most_different_words_" + name + ".txt")
    write1dArray(differences_list, "../data/movies/SVM/most_different_values_" + name + ".txt")



def convertToPPMIOld(freq_arrays_fn, term_names_fn):
    file = open(freq_arrays_fn)
    for line in file:
        print((len(line.split())))
    freq_arrays = np.asarray(import2dArray(freq_arrays_fn, "s"))
    term_names = import1dArray(term_names_fn)
    ppmi_arrays = []
    overall = 0.0
    for f in freq_arrays:
        overall += sum(f)
    entity_array = [0] * 15000
    # For each term
    for t in range(len(freq_arrays)):
        ppmi_array = []
        term = sum(freq_arrays[t, :])
        term_p = 0.0
        for f in freq_arrays[t, :]:
            term_p += f / overall
        for e in range(len(freq_arrays[t])):
            ppmi = 0.0
            freq = freq_arrays[t][e]
            if freq != 0:
                freq_p = freq / overall
                if entity_array[e] == 0:
                    entity = sum(freq_arrays[:, e])
                    entity_p = 0.0
                    for f in freq_arrays[:, e]:
                        entity_p += f / overall
                    entity_array[e] = entity_p
                proba = freq_p / (entity_array[e] * term_p)
                ppmi = np.amax([0.0, np.log(proba)])
            ppmi_array.append(ppmi)
        ppmi_arrays.append(ppmi_array)
        write1dArray(ppmi_array, "../data/movies/bow/ppmi/class-" + term_names[t])
    write2dArray(ppmi_arrays, "../data/movies/bow/ppmi/class-all")

def convertToPPMI(freq_arrays_fn, term_names_fn):
    freq_arrays = np.asarray(import2dArray(freq_arrays_fn, "i"))
    term_names = import1dArray(term_names_fn)
    ppmi_arrays = []
    overall = 0.0
    for f in freq_arrays:
        overall += sum(f)
    entity_array = [0] * 15000
    # For each term
    for t in range(len(freq_arrays)):
        ppmi_array = []
        term = sum(freq_arrays[t, :])
        term_p = term / overall
        for e in range(len(freq_arrays[t])):
            ppmi = 0.0
            freq = freq_arrays[t][e]
            if freq != 0:
                freq_p = freq / overall
                if entity_array[e] == 0:
                    entity = sum(freq_arrays[:, e])
                    entity_p = entity / overall
                    entity_array[e] = entity_p
                proba = freq_p / (entity_array[e] * term_p)
                ppmi = np.amax([0.0, np.log(proba)])
            ppmi_array.append(ppmi)
        print(ppmi_array)
        ppmi_arrays.append(ppmi_array)
        write1dArray(ppmi_array, "../data/movies/bow/ppmi/class-" + term_names[t])
    write2dArray(ppmi_arrays, "../data/movies/bow/ppmi/class-all")
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

#write2dArray(convertPPMI( sp.csr_matrix(import2dArray("../data/movies/bow/frequency/phrases/class-all"))), "../data/movies/bow/ppmi/class-all-l")

def getDifference(array1, array2):
    file1 = open(array1)
    file2 = open(array2)
    for line1 in file1:
        line1 = line1.split()
        line1 = [float(line1[v]) for v in range(len(line1))]
        print(line1)
        for line2 in file2:
            line2 = line2.split()
            line2 = [float(line2[v]) for v in range(len(line2))]
            print(line2)
            break
        break

original_ppmi = "../data/movies/bow/ppmi/class-all"
library_ppmi = "../data/movies/bow/ppmi/class-all-l"

#getDifference(original_ppmi, library_ppmi)

import random
"""
#Going to just use dropout instead
def saltAndPepper(movie_vectors_fn, chance_to_set_noise, salt, filename):
    movie_vectors = import2dArray(movie_vectors_fn)
    amount_to_noise = len(movie_vectors_fn) * chance_to_set_noise
    for m in range(len(movie_vectors)):
        for a in range(amount_to_noise):
            ri = random.choice(list(enumerate(movie_vectors[m])))
            if salt is True:
                movie_vectors[m][ri] = 0
            else:
                movie_vectors[m][ri] = 1
        if salt is True:
            filename += "SPN0NC" + str(chance_to_set_noise)
        else:
            filename += "SPN1NC" + str(chance_to_set_noise)
    write2dArray(movie_vectors, filename)

movie_vectors_fn = "../data/movies/bow/ppmi/class-all-normalized--1,1"

saltAndPepper(movie_vectors_fn, 0.5, True, "../data/movies/bow/ppmi/class-all-normalized--1,1")
"""

def convertPPMI_original(mat):
    """
    Compute the PPMI values for the raw co-occurrence matrix.
    PPMI values will be written to mat and it will get overwritten.
    """
    (nrows, ncols) = mat.shape
    colTotals = np.zeros(ncols, dtype="float")
    for j in range(0, ncols):
        colTotals[j] = np.sum(mat[:,j].data)
    print(colTotals)
    N = np.sum(colTotals)
    for i in range(0, nrows):
        row = mat[i,:]
        rowTotal = np.sum(row.data)
        for j in row.indices:
            val = np.log((mat[i,j] * N) / (rowTotal * colTotals[j]))
            mat[i, j] = max(0, val)
    return mat
#write2dArray(convertPPMI_original( np.asarray(import2dArray("../data/movies/bow/frequency/phrases/class-all"))), "../data/movies/bow/ppmi/class-all-lori")

def convertToTfIDF(freq_arrays_fn):
    freq = np.asarray(import2dArray(freq_arrays_fn))
    v = TfidfTransformer()
    x = v.fit_transform(freq)
    x = x.toarray()
    write2dArray(x, "../data/movies/bow/tfidf/class-all")
    writeClassAll("../data/movies/bow/tfidf/class-all", "../data/movies/bow/phrase_names.txt",
                  "../data/movies/bow/names/200.txt", "../data/movies/bow/tfidf/class-all-200")


def plotSpace(space):
    file_name = "../data/movies/bow/ppmi/class-all"
    file = open(file_name)
    single_values = []

    space = np.asarray(import2dArray(file_name))
    counter = 0
    for s in space:
        s = s[s != 0]
        single_values.extend(s)

    # basic plot
    sns.distplot(single_values, kde=False, rug=False)
    sns.plt.show()
    print ("now we here")


def scaleSpace(space, lower_bound, upper_bound, file_name):
    minmax_scale = MinMaxScaler(feature_range=(lower_bound, upper_bound), copy=True)
    space = minmax_scale.fit_transform(space)
    write2dArray(space, file_name)
    return space
"""
file = open("../data/movies/bow/ppmi/class-all-normalized--1,1")

for line in file:
    line = line.split()
    for l in range(len(line)):
        line[l] = float(line[l])
        if line[l] > 1 or line[l] < -1:
            print("FAILED!", line[l])
    print(line)

plotSpace(scaleSpace(import2dArray("../data/movies/bow/ppmi/class-all"), -1, 1, "../data/movies/bow/ppmi/class-all-normalized--1,1"))
"""
#convertToTfIDF("../data/movies/bow/frequency/phrases/class-All")
#convertToPPMI("../data/movies/bow/frequency/phrases/class-All", "../data/movies/bow/phrase_names.txt")

"""
file = np.asarray(import2dArray("../data/movies/bow/tfidf/class-All")).transpose()
phrase_names = import1dArray("../data/movies/bow/phrase_names.txt")
movie_names = import1dArray("../data/movies/nnet/spaces/filmNames.txt")
example = file[1644]
indexes = np.argsort(example)
for i in indexes:
    print(phrase_names[i])
"""