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
def convertLine(line):
    line = list(map(float, line.strip().split()))
    return line
def import1dArray(file_name, file_type="s"):
    with open(file_name, "r") as infile:
        if file_type == "f":
            array = []
            lines = infile.readlines()
            for line in lines:
                array.append(float(line.strip()))
        elif file_type == "i":
            array = [int(line.strip()) for line in infile]
        else:
            array = [line.strip() for line in infile]
    return array
def balanceClasses(movie_vectors, class_array):
    count = 0
    count2 = 0
    for i in class_array:
        if i == 0:
            count+=1
        else:
            count2+=1
    indexes_to_remove = []
    amount_to_balance_to = count-count2
    amount = 0
    while amount < amount_to_balance_to:
        index = random.randint(0, len(class_array) - 1)
        if class_array[index] == 0:
            indexes_to_remove.append(index)
            amount+=1
    movie_vectors = np.delete(movie_vectors, indexes_to_remove, axis=0)
    class_array = np.delete(class_array, indexes_to_remove)

    return movie_vectors, class_array

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

def importFirst2dArray(file_name, file_type="f", amount=100):
    array = []
    with open(file_name, "r") as infile:
        counter = 0
        for line in infile:
            if file_type == "i":
                array.append(list(map(int, line.strip().split())))
            elif file_type == "f":
                array.append(list(map(float, line.strip().split())))
            elif file_type == "discrete":
                to_add = list(line.strip().split())
                for v in range(len(to_add)):
                    to_add[v] = int(to_add[v][:-1])
            else:
                array.append(list(line.strip().split()))
            if counter > amount:
                return array
            counter += 1
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

def getFolder(folder_path):
    onlyfiles = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    two_d = []
    for name in onlyfiles:
        one_d = import1dArray(folder_path + name)
        two_d.append(one_d)
    return two_d

def balanceClasses(movie_vectors, class_array):
    count = 0
    count2 = 0
    for i in class_array:
        if i == 0:
            count+=1
        else:
            count2+=1
    indexes_to_remove = []
    amount_to_balance_to = count - count2*2
    amount = 0
    while amount < amount_to_balance_to:
        index = random.randint(0, len(class_array) - 1)
        if class_array[index] == 0:
            indexes_to_remove.append(index)
            amount+=1
    movie_vectors = np.delete(movie_vectors, indexes_to_remove, axis=0)
    class_array = np.delete(class_array, indexes_to_remove)

    return movie_vectors, class_array

def balance2dClasses(movie_vectors, movie_classes, min_occ):

    indexes_to_remove = []
    for m in range(len(movie_classes)):
        counter = 0
        for i in movie_classes[m]:
            if i > 0:
                counter+=1
        if counter < min_occ:
            indexes_to_remove.append(m)

    movie_vectors = np.delete(movie_vectors, indexes_to_remove, axis=0)
    movie_classes = np.delete(movie_classes, indexes_to_remove, axis=0)
    print("deleted", len(indexes_to_remove))
    return movie_vectors, movie_classes


"""

DATA EDITING TASKS

"""

def toBool(string):
    if string == "True":
        return True
    else:
        return False

def writeArrayDict(dict, name):
    file = open(name, "w")
    for key, value in dict.items():
        file.write(str(key) + ": ")
        for v in value:
            file.write(str(v) + " ")
        file.write("\n")
    file.close()

def writeArrayDict1D(dict, name):
    file = open(name, "w")
    for key, value in dict.items():
        file.write(str(key) + ": ")
        file.write(str(value) + " ")
        file.write("\n")
    file.close()

def readArrayDict(file_name):
    file = open(file_name)
    lines = file.readlines()
    dict = OrderedDict()
    for l in lines:
        l = l.split()
        if l[0][len(l[0])-1:] == ":":
            name = l[0][:-1]
        else:
            name = l[0]
        del l[0]
        dict[name] = l
        print(name)
    return dict

def readArrayDict1D(file_name):
    file = open(file_name)
    lines = file.readlines()
    dict = OrderedDict()
    for l in lines:
        l = l.split()
        if l[0][len(l[0])-1:] == ":":
            name = l[0][:-1]
        else:
            name = l[0]
        del l[0]
        dict[name] = int(l)
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

def allFnsAlreadyExist(all_fns):
    all_exist = 0
    for f in range(len(all_fns)):
        if fileExists(all_fns[f]):
            print(all_fns[f], "Already exists")
            all_exist += 1
        else:
            print(all_fns[f], "Doesn't exist")
    if all_exist == len(all_fns):
        return True
    return False

def write2dArray(array, name):
    file = open(name, "w")
    for i in range(len(array)):
        for n in range(len(array[i])):
            file.write(str(array[i][n]) + " ")
        file.write("\n")
    file.close()
def sortIndexesByArraySize(array):
    array_of_lens = []
    for i in range(len(array)):
        array_of_lens.append(len(array[i]))
    indexes = [i[0] for i in sorted(enumerate(array_of_lens), key=lambda x: x[1])]
    return indexes

import pandas as pd

def fileExists(fn):
    return os.path.exists(fn)

def write_to_csv(csv_fn, col_names, cols_to_add):
    df = pd.read_csv(csv_fn, index_col=0)
    for c in range(len(cols_to_add)):
        df[col_names[c]] = cols_to_add[c]
    df.to_csv(csv_fn)

def read_csv(csv_fn):
    return pd.read_csv(csv_fn, index_col=0)

def write_csv(csv_fn, col_names, cols_to_add, key):
    d = {}
    for c in range(len(cols_to_add)):
        d[col_names[c]] = cols_to_add[c]
    df = pd.DataFrame(d, index=key)
    df.to_csv(csv_fn)

def average_csv(csv_fns):
    csvs = []
    for i in range(len(csv_fns)):
        csvs.append(read_csv(csv_fns[i]))
    for c in csvs:
        for col in c:
            for val in col:
                print(val)
        print("Average")

csv_fns = []
for i in range(5):
    csv_fns.append("../data/wines/rules/tree_csv/" +
        "wines ppmi E200 DS[100, 100, 100] DN0.5 HAtanh CV5 S0 SFT0L050 ndcg0.9001100" + ".csv")
#average_csv(csv_fns)

def findDifference(string1, string2):
    index = 0
    for l in range(len(string1)):
        if string1[l] != string2[l]:
            index = l
            break
        else:
            index = len(string1)
    print(string1[index:])
    print(string2[index:])
"""
findDifference("moviesppmirankE200DS[200, 100, 50]DN0.3reluCV1SFT0S0L0200ndcg0.8400IT3000.txt",
               "moviesppmirankE200DS[200, 100, 50]DN0.3reluCV1SFT0S0L0200ndcg0.9400IT3000.txt")
"""


def write2dCSV(array, name):
    file = open(name, "w")

    for i in range(len(array[0])):
        if i >= len(array[0]) - 1:
            file.write(str(i) + "\n")
        else:
            file.write(str(i) + ",")
    for i in range(len(array)):
        for n in range(len(array[i])):
            if n >= len(array[i])-1:
                file.write(str(array[i][n]))
            else:
                file.write(str(array[i][n]) + ",")
        file.write("\n")
    file.close()


def writeCSV(features, classes, class_names, file_name, header=True):
    for c in range(len(class_names)):
        file = open(file_name + class_names[c] + ".csv", "w")
        if header:
            for i in range(len(features[0])):
                if i >= len(features[0]) - 1:
                    file.write(str(i) + "," + class_names[c] + "\n")
                else:
                    file.write(str(i) + ",")
        for i in range(len(features)):
            for n in range(len(features[i])):
                if n >= len(features[i]) - 1:
                    if classes[c][i] == 0:
                        file.write(str(features[i][n]) + ",FALSE")
                    else:
                        file.write(str(features[i][n]) + ",TRUE")
                else:
                    file.write(str(features[i][n]) + ",")
            file.write("\n")
        file.close()

def writeArff(features, classes, class_names, file_name, header=True):
    for c in range(len(class_names)):
        file = open(file_name + class_names[c] + ".arff", "w")
        file.write("@RELATION genres\n")
        if header:
            for i in range(len(features[0])):
                file.write("@ATTRIBUTE " + str(i) + " NUMERIC\n")
            file.write("@ATTRIBUTE " + class_names[c] + " {f,t}\n")
        file.write("@DATA\n")
        for i in range(len(features)):
            for n in range(len(features[i])):
                if n >= len(features[i]) - 1:
                    if classes[c][i] == 0:
                        file.write(str(features[i][n]) + ",f")
                    else:
                        file.write(str(features[i][n]) + ",t")
                else:
                    file.write(str(features[i][n]) + ",")
            file.write("\n")
        file.close()


def write1dArray(array, name):
    file = open(name, "w")
    for i in range(len(array)):
        file.write(str(array[i]) + "\n")
    file.close()

import io
def write1dLinux(array, name):
    file = io.open(name, "w", newline='\n')
    for i in range(len(array)):
        file.write(str(array[i]) + "\n")
    file.close()

def write1dCSV(array, name):
    file = open(name, "w")
    file.write("0\n")
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
# Finding the differences between two entities with different bag of words, but similar terms
def getIndexOfCommonElements(short_list, long_list):
    index = []
    for n in range(len(short_list)):
        for ni in range(len(long_list)):
            if short_list[n] == long_list[ni]:
                index.append(ni)
    return index


def getScoreDifferences(name_word_file1, name_score_file1, name_word_file2, name_score_file2, name, data_type):

    scores1 = import1dArray(name_score_file1, "f")
    scores2 = import1dArray(name_score_file2, "f")

    words1 = import1dArray(name_word_file1, "s")
    words2 = import1dArray(name_word_file2, "s")

    differences_list = []
    if len(words1) > len(words2):
        same_element_index = getIndexOfCommonElements(words2, words1)
        scores1 = np.asarray(scores1)[same_element_index]
        words1 = np.asarray(words1)[same_element_index]
    else:
        same_element_index = getIndexOfCommonElements(words1, words2)
        scores2 = np.asarray(scores2)[same_element_index]
        words2 = np.asarray(words2)[same_element_index]

    for i in range(len(scores1)):
        differences_list.append(scores1[i] - scores2[i])
    most_different_words = [x for (y,x) in sorted(zip(differences_list,words1))]
    differences_list = sorted(differences_list)
    write1dArray(most_different_words, "../data/"+data_type+"/SVM/difference/most_different_words_" + name + ".txt")
    write1dArray(differences_list, "../data/"+data_type+"/SVM/difference/most_different_values_" + name + ".txt")
data_type = "placetypes"
filepath = "../data/"+data_type+"/"
"""
getScoreDifferences(filepath + "bow/names/50-10-geonames.txt", filepath + "ndcg/placetypes mds E2000 DS[100] DN0.6 CTgeonames HAtanh CV1 S0 DevFalse LETrue SFT0L050.txt",
                    filepath + "bow/names/50-10-all.txt", filepath + "ndcg/placetypes mds E2000 DS[100] DN0.6 CTgeonames HAtanh CV1 S0 DevFalse LEFalse SFT0L050.txt",
                    "placetypes mds E3000 DS[100] DN0.6 CTgeonames HAtanh CV1 S0 DevFalse SFT0L0 LE", data_type)
print("done doing score diff")
"""
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


def writeIndividualClasses(overall_class_fn, names_fn, output_filename):
    overall_class = import2dArray(overall_class_fn, "f")
    names = import1dArray(names_fn)
    for n in range(len(names)):
        write1dArray(overall_class[n], output_filename + "class-" + names[n])
        print(names[n])

#writeIndividualClasses("../data/movies/bow/frequency/phrases/class-all-scaled0,1.txt", "../data/movies/bow/phrase_names.txt", "../data/movies/bow/normalized_frequency/")
#writeIndividualClasses("../data/movies/bow/ppmi/class-all-scaled0,1", "../data/movies/bow/phrase_names.txt", "../data/movies/bow/normalized_ppmi/")
def plotSpace(space):

    single_values = []

    counter = 0
    for s in space:
        single_values.extend(s)

    # basic plot
    sns.distplot(single_values, kde=False, rug=False)
    sns.plt.show()
    print ("now we here")

#plotSpace("../data/movies/nnet/spaces/films200L1100N0.5TermFrequencyN0.5FT.txt")

def getNamesFromDict(dict_fn, file_name):
    new_dict = import2dArray(dict_fn, "s")
    names = []
    for d in range(len(new_dict)):
        names.append(new_dict[d][0].strip())
    write1dArray(names, "../data/movies/cluster/hierarchy_names/" +file_name+".txt")

#getNamesFromDict("../data/movies/cluster/hierarchy_dict/films200L1100N0.50.8.txt", "films200L1100N0.50.8.txt")

def scaleSpace(space, lower_bound, upper_bound, file_name):
    minmax_scale = MinMaxScaler(feature_range=(lower_bound, upper_bound), copy=True)
    space = minmax_scale.fit_transform(space)
    write2dArray(space, file_name)
    return space

import math

def magnitude(v):
    return math.sqrt(sum(v[i]*v[i] for i in range(len(v))))

def add(u, v):
    return [ u[i]+v[i] for i in range(len(u)) ]

def sub(u, v):
    return [ u[i]-v[i] for i in range(len(u)) ]

def dot(u, v):
    return sum(u[i]*v[i] for i in range(len(u)))

def normalize(v):
    vmag = magnitude(v)
    return [ v[i]/vmag  for i in range(len(v)) ]

def scaleSpaceUnitVector(space, file_name):
    space = np.asarray(space).transpose()
    print(len(space), len(space[0]))
    scaled_vector = []
    for v in space:
        if np.sum(v) != 0:
            norm = normalize(v)
            scaled_vector.append(norm)
        else:
            scaled_vector.append(v)
    space = space.transpose()
    write2dArray(scaled_vector, file_name)

file_name = "../data/movies/finetune/films200L1100N0.5TermFrequency"

def concatenateArrays(arrays, file_name):
    new_array = arrays[0]
    for a in range(1, len(arrays)):
        new_array = np.concatenate((new_array, arrays[a]), axis=0)
    write2dArray(new_array, file_name)


def getTop10Clusters(file_name, ids):
    clusters = np.asarray(import2dArray("../data/movies/rank/discrete/" + file_name + "P1.txt", "s")).transpose()
    cluster_names = import2dArray("../data/movies/cluster/hierarchy_names/" + file_name+"0.8400.txt", "s")
    for c in range(len(cluster_names)):
        cluster_names[c] = cluster_names[c][0]
    to_get = []
    for i in ids:
        for v in range(len(clusters[i])):
            rank = int(clusters[i][v][:-1])
            if rank <= 3:
                print(cluster_names[v][6:])
        print("----------------------")

from sklearn import svm
from sklearn.metrics import cohen_kappa_score
def obtainKappaOnClusteredDirection(names, ranks):
    # For each discrete rank, obtain the Kappa score compared to the word occ
    kappas = np.empty(len(names))
    for n in range(len(names)):
        clf = svm.LinearSVC()
        ppmi = np.asarray(import1dArray("../data/movies/bow/binary/phrases/" + names[n], "i"))
        clf.fit(ranks, ppmi)
        y_pred = clf.predict(ranks)
        score = cohen_kappa_score(ppmi, y_pred)
        kappas[n] = score
    return kappas

# Takes as input a folder with a series of files
def concatenateDirections(folder_name):
    files = getFns(folder_name)
    all_directions = []
    all_names = []
    all_kappa = []
    for f in files:
        file = open(f)
        lines = file.readlines()
        all_directions.append(lines[0])
        all_names.append(f[:5])

# Get the amount of non-zero occurances for each class in a class-all
def getNonZero(class_names_fn, file_name):
    class_names = import1dArray(class_names_fn, "s")
    class_all = np.asarray(import2dArray(file_name)).transpose()
    for c in range(len(class_all)):
        print(np.count_nonzero(class_all[c]))

#getNonZero("../data/movies/classify/genres/names.txt", "../data/movies/classify/genres/class-all")
import string
import re
def keepNumbers(string):
    s = stripPunctuation(string)
    s = string.lower()
    s = "".join(s.split())
    numbers = re.compile('\d+(?:\.\d+)?')
    s = numbers.findall(s)
    return s
def removeEverythingFromString(string):
    string = stripPunctuation(string)
    string = string.lower()
    string = "".join(string.split())
    return string
def lowercaseSplit(string):
    string = string.lower()
    string = "".join(string.split())
    return string

def remove_indexes(indexes, array_fn):
    array = np.asarray(import1dArray(array_fn))
    array = np.delete(array, indexes, axis=0)
    write1dArray(array, array_fn)
    print("wrote", array_fn)

def averageCSVs(csv_array_fns):
    csv_array = []
    for csv_name in csv_array_fns:
        csv_array.append(read_csv(csv_name))
    for csv in range(1, len(csv_array)):
        for col in range(1, len(csv_array[csv])):
            for val in range(len(csv_array[csv].iloc[col])):
                csv_array[0].iloc[col][val] += csv_array[csv].iloc[col][val]
    for col in range(1, len(csv_array[0])):
        for val in range(len(csv_array[0].iloc[col])):
            print(csv_array[0].iloc[col][val])
            csv_array[0].iloc[col][val] = csv_array[0].iloc[col][val] / len(csv_array)
            print(csv_array[0].iloc[col][val])

    csv_array[0].to_csv(csv_array_fns[0][:len(csv_array_fns[0])-4] + "AVG.csv")

def stringToArray(string):
    array = string.split()
    for e in range(len(array)):
        array[e] = eval(removeEverythingFromString(array[e]))
    return array

def chunks(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]
import time
def compileSVMResults(file_name, chunk_amt, data_type):
    if fileExists("../data/"+data_type+"/svm/directions/"+file_name+".txt") is False:
        print("Compiling SVM results")
        randomcount = 0
        directions = []
        for c in range(chunk_amt):
            directions.append("../data/"+data_type+"/svm/directions/"+file_name + " CID" + str(c) + " CAMT" + str(chunk_amt)+".txt")
        kappa = []
        for c in range(chunk_amt):
            kappa.append("../data/"+data_type+"/svm/kappa/"+file_name + " CID" + str(c) + " CAMT" + str(chunk_amt)+".txt")
        for f in directions:
            while not fileExists(f):
                time.sleep(10)
        time.sleep(10)
        di = []
        for d in directions:
            di.extend(import2dArray(d))
        ka = []
        for k in kappa:
            ka.extend(import1dArray(k))
        write2dArray(di, "../data/"+data_type+"/svm/directions/"+file_name+".txt")
        write1dArray(ka, "../data/"+data_type+"/svm/kappa/"+file_name+".txt")
    else:
        print ("Skipping compile")

def shorten2dFloats(floats_fn):
    fa = import2dArray(floats_fn)
    for a in range(len(fa)):
        fa[a] = np.around(fa[a], decimals=4)
    return fa

def deleteAllButIndexes(array, indexes):
    old_ind = list(range(len(array)))
    del_ind = np.delete(old_ind, indexes)
    array = np.delete(array, del_ind)
    return array

def match_entities(entities, t_names, names):
    amount_found = 0
    for n in range(len(names)):
        names[n] = removeEverythingFromString(names[n])
    for n in range(len(t_names)):
        t_names[n] = removeEverythingFromString(t_names[n])
    matched_ids = []
    for n in range(len(t_names)):
        for ni in range(len(names)):
            matched_name = t_names[n]
            all_name = names[ni]
            if matched_name == all_name:
                matched_ids.append(ni)
                amount_found += 1
                break
    matched_entities = []
    for e in matched_ids:
        matched_entities.append(entities[e])
    print("Amount found", amount_found)
    return matched_entities

"""
write2dArray(deleteAllButIndexes(import2dArray("../data/movies/cluster/hierarchy_directions/films200-genres100ndcg0.9200.txt", "s"),
                                               import1dArray("../data/movies/cluster/hierarchy_names/human_ids films200genres.txt")),
                                 "../data/movies/cluster/hierarchy_directions/films200-genres100ndcg0.9200 human_prune.txt")

"""
#getTop10Clusters("films100L2100N0.5", [1644,164,4018,6390])

"""
from sklearn.datasets import dump_svmlight_file
genre_names = import1dArray("../data/movies/classify/genres/names.txt", "s")


genres = np.asarray(import2dArray("../data/movies/classify/genres/class-all", "i")).transpose()

class_all = []
#for i in range(23):
    #g = import1dArray("../data/movies/classify/genres/class-" + genre_names[i], "i")
    #class_all.append(g)

class_all = np.asarray(class_all).transpose()

#write2dArray(class_all, "../data/movies/classify/genres/class-all")

space_name = "films200-genres100ndcg0.85200 tdev3004FTL0 E100 DS[200] DN0.5 CTgenres HAtanh CV1 S0 DevFalse SFT0L0100ndcg0.95200MC1"
space = np.asarray(import2dArray("../data/movies/rank/numeric/"+space_name+".txt")).transpose()

writeArff(space, genres, genre_names, "../data/movies/keel/vectors/"+space_name+"genres", header=True)

#np.savetxt( "../data/movies/keel/vectors/"+space_name+"np.csv", space, delimiter=",")
"""



"""



fn = "films200L325N0.5"
cluster_names_fn = "../data/movies/cluster/names/" + fn + ".txt"
file_name = fn + "InClusterN0.5FTadagradcategorical_crossentropy100"
cluster_vectors_fn = "../data/movies/nnet/clusters/" + file_name + ".txt"
new_v = []
new_v.append(import1dArray(cluster_names_fn))

fn = "films200L250N0.5"
cluster_names_fn = "../data/movies/cluster/names/" + fn + ".txt"
file_name = fn + "InClusterN0.5FTadagradcategorical_crossentropy100"
cluster_vectors_fn = "../data/movies/nnet/clusters/" + file_name + ".txt"

new_v.append(import1dArray(cluster_names_fn))

fn = "films200L1100N0.5"
cluster_names_fn = "../data/movies/cluster/names/" + fn + ".txt"
file_name = fn + "InClusterN0.5FTadagradcategorical_crossentropy100"
cluster_vectors_fn = "../data/movies/nnet/clusters/" + file_name + ".txt"

new_v.append(import1dArray(cluster_names_fn))

concatenateArrays(new_v, cluster_names_fn+"ALL")
#space = import2dArray(file_name + ".txt")

#scaleSpaceUnitVector(space, file_name+"uvscaled.txt")

#scaleSpace(space, 0, 1, file_name +"scaled")
"""
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