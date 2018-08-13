import os
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy import sparse as sp

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


def write1dArray(array, name, encoding=None):
    try:
        file = open(name, "w", encoding=encoding)
        for i in range(len(array)):
            file.write(str(array[i]) + "\n")
        file.close()
    except FileNotFoundError:
        print("Failed")


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


def write_to_csv(csv_fn, col_names, cols_to_add):
    df = pd.read_csv(csv_fn, index_col=0)
    for c in range(len(cols_to_add)):
        df[col_names[c]] = cols_to_add[c]
    df.to_csv(csv_fn)


def write_to_csv_key(csv_fn, col_names, cols_to_add, keys):
    df = pd.read_csv(csv_fn, index_col=0)
    for c in range(len(cols_to_add)):
        for k in range(len(keys)):
            df[col_names[c]][k] = cols_to_add[c][k]
    df.to_csv(csv_fn)


def read_csv(csv_fn):
    csv = pd.read_csv(csv_fn, index_col=0)
    for col in range(0, len(csv)):
        for val in range(len(csv.iloc[col])):
            if np.isnan(csv.iloc[col][val] ):
                print("!NAN!", col, val)
    return csv

write_to_csv("../../data/newsgroups/rules/tree_csv/sns_ppmi3mdsnew200svmdualCV1S0 SFT0 allL03018836 LR acc KMeans CA400 MC1 MS0.4 ATS500 DS800 newsgroupsAVG.csv", "1", "1")
for i in range(2147000000):
    print(i)



def write_csv(csv_fn, col_names, cols_to_add, key):
    d = {}
    for c in range(len(cols_to_add)):
        d[col_names[c]] = cols_to_add[c]
    df = pd.DataFrame(d, index=key)
    df.to_csv(csv_fn)


def write2dArray(array, name):
    try:
        file = open(name, "w")
        print("starting array")
        for i in range(len(array)):
            for n in range(len(array[i])):
                file.write(str(array[i][n]) + " ")
            file.write("\n")
        file.close()
    except FileNotFoundError:
        print("FAILURE")
    try:
        if name[-4:] == ".txt":
            name = name[:-4]
        array = np.asarray(array)
        np.save(name, array)
    except:
        print("failed")

    print("successful write", name)


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

def isFloat(x):
    try:
        a = float(x)
    except ValueError:
        return False
    else:
        return True

def isInt(x):
    try:
        a = float(x)
        b = int(a)
    except ValueError:
        return False
    else:
        return a == b

def importArray(file_name, file_type="s", return_sparse=True):
    if file_name[-4:] == ".npz":
        if return_sparse is False:
            return sp.load_npz(file_name).toarray()
        return sp.load_npz(file_name)
    elif file_name[-4:] == ".npy":
        return np.load(file_name)#
    else:
        with open(file_name, "r") as infile:
            lines = infile.readlines()
            split = True
            for line in lines:
                val = list(line.split())
                if len(val) == 1:
                    split = False
            if split is True:
                if file_type == "i":
                    return [list(map(int, line.strip().split())) for line in lines]
                elif file_type == "f":
                    return [list(map(float, line.strip().split())) for line in lines]
                else:
                    return np.asarray([list(line.strip().split()) for line in lines])
            else:
                if file_type == "i":
                    return [int(line.strip()) for line in lines]
                elif file_type == "f":
                    return [float(line.strip()) for line in lines]
                else:
                    return [line.strip() for line in lines]
def isArray(N):
    if hasattr(N, '__len__') and (not isinstance(N, str)):
        return True
    else:
        return False

def writeArray(array, name):
    if isArray(array[0]):
        write2dArray(array, name)
    else:
        write1dArray(array, name)
    print("successful write", name)


def getFolder(folder_path):
    onlyfiles = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    two_d = []
    for name in onlyfiles:
        one_d = import1dArray(folder_path + name)
        two_d.append(one_d)
    return two_d


def import1dArray(file_name, file_type="s"):
    with open(file_name, "r", encoding="cp1252") as infile:
        if file_type == "f":
            array = []
            lines = infile.readlines()
            for line in lines:
                array.append(float(line.strip()))
        elif file_type == "i":
            array = [int(float(line.strip())) for line in infile]
        else:
            array = [line.strip() for line in infile]
    return np.asarray(array)

def import2dArray(file_name, file_type="f", return_sparse=False):
    if file_name[-4:] == ".npz":
        print("Loading sparse array")
        array = sp.load_npz(file_name)
        if return_sparse is False:
            array = array.toarray()
    elif file_name[-4:] == ".npy":
        print("Loading numpy array")
        array = np.load(file_name)#
    else:
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
                array = np.asarray([list(line.strip().split()) for line in infile])
        array = np.asarray(array)
    print("successful import", file_name)
    return array

def importNumpyVectors(numpy_vector_path=None):
    movie_vectors = np.load(numpy_vector_path)
    movie_vectors = np.ndarray.tolist(movie_vectors)
    movie_vectors = list(reversed(zip(*movie_vectors)))
    movie_vectors = np.asarray(movie_vectors)
    return movie_vectors