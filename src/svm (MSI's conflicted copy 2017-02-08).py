
from sklearn import svm
import numpy as np
from sklearn.metrics import cohen_kappa_score, mean_squared_error, f1_score, accuracy_score
import data as dt
from scipy import  linalg
import itertools


def runGaussianSVM(y_test, y_train, x_train, x_test, get_kappa, get_f1):
    clf = svm.SVC(kernel='rbf', class_weight='balanced')
    if get_f1:
        cross_val = cross_val_score(clf, x_train, y_train, scoring="f1", cv=5)
        f1 = np.average(cross_val)
    else:
        f1 = 0
    clf.fit(x_train, y_train)
    direction = clf.dual_coef_.tolist()[0]
    y_pred = clf.predict(x_test)
    y_pred = y_pred.tolist()
    if get_kappa:
        kappa_score = cohen_kappa_score(y_test, y_pred)
    else:
        kappa_score = 0

    ktau = 0
    #ppmi_score, ppmi_ratio = get_ppmi_score(y_pred, property_name)
    return kappa_score, direction, f1, 0, 0

from sklearn.model_selection import cross_val_score
def runSVM(x_train, x_test, property_name, get_kappa, get_f1, data_type):

    y = dt.import1dArray("../data/" + data_type + "/bow/binary/phrases/class-" + property_name)

    y_train, y_test = train_test_split(y, test_size=0.3, random_state=0)
    #x_train, y_train = dt.balanceClasses(x_train, y_train)
    clf = svm.LinearSVC(class_weight="balanced")
    """
    if get_f1:
        cross_val = cross_val_score(clf, x_train, y_train, scoring="f1", cv=5)
        f1 = np.average(cross_val)
    else:
        f1 = 0
    """
    try:
        clf.fit(x_train, y_train)
        direction = clf.coef_.tolist()[0]
        y_pred = clf.predict(x_test)
        y_pred = y_pred.tolist()
        if get_f1:
            f1 = f1_score(y_test, y_pred, average="macro")
        else:
            f1 = 0
        if get_kappa:
            kappa_score = cohen_kappa_score(y_test, y_pred)
        else:
            kappa_score = 0

        ktau = 0
        #ppmi_score, ppmi_ratio = get_ppmi_score(y_pred, property_name)
    except ValueError:
        kappa_score = 0
        f1 = 0
        direction = None
    return kappa_score, f1, direction,  0, 0
from sklearn.model_selection import cross_val_score

import random

import multiprocessing
import math
from functools import partial
from multiprocessing.dummy import Pool as ThreadPool

def runAllSVMs(y_test, y_train, x_train, x_test, property_names, file_name, svm_type, get_kappa, get_f1, getting_directions, data_type, threads):
    kappa_scores = [0] * len(property_names)
    directions = [None] * len(property_names)
    f1_scores = [0] * len(property_names)
    saved_x_trans = None
    saved_test_x_trans = None
    indexes_to_remove = []
    for y in range(0, len(property_names), threads):
        x_train_a = [None] * threads
        x_test_a = [None] * threads
        property_names_a = [None] * threads
        get_kappa_a = [None] * threads
        get_f1_a = [None] * threads
        data_type_a = [None] * threads
        for t in range(threads):
            try:
                x_train_a[t] = x_train
                x_test_a[t] = x_test
                property_names_a[t] = property_names[y+t]
                get_kappa_a[t] = get_kappa
                get_f1_a[t] = get_f1
                data_type_a[t] = data_type
            except IndexError as e:
                print(e)
                for x in range(len(x_train_a)-1, 0, -1):
                    print(x_train_a[x])
                    if property_names_a[x] is None or x_train_a[x] is None:
                        del x_train_a[x]
                        del x_test_a[x]
                        del property_names_a[x]
                        del get_kappa_a[x]
                        del get_f1_a[x]
                        del data_type_a[x]
                break

        pool = ThreadPool(threads)
        kappa = pool.starmap(runSVM, zip(x_train_a, x_test_a, property_names_a, get_kappa_a, get_f1_a, data_type_a))
        pool.close()
        pool.join()
        for t in range(len(kappa)):
            if kappa[t][2] is not None and math.isnan(kappa[t][0]) is False and kappa[t][0] != 0.0:
                kappa_scores[y+t] = kappa[t][0]
                f1_scores[y+t] = kappa[t][1]
                directions[y+t] = kappa[t][2]
                print(y, "Score", kappa[t][0],  kappa[t][1], property_names_a[t])
            else:
                indexes_to_remove.append(y+t)
        """
        kappa, f1, direction,  unused_variable, unused_variable_2 = runSVM(y_test, y_train, x_train,  x_test,
                                                                    property_names[y], get_kappa, get_f1, data_type)

        kappa_scores[y] = kappa
        directions[y] = direction
        f1_scores[y] = f1
        print(y, "Score", kappa, f1, property_names[y])
        """

    kappa_scores = np.delete(np.asarray(kappa_scores), indexes_to_remove, axis=0)
    directions = np.delete(np.asarray(directions), indexes_to_remove, axis=0)
    f1_scores = np.delete(np.asarray(f1_scores), indexes_to_remove, axis=0)

    return kappa_scores, directions, f1_scores

from sklearn.cross_validation import train_test_split

def getSVMResults(vector_path, class_path, property_names_fn, file_name, svm_type, training_size=10000,  lowest_count=200,
                  highest_count=21470000, get_kappa=True, get_f1=True, single_class=True, data_type="movies",
                  getting_directions=True, threads=1,
                 rewrite_files=False):
    directions_fn = "../data/" + data_type + "/svm/directions/" + file_name + ".txt"
    kappa_fn = "../data/" + data_type + "/svm/kappa/" + file_name + ".txt"
    ktau_scores_fn = "../data/" + data_type + "/svm/f1/" + file_name + ".txt"

    all_fns = [directions_fn, kappa_fn, ktau_scores_fn]
    if dt.allFnsAlreadyExist(all_fns) and not rewrite_files:
        print("Skipping task", "getSVMResults")
        return
    else:
        print("Running task", "getSVMResults")

    y_train = 0
    y_test = 0
    if get_f1:
        vectors = np.asarray(dt.import2dArray(vector_path)).transpose()
    else:
        vectors = np.asarray(dt.import2dArray(vector_path))
    if not getting_directions:
        classes = np.asarray(dt.import2dArray(class_path))
    property_names = dt.import1dArray(property_names_fn)

    if single_class and not getting_directions:
        classes = classes.transpose()

    if not getting_directions:
        x_train, x_test, y_train, y_test = train_test_split(vectors, classes, test_size=0.3, random_state=0)
    else:
        x_train, x_test = train_test_split(vectors,  test_size=0.3, random_state=0)

    if single_class and not getting_directions:
        y_train = y_train.transpose()
        y_test = y_test.transpose()

    kappa_scores, directions, ktau_scores = runAllSVMs(y_test, y_train, x_train, x_test, property_names, file_name,
                                                       svm_type, get_kappa, get_f1, getting_directions, data_type, threads)



    dt.write1dArray(kappa_scores, kappa_fn)
    dt.write2dArray(directions, directions_fn)
    dt.write1dArray(ktau_scores, ktau_scores_fn)

class SVM:
    def __init__(self, file_name="", class_names=None, vector_path=None, class_path=None, class_by_class=True, input_size=200,
                 training_size=10000, amount_of_scores=400,  low_kappa=0.1, high_kappa=0.5, rankSVM=False, lowest_count=100, largest_count=21470000):
            getSVMResults(vector_path, class_path, file_name)


def main(vectors_fn, classes_fn, property_names, training_size, file_name, lowest_count, largest_count):
    SVM(vectors_fn, classes_fn, property_names, lowest_count=lowest_count,
        training_size=training_size, file_name=file_name, largest_count=largest_count)
data_type = "wines"
file_name = "winesppmirankE500DS[1000, 500, 250, 100, 50]L0DN0.3reluSFT0L050ndcgSimilarityClusteringIT3000"
class_name = "types"
# Get SVM scores
svm_type = "svm"
lowest_count = 200
highest_count = 10000
cluster_amt = 200
split = 0.9
vector_path = "../data/" + data_type + "/rank/numeric/"+file_name+".txt"
class_path = "../data/" + data_type + "/classify/"+class_name+"/class-all"
property_names_fn = "../data/" + data_type + "/classify/"+class_name+"/names.txt"
file_name = file_name + "genre"
#getSVMResults(vector_path, class_path, property_names_fn, file_name, lowest_count=lowest_count, data_type=data_type, get_kappa=False, get_f1=True, highest_count=highest_count, svm_type=svm_type, rewrite_files=True)


"""
ppmi = np.asarray(dt.import2dArray("../data/movies/bow/ppmi/class-all")).transpose()

from sklearn import decomposition

pca = decomposition.PCA(n_components=100)
pca.fit(ppmi)
pca = pca.transform(ppmi)

dt.write2dArray(pca, "../data/movies/nnet/spaces/pca.txt")

file_name = "pca"

# Get SVM scores
lowest_count = 200
highest_count = 10000
vector_path = "../data/movies/nnet/spaces/"+file_name+".txt"
class_path = "../data/movies/bow/binary/phrases/class-all-200"
property_names_fn = "../data/movies/bow/names/" + str(lowest_count) + ".txt"
getSVMResults(vector_path, class_path, property_names_fn, file_name, lowest_count=lowest_count, highest_count=highest_count)


property_names_fn = "../data/movies/classify/keywords/names.txt"
class_path = "../data/movies/classify/keywords/class-All"
file_name = "filmsPPMIDropoutL1100DNNonerelusoftplusadagradkullback_leibler_divergence"
vector_path = "../data/movies/nnet/spaces/"+file_name+".txt"
file_name = "films100"
vector_path = "../data/movies/nnet/spaces/"+file_name+".txt"
getSVMResults(vector_path, class_path, property_names_fn, "LinearGenre"+file_name)
getSVMResults(vector_path, class_path, property_names_fn, "LinearGenre"+file_name)

path="newdata/spaces/"
#path="filmdata/films200.mds/"
#array = ["700", "400", "100"]
filenames = ["films100N0.6H75L1", "films100N0.6H50L2", "films100N0.6H25L3",
             "films100N0.6H50L4", "films100N0.6H75L5", "films100N0.6H100L6"]

"""

"AUTOENCODER0.2tanhtanhmse15tanh[1000]4SDA1","AUTOENCODER0.2tanhtanhmse60tanh[200]4SDA2","AUTOENCODER0.2tanhtanhmse30tanh[1000]4SDA3",
"AUTOENCODER0.2tanhtanhmse60tanh[200]4SDA4"
"""
cut = 100
for f in range(len(filenames)):
    newSVM = SVM(vector_path=path+filenames[f]+".mds", class_path="filmdata/classesPhrases/class-All", lowest_count=cut, training_size=10000, file_name=filenames[f]+"LS", largest_count=9999999999)
"""
"""
if  __name__ =='__main__':main(vectors, classes, property_names, file_name, training_size,  lowest_count, largest_count)
"""