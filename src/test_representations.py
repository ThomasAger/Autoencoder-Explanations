import newsgroups
import sentiment
from svm import linearSVMScore
from svm import multiClassLinearSVM
import data as dt
import csv
import numpy as np
from sklearn.decomposition import PCA

def getPCA(tf, depth):
    svd = PCA(n_components=depth, svd_solver="full") # use the scipy algorithm "arpack"
    pos = svd.fit_transform(tf)
    return pos

def testAll(name_array, rep_array, class_array, data_type):
    csv_rows = []

    for i in range(len(rep_array)):
        if data_type == "newsgroups":
            x_train, y_train, x_test, y_test, x_dev, y_dev = newsgroups.getSplits(rep_array[i], class_array[i])
            scores = multiClassLinearSVM(x_train, y_train, x_dev, y_dev)
            f1 = scores[0]
            acc = scores[1]
            macro_f1 = scores[2]
            csv_rows.append((name_array[i], acc, f1, macro_f1))
            print(csv_rows[i])
    with open("../data/raw/" + data_type + "/test/reps.csv", 'wt') as f:
        writer = csv.writer(f)
        writer.writerow(("name", "acc", "micro f1", "macro f1"))
        writer.writerows(csv_rows)

if __name__ == '__main__':

    testAll([ "ppmi"],
            [
             dt.import2dArray("../data/newsgroups/bow/frequency/phrases/simple_stopwords_bow30-0.999-all.npz").transpose()],
            [
             np.load("../data/raw/newsgroups/" + "simple_stopwords" + "_classes_categorical.npy")], "newsgroups")


    ppmi_fn = "../data/newsgroups/bow/ppmi/simple_stopwords_ppmi 2-gram50-0.99-all.npz"
    freq_fn = "../data/newsgroups/bow/frequency/phrases/simple_stopwords_bow 2-gram50-0.99-all.npz"
