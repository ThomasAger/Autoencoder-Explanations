import newsgroups
import sentiment
from svm import linearSVMScore
from svm import multiClassLinearSVM
import data as dt
import csv
import numpy as np
from sklearn.decomposition import PCA
import reuters
def getPCA(tf, depth):
    svd = PCA(n_components=depth, svd_solver="full") # use the scipy algorithm "arpack"
    pos = svd.fit_transform(tf)
    return pos

def testAll(name_array, rep_array, class_array, data_type):
    csv_rows = []

    for i in range(len(rep_array)):
        if data_type == "newsgroups":
            x_train, y_train, x_test, y_test, x_dev, y_dev = newsgroups.getSplits(rep_array[i], class_array[i])
        elif data_type == "sentiment":
            x_train, y_train, x_test, y_test, x_dev, y_dev = sentiment.getSplits(rep_array[i], class_array[i])
        elif data_type == "reuters":
            x_train, y_train, x_test, y_test, x_dev, y_dev = reuters.getSplits(rep_array[i], class_array[i])

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
    fn = "../data/newsgroups/bow/ppmi/class-all-"+str(30)+"-"+str(18836)+"-" + "all.npz"
    print("Testing", fn)
    testAll([ "mds", "finetune_space", "mds_rankings", "finetune_rankings"],
            [
             dt.import2dArray("../data/newsgroups/nnet/spaces/wvFIXED200.npy"),
            dt.import2dArray("../data/newsgroups/nnet/spaces/sns_ppmi3wvFIXED200CV1S0 SFT0 allL03018836 LR kappa KMeans CA200 MC1 MS0.4 ATS2000 DS400FT BOCFi NT[200]tanh300S6040V1.2L0.npy"),
            dt.import2dArray("../data/newsgroups/rank/numeric/sns_ppmi3wvFIXED200CV1S0 SFT0 allL03018836 LR kappa KMeans CA400 MC1 MS0.4 ATS500 DS800.npy").transpose(),
            dt.import2dArray("../data/newsgroups/nnet/clusters/sns_ppmi3wvFIXED200CV1S0 SFT0 allL03018836 LR kappa KMeans CA200 MC1 MS0.4 ATS2000 DS400FT BOCFi NT[200]tanh300S6040V1.2.npy").transpose()],
            [
                dt.import2dArray("../data/newsgroups/classify/newsgroups/class-all", "i"),
                dt.import2dArray("../data/newsgroups/classify/newsgroups/class-all", "i"),
                dt.import2dArray("../data/newsgroups/classify/newsgroups/class-all", "i"),
                dt.import2dArray("../data/newsgroups/classify/newsgroups/class-all", "i")
             #np.load("../data/raw/newsgroups/" + "simple_numeric_stopwords" + "_classes_categorical.npy")
            ], "newsgroups")

