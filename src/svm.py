
from sklearn import svm
import numpy as np
from sklearn.metrics import cohen_kappa_score, mean_squared_error, f1_score, accuracy_score
import helper.data as dt
from scipy import  linalg
import itertools

def transform_pairwise(X, y, just_y=False, file_name="NO FILENAME", just_x=False):
    """Transforms data into pairs with balanced labels for ranking
    Transforms a n-class ranking problem into a two-class classification
    problem. Subclasses implementing particular strategies for choosing
    pairs should override this method.
    In this method, all pairs are choosen, except for those that have the
    same target value. The output is an array of balanced classes, i.e.
    there are the same number of -1 as +1
    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        The data
    y : array, shape (n_samples,) or (n_samples, 2)
        Target labels. If it's a 2D array, the second column represents
        the grouping of samples, i.e., samples with different groups will
        not be considered.
    Returns
    -------
    X_trans : array, shape (k, n_feaures)
        Data as pairs
    y_trans : array, shape (k,)
        Output class labels, where classes have values {-1, +1}
    """
    if not just_y and not just_x:
        X_new = []
        y_new = []
        y = np.asarray(y)
        if y.ndim == 1:
            y = np.c_[y, np.ones(y.shape[0])]
        comb = itertools.combinations(range(X.shape[0]), 2)
        lister = list(comb)
        for k, (i, j) in enumerate(comb):
            if y[i, 0] == y[j, 0] or y[i, 1] != y[j, 1]:
                # skip if same target or different group
                continue
            # If the array doesn't have the same target, subtract the first array from the second and append that
            X_new.append(X[i] - X[j])
            y_new.append(np.sign(y[i, 0] - y[j, 0]))
            # output balanced classes
            if y_new[-1] != (-1) ** k:
                y_new[-1] = - y_new[-1]
                X_new[-1] = - X_new[-1]
        return np.asarray(X_new), np.asarray(y_new).ravel()
    elif just_y:
        y_new = []
        y = np.asarray(y)
        if y.ndim == 1:
            y = np.c_[y, np.ones(y.shape[0])]
        comb = itertools.combinations(range(X.shape[0]), 2)
        for k, (i, j) in enumerate(comb):
            if y[i, 0] == y[j, 0] or y[i, 1] != y[j, 1]:
                # skip if same target or different group
                continue
            y_new.append(np.sign(y[i, 0] - y[j, 0]))
            # output balanced classes
            if y_new[-1] != (-1) ** k:
                y_new[-1] = - y_new[-1]

        return np.asarray(y_new).ravel()
    elif just_x:
        X_new = []
        comb = itertools.combinations(range(X.shape[0]), 2)
        for k, (i, j) in enumerate(comb):
            X_new.append(X[i] - X[j])
        return np.asarray(X_new)


class RankSVM(svm.LinearSVC):
    """Performs pairwise ranking with an underlying LinearSVC model
    Input should be a n-class ranking problem, this object will convert it
    into a two-class classification problem, a setting known as
    `pairwise ranking`.
    See object :ref:`svm.LinearSVC` for a full description of parameters.
    """

    X_trans = None
    y_trans = None
    score_X_trans = None
    score_y_trans = None
    phrase_name = ""
    file_name = ""


    def fit(self, X, y, file_name, phrase_name, old_X_trans, old_test_X_trans):
        """
        Fit a pairwise ranking model.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        y : array, shape (n_samples,) or (n_samples, 2)
        Returns
        -------
        self
        """

        if old_X_trans is not None:
            self.X_trans = old_X_trans

        if old_test_X_trans is not None:
            self.score_X_trans = old_test_X_trans

        self.file_name = file_name
        file_name += "pairwise"
        full_fn = "../data/movies/nnet/spaces/" + file_name + ".txt"

        self.phrase_name = phrase_name
        phrase_name = phrase_name + "pairwise"
        full_phrase_fn = "../data/movies/bow/frequency/phrases/" + phrase_name

        if self.X_trans is None:
            no_x = False
            no_y = False
            try:
                file = open(full_fn)
                self.X_trans = dt.import2dArray(full_fn)
            except FileNotFoundError:
                no_x = True
            try:
                file = open(full_phrase_fn)
                self.y_trans = dt.import2dArray(full_phrase_fn)
            except FileNotFoundError:
                no_y = True
            if no_x and no_y:
                self.X_trans, self.y_trans = transform_pairwise(X, y, file_name=self.file_name)
                dt.write2dArray(self.X_trans, full_fn)
                dt.write1dArray(self.y_trans, full_phrase_fn)
            elif no_y and not no_x:
                self.y_trans = transform_pairwise(X, y, True)
                dt.write1dArray(self.y_trans, full_phrase_fn)
        else:
            if self.y_trans is None:
                try:
                    file = open(full_phrase_fn)
                    self.y_trans = dt.import2dArray(full_phrase_fn)
                except FileNotFoundError:
                    self.y_trans = transform_pairwise(X, y, True)
                    dt.write1dArray(self.y_trans, full_phrase_fn)

        super(RankSVM, self).fit(self.X_trans, self.y_trans)
        return self

    def get_x_trans(self):
        return self.X_trans, self.score_X_trans

    def predict(self, X):
        """
        Predict an ordering on X. For a list of n samples, this method
        returns a list from 0 to n-1 with the relative order of the rows of X.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        Returns
        -------
        ord : array, shape (n_samples,)
            Returns a list of integers representing the relative order of
            the rows in X.
        """
        if hasattr(self, 'coef_'):
            np.argsort(np.dot(X, self.coef_.T))
        else:
            raise ValueError("Must call fit() prior to predict()")

    def direction(self):
        return self.coef_


    def kappa_score(self, X, y):
        """
        Because we transformed into a pairwise problem, chance level is at 0.5
        """

        file_name = self.file_name + "test"
        full_fn = "../data/movies/nnet/spaces/" + file_name + ".txt"
        phrase_name = self.phrase_name + "test"
        full_phrase_fn = "../data/movies/bow/frequency/phrases/" + phrase_name

        if self.score_X_trans is None:
            no_x = False
            no_y = False
            try:
                file = open(full_fn)
                self.score_X_trans = dt.import2dArray(full_fn)
            except FileNotFoundError:
                no_x = True
            try:
                file = open(full_phrase_fn)
                self.score_y_trans = dt.import2dArray(full_phrase_fn)
            except FileNotFoundError:
                no_y = True
            if no_x and no_y:
                self.score_X_trans, self.score_y_trans = transform_pairwise(X, y, file_name=self.file_name)
                dt.write2dArray(self.score_X_trans, full_fn)
                dt.write1dArray(self.score_y_trans, full_phrase_fn)
            elif no_y and not no_x:
                self.score_y_trans = transform_pairwise(X, y, True)
                dt.write1dArray(self.score_y_trans, full_phrase_fn)
        else:
            if self.score_y_trans is None:
                try:
                    file = open(full_phrase_fn)
                    self.score_y_trans = dt.import2dArray(full_phrase_fn)
                except FileNotFoundError:
                    self.score_y_trans = transform_pairwise(X, y, True)
                    dt.write1dArray(self.score_y_trans, full_phrase_fn)

        kappa_score = cohen_kappa_score(self.score_y_trans, super(RankSVM, self).predict(self.score_X_trans))
        return kappa_score

    def score(self, X, y):
        """
        Because we transformed into a pairwise problem, chance level is at 0.5
        """
        file_name = self.file_name + "test"
        full_fn = "../data/movies/nnet/spaces/" + file_name + ".txt"
        phrase_name = self.phrase_name + "test"
        full_phrase_fn = "../data/movies/bow/frequency/phrases/class-" + phrase_name

        if self.score_X_trans is None:
            no_x = False
            no_y = False
            try:
                file = open(full_fn)
                self.score_X_trans = dt.import2dArray(full_fn)
            except FileNotFoundError:
                no_x = True
            try:
                file = open(full_phrase_fn)
                self.score_y_trans = dt.import2dArray(full_phrase_fn)
            except FileNotFoundError:
                no_y = True
            if no_x and no_y:
                self.score_X_trans, self.score_y_trans = transform_pairwise(X, y, file_name=self.file_name)
                dt.write2dArray(self.score_X_trans, full_fn)
                dt.write1dArray(self.score_y_trans, full_phrase_fn)
            elif no_y and not no_x:
                self.score_y_trans = transform_pairwise(X, y, True)
                dt.write1dArray(self.score_y_trans, full_phrase_fn)
        else:
            if self.score_y_trans is None:
                try:
                    file = open(full_phrase_fn)
                    self.score_y_trans = dt.import2dArray(full_phrase_fn)
                except FileNotFoundError:
                    self.score_y_trans = transform_pairwise(X, y, True)
                    dt.write1dArray(self.score_y_trans, full_phrase_fn)

        return np.mean(super(RankSVM, self).predict(self.score_X_trans) == self.score_y_trans)


def runRankSVM(y_test, y_train, x_train, x_test, property_name, file_name, saved_x_trans, saved_test_x_trans):
    rank_svm = RankSVM().fit(x_train, y_train, file_name, property_name, saved_x_trans, saved_test_x_trans)
    direction = rank_svm.direction()
    kappa_score = rank_svm.kappa_score(x_test, y_test)
    ktau = rank_svm.score(x_test, y_test)
    saved_x_trans, saved_test_x_trans = rank_svm.get_x_trans()
    print('Performance of ranking ', ktau, property_name)
    return 0, [0], ktau, saved_x_trans, saved_test_x_trans

def get_ppmi_score(y_pred, property_name, data_type):
    term_frequency = dt.import1dArray("../data/" + data_type + "/bow/frequency/phrases/"+property_name)
#    term_frequency = [int(term_frequency[f]) for f in term_frequency]
    total_y = 0
    total_x = 0
    counter_y = 0
    counter_x = 0
    for t in range(len(term_frequency)):
        term_frequency[t] = int(term_frequency[t])
    for i in range(len(y_pred)):
        if i == 0:
            total_x += term_frequency[i]
            counter_x += 1
        else:
            total_y += term_frequency[i]
            counter_y += 1
    avg_n = total_x / counter_x
    avg_p = total_y / counter_y
    score = avg_p - avg_n
    ratio = (100 / (total_y + total_x)) * total_y
    return score, ratio

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
def runSVM(y_test, y_train, x_train, x_test, property_name, get_kappa, get_f1, data_type):
    y = dt.import1dArray("../data/" + data_type + "/bow/binary/phrases/class-trimmed-" + property_name)
    y_train, y_test = train_test_split(y, test_size=0.3, random_state=0)
    clf = svm.LinearSVC(class_weight='balanced')
    if get_f1:
        cross_val = cross_val_score(clf, x_train, y_train, scoring="f1", cv=5)
        f1 = np.average(cross_val)
    else:
        f1 = 0
    clf.fit(x_train, y_train)
    direction = clf.coef_.tolist()[0]
    y_pred = clf.predict(x_test)
    y_pred = y_pred.tolist()
    if get_kappa:
        kappa_score = cohen_kappa_score(y_test, y_pred)
    else:
        kappa_score = 0

    ktau = 0
    #ppmi_score, ppmi_ratio = get_ppmi_score(y_pred, property_name)
    return kappa_score, f1, direction,  0, 0
from sklearn.model_selection import cross_val_score
def runLibSVM(y_test, y_train, x_train, x_test, property_name, get_kappa, get_f1, wasted_variable3):
    clf = svm.SVC(kernel='linear', class_weight='balanced')
    if get_f1:
        cross_val = cross_val_score(clf, x_train, y_train, scoring="f1", cv=5)
        f1 = np.average(cross_val)
    else:
        f1 = 0
    clf.fit(x_train, y_train)
    direction = clf.coef_.tolist()[0]
    y_pred = clf.predict(x_test)
    y_pred = y_pred.tolist()
    if get_kappa:
        kappa_score = cohen_kappa_score(y_test, y_pred)
    else:
        kappa_score = 0

    ktau = 0
    #ppmi_score, ppmi_ratio = get_ppmi_score(y_pred, property_name)
    return kappa_score, direction, f1, 0, 0

import random

def runSVR(y_test, y_train, x_train, x_test, property_name, vectors, classes, wasted_variable3):
    x_train, y_train = dt.balanceClasses(x_train, y_train)
    clf = svm.LinearSVR()
    clf.fit(x_train, y_train)
    direction = clf.coef_.tolist()
    y_pred = clf.predict(x_test)
    y_pred = y_pred.tolist()
    kappa_score = mean_squared_error(y_test, y_pred)
    ktau = 0
    #ppmi_score, ppmi_ratio = get_ppmi_score(y_pred, property_name)
    return kappa_score, direction, ktau, 0, 0
import multiprocessing
def runAllSVMs(y_test, y_train, x_train, x_test, property_names, file_name, svm_type, get_kappa, get_f1, getting_directions, data_type, threads):
    kappa_scores = [0] * len(property_names)
    directions = [None] * len(property_names)
    f1_scores = [0] * len(property_names)
    saved_x_trans = None
    saved_test_x_trans = None
    for y in range(0, len(property_names)):
        """
        pool = multiprocessing.Pool(processes=4)
        kappa = pool.map(runSVM(y_test, y_train, x_train,
                                        x_test, property_names[y], get_kappa,get_f1, data_type), range(threads))

        for t in range(len(kappa)):
            kappa_scores[y+t] = kappa[t][0]
            directions[y+t] = kappa[t][1]
            ktau_scores[y+t] = kappa[t][2]
            print(y, "Score", kappa[t][0],  kappa[t][1], property_names[y])
        """
        kappa, f1, direction,  unused_variable, unused_variable_2 = runSVM(y_test, y_train, x_train,  x_test,
                                                                    property_names[y], get_kappa, get_f1, data_type)

        kappa_scores[y] = kappa
        directions[y] = direction
        f1_scores[y] = f1
        print(y, "Score", kappa, f1, property_names[y])

    return kappa_scores, directions, f1_scores

from sklearn.cross_validation import train_test_split

def getSVMResults(vector_path, class_path, property_names_fn, file_name, svm_type, training_size=10000,  lowest_count=200,
                  highest_count=21470000, get_kappa=True, get_f1=True, single_class=True, data_type="movies",
                  getting_directions=True, threads=1):
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

    directions_fn = "../data/" + data_type + "/svm/directions/" + file_name + ".txt"
    kappa_fn = "../data/" + data_type + "/svm/kappa/" + file_name + ".txt"
    ktau_scores_fn = "../data/" + data_type + "/svm/f1/" + file_name + ".txt"
    ppmI_ratios_fn = "../data/" + data_type + "/svm/ppmi/" + file_name + "ratio.txt"

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

file_name = "films100svmndcg0.9200"
# Get SVM scores
svm_type = "svm"
lowest_count = 200
highest_count = 10000
cluster_amt = 200
split = 0.9
#vector_path = "../data/movies/nnet/spaces/" + file_name + ".txt"#
vector_path = "../data/movies/rank/numeric/"+file_name+".txt"
#class_path = "../data/movies/bow/ppmi/class-all-" + str(lowest_count)
class_path = "../data/movies/classify/genres/class-all"
property_names_fn = "../data/movies/classify/genres/names.txt"
file_name = file_name + "genre"
#getSVMResults(vector_path, class_path, property_names_fn, file_name, lowest_count=lowest_count, highest_count=highest_count, svm_type=svm_type)


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