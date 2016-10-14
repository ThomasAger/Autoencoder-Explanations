
from sklearn import svm
import numpy as np
from sklearn.metrics import cohen_kappa_score
import helper.data as dt
from scipy import  linalg



def runRankSVM(y_test, y_train, x_train, x_test, class_type, input_size, property_names, keyword):
    clf = svm.SVC(kernel='linear', C=.1)
    clf.fit(x_train[keyword], y_train[keyword])
    direction = clf.coef_
    return direction

def get_ppmi_score(y_pred, property_name):
    term_frequency = dt.import1dArray("../data/movies/bow/frequency/phrases/"+property_name)
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

def runGaussianSVM(y_test, y_train, x_train, x_test, property_name):
    clf = svm.SVC(kernel='rbf', class_weight='auto')
    clf.fit(x_train, y_train)
    direction = clf.dual_coef_.tolist()[0]
    y_pred = clf.predict(x_test)
    y_pred = y_pred.tolist()
    kappa_score = cohen_kappa_score(y_test, y_pred)
    #ppmi_score, ppmi_ratio = get_ppmi_score(y_pred, property_name)
    return kappa_score, direction, 0, 0

def runSVM(y_test, y_train, x_train, x_test, property_name):
    clf = svm.LinearSVC(class_weight='auto')
    clf.fit(x_train, y_train)
    direction = clf.coef_.tolist()[0]
    y_pred = clf.predict(x_test)
    y_pred = y_pred.tolist()
    kappa_score = cohen_kappa_score(y_test, y_pred)
    #ppmi_score, ppmi_ratio = get_ppmi_score(y_pred, property_name)
    return kappa_score, direction, 0, 0

def runAllSVMs(y_test, y_train, x_train, x_test, property_names):
    kappa_scores = []
    directions = []
    ppmi_scores = []
    ppmi_ratios = []
    for y in range(len(y_train)):
        kappa, direction, ppmi_score, ppmi_ratio = runSVM(y_test[y], y_train[y], x_train, x_test, property_names[y])
        kappa_scores.append(kappa)
        directions.append(direction)
        ppmi_scores.append(ppmi_score)
        ppmi_ratios.append(ppmi_ratio)
        print(y, "kappa", kappa,  ppmi_score, ppmi_ratio, property_names[y])

    return kappa_scores, directions, ppmi_scores, ppmi_ratios


def getSVMResults(vector_path, class_path, property_names_fn, file_name, training_size=10000,  lowest_count=200, highest_count=21470000):
    vectors = dt.import2dArray(vector_path)
    classes = dt.import2dArray(class_path)
    property_names = dt.import1dArray(property_names_fn)

    x_train = np.asarray(vectors[:training_size])
    x_test = np.asarray(vectors[training_size:])

    #property_names, classes = getSampledData(property_names, classes, lowest_count, highest_count)

    classes = np.asarray(classes)

    if len(classes) != len(vectors):
        classes = classes.transpose()

    y_train = np.asarray(classes[:training_size])
    y_test = np.asarray(classes[training_size:])

    y_train = y_train.transpose()
    y_test = y_test.transpose()

    kappa_scores, directions, ppmi_scores, ppmi_ratios = runAllSVMs(y_test, y_train, x_train, x_test, property_names)

    directions_fn = "../data/movies/svm/directions/" + file_name + str(lowest_count) + ".txt"
    kappa_fn = "../data/movies/svm/kappa/" + file_name + str(lowest_count) + ".txt"
    ppmi_scores_fn = "../data/movies/svm/ppmi/" + file_name + str(lowest_count) + ".txt"
    ppmI_ratios_fn = "../data/movies/svm/ppmi/" + file_name + str(lowest_count) + "ratio.txt"

    dt.write1dArray(kappa_scores, kappa_fn)
    dt.write2dArray(directions, directions_fn)
    dt.write1dArray(ppmi_scores, ppmi_scores_fn)
    dt.write1dArray(ppmi_ratios, ppmI_ratios_fn)

class SVM:
    def __init__(self, file_name="", class_names=None, vector_path=None, class_path=None, class_by_class=True, input_size=200,
                 training_size=10000, amount_of_scores=400,  low_kappa=0.1, high_kappa=0.5, rankSVM=False, lowest_count=100, largest_count=21470000):
            getSVMResults(vector_path, class_path, file_name)


def main(vectors_fn, classes_fn, property_names, training_size, file_name, lowest_count, largest_count):
    SVM(vectors_fn, classes_fn, property_names, lowest_count=lowest_count,
        training_size=training_size, file_name=file_name, largest_count=largest_count)
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
"""


property_names_fn = "../data/movies/classify/keywords/names.txt"
class_path = "../data/movies/classify/keywords/class-All"
file_name = "filmsPPMIDropoutL1100DNNonerelusoftplusadagradkullback_leibler_divergence"
vector_path = "../data/movies/nnet/spaces/"+file_name+".txt"
file_name = "films100"
vector_path = "../data/movies/nnet/spaces/"+file_name+".txt"
getSVMResults(vector_path, class_path, property_names_fn, "LinearGenre"+file_name)
getSVMResults(vector_path, class_path, property_names_fn, "LinearGenre"+file_name)


"""
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