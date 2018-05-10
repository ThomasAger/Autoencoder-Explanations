from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
import data as dt
import numpy as np
import MovieTasks as mt
import scipy.sparse as sp
# Import the newsgroups


def getSplits(vectors, classes, dev=0.8):
    if len(vectors) > 18846:
        print("This is not the standard size of 20NG, expected 18846")
        return False
    x_train = vectors[:11314]
    y_train = classes[:11314]
    x_test = vectors[11314:]
    y_test = classes[11314:]
    print("Returning development splits", dev)
    x_dev = x_train[int(len(x_train) * 0.8):]
    y_dev = y_train[int(len(y_train) * 0.8):]
    x_train = x_train[:int(len(x_train) * 0.8)]
    y_train = y_train[:int(len(y_train) * 0.8)]
    print(len(x_test), len(x_test[0]), "x_test")
    print(len(y_test), len(y_test[0]), "y_test")
    print(len(x_dev), len(x_dev[0]), "x_dev")
    print(len(y_dev), len(y_dev[0]), "y_dev")
    print(len(x_train), len(x_train[0]), "x_train")
    print(len(y_train), len(y_train[0]), "y_train")
    return x_train, y_train, x_test, y_test, x_dev, y_dev
def regularNewsgroupsStuff(): # Rename later
    classification = "all"
    highest_amt = 18836

    lowest_amt = 30
    all_fn = "../data/newsgroups/bow/frequency/phrases/class-all-"+str(lowest_amt)+"-"+str(highest_amt)+"-" + classification
    #newsgroups_train = fetch_20newsgroups(subset='train', shuffle=False, remove=("headers", "footers", "quotes"))
    #newsgroups_test = fetch_20newsgroups(subset='test', shuffle=False, remove=("headers", "footers", "quotes"))

    all = fetch_20newsgroups(subset='all', shuffle=False, remove=("headers", "footers", "quotes"))
    
    train_len = len(all.data)
    
    print(all.target[train_len-1])
    print(all.target[train_len-2])
    print(all.target[train_len-3])
    print(all.target[0])
    print(all.target[1])
    print(all.target[2])
    
    
    vectors = all.data
    classes = all.target
    
    ac_x_train = vectors[:11314]
    ac_x_test = vectors[11314:]
    ac_y_train = classes[:11314]
    ac_y_test = classes[11314:]
    
    print(classes[train_len-1])
    print(classes[train_len-2])
    print(classes[train_len-3])
    
    tf_vectorizer = CountVectorizer(max_df=highest_amt, min_df=lowest_amt, stop_words='english')
    print("completed vectorizer")
    tf = tf_vectorizer.fit(vectors)
    feature_names = tf.get_feature_names()
    dt.write1dArray(feature_names, "../data/newsgroups/bow/names/" + str(lowest_amt) + "-" + str(highest_amt) + "-" + classification + ".txt")
    dict = tf.vocabulary_
    tf = tf_vectorizer.transform(vectors)
    dense = FunctionTransformer(lambda x: x.todense(), accept_sparse=True)
    tf = dense.fit_transform(tf)
    tf = np.squeeze(np.asarray(tf))
    tf = np.asarray(tf, dtype=np.int32)
    tf = tf.transpose()
    freqs = []
    for t in tf:
        freq = 0
        for i in range(len(t)):
            if t[i] != 0:
                freq += t[i]
        freqs.append(freq)
    print("Amount of terms:", len(tf))
    dt.write1dArray(freqs, "../data/newsgroups/bow/freq_count/"+str(lowest_amt)+"-"+str(highest_amt))
    #dt.write2dArray(tf, all_fn)
    #mt.printIndividualFromAll("newsgroups",  "frequency/phrases", lowest_amt, highest_amt, classification, all_fn=all_fn, names_array=feature_names)
    ppmi_fn = "../data/newsgroups/bow/ppmi/class-all-"+str(lowest_amt)+"-"+str(highest_amt)+"-" + classification
    #if dt.fileExists(ppmi_fn) is False:
    tf = sp.csr_matrix(tf)
    sp.save_npz(all_fn, tf)
    ppmi = mt.convertPPMI( tf)
    #dt.write2dArray(ppmi, ppmi_fn)
    ppmi_sparse = sp.csr_matrix(ppmi)
    sp.save_npz(ppmi_fn, ppmi_sparse)
    mt.printIndividualFromAll("newsgroups",  "ppmi", lowest_amt, highest_amt, classification, all_fn=all_fn, names_array=feature_names)
    
    print("1")
    classes = np.asarray(classes, dtype=np.int32)
    print(2)
    classes_dense = np.zeros(shape=(len(classes), np.amax(classes)+1 ), dtype=np.int8)
    print(3)
    for c in range(len(classes)):
        classes_dense[c][classes[c]] = 1
    print(4)
    names = list(all.target_names)
    dt.write1dArray(names, "../data/newsgroups/classify/newsgroups/names.txt")
    classes_dense = classes_dense.transpose()
    for c in range(len(classes_dense)):
        dt.write1dArray(classes_dense[c], "../data/newsgroups/classify/newsgroups/class-" + names[c])
    classes_dense = classes_dense.transpose()
    
    dt.write2dArray(classes_dense,"../data/newsgroups/classify/newsgroups/class-all")

    feature_names = dt.import1dArray("../data/newsgroups/bow/names/"+str(lowest_amt)+"-"+str(highest_amt)+"-all.txt")

    freq = dt.import2dArray(all_fn)

    binary = np.zeros(shape=(len(freq), len(freq[0])))
    for i in range(len(freq)):
        for j in range(len(freq[i])):
            if freq[i][j] > 0:
                binary[i][j] = 1
    binary_all_fn = "../data/newsgroups/bow/binary/phrases/class-all-"+str(lowest_amt)+"-"+str(highest_amt)+"-" + classification
    binary = sp.csr_matrix(binary)
    sp.save_npz(binary_all_fn, binary)
    #dt.write2dArray(binary, binary_all_fn)

    #mt.printIndividualFromAll("newsgroups",  "binary/phrases", lowest_amt, highest_amt, classification, all_fn=all_fn, names_array=feature_names)
    #ppmi_fn = "../data/newsgroups/bow/ppmi/class-all-"+str(lowest_amt)+"-"+str(highest_amt)+"-" + classification

#regularNewsgroupsStuff()