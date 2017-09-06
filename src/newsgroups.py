from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import FunctionTransformer
import data as dt
import numpy as np
import MovieTasks as mt
import scipy.sparse as sp
# Import the newsgroups

newsgroups_train = fetch_20newsgroups(subset='train', shuffle=False)
newsgroups_test = fetch_20newsgroups(subset='test', shuffle=False)

train_len = len(newsgroups_train.data)
test_len = len(newsgroups_test.data)

print(newsgroups_train.target[train_len-1])
print(newsgroups_train.target[train_len-2])
print(newsgroups_train.target[train_len-3])
print(newsgroups_test.target[0])
print(newsgroups_test.target[1])
print(newsgroups_test.target[2])


vectors = np.concatenate((newsgroups_train.data, newsgroups_test.data), axis=0)
classes = np.concatenate((newsgroups_train.target, newsgroups_test.target), axis=0)

print(classes[train_len-1])
print(classes[train_len-2])
print(classes[train_len-3])
print(classes[train_len])
print(classes[train_len+1])
print(classes[train_len+2])

classification = "all"
lowest_amt = 50
highest_amt = 0.95
all_fn = "../data/newsgroups/bow/frequency/phrases/class-all-"+str(lowest_amt)+"-"+str(highest_amt)+"-" + classification
if dt.fileExists(all_fn):
    tf = dt.import2dArray(all_fn)
else:
    tf_vectorizer = CountVectorizer(max_df=highest_amt, min_df=lowest_amt, stop_words='english')
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
    dt.write2dArray(tf, all_fn)
    mt.printIndividualFromAll("newsgroups",  "frequency/phrases", lowest_amt, 0.95, classification, all_fn=all_fn, names_array=feature_names)
tf = sp.csr_matrix(tf)
dt.write2dArray(mt.convertPPMI( tf), "../data/newsgroups/bow/ppmi/class-all-"+str(lowest_amt)+"-"+str(highest_amt)+"-" + classification)
mt.printIndividualFromAll("newsgroups",  "ppmi", lowest_amt, 0.95, classification, all_fn=all_fn, names_array=feature_names)

# Convert them to vectors





# Convert vectors to our format




# Convert classes to our format




