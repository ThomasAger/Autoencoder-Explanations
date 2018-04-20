#python example to train doc2vec model (with or without pre-trained word embeddings)

import gensim.models as g
import logging
import numpy as np
import os
import data as dt
from svm import LinearSVMScore
from newsgroups import getSplits
from sklearn.datasets import fetch_20newsgroups

def doc2Vec(embedding_fn, corpus_fn, vector_size, window_size, min_count, sampling_threshold,
                negative_size, train_epoch, dm, worker_count):

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    docs = g.doc2vec.TaggedLineDocument(corpus_fn)
    model = g.Doc2Vec(docs, size=vector_size, window=window_size, min_count=min_count, sample=sampling_threshold,
                      workers=worker_count, hs=0, dm=dm, negative=negative_size, dbow_words=1, dm_concat=1,
                      pretrained_emb=embedding_fn, iter=train_epoch)
    vectors = []
    i =0
    for d in range(len(model.docvecs)):
        vectors.append(model.docvecs[d])
    return vectors, model

#doc2vec parameters
vector_size = 300
window_size = 15
min_count = 1
sampling_threshold = 1e-5
negative_size = 5
train_epoch = 400
dm = 0
worker_count = 1


data_type = "newsgroups"
file_name = "0-0Doc2Vec"
corpus_fn = "../data/raw/"+data_type+"/corpus_processed.txt"

if os.path.exists(corpus_fn) is False:
    x_train = np.load("../data/raw/"+data_type+"/x_train_w.npy")
    x_test = np.load("../data/raw/"+data_type+"/x_test_w.npy")
    corpus = np.concatenate((x_train, x_test), axis=0)
    text_corpus = np.empty(len(corpus), dtype=np.unicode_)
    for i in range(len(corpus)):
        text_corpus[i] = " ".join(corpus[i])
    dt.write1dArray(text_corpus, corpus_fn)

embedding_fn = "/home/tom/Downloads/glove.6B/glove.6B.300d.txt"

vectors, model = doc2Vec(embedding_fn, corpus_fn, vector_size, window_size, min_count, sampling_threshold,
                negative_size, train_epoch, dm, worker_count)

classes = dt.import1dArray("../data/" + data_type + "/classify/" + data_type + "/class-all")

x_train, y_train, x_test, y_test = getSplits(vectors, classes)

scores = LinearSVMScore(x_train, y_train, x_test, y_test)

print(scores)

vector_fn = "../data/" + data_type + "/nnet/spaces/" + file_name + "doc2vec.npy"
model_fn = "../data/"+data_type+"/doc2vec/"+file_name+".bin"
score_fn = "../data/"+data_type+"/doc2vec/"+file_name+".score"

dt.write1dArray(scores, score_fn)
np.save(vector_fn, vectors)
model.save(model_fn)