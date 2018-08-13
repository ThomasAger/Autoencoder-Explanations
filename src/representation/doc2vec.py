#python example to train doc2vec model (with or without pre-trained word embeddings)

import logging
import os

import gensim.models as g
import numpy as np

import newsgroups
import sentiment
from svm import linearSVMScore, multiClassLinearSVM
from util import proj_util as dt


def doc2Vec(embedding_fn, corpus_fn, vector_size, window_size, min_count, sampling_threshold,
                negative_size, train_epoch, dm, worker_count, train_wv, concatenate_wv, use_hierarchical_softmax):

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    docs = g.doc2vec.TaggedLineDocument(corpus_fn) #
    model = g.Doc2Vec(docs, size=vector_size,window=window_size, iter=train_epoch, min_count=min_count,
                        sample=sampling_threshold,negative=negative_size,
                      workers=worker_count, hs=use_hierarchical_softmax, dm=dm, dbow_words=train_wv, dm_concat=concatenate_wv,
                      pretrained_emb=embedding_fn)
    return model

def main(data_type, vector_size, window_size, min_count, sampling_threshold, negative_size,
                               train_epoch, dm, worker_count, train_wv, concatenate_wv, use_hierarchical_softmax):
    file_name = "Doc2Vec" + " VS" + str(vector_size) + " WS" + str(window_size) + " MC" + str(min_count) + " ST" + str(
        sampling_threshold) + \
                " NS" + str(negative_size) + " TE" + str(train_epoch) + " DM" + str(dm) + " WC" + str(
        worker_count) + "spacy"
    " NS" + str(negative_size) + " TE" + str(train_epoch) + " DM" + str(dm) + " WC" + str(worker_count) + \
    " TW" + str(train_wv) + " CW" + str(concatenate_wv) + " HS" + str(use_hierarchical_softmax)

    corpus_fn = "../data/raw/" + data_type + "/corpus_processed.txt"

    if os.path.exists(corpus_fn) is False:
        x_train = np.load("../data/raw/" + data_type + "/x_train_w.npy")
        x_test = np.load("../data/raw/" + data_type + "/x_test_w.npy")
        corpus = np.concatenate((x_train, x_test), axis=0)
        text_corpus = np.empty(len(corpus), dtype=np.object)
        for i in range(len(corpus)):
            text_corpus[i] = " ".join(corpus[i])
            print(text_corpus[i])
        dt.write1dArray(text_corpus, corpus_fn)

    embedding_fn = "/home/tom/Downloads/glove.6B/glove.6B.300d.txt"

    model_fn = "../data/" + data_type + "/doc2vec/" + file_name + ".bin"
    vector_fn = "../data/" + data_type + "/nnet/spaces/" + file_name + ".npy"
    score_fn = "../data/" + data_type + "/doc2vec/" + file_name + "catacc.score"

    if os.path.exists(model_fn):
        print("Imported model")
        model = g.utils.SaveLoad.load(model_fn)
    elif file_name[:7] == "Doc2Vec":
        model = doc2Vec(embedding_fn, corpus_fn, vector_size, window_size, min_count, sampling_threshold,
                        negative_size, train_epoch, dm, worker_count, train_wv, concatenate_wv, use_hierarchical_softmax)
        model.save(model_fn)

    if os.path.exists(vector_fn) is False:
        vectors = []
        for d in range(len(model.docvecs)):
            vectors.append(model.docvecs[d])
        np.save(vector_fn, vectors)
    else:
        print("Imported vectors")
        vectors = np.load(vector_fn)

    if os.path.exists(score_fn) is False or file_name[:6] != "Doc2Vec":
        print("Getting score")
        if data_type == "sentiment":
            classes = dt.import1dArray("../data/" + data_type + "/classify/" + data_type + "/class-all", "i")
            x_train, y_train, x_test, y_test = sentiment.getSplits(vectors, classes)
            scores = linearSVMScore(x_train, y_train, x_test, y_test)
        else:
            classes = dt.import2dArray("../data/" + data_type + "/classify/" + data_type + "/class-all", "i")
            x_train, y_train, x_test, y_test = newsgroups.getSplits(vectors, classes)
            scores = multiClassLinearSVM(x_train, y_train, x_test, y_test)
        print(scores)
        dt.write1dArray(scores, score_fn)

vector_size = 200
window_size = 15
min_count = 1
sampling_threshold = 1e-5
negative_size = 5
train_epoch = 400
dm = 0
worker_count = 10
train_wv = 1
concatenate_wv = 1
use_hierarchical_softmax = 1
data_type = "newsgroups"

if  __name__ =='__main__':main(data_type, vector_size, window_size, min_count, sampling_threshold, negative_size,
                               train_epoch, dm, worker_count, train_wv, concatenate_wv, use_hierarchical_softmax)
