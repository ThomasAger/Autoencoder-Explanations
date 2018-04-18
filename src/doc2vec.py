#python example to train doc2vec model (with or without pre-trained word embeddings)

import gensim.models as g
import logging
import numpy as np

def doc2Vec(embedding_fn, corpus_fn, vector_size, window_size, min_count, sampling_threshold,
                negative_size, train_epoch, dm, worker_count):

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    docs = np.load(corpus_fn)
    model = g.Doc2Vec(docs, size=vector_size, window=window_size, min_count=min_count, sample=sampling_threshold,
                      workers=worker_count, hs=0, dm=dm, negative=negative_size, dbow_words=1, dm_concat=1,
                      pretrained_emb=embedding_fn, iter=train_epoch)
    vectors = [model.infer_vector(sent) for sent in docs]
    return vectors, model

#doc2vec parameters
vector_size = 300
window_size = 15
min_count = 1
sampling_threshold = 1e-5
negative_size = 5
train_epoch = 100
dm = 0
worker_count = 1


data_type = "sentiment"
file_name = "0-0Doc2Vec"
corpus_fn = "../data/raw/sentiment/corpus.npy"

embedding_fn = "/home/tom/Downloads/glove.6B/glove.6B.300d.txt"

vectors, model = doc2Vec(embedding_fn, corpus_fn, vector_size, window_size, min_count, sampling_threshold,
                negative_size, train_epoch, dm, worker_count)

vector_fn = "../data/" + data_type + "/nnet/vectors/" + file_name + "doc2vec.npy"
model_fn = "../data/"+data_type+"/doc2vec/"+file_name+".bin"

np.save(vector_fn, vectors)
model.save(model_fn)