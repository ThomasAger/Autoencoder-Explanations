'''This example demonstrates the use of fasttext for text classification
Based on Joulin et al's paper:
Bags of Tricks for Efficient Text Classification
https://arxiv.org/abs/1607.01759
Results on IMDB datasets with uni and bi-gram embeddings:
    Uni-gram: 0.8813 test accuracy after 5 epochs. 8s/epoch on i7 cpu.
    Bi-gram : 0.9056 test accuracy after 5 epochs. 2s/epoch on GTx 980M gpu.
'''

from __future__ import print_function

import os

import numpy as np
from keras.callbacks import TensorBoard
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.metrics import categorical_accuracy
from keras.models import Model
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing import sequence

import newsgroups
from util import nnet_util

data_type = "newsgroups"
# Set parameters:
# ngram_range = 2 will add bi-grams features
if data_type == "sentiment":
    ngram_range = 1
    max_features = 20000
    maxlen = 400
    batch_size = 32
    embedding_dims = 50
    epochs = 20
    top_n = 5
    test = False
    use_pretrained_vectors = False
else:
    ngram_range = 1
    max_features = 20000
    maxlen = 200
    batch_size = 32
    embedding_dims = 300
    epochs = 20
    top_n = 5
    test = False
    use_pretrained_vectors = True

embedding_matrix = None
if use_pretrained_vectors:
    matrix_fn =  "../../Interpretable LSTM/data/"+data_type+"/matrix/google_news_no_bigram_IMDB_words.npy"
    embedding_matrix = np.load(matrix_fn)
    max_features = len(embedding_matrix)
    embedding_matrix = [embedding_matrix]
    embedding_dims = 300

corpus_type = "simple_numeric_stopwords" # simple_stopwords_corpus
gram = "" # " 2-gram", " 3-gram"

file_name = "fastText " + corpus_type + gram + " E" + str(embedding_dims) + " ML" + str(maxlen) + " MF" + str(max_features) + " E" + str(epochs) + " NG" + str(ngram_range) + " PRE" + str(use_pretrained_vectors)

print(file_name)

folder_name = "../data/raw/" + data_type + "/"


if data_type == "sentiment":
    file_name = folder_name + "MF " + str(max_features) + " NG" + str(ngram_range) + " ML" + str(
        maxlen) + " Test" + str(test)
    x_train_fn = file_name + " x_train.npy"
    x_test_fn = file_name + " x_train.npy"
    x_dev_fn = file_name + " x_train.npy"
    y_train_fn = file_name + " x_train.npy"
    y_test_fn = file_name + " x_train.npy"
    y_dev_fn = file_name + " x_train.npy"
    if os.path.exists(x_train_fn) is False:
        x_train, x_test, y_train, y_test = nnet_util.getIMDBSequences(max_features, ngram_range, maxlen)
        np.save(x_train_fn, x_train)
        np.save(y_train_fn, y_train)
        np.save(x_test_fn, x_test)
        np.save(y_test_fn, y_test)
    else:
        x_train = np.load(x_train_fn)
        x_test = np.load(x_test_fn)
        y_train = np.load(y_train_fn)
        y_test = np.load(y_test_fn)
    output_size = 1
    output_activation = "sigmoid"
    metric = 'accuracy'
    loss = 'binary_crossentropy'
else:
    print("Loaded corpus", corpus_type + "_tokenized_corpus" + gram + ".npy")
    print("Loaded classes", corpus_type + "_classes_categorical.npy")
    corpus = np.load(folder_name + corpus_type + "_tokenized_corpus" + gram + ".npy")
    classes = np.load(folder_name + corpus_type + "_classes_categorical.npy")
    corpus = sequence.pad_sequences(corpus, maxlen=maxlen)
    x_train, y_train, x_test, y_test, x_dev, y_dev = newsgroups.getSplits(corpus, classes)
    output_size = len(y_test[0])
    output_activation = "softmax"
    metric = categorical_accuracy
    loss = 'categorical_crossentropy'

if test:
    x_train = x_train[:100]
    x_test = x_test[:100]
    y_train = y_train[:100]
    y_test = y_test[:100]

model_fn = "../data/"+data_type+"/fastText/model/" + file_name + ".model"
score_fn = "../data/"+data_type+"/fastText/score/" + file_name + ".txt"

if os.path.exists(model_fn) is False:

    tensorboard = TensorBoard(log_dir='/home/tom/Desktop/Logs/'+str(data_type)+"/"+file_name+'/', histogram_freq=0,
                                  write_graph=True, write_images=True)

    model = Sequential()


    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(max_features, embedding_dims, weights=embedding_matrix, input_length=maxlen, trainable=True))

    # we add a GlobalAveragePooling1D, which will average the embeddings
    # of all words in the document
    model.add(GlobalAveragePooling1D())

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(output_size, activation=output_activation))

    model.compile(loss=loss,
                  optimizer='adam',
                  metrics=[metric])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_dev, y_dev), callbacks=[tensorboard])
    scores = model.evaluate(x_dev, y_dev, batch_size=batch_size)
    model.save(model_fn)
    np.savetxt(score_fn, scores)
    print(scores)

vector_path = "../data/"+data_type+"/fastText/vectors/" + file_name + ".npy"
if os.path.exists(vector_path) is False:
    model = load_model(model_fn)
    target_layer = model.layers[-2]
    outputs = target_layer(target_layer.input)
    m = Model(model.input, outputs)
    hidden_state = m.predict(np.concatenate((x_train, x_dev, x_test), axis=0))
    np.save(vector_path, hidden_state)
