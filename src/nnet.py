# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import numpy as np
from keras.layers.noise import GaussianNoise
import helper.data as dt
from keras.regularizers import l2, activity_l2
from keras.layers import Input, Dense, Dropout
from keras.optimizers import SGD, Adagrad, Adadelta, Adam, RMSprop
from keras.models import Sequential
from keras.models import model_from_json
from sklearn import preprocessing
import matplotlib.pyplot as plt
import gini
import cluster
import rank
import finetune_outputs as fto
import svm

class NeuralNetwork:

    # The shared model
    end_space = None
    model = None

    # Shared variables
    training_data = None
    class_path = None
    learn_rate = None
    epochs = None
    loss = None
    batch_size = None
    hidden_activation = None
    layer_init = None
    output_activation = None
    hidden_layer_size = None
    file_name = None
    vector_path = None
    optimizer = None
    dropout_noise = None
    reg = 0.0
    activity_reg = 0.0

    def __init__(self, training_data=10000, class_path=None, network_type="ft",  randomize_finetune_weights=False, dropout_noise = None,
                 epochs=1,  learn_rate=0.01, loss="mse", batch_size=1, past_model_bias_fn=None, identity_swap=False, reg=0.0,
                 hidden_activation="tanh", layer_init="glorot_uniform", output_activation="tanh", deep_size = None, corrupt_finetune_weights = False,
                   hidden_layer_size=100, file_name="unspecified_filename", vector_path=None, is_identity=False, activity_reg=0.0,
                 optimizer_name="rmsprop", noise=0.0, fine_tune_weights_fn=None, past_model_weights_fn=None):

        self.model = Sequential()
        self.training_data = training_data
        self.class_path = class_path
        self.learn_rate = learn_rate
        self.epochs = epochs
        self.loss = loss
        self.batch_size = batch_size
        self.hidden_activation = hidden_activation
        self.layer_init = layer_init
        self.output_activation = output_activation
        self.hidden_layer_size = hidden_layer_size
        self.file_name = file_name
        self.vector_path = vector_path
        self.dropout_noise = dropout_noise
        self.reg = reg
        self.activity_reg = activity_reg

        if optimizer_name == "adagrad":
            self.optimizer = Adagrad(lr=learn_rate, epsilon=1e-15)
        else:
            self.optimizer = SGD(lr=learn_rate, momentum=0.0, decay=0.0, nesterov=False)
        file = open(vector_path)

        if network_type == "ft":
            movie_vectors, movie_classes = self.fineTuneNetwork(past_model_weights_fn, past_model_bias_fn,
                                                                fine_tune_weights_fn, is_identity, identity_swap,
                                                                randomize_finetune_weights, corrupt_finetune_weights, deep_size)
        elif network_type == "da":
            movie_vectors, movie_classes = self.denoisingAutoencoder(noise, deep_size)

        # Compile the model and fit it to the data
        self.model.fit(movie_vectors, movie_classes, nb_epoch=self.epochs, batch_size=self.batch_size, verbose=1)

        truncated_model = Sequential()
        total_file_name = "../data/movies/nnet/spaces/" + self.file_name + ".txt"

        truncated_model.add(self.model.layers[0])
        truncated_model.add(self.model.layers[1])
        if is_identity or deep_size or dropout_noise is not None:
            truncated_model.add(self.model.layers[2])
        truncated_model.compile(loss=self.loss, optimizer="sgd")
        self.end_space = truncated_model.predict(movie_vectors)

        for l in range(len(self.model.layers)):
            try:
                dt.write2dArray(self.model.layers[l].get_weights()[0],
                                "../data/movies/nnet/weights/L" + str(l) + file_name + ".txt")
                dt.write1dArray(self.model.layers[l].get_weights()[1],
                                "../data/movies/nnet/bias/L" + str(l) + file_name + ".txt")
            except IndexError:
                print("Layer ", str(l), "Failed")

        dt.write2dArray(self.end_space, total_file_name)


    def fineTuneNetwork(self, past_weights_fn, past_model_bias_fn, fine_tune_weights_fn, is_identity, identity_swap,
                        randomize_finetune_weights, corrupt_finetune_weights, deep_size):
        movie_vectors = np.asarray(dt.import2dArray(self.vector_path))
        movie_classes = np.asarray(dt.import2dArray(self.class_path))
        if len(movie_vectors) != 15000:
            movie_vectors = movie_vectors.transpose()
        input_size = len(movie_vectors[0])
        output_size = len(movie_classes[0])

        past_model_weights = np.asarray(dt.import2dArray(past_weights_fn), dtype="float64")
        past_model_bias = np.asarray(dt.import1dArray(past_model_bias_fn, "f"), dtype="float64")

        past_model_weights = np.around(past_model_weights, decimals=6)
        past_model_bias = np.around(past_model_bias, decimals=6)

        past_weights = []
        past_weights.append(past_model_weights)
        past_weights.append(past_model_bias)

        weights = []
        if fine_tune_weights_fn is not None:
            for f in fine_tune_weights_fn:
                weights.extend(dt.import2dArray(f))

        r = np.asarray(weights, dtype="float64")

        for a in range(len(r)):
            r[a] = np.around(r[a], decimals=6)

        for a in range(len(movie_classes)):
            movie_classes[a] = np.around(movie_classes[a], decimals=6)

        fine_tune_weights = []
        fine_tune_weights.append(r.transpose())
        fine_tune_weights.append(np.empty(shape=output_size, dtype="float64"))

        self.model.add(GaussianNoise(0.0, input_shape=(input_size,)))

        # If we want to swap the identity layer to before the hidden layer
        if identity_swap:
            self.model.add(Dense(output_dim=self.hidden_layer_size, input_dim=self.hidden_layer_size,
                                 activation=self.hidden_activation,
                                 init="identity"))

        if deep_size is not None:
            self.model.add(Dense(output_dim=deep_size, input_dim=self.hidden_layer_size, init=self.layer_init,
                                 activation=self.hidden_activation, W_regularizer=l2(self.reg)))

        self.model.add(Dense(output_dim=self.hidden_layer_size, input_dim=input_size, activation=self.hidden_activation,
                             weights=past_weights, W_regularizer=l2(self.reg)))

        # Add an identity layer that has equal values to the input space to find some more nonlinear relationships
        if is_identity:
            self.model.add(Dense(output_dim=self.hidden_layer_size, input_dim=self.hidden_layer_size, activation=self.hidden_activation,
                      init="identity"))

        if randomize_finetune_weights:
            self.model.add(Dense(output_dim=output_size, input_dim=self.hidden_layer_size, activation=self.output_activation,
                                 init=self.layer_init))
        elif corrupt_finetune_weights:
            self.model.add(Dense(output_dim=output_size, input_dim=self.hidden_layer_size, activation=self.output_activation,
                                 weights=fine_tune_weights))
        else:
            self.model.add(Dense(output_dim=output_size, input_dim=self.hidden_layer_size, activation=self.output_activation,
                                 weights=fine_tune_weights))

        self.model.compile(loss=self.loss, optimizer=self.optimizer)
        self.output_clusters = self.model.predict(movie_vectors)
        dt.write2dArray(self.output_clusters.transpose(), "../data/movies/nnet/clusters/" + self.file_name + ".txt")
        return movie_vectors, movie_classes


    def denoisingAutoencoder(self, noise, deep_size):
        movie_vectors = np.asarray(dt.import2dArray(self.vector_path))
        if len(movie_vectors) != 15000:
            movie_vectors = movie_vectors.transpose()
        if self.class_path is None:
            movie_classes = movie_vectors
        else:
            movie_classes = np.asarray(dt.import2dArray(self.class_path))
        input_size = len(movie_vectors[0])
        output_size = len(movie_classes[0])
        if self.dropout_noise is None:
            self.model.add(GaussianNoise(noise, input_shape=(input_size,)))
        else:
            self.model.add(Dropout(self.dropout_noise[0], input_shape=(input_size,)))
        if deep_size is not None:
            self.model.add(Dense(output_dim=deep_size, input_dim=self.hidden_layer_size, init=self.layer_init,
                                 activation=self.hidden_activation, W_regularizer=l2(self.reg), activity_regularizer=activity_l2(self.activity_reg)))
        self.model.add(Dense(output_dim=self.hidden_layer_size, input_dim=input_size, init=self.layer_init,
                             activation=self.hidden_activation, W_regularizer=l2(self.reg)))
        if self.dropout_noise is not None:
            self.model.add(Dropout(self.dropout_noise[1]))
        self.model.add(Dense(output_dim=output_size, init=self.layer_init, activation=self.output_activation, W_regularizer=l2(self.reg)))
        self.model.compile(loss=self.loss, optimizer=self.optimizer)
        return movie_vectors, movie_classes


    def getEndSpace(self):
        return self.end_space

    def getEncoder(self):
        return self.model.layers[1]



def main():
    hidden_layer_sizes = [100,50,25]
    file_names = []
    for f in range(len(hidden_layer_sizes)):
        #file_names.append("filmsBOWL" + str(f + 1) + "" + str(hidden_layer_sizes[f]))
        file_names.append("filmsPPMIDropoutL"+str(f+1)+""+str(hidden_layer_sizes[f]))
        #file_names.append("films200L" + str(f + 1) + "" + str(hidden_layer_sizes[f]))
    #init_vector_path= "../data/movies/bow/binary/phrases/class-all"
    #init_vector_path = "../data/movies/bow/ppmi/class-all"
    init_vector_path="../data/movies/nnet/spaces/films200.txt"
    # Class and vector inputs
    for i in range(len(file_names)):
        #These are the parameter values
        hidden_layer_size = hidden_layer_sizes[i]

        batch_size = 200
        epochs = 10
        reg = 0.0

        if i == 0:
            epochs = 20
            noise = 0.0
            dropout_noise = None
            hidden_activation = "relu"
            output_activation = "softplus"
            optimizer_name = "adagrad"
            loss = "mse"
            reg = 0.0
            activity_reg = 0.01

            dropout_noise = None
            hidden_activation = "relu"
            output_activation = "softplus"
            optimizer_name = "adagrad"
            loss = "kullback_leibler_divergence"
            reg = 0.0
            activity_reg = 0.0

            """
            # Binary BOW
            dropout_noise = None
            hidden_activation = "relu"
            output_activation = "sigmoid"
            optimizer_name = "adagrad"
            loss = "kullback_leibler_divergence"
            """

            file_name = file_names[i] + "DN" + str(dropout_noise) + hidden_activation + output_activation + optimizer_name + loss

        else:
            noise = 0.5
            dropout_noise = None
            file_name = file_names[i] + "N" +str(noise)
            hidden_activation = "relu"
            output_activation = "softplus"
            optimizer_name = "adagrad"
            loss = "mse"

        """
        noise = 0.5
        dropout_noise = None
        file_name = file_names[i] + "N" + str(noise)
        hidden_activation = "tanh"
        output_activation = "tanh"
        optimizer_name = "sgd"
        epochs = 200
        activity_reg = 0
        loss = "mse"
        """

        class_path = None
        print(file_name)

        #deep_size = 100 - (i * 25)
        deep_size = None
        if deep_size is not None:
            file_name = file_name + "DL" + str(deep_size)
        #NN Setup
        """
        SDA = NeuralNetwork( noise=noise, optimizer_name=optimizer_name, batch_size=batch_size, epochs=epochs, dropout_noise=dropout_noise,
                     vector_path=init_vector_path,  hidden_layer_size=hidden_layer_size, class_path=class_path, reg=reg,
                           hidden_activation=hidden_activation, output_activation=output_activation,
                          file_name=file_name, network_type="da", deep_size=deep_size, activity_reg=activity_reg)
        """
        past_model_weights_fn = "../data/movies/nnet/weights/L1" + file_name + ".txt"
        past_model_bias_fn = "../data/movies/nnet/bias/L1" + file_name + ".txt"
        hidden_space_fn = "../data/movies/nnet/spaces/"+file_name+".txt"
        # Get SVM scores
        lowest_count = 200
        highest_count = 10000
        vector_path = "../data/movies/nnet/spaces/"+file_name+".txt"
        class_path = "../data/movies/bow/binary/phrases/class-all-200"
        property_names_fn = "../data/movies/bow/names/" + str(lowest_count) + ".txt"
        #svm.getSVMResults(vector_path, class_path, property_names_fn, file_name, lowest_count=lowest_count, highest_count=highest_count)

        # Get rankings
        vector_names_fn = "../data/movies/nnet/spaces/filmNames.txt"
        class_names_fn = "../data/movies/bow/names/" + str(lowest_count)+".txt"
        directions_fn = "../data/movies/svm/directions/" + file_name + str(lowest_count)+".txt"

        #rank.getAllPhraseRankings(directions_fn, vector_path, class_names_fn, vector_names_fn, file_name)

        # Get gini scores
        ppmi_fn = "../data/movies/bow/ppmi/class-all-200"
        discrete_labels_fn = "../data/movies/rank/discrete/"+file_name+".txt"
        #gini.getGinis(ppmi_fn, class_names_fn, discrete_labels_fn, file_name)

        # Get clusters
        is_gini = False
        #scores_fn = "../data/movies/gini/"+file_name+".txt"
        amt_high_directions = hidden_layer_size*6
        amt_low_directions = hidden_layer_size*60
        amt_of_clusters = hidden_layer_size*4
        scores_fn = "../data/movies/svm/kappa/"+file_name+"200.txt"
        #cluster.getClusters(directions_fn, scores_fn, class_names_fn, is_gini,  amt_high_directions, amt_low_directions, file_name, amt_of_clusters)

        # Get rankings
        clusters_fn = "../data/movies/cluster/clusters/" +file_name+".txt"
        property_names_fn = "../data/movies/cluster/names/" +file_name+".txt"
        percentage_bin = 1
        #rank.getAllRankings(clusters_fn, vector_path, property_names_fn, vector_names_fn, 0.2, percentage_bin, False, file_name)
        # Get PAV
        rankings_fn = "../data/movies/rank/numeric/"+file_name+".txt"
        property_names_fn = "../data/movies/cluster/names/"+file_name+".txt"
        fto.getPAVNoAverage(property_names_fn, rankings_fn, file_name)

        discrete_labels_fn = "../data/movies/rank/discrete/" + file_name + "P1.txt"
        #pav.getPAV(property_names_fn, discrete_labels_fn, file_name)

        # Use PAV as class vectors
        class_path = "../data/movies/pav/"+file_name+".txt"
        #class_path = "../data/movies/rank/numeric/" + file_name + ".txt"
        fine_tune_weights_fn = ["../data/movies/cluster/clusters/" +file_name+".txt"]
        epochs = 10
        batch_size = 200
        learn_rate = 0.01
        loss = "mse"
        optimizer_name = "adagrad"
        #hidden_activation = "tanh"
        hidden_activation = "relu"
        output_activation = "linear"
        is_identity = False
        identity_swap = False
        randomize_finetune_weights = False
        corrupt_finetune_weights = False
        if randomize_finetune_weights:
            file_name = file_names[i] + "N" +str(noise)+"FTR"
        elif corrupt_finetune_weights:
            file_name = file_names[i] + "N" + str(noise) + "FTC"
        else:
            file_name = file_names[i] + "N" + str(noise) + "FT"

        if is_identity:
            epochs = 20
            file_name = file_name + "IT"
        if identity_swap:
            epochs = 20
            is_identity = False
            file_name = file_name + "ITS"

        SDA = NeuralNetwork( noise=0, fine_tune_weights_fn=fine_tune_weights_fn, optimizer_name=optimizer_name, network_type="ft", past_model_bias_fn=past_model_bias_fn, randomize_finetune_weights=randomize_finetune_weights,
                             vector_path=init_vector_path,  hidden_layer_size=hidden_layer_size, class_path=class_path,  identity_swap=identity_swap,
                                   hidden_activation=hidden_activation, output_activation=output_activation, epochs=epochs, learn_rate=learn_rate, is_identity=is_identity,
                             batch_size=batch_size, past_model_weights_fn = past_model_weights_fn, loss=loss, file_name=file_name)

        init_vector_path = "../data/movies/nnet/spaces/"+file_name+".txt"

        # Get SVM scores
        lowest_count = 200
        highest_count = 10000
        vector_path = "../data/movies/nnet/spaces/"+file_name+".txt"
        class_path = "../data/movies/bow/binary/phrases/class-all-200"
        property_names_fn = "../data/movies/bow/names/" + str(lowest_count) + ".txt"
        svm.getSVMResults(vector_path, class_path, property_names_fn, file_name, lowest_count=lowest_count, highest_count=highest_count)
        """
        """
        # Get rankings
        vector_names_fn = "../data/movies/nnet/spaces/filmNames.txt"
        class_names_fn = "../data/movies/bow/names/" + str(lowest_count)+".txt"
        directions_fn = "../data/movies/svm/directions/" + file_name + str(lowest_count)+".txt"

        rank.getAllPhraseRankings(directions_fn, vector_path, class_names_fn, vector_names_fn, file_name)

        # Get gini scores
        ppmi_fn = "../data/movies/bow/frequency/phrases/class-all-200"
        discrete_labels_fn = "../data/movies/rank/discrete/"+file_name+".txt"
        gini.getGinis(ppmi_fn, class_names_fn, discrete_labels_fn, file_name)

        # Get clusters
        is_gini = False
        amt_high_directions = hidden_layer_size*4
        amt_low_directions = hidden_layer_size*40
        amt_of_clusters = hidden_layer_size*2
        scores_fn = "../data/movies/svm/kappa/"+file_name+"200.txt"
        cluster.getClusters(directions_fn, scores_fn, class_names_fn, is_gini,  amt_high_directions, amt_low_directions, file_name, amt_of_clusters)

        # Get rankings on gini score
        clusters_fn = "../data/movies/cluster/clusters/" +file_name+".txt"
        property_names_fn = "../data/movies/cluster/names/" +file_name+".txt"
        percentage_bin = 1
        rank.getAllRankings(clusters_fn, vector_path, property_names_fn, vector_names_fn, 0.2, percentage_bin, False, file_name)
        """

#path = "../data/movies/"
#getScoreDifferences("../data/movies/bow/names/200.txt", path+"svm/kappa/films100N0.5H75L1FTW150Kappa200.txt", path+"svm/kappa/films100N0.5H75L1ginikappaFTW500200.txt","Kappa VS Gini")
"""
labels = dt.import2dArray("../data/movies/bow/frequency/phrases/class-all")
scaled_labels = []
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
for l in labels:
    scaled_labels.append(min_max_scaler.fit_transform(l))

dt.write2dArray(scaled_labels, "../data/movies/bow/frequency/phrases/class-all-scaled0,1.txt")
"""
if  __name__ =='__main__':main()