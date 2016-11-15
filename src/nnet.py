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
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
import gini
import cluster
import rank
import finetune_outputs as fto
import svm
import tree
import hierarchy
import ndcg

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
                 epochs=1,  learn_rate=0.01, loss="mse", batch_size=1, past_model_bias_fn=None, identity_swap=False, reg=0.0, amount_of_finetune=1,
                 hidden_activation="tanh", layer_init="glorot_uniform", output_activation="tanh", deep_size = None, corrupt_finetune_weights = False,
                   hidden_layer_size=100, file_name="unspecified_filename", vector_path=None, is_identity=False, activity_reg=0.0,
                 optimizer_name="rmsprop", noise=0.0, fine_tune_weights_fn=None, past_model_weights_fn=None, from_ae=True):

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
        self.amount_of_finetune = amount_of_finetune

        if optimizer_name == "adagrad":
            self.optimizer = Adagrad()
        else:
            self.optimizer = SGD(lr=learn_rate, momentum=0.0, decay=0.0, nesterov=False)

        if network_type == "ft":
            movie_vectors, movie_classes = self.fineTuneNetwork(past_model_weights_fn, past_model_bias_fn,
                                                                fine_tune_weights_fn, is_identity, identity_swap,
                                                                randomize_finetune_weights, corrupt_finetune_weights,
                                                                deep_size, from_ae)
        elif network_type == "da":
            movie_vectors, movie_classes = self.denoisingAutoencoder(noise, deep_size)

        x_train, x_test, y_train, y_test = train_test_split( movie_vectors, movie_classes, test_size = 0.3, random_state = 0)


        # Compile the model and fit it to the data
        self.model.fit(x_train, y_train, nb_epoch=self.epochs, batch_size=self.batch_size, verbose=1)


        if network_type == "ft":
            scores = []
            y_pred = self.model.predict(x_test)
            y_pred[y_pred >= 0.5] = 1
            y_pred[y_pred < 0.5] = 0
            f1 = f1_score(y_test, y_pred, average="macro")

            accuracy_array = []
            for y in range(len(y_pred)):
                accuracy_array.append(accuracy_score(y_test[y], y_pred[y]))
            accuracy = np.mean(accuracy_array)

            scores.append(f1)
            scores.append(accuracy)
            dt.write1dArray(scores, "../data/movies/nnet/scores/" + self.file_name + ".txt")
            print(scores)

            self.output_clusters = self.model.predict(movie_vectors)
            dt.write2dArray(self.output_clusters.transpose(), "../data/movies/nnet/clusters/" + self.file_name + ".txt")

        total_file_name = "../data/movies/nnet/spaces/" + self.file_name
        for l in range(1, len(self.model.layers) - 1):
            truncated_model = Sequential()
            for a in range(l+1):
                truncated_model.add(self.model.layers[a])
            truncated_model.compile(loss=self.loss, optimizer="sgd")
            self.end_space = truncated_model.predict(movie_vectors)
            dt.write2dArray(self.end_space, total_file_name + "L" + str(l) + ".txt")

        for l in range(len(self.model.layers)):
            try:
                dt.write2dArray(self.model.layers[l].get_weights()[0],
                                "../data/movies/nnet/weights/L" + str(l) + file_name + ".txt")
                dt.write1dArray(self.model.layers[l].get_weights()[1],
                                "../data/movies/nnet/bias/L" + str(l) + file_name + ".txt")
            except IndexError:
                print("Layer ", str(l), "Failed")



    def fineTuneNetwork(self, past_weights_fn, past_model_bias_fn, fine_tune_weights_fn, is_identity, identity_swap,
                        randomize_finetune_weights, corrupt_finetune_weights, deep_size, from_ae):
        movie_vectors = np.asarray(dt.import2dArray(self.vector_path))
        movie_classes = np.asarray(dt.import2dArray(self.class_path))

        if len(movie_classes) != 15000:
            movie_classes = movie_classes.transpose()
        if len(movie_vectors) != 15000:
            movie_vectors = movie_vectors.transpose()
        input_size = len(movie_vectors[0])
        output_size = len(movie_classes[0])
        print(input_size, output_size)

        past_weights = []

        if from_ae:
            past_model_weights = []
            for p in past_weights_fn:
                past_model_weights.append(np.asarray(dt.import2dArray(p), dtype="float64"))
            past_model_bias = []
            for p in past_model_bias_fn:
                past_model_bias.append(np.asarray(dt.import1dArray(p, "f"), dtype="float64"))

            for p in range(len(past_model_weights)):
                past_model_weights[p] = np.around(past_model_weights[p], decimals=6)
                past_model_bias[p] = np.around(past_model_bias[p], decimals=6)

            for p in range(len(past_model_weights)):
                past_weights.append([])
                past_weights[p].append(past_model_weights[p])
                past_weights[p].append(past_model_bias[p])

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
            print("Identity swapped layer")
            self.model.add(Dense(output_dim=self.hidden_layer_size, input_dim=input_size,
                                 activation=self.hidden_activation,
                                 init="identity"))

        if deep_size is not None:
            print("Deep layer")
            self.model.add(Dense(output_dim=deep_size, input_dim=self.hidden_layer_size, init=self.layer_init,
                                 activation=self.hidden_activation, W_regularizer=l2(self.reg)))
        if from_ae:
            for p in past_weights:
                print("Past AE layer")
                self.model.add(Dense(output_dim=self.hidden_layer_size, input_dim=input_size, activation=self.hidden_activation,
                                     weights=p, W_regularizer=l2(self.reg)))
            if self.dropout_noise is not None:
                print("Dropout layer")
                self.model.add(Dropout(self.dropout_noise))

        # Add an identity layer that has equal values to the input space to find some more nonlinear relationships
        if is_identity:
            print("Identity layer")
            for a in range(self.amount_of_finetune):
                self.model.add(Dense(output_dim=self.hidden_layer_size, input_dim=self.hidden_layer_size, activation=self.hidden_activation,
                          init="identity"))

        if randomize_finetune_weights:
            print("Randomize finetune weights")
            self.model.add(Dense(output_dim=output_size, input_dim=self.hidden_layer_size, activation=self.output_activation,
                                 init=self.layer_init))
        elif corrupt_finetune_weights:
            print("Corrupt finetune weights")
            self.model.add(Dense(output_dim=output_size, input_dim=self.hidden_layer_size, activation=self.output_activation,
                                 weights=fine_tune_weights))
        else:
            print("Fine tune weights")
            self.model.add(Dense(output_dim=output_size, input_dim=self.hidden_layer_size, activation=self.output_activation,
                                 weights=fine_tune_weights))

        print("Compiling")
        self.model.compile(loss=self.loss, optimizer=self.optimizer, class_mode="binary")

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
        self.model.add(Dense(output_dim=output_size, init=self.layer_init, activation=self.output_activation, W_regularizer=l2(self.reg)))
        self.model.compile(loss=self.loss, optimizer=self.optimizer)
        return movie_vectors, movie_classes


    def getEndSpace(self):
        return self.end_space

    def getEncoder(self):
        return self.model.layers[1]



def main():
    hidden_layer_sizes = [100,100,100,100,100,100]
    file_names = []
    for f in range(len(hidden_layer_sizes)):
        #file_names.append("filmsBOWL" + str(f + 1) + "" + str(hidden_layer_sizes[f]))
        #file_names.append("filmsPPMIDropoutL"+str(f+1)+""+str(hidden_layer_sizes[f]))
        file_names.append("films100L" + str(f + 1) + "" + str(hidden_layer_sizes[f]))
    #init_vector_path= "../data/movies/bow/binary/phrases/class-all"
    #init_vector_path = "../data/movies/bow/ppmi/class-all"
    #init_vector_path="../data/movies/nnet/spaces/films200L1100N0.5pavPPMIN0.5FTadagradcategorical_crossentropy100.txt"
    init_vector_path = "../data/movies/nnet/spaces/films100.txt"
    end_file_names = []

    # Class and vector inputs
    for i in range(len(file_names)):
        #These are the parameter values
        hidden_layer_size = hidden_layer_sizes[i]
        batch_size = 200
        reg = 0.0
        noise = 0.5
        dropout_noise = None
        file_name = file_names[i] + "N" + str(noise)
        hidden_activation = "tanh"
        output_activation = "tanh"
        optimizer_name = "sgd"
        learn_rate = 0.01
        epochs = 500
        activity_reg = 0
        loss = "mse"
        class_path = None
        print(file_name)
        #deep_size = hidden_layer_sizes[i]
        deep_size = None
        if deep_size is not None:
            file_name = file_name + "DL" + str(deep_size)
        #NN Setup
        """
        SDA = NeuralNetwork( noise=noise, optimizer_name=optimizer_name, batch_size=batch_size, epochs=epochs, dropout_noise=dropout_noise,
                         vector_path=init_vector_path,  hidden_layer_size=hidden_layer_size, class_path=class_path, reg=reg,
                               hidden_activation=hidden_activation, output_activation=output_activation, learn_rate=learn_rate,
                              file_name=file_name, network_type="da", deep_size=deep_size, activity_reg=activity_reg)
        """
        past_model_weights_fn = ["../data/movies/nnet/weights/L1" + file_name + ".txt"]
        past_model_bias_fn = ["../data/movies/nnet/bias/L1" + file_name + ".txt"]
        hidden_space_fn = "../data/movies/nnet/spaces/"+file_name+".txt"
        file_name = "films100ppmi"
        # Get SVM scores
        lowest_count = 200
        highest_count = 10000
        vector_path = "../data/movies/nnet/spaces/"+file_name+"L1.txt"
        class_path = "../data/movies/bow/binary/phrases/class-all-" +str(lowest_count)
        property_names_fn = "../data/movies/bow/names/" + str(lowest_count) + ".txt"
        #svm.getSVMResults(vector_path, class_path, property_names_fn, file_name, lowest_count=lowest_count, highest_count=highest_count)

        # Get rankings
        vector_names_fn = "../data/movies/nnet/spaces/filmNames.txt"
        class_names_fn = "../data/movies/bow/names/" + str(lowest_count)+".txt"
        directions_fn = "../data/movies/svm/directions/" + file_name + str(lowest_count)+".txt"
        #rank.getAllPhraseRankings(directions_fn, vector_path, class_names_fn, vector_names_fn, file_name)
        #file_name = file_name + "ndcg"
        #ndcg.getNDCG("../data/movies/rank/numeric/"+file_name+"ALL.txt",file_name)

        # Get clusters
        scores_fn = "../data/movies/ndcg/"+file_name+".txt"
        top_directions = 1500
        amt_high_directions = 1000
        amt_low_directions = 2500
        amt_of_clusters = 400
        #cluster.getClusters(directions_fn, scores_fn, class_names_fn, False,  amt_high_directions, amt_low_directions, file_name, amt_of_clusters)
        vector_fn = "../data/movies/nnet/spaces/" + file_name + "L1.txt"
        directions_fn = "../data/movies/svm/directions/" + file_name + "200.txt"
        #scores_fn = "../data/movies/svm/kappa/" + file_name + "200.txt"
        names_fn = "../data/movies/bow/names/200.txt"
        similarity_threshold = 0.6
        cluster_amt = 400
        amount_to_start = 1000
        score_limit = 0.6
        #file_name = file_name + "kappa"
        print(file_name)
        hierarchy.initClustering(vector_fn, directions_fn, scores_fn, names_fn, amount_to_start, False, similarity_threshold,  cluster_amt, score_limit, file_name)
        # Get rankings
        clusters_fn = "../data/movies/cluster/hierarchy_directions/" + file_name + str(score_limit) + str(cluster_amt) + ".txt"
        property_names_fn = "../data/movies/cluster/hierarchy_names/" + file_name + str(score_limit) + str(cluster_amt) + ".txt"
        vector_names_fn = "../data/movies/nnet/spaces/filmnames.txt"
        file_name = file_name + str(cluster_amt)
        #rank.getAllRankings(clusters_fn, vector_path, property_names_fn, vector_names_fn, 0.2, 1, False, file_name, False)
        # Get PAV
        ranking_fn = "../data/movies/rank/numeric/"+file_name+".txt"

        #fto.pavPPMI(property_names_fn, ranking_fn, file_name)

        #fto.pavTermFrequency(ranking_fn, cluster_names_fn, file_name, False)
        #fto.binaryClusterTerm(cluster_names_fn, file_name)
        #fto.binaryInCluster(property_names_fn, file_name)
        discrete_labels_fn = "../data/movies/rank/discrete/" + file_name + "P1.txt"

        # Use PAV as class vectors
        fine_tune_weights_fn = [clusters_fn]
        epochs = 100
        batch_size = 200
        learn_rate = 0.01
        is_identity = True
        identity_swap = False
        randomize_finetune_weights = False
        corrupt_finetune_weights = False
        from_ae = False

        fn = file_name

        # Running Finetune on original space
        file_name = file_name + "pavPPMI"
        class_path = "../data/movies/finetune/" + file_name +  ".txt"

        if randomize_finetune_weights:
            fine_tune_weights_fn = None
            file_name = file_name + "N" +str(noise)+"FTR"
        elif corrupt_finetune_weights:
            file_name = file_name + "N" + str(noise) + "FTC"
        else:
            file_name = file_name + "N" + str(noise) + "FT"

        if is_identity:
            file_name = file_name + "IT"
        if identity_swap:
            is_identity = False
            file_name = file_name + "ITS"
        print(file_name)

        loss = "mse"
        optimizer_name = "adagrad"
        hidden_activation = "tanh"
        output_activation = "linear"
        file_name = file_name + optimizer_name + loss + str(epochs)

        amount_of_finetune = 8
        """
        for a in range(1, amount_of_finetune):
            file_name = file_name + str(a)
            SDA = NeuralNetwork( noise=0, fine_tune_weights_fn=fine_tune_weights_fn, optimizer_name=optimizer_name, network_type="ft", past_model_bias_fn=past_model_bias_fn, randomize_finetune_weights=randomize_finetune_weights,
                             vector_path=init_vector_path,  hidden_layer_size=hidden_layer_size, class_path=class_path,  identity_swap=identity_swap, amount_of_finetune=a,
                                   hidden_activation=hidden_activation, output_activation=output_activation, epochs=epochs, learn_rate=learn_rate, is_identity=is_identity,
                             batch_size=batch_size, past_model_weights_fn = past_model_weights_fn, loss=loss, file_name=file_name, from_ae=from_ae)
        """
        init_vector_path = "../data/movies/nnet/spaces/"+file_name+"L1.txt"

        # Get SVM scores
        lowest_count = 500
        highest_count = 10000
        vector_path = "../data/movies/nnet/spaces/"+file_name+"L1.txt"
        class_path = "../data/movies/bow/binary/phrases/class-all-" +str(lowest_count)
        property_names_fn = "../data/movies/bow/names/" + str(lowest_count) + ".txt"
        #svm.getSVMResults(vector_path, class_path, property_names_fn, file_name, lowest_count=lowest_count, highest_count=highest_count)
        """
        vector_names_fn = "../data/movies/nnet/spaces/filmNames.txt"
        class_names_fn = "../data/movies/bow/names/" + str(lowest_count) + ".txt"
        directions_fn = "../data/movies/svm/directions/" + file_name + str(lowest_count) + ".txt"
        rank.getAllPhraseRankings(directions_fn, vector_path, class_names_fn, vector_names_fn, file_name)

        ndcg.getNDCG("../data/movies/rank/numeric/" + file_name + str(lowest_count) + ".txt", file_name)
        """
        cluster_to_classify = -1
        max_depth = 50
        label_names_fn = "../data/movies/classify/keywords/names.txt"
        cluster_labels_fn = "../data/movies/classify/keywords/class-All"
        cluster_names_fn = "../data/movies/cluster/hierarchy_names/" + fn + str(score_limit) + ".txt"
        #clf = tree.DecisionTree(clusters_fn, cluster_labels_fn, label_names_fn, cluster_names_fn, file_name, 10000, max_depth)



    fn_to_place = "films100L3100N0.5"
    score_limit = 0.8
    cluster_amt = 400
    property_names_fn = "../data/movies/cluster/hierarchy_names/" + fn_to_place + str(score_limit) + str(
        cluster_amt) + ".txt"

    ranking_fn = "../data/movies/rank/numeric/" + fn_to_place + ".txt"

    #fto.pavPPMI(property_names_fn, ranking_fn, fn_to_place)


    end_file_names = ["L1films100L3100N0.5InClusterN0.5FTadagradcategorical_crossentropy100Genres100L3",
                      "L2films100L3100N0.5InClusterN0.5FTadagradcategorical_crossentropy100Genres100L3",
                      "L3films100L3100N0.5InClusterN0.5FTadagradcategorical_crossentropy100Genres100L3"]
    init_vector_path = "../data/movies/nnet/spaces/films100.txt"
    past_model_weights_fn = []
    past_model_bias_fn = []

    for f in end_file_names:
        past_model_weights_fn.append("../data/movies/nnet/weights/" + f + ".txt")
        past_model_bias_fn.append("../data/movies/nnet/bias/" + f + ".txt")

    class_path = "../data/movies/classify/genres/class-all"
    loss = "binary_crossentropy"
    output_activation = "sigmoid"
    optimizer_name = "adagrad"
    hidden_activation = "tanh"
    learn_rate = 0.01
    fine_tune_weights_fn = None
    randomize_finetune_weights = False
    epochs = 100
    batch_size = 200
    hidden_layer_size = 400
    is_identity = False
    dropout_noise = None
    from_ae = True
    identity_swap = False
    file_name = end_file_names[len(end_file_names)-1]
    """
    score_limit = 0.8
    cluster_amt = 400
    clusters_fn = "../data/movies/cluster/hierarchy_directions/" + fn_to_place + str(score_limit) + str(
        cluster_amt) + ".txt"
    fine_tune_weights_fn = [clusters_fn]
    randomize_finetune_weights = False
    class_path ="../data/movies/finetune/" + fn_to_place + "pavPPMI.txt"
    loss = "mse"
    output_activation = "linear"
    batch_size = 200
    hidden_layer_size = 100
    epochs = 250
    file_name = file_name + "Genres" + str(epochs) + "L" + str(len(end_file_names))
    """

    deep_size = 400
    epochs = 299
    from_ae = False
    past_model_weights_fn = None
    past_model_bias_fn = None
    fine_tune_weights_fn = None
    is_identity = True
    amount_of_finetune = 0
    randomize_finetune_weights = True
    init_vector_path = "../data/movies/rank/numeric/films100400.txt"
    file_name = file_name + "rank" + "E" + str(epochs) + "DS" + str(deep_size) + "L" +  str(amount_of_finetune)
    SDA = NeuralNetwork(noise=0, fine_tune_weights_fn=fine_tune_weights_fn, optimizer_name=optimizer_name,
                        network_type="ft", past_model_bias_fn=past_model_bias_fn, deep_size=deep_size,
                        randomize_finetune_weights=randomize_finetune_weights, amount_of_finetune=amount_of_finetune,
                        vector_path=init_vector_path, hidden_layer_size=hidden_layer_size, class_path=class_path,
                        identity_swap=identity_swap, dropout_noise=dropout_noise,
                        hidden_activation=hidden_activation, output_activation=output_activation, epochs=epochs,
                        learn_rate=learn_rate, is_identity=is_identity,
                        batch_size=batch_size, past_model_weights_fn=past_model_weights_fn, loss=loss,
                        file_name=file_name, from_ae=from_ae)

if  __name__ =='__main__':main()