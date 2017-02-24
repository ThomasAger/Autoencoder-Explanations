# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import numpy as np
from keras.layers.noise import GaussianNoise
import data as dt
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
    finetune_size = 0
    class_outputs = False
    amount_of_hidden = 0
    finetune_activation = ""

    def __init__(self, training_data=10000, class_path=None, network_type="ft",  randomize_finetune_weights=False, dropout_noise = None, amount_of_hidden=0,
                 epochs=1,  learn_rate=0.01, loss="mse", batch_size=1, past_model_bias_fn=None, identity_swap=False, reg=0.0, amount_of_finetune=1, output_size=25,
                 hidden_activation="tanh", layer_init="glorot_uniform", output_activation="tanh", deep_size = None, corrupt_finetune_weights = False,
                   hidden_layer_size=100, file_name="unspecified_filename", vector_path=None, is_identity=False, activity_reg=0.0, finetune_size=0, data_type="movies",
                 optimizer_name="rmsprop", noise=0.0, fine_tune_weights_fn=None, past_model_weights_fn=None, from_ae=True, class_outputs=False, finetune_activation="linear"):

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
        self.finetune_size = finetune_size
        self.class_outputs = class_outputs
        self.reg = reg
        self.activity_reg = activity_reg
        self.activity_reg = activity_reg
        self.amount_of_finetune = amount_of_finetune
        self.amount_of_hidden = amount_of_hidden
        self.output_size = output_size
        self.finetune_activation = finetune_activation

        print(data_type)

        if optimizer_name == "adagrad":
            self.optimizer = Adagrad()
        else:
            self.optimizer = SGD(lr=learn_rate, momentum=0.0, decay=0.0, nesterov=False)

        entity_vectors, entity_classes = None, None

        if network_type == "ft":
            entity_vectors, entity_classes = self.fineTuneNetwork(past_model_weights_fn, past_model_bias_fn,
                                                                fine_tune_weights_fn, is_identity, identity_swap,
                                                                randomize_finetune_weights, corrupt_finetune_weights,
                                                                deep_size, from_ae)
        elif network_type == "da":
            entity_vectors, entity_classes = self.denoisingAutoencoder(noise, deep_size)

        x_train, x_test, y_train, y_test = train_test_split( entity_vectors, entity_classes, test_size = 0.3, random_state = 0)

        #x_train, y_train = dt.balance2dClasses(x_train, y_train, 1)

        # Compile the model and fit it to the data
        self.model.fit(x_train, y_train, nb_epoch=self.epochs, batch_size=self.batch_size, verbose=1)



        if network_type == "ft":
            if class_outputs:
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
                dt.write1dArray(scores, "../data/" + data_type + "/nnet/scores/" + self.file_name + ".txt")
                print(scores)
            self.output_clusters = self.model.predict(entity_vectors)
            dt.write2dArray(self.output_clusters.transpose(), "../data/" + data_type + "/nnet/clusters/" + self.file_name + ".txt")

        total_file_name = "../data/" + data_type + "/nnet/spaces/" + self.file_name
        for l in range(0, len(self.model.layers) - 1):
            if dropout_noise is not None or dropout_noise > 0.0:
                if l % 2 == 1:
                    continue
            print("Writing", l, "layer")
            truncated_model = Sequential()
            for a in range(l+1):
                truncated_model.add(self.model.layers[a])
            truncated_model.compile(loss=self.loss, optimizer="sgd")
            self.end_space = truncated_model.predict(entity_vectors)
            dt.write2dArray(self.end_space, total_file_name + "L" + str(l) + ".txt")

        for l in range(len(self.model.layers)):
            try:
                dt.write2dArray(self.model.layers[l].get_weights()[0],
                                "../data/" + data_type + "/nnet/weights/L" + str(l) + file_name + ".txt")
                dt.write1dArray(self.model.layers[l].get_weights()[1],
                                "../data/" + data_type + "/nnet/bias/L" + str(l) + file_name + ".txt")
            except IndexError:
                print("Layer ", str(l), "Failed")



    def fineTuneNetwork(self, past_weights_fn, past_model_bias_fn, fine_tune_weights_fn, is_identity, identity_swap,
                        randomize_finetune_weights, corrupt_finetune_weights, deep_size, from_ae):
        entity_vectors = np.asarray(dt.import2dArray(self.vector_path))
        print("Imported vectors", len(entity_vectors), len(entity_vectors[0]))
        entity_classes = np.asarray(dt.import2dArray(self.class_path))
        print("Imported classes", len(entity_classes), len(entity_classes[0]))


        if len(entity_classes) != len(entity_vectors):
            entity_classes = entity_classes.transpose()
            print("Transposed classes, now in form", len(entity_classes), len(entity_classes[0]))
        if len(entity_vectors) != len(entity_classes):
            entity_vectors = entity_vectors.transpose()
            print("Transposed vectors, now in form", len(entity_vectors), len(entity_vectors[0]))
        input_size = len(entity_vectors[0])
        print(input_size, self.hidden_layer_size, self.finetune_size, self.output_size)

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

            for a in range(len(entity_classes)):
                entity_classes[a] = np.around(entity_classes[a], decimals=6)

            fine_tune_weights = []
            fine_tune_weights.append(r.transpose())
            fine_tune_weights.append(np.empty(shape=len(r), dtype="float64"))

        #self.model.add(GaussianNoise(0.0, input_shape=(input_size,)))

        # If we want to swap the identity layer to before the hidden layer
        if identity_swap:
            print("Identity swapped layer", input_size, self.hidden_layer_size, self.hidden_activation)
            for a in range(self.amount_of_finetune):
                self.model.add(Dense(output_dim=self.hidden_layer_size, input_dim=input_size,
                                     activation=self.hidden_activation,
                                     init="identity"))

        if deep_size is not None:
            print(a, "Deep layer", input_size, deep_size[0], self.hidden_activation)
            self.model.add(Dense(output_dim=deep_size[0], input_dim=input_size, init=self.layer_init,
                                 activation=self.hidden_activation, W_regularizer=l2(self.reg)))
            if self.dropout_noise is not None:
                print("Dropout layer")
                self.model.add(Dropout(self.dropout_noise))

            for a in range(1, len(deep_size)):
                print(a, "Deep layer", deep_size[a-1], deep_size[a], self.hidden_activation)
                self.model.add(Dense(output_dim=deep_size[a], input_dim=deep_size[a-1], init=self.layer_init,
                                 activation=self.hidden_activation, W_regularizer=l2(self.reg)))
                if self.dropout_noise is not None:
                    print("Dropout layer")
                    self.model.add(Dropout(self.dropout_noise))

        if from_ae:
            for p in range(len(past_weights)):
                print(p, "Past AE layer", input_size, self.hidden_layer_size, self.hidden_activation)
                self.model.add(Dense(output_dim=self.hidden_layer_size, input_dim=input_size, activation=self.hidden_activation,
                                     weights=past_weights[p], W_regularizer=l2(self.reg)))
            if self.dropout_noise is not None:
                print("Dropout layer")
                self.model.add(Dropout(self.dropout_noise))

        # Add an identity layer that has equal values to the input space to find some more nonlinear relationships
        if is_identity:
            print("Identity layer", self.hidden_layer_size, self.hidden_layer_size, self.hidden_activation)
            for a in range(self.amount_of_finetune):
                self.model.add(Dense(output_dim=self.hidden_layer_size, input_dim=self.hidden_layer_size, activation=self.hidden_activation,
                          init="identity"))

        if randomize_finetune_weights:
            print("Randomize finetune weights", self.hidden_layer_size, self.finetune_size, self.finetune_activation)
            self.model.add(Dense(output_dim=self.finetune_size, input_dim=self.hidden_layer_size, activation=self.finetune_activation,
                                 init=self.layer_init))
        elif corrupt_finetune_weights:
            print("Corrupt finetune weights", self.hidden_layer_size, self.finetune_size, self.finetune_activation)
            self.model.add(Dense(output_dim=self.finetune_size, input_dim=self.hidden_layer_size, activation=self.finetune_activation,
                                 weights=fine_tune_weights))
        elif deep_size is None:
            print("Fine tune weights", self.hidden_layer_size, len(fine_tune_weights[0][0]), self.finetune_activation)
            self.model.add(Dense(output_dim=len(fine_tune_weights[0][0]), input_dim=self.hidden_layer_size, activation=self.finetune_activation,
                                 weights=fine_tune_weights))
        if self.class_outputs:

            if randomize_finetune_weights or corrupt_finetune_weights or len(fine_tune_weights_fn) > 0:
                print("Class outputs", self.finetune_size, self.output_size, self.output_activation)
                self.model.add(
                    Dense(output_dim=self.output_size, input_dim=self.finetune_size, activation=self.output_activation,
                          init=self.layer_init))
            else:
                print("Class outputs", deep_size[len(deep_size)-1], self.output_size, self.output_activation)
                self.model.add(
                    Dense(output_dim=self.output_size, input_dim=deep_size[len(deep_size)-1],
                          activation=self.output_activation,
                          init=self.layer_init))
        print("Compiling")
        self.model.compile(loss=self.loss, optimizer=self.optimizer)

        return entity_vectors, entity_classes


    def denoisingAutoencoder(self, noise, deep_size):
        entity_vectors = np.asarray(dt.import2dArray(self.vector_path))
        if len(entity_vectors) != 15000:
            entity_vectors = entity_vectors.transpose()
        if self.class_path is None:
            entity_classes = entity_vectors
        else:
            entity_classes = np.asarray(dt.import2dArray(self.class_path))
        input_size = len(entity_vectors[0])
        output_size = len(entity_classes[0])
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
        return entity_vectors, entity_classes


    def getEndSpace(self):
        return self.end_space

    def getEncoder(self):
        return self.model.layers[1]



def main():
    hidden_layer_sizes = [100,100,100,100,100,100]
    file_names = []

    data_type = "wines"

    for f in range(len(hidden_layer_sizes)):
        #file_names.append("filmsBOWL" + str(f + 1) + "" + str(hidden_layer_sizes[f]))
        #file_names.append("filmsPPMIDropoutL"+str(f+1)+""+str(hidden_layer_sizes[f]))
        file_names.append(data_type + "100L" + str(f + 1) + "" + str(hidden_layer_sizes[f]))

    #init_vector_path= "../data/" + data_type + "/bow/binary/phrases/class-all"
    #init_vector_path = "../data/" + data_type + "/bow/ppmi/class-all"
    #init_vector_path="../data/" + data_type + "/nnet/spaces/films200L1100N0.5pavPPMIN0.5FTadagradcategorical_crossentropy100.txt"
    init_vector_path = "../data/" + data_type + "/nnet/spaces/wines100.txt"
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
                         vector_path=init_vector_path,  hidden_layer_size=hidden_layer_size, class_path=class_path, reg=reg, data_type=data_type,
                               hidden_activation=hidden_activation, output_activation=output_activation, learn_rate=learn_rate,
                              file_name=file_name, network_type="da", deep_size=deep_size, activity_reg=activity_reg)
        """
        file_name = "wines100trimmed"

        vector_path = "../data/" + data_type + "/nnet/spaces/"+file_name+".txt"
        init_vector_path = "../data/" + data_type + "/nnet/spaces/"+file_name+".txt"
        past_model_weights_fn = ["../data/" + data_type + "/nnet/weights/L1" + file_name + ".txt"]
        past_model_bias_fn = ["../data/" + data_type + "/nnet/bias/L1" + file_name + ".txt"]
        hidden_space_fn = "../data/" + data_type + "/nnet/spaces/"+file_name+".txt"

        # Get SVM scores

        lowest_count = 50
        highest_count = 0
        #vector_path = "../data/" + data_type + "/nnet/spaces/"+file_name+"L1.txt"
        class_path = "../data/" + data_type + "/bow/binary/phrases/class-all-" +str(lowest_count)
        property_names_fn = "../data/" + data_type + "/bow/names/" + str(lowest_count) + ".txt"
        svm_type = "svm"
        file_name = file_name + svm_type
        """
        svm.getSVMResults(vector_path, class_path, property_names_fn, file_name, lowest_count=lowest_count, highest_count=highest_count,
                          svm_type=svm_type, get_kappa=True, get_f1=False, single_class=True, data_type=data_type)
        """
        directions_fn = "../data/" + data_type + "/svm/directions/" + file_name + str(lowest_count) + ".txt"
        # Get rankings
        vector_names_fn = "../data/" + data_type + "/nnet/spaces/entitynames.txt"
        class_names_fn = "../data/" + data_type + "/bow/names/" + str(lowest_count)+".txt"
        directions_fn = "../data/" + data_type + "/svm/directions/" + file_name + str(lowest_count)+".txt"

        #rank.getAllPhraseRankings(directions_fn, vector_path, class_names_fn, vector_names_fn, file_name, data_type=data_type)

        #ndcg.getNDCG("../data/" + data_type + "/rank/numeric/"+file_name+"ALL.txt",file_name, data_type, lowest_count)

        scores_fn = "../data/" + data_type + "/ndcg/"+file_name+".txt"
        file_name = file_name + "ndcg"
        kappa = False
        #scores_fn = "../data/" + data_type + "/svm/kappa/" + file_name + str(lowest_count)+".txt"
        #file_name = file_name + "kappa"
        #kappa = True

        # Get clusters
        amt_high_directions = hidden_layer_size*2
        amt_low_directions = 13000
        amt_of_clusters = hidden_layer_size*2
        #scores_fn = "../data/" + data_type + "/svm/kappa/"+file_name+"200.txt"
        #file_name = file_name + "similarityclustering"
        #cluster.getClusters(directions_fn, scores_fn, class_names_fn, False,  amt_high_directions, amt_low_directions, file_name, amt_of_clusters)
        clusters_fn = "../data/" + data_type + "/cluster/clusters/" +file_name+".txt"
        property_names_fn = "../data/" + data_type + "/cluster/names/" +file_name+".txt"
        percentage_bin = 1
        #rank.getAllRankings(clusters_fn, vector_path, property_names_fn, vector_names_fn, 0.2, 1, False, file_name, False, data_type)

        names_fn = "../data/" + data_type + "/bow/names/"+str(lowest_count)+".txt"
        dissimilarity_threshold = 0.5
        similarity_threshold = 0.9
        cluster_amt = 200
        amount_to_start = 8000
        score_limit = 0.95
        print(file_name)
        add_all_terms = False
        file_name = file_name + "not all terms" + str(score_limit)
        hierarchy.initClustering(vector_path, directions_fn, scores_fn, names_fn, amount_to_start, False,
                                 dissimilarity_threshold,  cluster_amt, score_limit, file_name, kappa,
                                 similarity_threshold, add_all_terms, data_type )

        # Get rankings
        clusters_fn = "../data/" + data_type + "/cluster/hierarchy_directions/" + file_name + ".txt"
        property_names_fn = "../data/" + data_type + "/cluster/hierarchy_names/" + file_name +  ".txt"
        vector_names_fn = "../data/" + data_type + "/nnet/spaces/entitynames.txt"


        rank.getAllRankings(clusters_fn, vector_path, property_names_fn, vector_names_fn, 0.2, 1, False, file_name,
                            False, data_type)

        #file_name = "films100previouswork"
        # Get PAV
        ranking_fn = "../data/" + data_type + "/rank/numeric/"+file_name+".txt"

        #fto.pavPPMI(property_names_fn, ranking_fn, file_name, data_type)

        #fto.pavTermFrequency(ranking_fn, cluster_names_fn, file_name, False)
        #fto.binaryClusterTerm(cluster_names_fn, file_name)
        #fto.binaryInCluster(property_names_fn, file_name)
        discrete_labels_fn = "../data/" + data_type + "/rank/discrete/" + file_name + "P1.txt"

        # Use PAV as class vectors
        fine_tune_weights_fn = [clusters_fn]
        epochs = 2000
        batch_size = 200
        learn_rate = 0.001
        is_identity = True
        identity_swap = False
        randomize_finetune_weights = False
        corrupt_finetune_weights = False
        from_ae = True
        #from_ae = False
        finetune_size = 200
        fn = file_name

        # Running Finetune on original space
        file_name = file_name + "pavPPMI"
        class_path = "../data/" + data_type + "/finetune/" + file_name +  ".txt"

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
            file_name = file_name + "ITS"
            file_name = file_name + "ITS"
        print(file_name)

        loss = "mse"
        optimizer_name = "sgd"
        hidden_activation = "tanh"
        finetune_activation = "linear"
        file_name = file_name + optimizer_name + loss + str(epochs)

        print(file_name)
        amount_of_finetune = 1
        """
        SDA = NeuralNetwork( noise=0, fine_tune_weights_fn=fine_tune_weights_fn, optimizer_name=optimizer_name,  network_type="ft",
                             past_model_bias_fn=past_model_bias_fn,  randomize_finetune_weights=randomize_finetune_weights,
                             vector_path=init_vector_path,  hidden_layer_size=hidden_layer_size, class_path=class_path,
                             amount_of_finetune=amount_of_finetune, identity_swap=identity_swap,
                               hidden_activation=hidden_activation, output_activation=output_activation, epochs=epochs,
                             learn_rate=learn_rate, is_identity=is_identity, finetune_activation=finetune_activation,
                         batch_size=batch_size, past_model_weights_fn = past_model_weights_fn, loss=loss,
                             file_name=file_name, from_ae=from_ae, finetune_size=finetune_size, data_type=data_type)
        """
        init_vector_path = "../data/" + data_type + "/nnet/spaces/"+file_name+"L1.txt"

        # Get SVM scores
        lowest_count = 200
        highest_count = 10000
        vector_path = "../data/" + data_type + "/nnet/spaces/"+file_name+"L1.txt"
        class_path = "../data/" + data_type + "/bow/binary/phrases/class-all-" +str(lowest_count)
        property_names_fn = "../data/" + data_type + "/bow/names/" + str(lowest_count) + ".txt"
        svm_type = "svm"
        file_name = file_name + svm_type
        #svm.getSVMResults(vector_path, class_path, property_names_fn, file_name, lowest_count=lowest_count, highest_count=highest_count, svm_type=svm_type, get_kappa=False, get_f1=False)
        # Get rankings
        vector_names_fn = "../data/" + data_type + "/nnet/spaces/entitynames.txt"
        class_names_fn = "../data/" + data_type + "/bow/names/" + str(lowest_count) + ".txt"
        directions_fn = "../data/" + data_type + "/svm/directions/" + file_name + str(lowest_count) + ".txt"
        #rank.getAllPhraseRankings(directions_fn, vector_path, property_names_fn, vector_names_fn, file_name)
        # file_name = file_name + "ndcg"
        #ndcg.getNDCG("../data/" + data_type + "/rank/numeric/"+file_name+"ALL.txt",file_name)

        names_fn = "../data/" + data_type + "/bow/names/"+str(lowest_count)+".txt"
        similarity_threshold = 0.5
        cluster_amt = 200
        amount_to_start = 8000
        score_limit = 0.9
        print(file_name)
        #hierarchy.initClustering(vector_path, directions_fn, scores_fn, names_fn, amount_to_start, False, similarity_threshold,  cluster_amt, score_limit, file_name, kappa)

        """
        scores_fn = "../data/" + data_type + "/svm/kappa/" + file_name + "200.txt"
        file_name = file_name + "kappa"
                kappa = True
        hierarchy.initClustering(vector_path, directions_fn, scores_fn, names_fn, amount_to_start, False,
                                 similarity_threshold, cluster_amt, score_limit, file_name, kappa)
        """
        # Get rankings
        clusters_fn = "../data/" + data_type + "/cluster/hierarchy_directions/" + file_name + str(score_limit) + str(cluster_amt) + ".txt"
        property_names_fn = "../data/" + data_type + "/cluster/hierarchy_names/" + file_name + str(score_limit) + str(cluster_amt) + ".txt"
        vector_names_fn = "../data/" + data_type + "/nnet/spaces/entitynames.txt"
        #rank.getAllRankings(clusters_fn, vector_path, property_names_fn, vector_names_fn, 0.2, 1, False, file_name, False)

        cluster_to_classify = -1
        max_depth = 50
        label_names_fn = "../data/" + data_type + "/classify/keywords/names.txt"
        cluster_labels_fn = "../data/" + data_type + "/classify/keywords/class-All"
        cluster_names_fn = "../data/" + data_type + "/cluster/hierarchy_names/" + fn + str(score_limit) + ".txt"
        #clf = tree.DecisionTree(clusters_fn, cluster_labels_fn, label_names_fn, cluster_names_fn, file_name, 10000, max_depth)



    fn_to_place = "films100L3100N0.5"
    score_limit = 0.8
    cluster_amt = 200
    property_names_fn = "../data/" + data_type + "/cluster/hierarchy_names/" + fn_to_place + str(score_limit) + str(
        cluster_amt) + ".txt"

    ranking_fn = "../data/" + data_type + "/rank/numeric/" + fn_to_place + ".txt"

    #fto.pavPPMI(property_names_fn, ranking_fn, fn_to_place)


    end_file_names = ["L1films100L3100N0.5InClusterN0.5FTadagradcategorical_crossentropy100Genres100L3",
                      "L2films100L3100N0.5InClusterN0.5FTadagradcategorical_crossentropy100Genres100L3",
                      "L3films100L3100N0.5InClusterN0.5FTadagradcategorical_crossentropy100Genres100L3"]
    init_vector_path = "../data/" + data_type + "/nnet/spaces/films100.txt"
    past_model_weights_fn = []
    past_model_bias_fn = []

    for f in end_file_names:
        past_model_weights_fn.append("../data/" + data_type + "/nnet/weights/" + f + ".txt")
        past_model_bias_fn.append("../data/" + data_type + "/nnet/bias/" + f + ".txt")

    class_path = "../data/" + data_type + "/classify/genres/class-all"
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
    clusters_fn = "../data/" + data_type + "/cluster/hierarchy_directions/" + fn_to_place + str(score_limit) + str(
        cluster_amt) + ".txt"
    fine_tune_weights_fn = [clusters_fn]
    randomize_finetune_weights = False
    class_path ="../data/" + data_type + "/finetune/" + fn_to_place + "pavPPMI.txt"
    loss = "mse"
    output_activation = "linear"
    batch_size = 200
    hidden_layer_size = 100
    epochs = 250
    file_name = file_name + "Genres" + str(epochs) + "L" + str(len(end_file_names))
    """
    """
    deep_size = 400
    epochs = 299
    from_ae = False
    past_model_weights_fn = None
    past_model_bias_fn = None
    fine_tune_weights_fn = None
    is_identity = True
    amount_of_finetune = 5
    randomize_finetune_weights = True
    file_name = "films100"
    finetune_size = cluster_amt
    init_vector_path = "../data/" + data_type + "/rank/numeric/"+file_name+".txt"
    file_name = file_name + "rank" + "E" + str(epochs) + "DS" + str(deep_size) + "L" +  str(amount_of_finetune)
    SDA = NeuralNetwork(noise=0, fine_tune_weights_fn=fine_tune_weights_fn, optimizer_name=optimizer_name,
                        network_type="ft", past_model_bias_fn=past_model_bias_fn, deep_size=deep_size,
                        randomize_finetune_weights=randomize_finetune_weights, amount_of_finetune=amount_of_finetune,
                        vector_path=init_vector_path, hidden_layer_size=hidden_layer_size, class_path=class_path,
                        identity_swap=identity_swap, dropout_noise=dropout_noise,
                        hidden_activation=hidden_activation, output_activation=output_activation, epochs=epochs,
                        learn_rate=learn_rate, is_identity=is_identity, finetune_size = finetune_size,
                        batch_size=batch_size, past_model_weights_fn=past_model_weights_fn, loss=loss,
                        file_name=file_name, from_ae=from_ae)
    """
    deep_size = 400
    epochs = 299
    from_ae = True
    #past_model_weights_fn = None
    #past_model_bias_fn = None
    #file_name = "films100"
    fine_tune_weights_fn = None
    is_identity = False
    amount_of_finetune = 0
    randomize_finetune_weights = False
    #file_name = end_file_names[len(end_file_names)-1]
    #init_vector_path = "../data/" + data_type + "/nnet/spaces/films100.txt"
    score_limit = 0.9
    cluster_amt = 400
    output_size = 23
    hidden_layer_size = 100
    epochs = 200
    class_outputs = True
    optimizer_name ="adagrad"
    learn_rate = 0.01
    output_activation = "sigmoid"
    finetune_activation = "linear"
    hidden_activation = "tanh"
    finetune_size = cluster_amt
    file_name = "films100"
    original_fn = file_name
    init_vector_path = "../data/" + data_type + "/rank/numeric/" +file_name+ "svmndcg0.9"+str(cluster_amt)+ ".txt"
    clusters_fn = "../data/" + data_type + "/cluster/hierarchy_directions/" + file_name + "svmndcg0.9"+str(cluster_amt)+".txt"
    deep_size = [100, 100, 100]
    fine_tune_weights_fn = [clusters_fn]
    fine_tune_weights_fn = ""
    class_path = "../data/" + data_type + "/classify/genres/class-All"
    from_ae = False
    file_name = file_name + "rank" + "E" + str(epochs) + "DS" + str(deep_size) + "L" + str(len(deep_size)) + str(cluster_amt)
    """
    SDA = NeuralNetwork(noise=0, fine_tune_weights_fn=fine_tune_weights_fn, optimizer_name=optimizer_name,
                        network_type="ft", past_model_bias_fn=past_model_bias_fn, deep_size=deep_size,
                        finetune_activation=finetune_activation,
                        randomize_finetune_weights=randomize_finetune_weights, amount_of_finetune=amount_of_finetune,
                        vector_path=init_vector_path, hidden_layer_size=hidden_layer_size, class_path=class_path,
                        identity_swap=identity_swap, dropout_noise=dropout_noise, class_outputs=class_outputs,
                        hidden_activation=hidden_activation, output_activation=output_activation, epochs=epochs,
                        learn_rate=learn_rate, is_identity=is_identity, output_size=output_size,
                        finetune_size=finetune_size,
                        batch_size=batch_size, past_model_weights_fn=past_model_weights_fn, loss=loss,
                        file_name=file_name, from_ae=from_ae)
    """
    data_type = "wines"
    classification_task = "types"
    file_name = "wines100trimmed"
    init_vector_path = "../data/" + data_type + "/nnet/spaces/" + file_name + ".txt"

    #file_name = "winesppmi"
    #init_vector_path = "../data/wines/bow/ppmi/class-trimmed-all-50"

    deep_size = [100,100,100]
    for d in range(len(deep_size)):
        print(deep_size, init_vector_path)
        loss = "binary_crossentropy"
        output_activation = "sigmoid"
        optimizer_name = "adagrad"
        hidden_activation = "tanh"
        classification_path = "../data/" + data_type + "/classify/" + classification_task + "/class-all"
        learn_rate = 0.01
        fine_tune_weights_fn = None
        epochs = 500
        batch_size = 200
        class_outputs = True
        dropout_noise = 0.3
        is_identity = False
        identity_swap = False
        randomize_finetune_weights = False
        hidden_layer_size = 100
        output_size = 10
        randomize_finetune_weights = False
        corrupt_finetune_weights = False
        fine_tune_weights_fn = []

        #init_vector_path = "../data/" + data_type + "/movies/bow/binary/phrases/class-all"
        if d == 0:
            file_name = file_name + "rank" + "E" + str(epochs) + "DS" + str(deep_size) + "L" + str(amount_of_finetune)\
                        + "DN" + str(dropout_noise) + hidden_activation + "SFT" + str(d)
        else:
            file_name = file_name + "SFT" + str(d)
        print("!!!!!!!!!!!!!!!", deep_size)

        SDA = NeuralNetwork(noise=0, fine_tune_weights_fn=fine_tune_weights_fn, optimizer_name=optimizer_name,
                network_type="ft", past_model_bias_fn=past_model_bias_fn, deep_size=deep_size, finetune_activation=finetune_activation,
                randomize_finetune_weights=randomize_finetune_weights, amount_of_finetune=amount_of_finetune,
                vector_path=init_vector_path, hidden_layer_size=hidden_layer_size, class_path=classification_path,
                identity_swap=identity_swap, dropout_noise=dropout_noise, class_outputs=class_outputs,
                hidden_activation=hidden_activation, output_activation=output_activation, epochs=epochs,
                learn_rate=learn_rate, is_identity=is_identity, output_size=output_size, finetune_size=finetune_size,
                batch_size=batch_size, past_model_weights_fn=past_model_weights_fn, loss=loss,
                file_name=file_name, from_ae=from_ae, data_type=data_type)
        new_file_names = []
        if dropout_noise is not None and dropout_noise > 0.0:
            for j in range(0, len(deep_size)*2 + 1, 2):
                new_fn = file_name + "L" + str(j)
                new_file_names.append(new_fn)
        else:
            for j in range(0, len(deep_size) + 1):
                new_fn = file_name + "L" + str(j)
                new_file_names.append(new_fn)

        for j in range(len(new_file_names)):
            #file_name = "wines100trimmed"
            #file_name = "films100rankE200DS[100, 100, 100]L3300L1svmndcg0.9200pavPPMIN0.5FTITsgdmse2000L1rankE100DS[100, 100]L0"
            file_name = new_file_names[j]
            past_model_weights_fn = ["../data/" + data_type + "/nnet/weights/" + file_name + ".txt"]
            past_model_bias_fn = ["../data/" + data_type + "/nnet/bias/"+ file_name + ".txt"]
            # Get SVM scores

            if data_type is "wines" or "placetypes":
                lowest_count = 50
            else:
                lowest_count = 200
            highest_count = 10000
            vector_path = "../data/" + data_type + "/nnet/spaces/"+file_name+".txt"
            class_path = "../data/" + data_type + "/bow/binary/phrases/class-all-" + str(lowest_count)
            property_names_fn = "../data/" + data_type + "/bow/names/" + str(lowest_count) + ".txt"
            svm_type = "svm"
            threads = 4
            file_name = file_name + svm_type
            svm.getSVMResults(vector_path, class_path, property_names_fn, file_name, lowest_count=lowest_count,
                  highest_count=highest_count, svm_type=svm_type, data_type=data_type, get_kappa=True,
                  get_f1=False, getting_directions=True, threads=4)

            directions_fn = "../data/" + data_type + "/svm/directions/" + file_name + str(lowest_count) + ".txt"
            # Get rankings
            vector_names_fn = "../data/" + data_type + "/nnet/spaces/entitynames.txt"
            class_names_fn = "../data/" + data_type + "/bow/names/" + str(lowest_count) + ".txt"
            directions_fn = "../data/" + data_type + "/svm/directions/" + file_name + str(lowest_count) + ".txt"

            """
            scores_fn = "../data/" + data_type + "/svm/kappa/" + file_name + str(lowest_count) + ".txt"
            kappa = True
            if d == 0:
                file_name = file_name + "kappa"
            """

            rank.getAllPhraseRankings(directions_fn, vector_path, class_names_fn, vector_names_fn, file_name,
                                  data_type=data_type)
            ndcg.getNDCG("../data/" + data_type + "/rank/numeric/" + file_name + "ALL.txt", file_name,
                         data_type=data_type, lowest_count=lowest_count)
            scores_fn = "../data/" + data_type + "/ndcg/" + file_name + ".txt"
            kappa = False
            if d == 0:
                file_name = file_name + "ndcg"


            names_fn = "../data/" + data_type + "/bow/names/"+str(lowest_count)+".txt"
            similarity_threshold = 0.5
            cluster_amt = deep_size[j] * 2
            amount_to_start = 8000
            score_limit = 0.9
            dissimilarity_threshold = 0.9


            file_name = file_name + str(score_limit) + str(cluster_amt)


            hierarchy.initClustering(vector_path, directions_fn, scores_fn, names_fn, amount_to_start, False,
                             similarity_threshold,  cluster_amt, score_limit, file_name, kappa, dissimilarity_threshold, data_type=data_type)



            # Get rankings
            clusters_fn = "../data/" + data_type + "/cluster/hierarchy_directions/" + file_name + ".txt"
            property_names_fn = "../data/" + data_type + "/cluster/hierarchy_names/" + file_name + ".txt"
            vector_names_fn = "../data/" + data_type + "/nnet/spaces/entitynames.txt"

            rank.getAllRankings(clusters_fn, vector_path, property_names_fn , vector_names_fn, 0.2, 1, False, file_name, False, data_type=data_type)


            # Get PAV
            ranking_fn = "../data/" + data_type + "/rank/numeric/" + file_name + ".txt"
            label_names_fn = "../data/" + data_type + "/classify/" + classification_task + "/names.txt"

            tree.DecisionTree(ranking_fn, classification_path, label_names_fn, property_names_fn, file_name, 10000,
                              3, balance="balanced", criterion="entropy", save_details=False,
                              data_type=data_type)

            tree.DecisionTree(ranking_fn, classification_path, label_names_fn, property_names_fn, file_name, 10000,
                                  None, balance="balanced", criterion="entropy", save_details=False,
                                  data_type=data_type)

            if d == 0:
                file_name = file_name + "pavPPMI"

            fto.pavPPMI(property_names_fn, ranking_fn, file_name, data_type=data_type)
            discrete_labels_fn = "../data/" + data_type + "/rank/discrete/" + file_name + "P1.txt"

            # Use PAV as class vectors
            fine_tune_weights_fn = [clusters_fn]
            epochs = 1000
            batch_size = 200
            learn_rate = 0.001
            is_identity = True
            identity_swap = False
            randomize_finetune_weights = False
            # from_ae = False
            finetune_size = cluster_amt
            fn = file_name

            # Running Finetune on original space
            class_path = "../data/" + data_type + "/finetune/" + file_name + ".txt"
            if d == 0:
                file_name = file_name + "IT"
            print(file_name)

            loss = "mse"
            optimizer_name = "sgd"
            hidden_activation = "tanh"
            finetune_activation = "linear"
            hidden_layer_size = deep_size[j]
            if d == 0:
                file_name = file_name + optimizer_name + loss + str(epochs)
            from_ae = True
            past_model_weights_fn = ["../data/" + data_type + "/nnet/weights/L" + new_file_names[j] + ".txt"]
            past_model_bias_fn = ["../data/" + data_type + "/nnet/bias/L" + new_file_names[j] + ".txt"]

            print(file_name)
            amount_of_finetune = 1

            SDA = NeuralNetwork(noise=0, fine_tune_weights_fn=fine_tune_weights_fn, optimizer_name=optimizer_name,
                                network_type="ft", past_model_bias_fn=past_model_bias_fn,
                                randomize_finetune_weights=randomize_finetune_weights,
                                vector_path=init_vector_path, hidden_layer_size=hidden_layer_size, class_path=class_path,
                                identity_swap=identity_swap, amount_of_finetune=amount_of_finetune,
                                hidden_activation=hidden_activation, output_activation=output_activation, epochs=epochs,
                                learn_rate=learn_rate, is_identity=is_identity, finetune_activation=finetune_activation,
                                batch_size=batch_size, past_model_weights_fn=past_model_weights_fn, loss=loss,
                                file_name=file_name, from_ae=from_ae, finetune_size=finetune_size, data_type=data_type)
            new_file_names[j-1] = file_name

            ranking_fn = "../data/" + data_type + "/nnet/clusters/" + file_name + ".txt"

            tree.DecisionTree(ranking_fn, classification_path, label_names_fn, property_names_fn, file_name, 10000,
                              3, balance="balanced", criterion="entropy", save_details=False,
                              data_type=data_type)

            tree.DecisionTree(ranking_fn, classification_path, label_names_fn, property_names_fn, file_name, 10000,
                              None, balance="balanced", criterion="entropy", save_details=False,
                              data_type=data_type)

        """
        file_name ="films100rankE200DS[100, 100, 100]L3300L1svmndcg0.9200pavPPMIN0.5FTITsgdmse2000L1"
        loss = "binary_crossentropy"
        output_activation = "sigmoid"
        optimizer_name = "adagrad"
        hidden_activation = "tanh"
        class_path = "../data/" + data_type + "/classify/genres/class-all"
        learn_rate = 0.01
        fine_tune_weights_fn = None
        epochs = 100
        batch_size = 200
        class_outputs = True
        dropout_noise = None
        deep_size = [100, 100]
        hidden_layer_size = 100
        output_size = 23
        randomize_finetune_weights = False
        corrupt_finetune_weights = False
        fine_tune_weights_fn = []
        init_vector_path = "../data/" + data_type + "/nnet/clusters/" + file_name + ".txt"
        file_name = file_name + "rank" + "E" + str(epochs) + "DS" + str(deep_size) + "L" + str(amount_of_finetune)

        SDA = NeuralNetwork(noise=0, fine_tune_weights_fn=fine_tune_weights_fn, optimizer_name=optimizer_name,
                            network_type="ft", past_model_bias_fn=past_model_bias_fn, deep_size=deep_size,
                            randomize_finetune_weights=randomize_finetune_weights, output_size=output_size,
                            amount_of_finetune=amount_of_finetune, class_outputs=class_outputs,
                            vector_path=init_vector_path, hidden_layer_size=hidden_layer_size, class_path=class_path,
                            identity_swap=identity_swap, dropout_noise=dropout_noise,
                            hidden_activation=hidden_activation, output_activation=output_activation, epochs=epochs,
                            learn_rate=learn_rate, is_identity=is_identity, finetune_size=finetune_size,
                            batch_size=batch_size, past_model_weights_fn=past_model_weights_fn, loss=loss,
                            file_name=file_name, from_ae=from_ae)
        """
        file_name = new_file_names[0]
        init_vector_path = "../data/" + data_type + "/nnet/spaces/" + file_name + "L0.txt"
        deep_size = deep_size[:len(deep_size)-1]


if  __name__ =='__main__':main()