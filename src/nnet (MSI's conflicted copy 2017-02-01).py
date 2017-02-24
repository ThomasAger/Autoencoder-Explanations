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
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import gini
import cluster
import rank
import finetune_outputs as fto
import svm
import tree
import hierarchy
import ndcg
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

class NeuralNetwork:

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
    save_outputs = False
    amount_of_hidden = 0
    identity_swap = None
    deep_size = None
    from_ae = None
    is_identity = None
    randomize_finetune_weights = None
    corrupt_finetune_weights = None
    deep_size = None
    fine_tune_weights_fn = None
    finetune_weights = None
    past_weights = None

    def __init__(self, training_data=10000, class_path=None, get_scores=False,  randomize_finetune_weights=False, dropout_noise = None, amount_of_hidden=0,
                 epochs=1,  learn_rate=0.01, loss="mse", batch_size=1, past_model_bias_fn=None, identity_swap=False, reg=0.0, amount_of_finetune=1, output_size=25,
                 hidden_activation="tanh", layer_init="glorot_uniform", output_activation="tanh", deep_size = None, corrupt_finetune_weights = False, cross_val=False,
                   hidden_layer_size=100, file_name="unspecified_filename", vector_path=None, is_identity=False, activity_reg=0.0, finetune_size=0, data_type="movies",
                 optimizer_name="rmsprop", noise=0.0, fine_tune_weights_fn=None, past_model_weights_fn=None, from_ae=True, save_outputs=False,
                 rewrite_files=False, cv_splits=1, tuning_parameters=False):

        total_file_name = "../data/" + data_type + "/nnet/spaces/S0" + file_name
        space_fn = total_file_name + "L0.txt"
        weights_fn = "../data/" + data_type + "/nnet/weights/S0" + file_name + "L0.txt"
        bias_fn = "../data/" + data_type + "/nnet/bias/S0" + file_name + "L0.txt"
        rank_fn = "../data/" + data_type + "/nnet/clusters/" + file_name + ".txt"
        f1_fn = "../data/" + data_type + "/nnet/scores/F1 " + file_name + ".txt"
        acc_fn = "../data/" + data_type + "/nnet/scores/ACC " + file_name + ".txt"

        all_fns = [space_fn, weights_fn, bias_fn, rank_fn, f1_fn, acc_fn]
        if dt.allFnsAlreadyExist(all_fns) and not rewrite_files:
            print("Skipping task", "nnet")
            return
        else:
            print("Running task", "nnet")

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
        self.get_scores = get_scores
        self.reg = reg
        self.activity_reg = activity_reg
        self.activity_reg = activity_reg
        self.amount_of_finetune = amount_of_finetune
        self.amount_of_hidden = amount_of_hidden
        self.output_size = output_size
        self.identity_swap = identity_swap
        self.deep_size = deep_size
        self.from_ae = from_ae
        self.is_identity = is_identity
        self.randomize_finetune_weights = randomize_finetune_weights
        self.corrupt_finetune_weights = corrupt_finetune_weights
        self.deep_size = deep_size
        self.fine_tune_weights_fn = fine_tune_weights_fn

        print(data_type)

        if optimizer_name == "adagrad":
            self.optimizer = Adagrad()
        else:
            self.optimizer = SGD(lr=learn_rate, momentum=0.0, decay=0.0, nesterov=False)

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

        self.input_size = len(entity_vectors[0])
        self.output_size = len(entity_classes[0])

        if fine_tune_weights_fn is not None:
            model_builder = self.fineTuneNetwork
            weights = []
            if from_ae:
                self.past_weights = []
                past_model_weights = []
                for p in past_model_weights_fn:
                    past_model_weights.append(np.asarray(dt.import2dArray(p), dtype="float64"))
                past_model_bias = []
                for p in past_model_bias_fn:
                    past_model_bias.append(np.asarray(dt.import1dArray(p, "f"), dtype="float64"))

                for p in range(len(past_model_weights)):
                    past_model_weights[p] = np.around(past_model_weights[p], decimals=6)
                    past_model_bias[p] = np.around(past_model_bias[p], decimals=6)

                for p in range(len(past_model_weights)):
                    self.past_weights.append([])
                    self.past_weights[p].append(past_model_weights[p])
                    self.past_weights[p].append(past_model_bias[p])
            for f in fine_tune_weights_fn:
                weights.extend(dt.import2dArray(f))

            r = np.asarray(weights, dtype="float64")

            for a in range(len(r)):
                r[a] = np.around(r[a], decimals=6)

            for a in range(len(entity_classes)):
                entity_classes[a] = np.around(entity_classes[a], decimals=6)

            self.fine_tune_weights = []
            self.fine_tune_weights.append(r.transpose())
            self.fine_tune_weights.append(np.empty(shape=len(r), dtype="float64"))
        else:
            model_builder = self.classifierNetwork

        models = []
        x_train = []
        y_train = []
        x_test = []
        y_test = []

        for i in range(cv_splits):
            models.append(model_builder())

        f1_scores = []
        accuracy_scores = []
        f1_averages = []
        accuracy_averages = []
        k_fold = KFold(n_splits=cv_splits, shuffle=False, random_state=None)
        if cv_splits > 1:
            c = 0
            for train, test in k_fold.split(entity_vectors):
                x_train.append(entity_vectors[train])
                y_train.append(entity_classes[train])
                x_test.append(entity_vectors[test])
                y_test.append(entity_classes[test])
                models[c].fit(entity_vectors[train], entity_classes[train], nb_epoch=self.epochs,
                              batch_size=self.batch_size, verbose=1)
                c += 1
        else:
            x_tr, x_te, y_tr, y_te = train_test_split(entity_vectors, entity_classes, test_size=0.33, random_state=0)
            x_train.append(x_tr)
            x_test.append(x_te)
            y_train.append(y_tr)
            y_test.append(y_te)
            models[0].fit(x_train, y_train, nb_epoch=self.epochs, batch_size=self.batch_size, verbose=1)

        original_fn = file_name
        for m in range(len(models)):
            if cv_splits > 1:
                file_name = original_fn + "S" + str(m)
            if get_scores:

                y_pred = models[m].predict(x_test[m])
                y_pred[y_pred >= 0.5] = 1
                y_pred[y_pred < 0.5] = 0
                f1_array = []
                accuracy_array = []
                for y in range(len(y_pred[m])):
                    accuracy_array.append(accuracy_score(y_test[m][y], y_pred[y]))
                    f1_array.append(f1_score(y_test[m][y], y_pred[y]))
                cv_f1_fn = "../data/" + data_type + "/nnet/scores/F1 " + file_name + ".txt"
                cv_acc_fn = "../data/" + data_type + "/nnet/scores/ACC " + file_name + ".txt"
                dt.write1dArray(f1_array, cv_f1_fn)
                dt.write1dArray(accuracy_array, cv_acc_fn)
                f1_scores.append(f1_array)
                accuracy_scores.append(accuracy_array)
                f1_average = np.average(f1_array)
                accuracy_average = np.average(accuracy_array)
                f1_averages.append(f1_average)
                accuracy_averages.append(accuracy_average)
                print("Average F1", f1_average, "Acc", accuracy_average)

            if save_outputs:
                self.output_clusters = models[m].predict(entity_vectors)
                dt.write2dArray(self.output_clusters.transpose(), rank_fn)


            for l in range(0, len(models[m].layers) - 1):
                if dropout_noise is not None and dropout_noise > 0.0:
                    if l % 2 == 1:
                        continue
                print("Writing", l, "layer")
                truncated_model = Sequential()
                for a in range(l+1):
                    truncated_model.add(models[m].layers[a])
                truncated_model.compile(loss=self.loss, optimizer="sgd")
                self.end_space = truncated_model.predict(entity_vectors)
                total_file_name = "../data/" + data_type + "/nnet/spaces/" + file_name
                dt.write2dArray(self.end_space, total_file_name + "L" + str(l) + ".txt")

            for l in range(len(models[m].layers)):
                try:
                    dt.write2dArray(models[m].layers[l].get_weights()[0],
                                    "../data/" + data_type + "/nnet/weights/" + file_name + "L" + str(l) + ".txt")
                    dt.write1dArray(models[m].layers[l].get_weights()[1],
                                    "../data/" + data_type + "/nnet/bias/" + file_name + "L" +  str(l) + ".txt")
                except IndexError:
                    print("Layer ", str(l), "Failed")

        if cv_splits > 1:
            class_f1_averages = []
            class_accuracy_averages = []
            f1_scores = np.asarray(f1_scores).transpose()
            accuracy_scores = np.asarray(accuracy_scores).transpose()

            for c in range(len(f1_scores)):
                class_f1_averages.append(np.average(f1_scores[c]))
                class_accuracy_averages.append(np.average(accuracy_scores[c]))

            f1_fn = "../data/" + data_type + "/nnet/scores/F1 " + file_name + ".txt"
            acc_fn = "../data/" + data_type + "/nnet/scores/ACC " + file_name + ".txt"
            dt.write1dArray(class_f1_averages, f1_fn)
            dt.write1dArray(class_accuracy_averages, acc_fn)
            overall_f1_average = np.average(f1_averages)
            overall_accuracy_average = np.average(accuracy_averages)

            print("Overall average F1", overall_f1_average, "Overall average accuracy", overall_accuracy_average)

    def classifierNetwork(self):
        print("CLASSIFIER")
        model = Sequential()
        print(self.input_size, self.hidden_layer_size, self.finetune_size, self.output_size)

        print(0, "Deep layer", self.input_size, self.deep_size[0], self.hidden_activation)
        model.add(Dense(output_dim=self.deep_size[0], input_dim=self.input_size, init=self.layer_init,
                        activation=self.hidden_activation, W_regularizer=l2(self.reg)))

        if self.dropout_noise is not None:
            print("Dropout layer")
            model.add(Dropout(self.dropout_noise))

        for a in range(1, len(self.deep_size)):
            print(a, "Deep layer", self.deep_size[a - 1], self.deep_size[a], self.hidden_activation)
            model.add(Dense(output_dim=self.deep_size[a], input_dim=self.deep_size[a - 1], init=self.layer_init,
                            activation=self.hidden_activation, W_regularizer=l2(self.reg)))
            if self.dropout_noise is not None:
                print("Dropout layer")
                model.add(Dropout(self.dropout_noise))

        print("Class outputs", self.deep_size[len(self.deep_size) - 1], self.output_size,
              self.output_activation)
        model.add(
            Dense(output_dim=self.output_size, input_dim=self.deep_size[len(self.deep_size) - 1],
                  activation=self.output_activation,
                  init=self.layer_init))

        print("Compiling")
        model.compile(loss=self.loss, optimizer=self.optimizer)

        return model

    def fineTuneNetwork(self):
        print("FINETUNER")
        model = Sequential()
        print(self.input_size, self.hidden_layer_size, self.finetune_size, self.output_size)

        #self.model.add(GaussianNoise(0.0, input_shape=(input_size,)))
        
        # If we want to swap the identity layer to before the hidden layer
        if self.identity_swap:
            print("Identity swapped layer", self.input_size, self.hidden_layer_size, self.hidden_activation)
            for a in range(self.amount_of_finetune):
                model.add(Dense(output_dim=self.hidden_layer_size, input_dim=self.input_size,
                                     activation=self.hidden_activation,
                                     init="identity"))


        if self.from_ae:
            for p in range(len(self.past_weights)):
                print(p, "Past AE layer", self.input_size, self.hidden_layer_size, self.hidden_activation)
                model.add(Dense(output_dim=self.hidden_layer_size, input_dim=self.input_size, activation=self.hidden_activation,
                                     weights=self.past_weights[p], W_regularizer=l2(self.reg)))
            if self.dropout_noise is not None:
                print("Dropout layer")
                model.add(Dropout(self.dropout_noise))

        # Add an identity layer that has equal values to the input space to find some more nonlinear relationships
        if self.is_identity:
            print("Identity layer", self.hidden_layer_size, self.hidden_layer_size, self.hidden_activation)
            for a in range(self.amount_of_finetune):
                model.add(Dense(output_dim=self.hidden_layer_size, input_dim=self.hidden_layer_size, activation=self.hidden_activation,
                          init="identity"))
        
        if self.randomize_finetune_weights:
            print("Randomize finetune weights", self.hidden_layer_size, self.finetune_size, "linear")
            model.add(Dense(output_dim=self.finetune_size, input_dim=self.hidden_layer_size, activation="linear",
                                 init=self.layer_init))
        elif self.corrupt_finetune_weights:
            print("Corrupt finetune weights", self.hidden_layer_size, self.finetune_size, "linear")
            model.add(Dense(output_dim=self.finetune_size, input_dim=self.hidden_layer_size, activation="linear",
                                 weights=self.fine_tune_weights))
        else:
            print("Fine tune weights", self.hidden_layer_size, len(self.fine_tune_weights[0][0]), "linear")
            model.add(Dense(output_dim=len(self.fine_tune_weights[0][0]), input_dim=self.hidden_layer_size, activation="linear",
                                 weights=self.fine_tune_weights))
        if self.get_scores:
            if self.randomize_finetune_weights or self.corrupt_finetune_weights or len(self.fine_tune_weights_fn) > 0:
                print("Class outputs", self.finetune_size, self.output_size, self.output_activation)
                model.add(
                    Dense(output_dim=self.output_size, input_dim=self.finetune_size, activation=self.output_activation,
                          init=self.layer_init))

        print("Compiling")
        model.compile(loss=self.loss, optimizer=self.optimizer)

        return model


def main():
    data_type = "wines"
    classification_task = "types"
    #file_name = "wines100trimmed"
    #init_vector_path = "../data/" + data_type + "/nnet/spaces/" + file_name + ".txt"

    file_name = "winesppmi"
    if data_type is "wines" or "placetypes":
        lowest_count = 50
    else:
        lowest_count = 200
    init_vector_path = "../data/"+data_type+"/bow/ppmi/class-all-" + str(lowest_count)

    deep_size = [1000, 500, 250, 100, 50]
    for d in range(len(deep_size)):
        print(deep_size, init_vector_path)
        loss = "binary_crossentropy"
        output_activation = "sigmoid"
        optimizer_name = "adagrad"
        hidden_activation = "relu"
        classification_path = "../data/" + data_type + "/classify/" + classification_task + "/class-all"
        learn_rate = 0.01
        fine_tune_weights_fn = None
        epochs = 200
        amount_of_finetune = 0
        batch_size = 200
        save_outputs = True
        dropout_noise = 0.3
        is_identity = False
        identity_swap = False
        from_ae = False
        cv_splits = 5
        past_model_weights_fn = None
        past_model_bias_fn = None
        randomize_finetune_weights = False
        hidden_layer_size = 100
        output_size = 10
        randomize_finetune_weights = False
        corrupt_finetune_weights = False
        rewrite_files = False
        get_scores = True
        #init_vector_path = "../data/" + data_type + "/movies/bow/binary/phrases/class-all"
        if d == 0:
            file_name = file_name + "rank" + "E" + str(epochs) + "DS" + str(deep_size) + "DN" +\
                        str(dropout_noise) + str(hidden_activation) + "CV" +  str(cv_splits) + "SFT" + str(d)
        else:
            file_name = file_name + "SFT" + str(d)

        print(file_name)

        csv_name = "../data/"+data_type+"/rules/tree_csv/"+file_name+".csv"

        SDA = NeuralNetwork(noise=0, fine_tune_weights_fn=fine_tune_weights_fn, optimizer_name=optimizer_name,
                get_scores=get_scores, past_model_bias_fn=past_model_bias_fn, deep_size=deep_size,
                randomize_finetune_weights=randomize_finetune_weights, amount_of_finetune=amount_of_finetune,
                vector_path=init_vector_path, hidden_layer_size=hidden_layer_size, class_path=classification_path,
                identity_swap=identity_swap, dropout_noise=dropout_noise, save_outputs=save_outputs,
                hidden_activation=hidden_activation, output_activation=output_activation, epochs=epochs,
                learn_rate=learn_rate, is_identity=is_identity, output_size=output_size,
                batch_size=batch_size, past_model_weights_fn=past_model_weights_fn, loss=loss, cv_splits=cv_splits,
                file_name=file_name, from_ae=from_ae, data_type=data_type, rewrite_files=rewrite_files)
        original_fn = file_name
        for splits in range(cv_splits):
            if cv_splits > 1:
                file_name = original_fn + "S" + str(splits)
            new_file_names = []
            if dropout_noise is not None and dropout_noise > 0.0:
                for j in range(0, len(deep_size)*2, 2):
                    new_fn = file_name + "L" + str(j)
                    new_file_names.append(new_fn)
            else:
                for j in range(0, len(deep_size) + 1):
                    new_fn = file_name + "L" + str(j)
                    new_file_names.append(new_fn)

            for j in range(len(new_file_names)):
                file_name = new_file_names[j]

                """ Begin Filename """

                is_identity = True
                breakoff = True
                kappa = False

                file_name = file_name + str(lowest_count)


                if kappa is False:
                    file_name = file_name + "ndcg"
                else:
                    file_name = file_name + "kappa"

                if breakoff:
                    score_limit = 0.9
                    cluster_amt = deep_size[j] * 2
                    file_name = file_name + str(score_limit) + str(cluster_amt)
                else:
                    file_name = file_name + "SimilarityClustering"


                if is_identity:
                    file_name = file_name + "IT"

                epochs = 3000
                file_name = file_name + str(epochs)

                """ Begin Parameters """

                """ SVM """
                svm_type = "svm"
                highest_count = 10000
                vector_path = "../data/" + data_type + "/nnet/spaces/"+new_file_names[j]+".txt"
                bow_path = "../data/" + data_type + "/bow/binary/phrases/class-all-" + str(lowest_count)
                property_names_fn = "../data/" + data_type + "/bow/names/" + str(lowest_count) + ".txt"
                directions_fn = "../data/" + data_type + "/svm/directions/" + file_name + ".txt"
                threads = 4


                """ DIRECTION RANKINGS """
                # Get rankings
                vector_names_fn = "../data/" + data_type + "/nnet/spaces/entitynames.txt"
                class_names_fn = "../data/" + data_type + "/bow/names/" + str(lowest_count) + ".txt"


                """ CLUSTERING """
                # Choosing the score-type
                if kappa is False:
                    scores_fn = "../data/" + data_type + "/ndcg/" + file_name + ".txt"
                else:
                    scores_fn = "../data/" + data_type + "/svm/kappa/" + file_name + str(lowest_count) + ".txt"
                names_fn = "../data/" + data_type + "/bow/names/" + str(lowest_count) + ".txt"

                if breakoff:
                    similarity_threshold = 0.5
                    amount_to_start = 8000
                    dissimilarity_threshold = 0.9
                    add_all_terms = False
                    clusters_fn = "../data/" + data_type + "/cluster/hierarchy_directions/" + file_name + ".txt"
                    cluster_names_fn = "../data/" + data_type + "/cluster/hierarchy_names/" + file_name + ".txt"
                else:
                    high_threshold = 0.5
                    low_threshold = 0.1
                    amt_of_clusters = hidden_layer_size * 2
                    clusters_fn = "../data/" + data_type + "/cluster/clusters/" + file_name + ".txt"
                    cluster_names_fn = "../data/" + data_type + "/cluster/names/" + file_name + ".txt"

                """ CLUSTER RANKING """
                vector_names_fn = "../data/" + data_type + "/nnet/spaces/entitynames.txt"

                # Decision tree
                ranking_fn = "../data/" + data_type + "/rank/numeric/" + file_name + ".txt"
                label_names_fn = "../data/" + data_type + "/classify/" + classification_task + "/names.txt"

                """ FINETUNING """
                fine_tune_weights_fn = [clusters_fn]

                batch_size = 200
                learn_rate = 0.001
                identity_swap = False
                randomize_finetune_weights = False
                from_ae = True
                finetune_size = cluster_amt

                class_path = "../data/" + data_type + "/finetune/" + file_name + ".txt"
                loss = "mse"
                optimizer_name = "sgd"

                hidden_layer_size = deep_size[j]

                past_model_weights_fn = ["../data/" + data_type + "/nnet/weights/" + new_file_names[j] + ".txt"]
                past_model_bias_fn = ["../data/" + data_type + "/nnet/bias/" + new_file_names[j] + ".txt"]

                amount_of_finetune = 1

                """ DECISION TREES FOR NNET RANKINGS """
                nnet_ranking_fn = "../data/" + data_type + "/nnet/clusters/" + file_name + "FT.txt"


                """ Begin Methods """
                print(file_name)
                svm.getSVMResults(vector_path, bow_path, property_names_fn, file_name, lowest_count=lowest_count,
                  highest_count=highest_count, data_type=data_type, get_kappa=True,
                  get_f1=False, svm_type=svm_type, getting_directions=True, threads=4, rewrite_files=rewrite_files)

                if not kappa:
                    rank.getAllPhraseRankings(directions_fn, vector_path, class_names_fn, vector_names_fn, file_name,
                                  data_type=data_type, rewrite_files=rewrite_files)
                    ndcg.getNDCG("../data/" + data_type + "/rank/numeric/" + file_name + "ALL.txt", file_name,
                             data_type=data_type, lowest_count=lowest_count, rewrite_files=rewrite_files)


                if breakoff:
                    hierarchy.initClustering(vector_path, directions_fn, scores_fn, names_fn, amount_to_start, False,
                         similarity_threshold,  cluster_amt, score_limit, file_name, kappa, dissimilarity_threshold,
                                 add_all_terms=add_all_terms, data_type=data_type, rewrite_files=rewrite_files)
                else:
                    cluster.getClusters(directions_fn, scores_fn, class_names_fn, False,  0, 0, file_name, amt_of_clusters,
                                        high_threshold, low_threshold, data_type, rewrite_files=rewrite_files)

                rank.getAllRankings(clusters_fn, vector_path, cluster_names_fn , vector_names_fn, 0.2, 1, False, file_name,
                                    False, data_type=data_type, rewrite_files=rewrite_files)

                tree.DecisionTree(ranking_fn, classification_path, label_names_fn, cluster_names_fn, file_name + str(3), 10000,
                          max_depth=3, balance="balanced", criterion="entropy", save_details=True, cv_splits=cv_splits,
                          data_type=data_type, csv_fn=csv_name, rewrite_files=rewrite_files)

                tree.DecisionTree(ranking_fn, classification_path, label_names_fn, cluster_names_fn, file_name + "None", 10000,
                                      max_depth=None, balance="balanced", criterion="entropy", save_details=False,
                                  data_type=data_type, csv_fn=csv_name, rewrite_files=rewrite_files,
                                  cv_splits=cv_splits)

                fto.pavPPMI(cluster_names_fn, ranking_fn, file_name, data_type=data_type, rewrite_files=rewrite_files)

                file_name = file_name + "FT"

                SDA = NeuralNetwork(noise=0, fine_tune_weights_fn=fine_tune_weights_fn, optimizer_name=optimizer_name,
                            get_scores=get_scores, past_model_bias_fn=past_model_bias_fn,
                            randomize_finetune_weights=randomize_finetune_weights,
                            vector_path=init_vector_path, hidden_layer_size=hidden_layer_size, class_path=class_path,
                            identity_swap=identity_swap, amount_of_finetune=amount_of_finetune,
                            hidden_activation=hidden_activation, output_activation=output_activation, epochs=epochs,
                            learn_rate=learn_rate, is_identity=is_identity, 
                            batch_size=batch_size, past_model_weights_fn=past_model_weights_fn, loss=loss, rewrite_files=rewrite_files,
                            file_name=file_name, from_ae=from_ae, finetune_size=finetune_size, data_type=data_type)

                new_file_names[j] = file_name

                tree.DecisionTree(nnet_ranking_fn, classification_path, label_names_fn, cluster_names_fn, file_name + str(3), 10000,
                                  3, balance="balanced", criterion="entropy", save_details=True, cv_splits=cv_splits,
                                  data_type=data_type, csv_fn=csv_name, rewrite_files=rewrite_files)

                tree.DecisionTree(nnet_ranking_fn, classification_path, label_names_fn, cluster_names_fn, file_name + "None", 10000,
                                  None, balance="balanced", criterion="entropy", save_details=False, cv_splits=cv_splits,
                                  data_type=data_type, csv_fn=csv_name, rewrite_files=rewrite_files)

                init_vector_path = vector_path

            file_name = new_file_names[0]
            init_vector_path = "../data/" + data_type + "/nnet/spaces/" + file_name + "L0.txt"
            deep_size = deep_size[:len(deep_size)-1]


if  __name__ =='__main__':main()