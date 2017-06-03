# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import numpy as np
import data as dt
from keras.regularizers import l2
from keras.optimizers import SGD, Adagrad, Adam, RMSprop, Adadelta, Adamax, Nadam
from sklearn.metrics import f1_score, accuracy_score
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import cluster
import rank
import finetune_outputs as fto
import svm
import tree
import hierarchy
import ndcg
import nnet
import wekatree
from sklearn.model_selection import KFold
from sklearn.preprocessing import MultiLabelBinarizer
import weka.core.jvm as jvm
import random
import sys
import time

jvm.start(max_heap_size="512m")
def main(data_type, classification_task, file_name, init_vector_path, hidden_activation, is_identity, amount_of_finetune,
         breakoff, kappa, score_limit, rewrite_files, cluster_multiplier, threads, dropout_noise, learn_rate, epochs, cross_val, ep,
         output_activation, cs, deep_size, classification, direction_count, lowest_amt, loss, development, add_all_terms,
         average_ppmi, optimizer_name, class_weight, amount_to_start, chunk_amt, chunk_id, lr, vector_path_replacement, dt_dev,
         use_pruned, max_depth, min_score, min_size, limit_entities, svm_classify, get_nnet_vectors_path, arcca, loc, largest_cluster,
         skip_nn):


    #jvm.start(max_heap_size="512m")
    if isinstance(deep_size, str):
        epochs = int(epochs)
        ep = int(ep)
        dropout_noise = float(dropout_noise)
        cluster_multiplier = float(cluster_multiplier)
        learn_rate = float(learn_rate)
        cross_val = int(cross_val)
        lowest_amt = int(lowest_amt)
        threads = int(threads)
        amount_to_start = int(amount_to_start)
        deep_size = dt.stringToArray(deep_size)
        if class_weight != 'None':
            class_weight = dt.stringToArray(class_weight)
        else:
            class_weight = None
        rewrite_files = dt.toBool(rewrite_files)
        development = dt.toBool(development)
        is_identity = dt.toBool(is_identity)
        average_ppmi = dt.toBool(average_ppmi)
        breakoff = dt.toBool(breakoff)
        score_limit = float(score_limit)
        amount_of_finetune = int(amount_of_finetune)
        svm_classify = dt.toBool(svm_classify)
        limit_entities = dt.toBool(limit_entities)
        min_size = int(min_size)
        min_score = float(min_score)
        max_depth = int(max_depth)
        use_pruned = dt.toBool(use_pruned)
        dt_dev = dt.toBool(dt_dev)
        lr = float(lr)
        chunk_id = int(chunk_id)
        largest_cluster = int(largest_cluster)
        chunk_amt = int(chunk_amt)
        cs = float(cs)
        arcca = dt.toBool(arcca)
        amount_to_start = int(amount_to_start)
        kappa = dt.toBool(kappa)
        add_all_terms = dt.toBool(add_all_terms)
        if vector_path_replacement == 'None':
            vector_path_replacement = None
        if get_nnet_vectors_path == 'None':
            get_nnet_vectors_path = None
        skip_nn = dt.toBool(skip_nn)


    cv_splits = cross_val
    csv_fns_dt_a = []
    csv_fns_nn_a = []

    copy_size = np.copy(deep_size)
    while len(copy_size) is not 1:
        for d in range(len(copy_size)):
            csv_fns_dt_a.append([])
        copy_size = copy_size[1:]
    csv_fns_dt_a.append([])


    for d in range(len(deep_size)):
        csv_fns_nn_a.append([])

    split_fns = []
    for s in range(cv_splits):
        if skip_nn is False:
            fn = file_name + " E" + str(ep) + " DS" + str(deep_size) + " DN" + str(dropout_noise) + " CT" + classification_task + \
                        " HA" + str(hidden_activation) + " CV" + str(cv_splits)  +  " S" + str(s) + " Dev" + str(development) + " LE" + str(limit_entities)
        else:
            fn = file_name  + " CV" + str(cv_splits)  +  " S" + str(s) + " LE" + str(limit_entities)

        split_fns.append(fn)
    original_deep_size = deep_size
    for splits in range(cv_splits):
        deep_size = original_deep_size
        file_name = split_fns[0]
        csv_fns_dt = []
        csv_fns_nn = []
        copy_size = np.copy(deep_size)
        nn_counter = 0
        while len(copy_size) is not 1:
            for d in range(len(copy_size)):
                csv_fns_dt.append("")
            copy_size = copy_size[1:]
        csv_fns_dt.append("")
        for d in range(len(deep_size)):
            csv_fns_nn.append([])
        data_type = data_type
        threads = threads
        classification_task = classification_task
        if data_type == "wines" or data_type == "placetypes":
            lowest_amt = 50
        else:
            lowest_amt = 100
        print(file_name)
        print("SPLIT", str(splits), rewrite_files, arcca)

        deep_fns = []
        for s in range(len(deep_size)):
            deep_fns.append(split_fns[splits] + " SFT" + str(s))
        csv_fns = []
        counter = 0
        for d in range(len(deep_size)):
            file_name = deep_fns[d]
            print(deep_size, init_vector_path)
            loss = loss
            output_activation = output_activation
            optimizer_name = optimizer_name
            hidden_activation = hidden_activation
            classification_path = loc + data_type + "/classify/" + classification_task + "/class-all"
            label_names_fn = loc + data_type + "/classify/" + classification_task + "/names.txt"
            fine_tune_weights_fn = None
            ep = ep
            batch_size = 200
            save_outputs = True
            dropout_noise = dropout_noise
            identity_swap = False
            from_ae = False
            past_model_weights_fn = None
            past_model_bias_fn = None
            randomize_finetune_weights = False
            hidden_layer_size = 100
            output_size = 10
            randomize_finetune_weights = False
            corrupt_finetune_weights = False
            get_scores = True

            csv_fns_nn[nn_counter] = loc + data_type + "/nnet/csv/" + file_name + ".csv"
            nn_counter+=1
            print("nnet hi", arcca)
            if not arcca and not skip_nn:
                print ("nnet hello?")
                SDA = nnet.NeuralNetwork(noise=0, fine_tune_weights_fn=fine_tune_weights_fn, optimizer_name=optimizer_name,
                    get_scores=get_scores, past_model_bias_fn=past_model_bias_fn, deep_size=deep_size,  cutoff_start=cs,
                    randomize_finetune_weights=randomize_finetune_weights, amount_of_finetune=amount_of_finetune,
                    vector_path=init_vector_path, hidden_layer_size=hidden_layer_size, class_path=classification_path,
                    identity_swap=identity_swap, dropout_noise=dropout_noise, save_outputs=save_outputs,
                    hidden_activation=hidden_activation, output_activation=output_activation, epochs=ep,
                    learn_rate=lr, is_identity=is_identity, output_size=output_size, split_to_use=splits, label_names_fn=label_names_fn,
                    batch_size=batch_size, past_model_weights_fn=past_model_weights_fn, loss=loss, cv_splits=cv_splits, csv_fn = file_name,
                    file_name=file_name, from_ae=from_ae, data_type=data_type, rewrite_files=rewrite_files, development=development,
                                         class_weight=class_weight, get_nnet_vectors_path=get_nnet_vectors_path)

            new_file_names = []

            name_amt = len(deep_size)
            if dropout_noise is not None and dropout_noise > 0.0:
                for j in range(0, name_amt*2, 2):
                    new_fn = file_name + "L" + str(j)
                    new_file_names.append(new_fn)
            else:
                for j in range(0, name_amt + 1):
                    new_fn = file_name + "L" + str(j)
                    new_file_names.append(new_fn)

            #for j in range(len(new_file_names)):
            for x in range(len(deep_size)):
            #for x in range(len([0])):

                if limit_entities is False:
                    new_classification_task = "all"
                else:
                    new_classification_task = classification_task
                file_name = new_file_names[x]

                if vector_path_replacement is not None:
                    vector_path = vector_path_replacement
                """ Begin Filename """

                is_identity = is_identity
                breakoff = breakoff
                kappa = kappa

                file_name = file_name + str(lowest_amt)

                """ Begin Parameters """
                """ SVM """
                svm_type = "svm"
                highest_count = direction_count

                if vector_path_replacement is None:
                    vector_path = loc + data_type + "/nnet/spaces/"+new_file_names[x]+".txt"
                #vector_path = loc + data_type + "/nnet/spaces/"+vector_path_replacement+".txt"
                bow_path = loc + data_type + "/bow/binary/phrases/class-all-"+str(lowest_amt)+"-"+str(highest_count)+"-"+new_classification_task
                property_names_fn = loc + data_type + "/bow/names/" + str(lowest_amt) + "-" +str(highest_count)+"-"+ new_classification_task +".txt"
                directions_fn = loc + data_type + "/svm/directions/" + file_name + ".txt"


                """ DIRECTION RANKINGS """
                # Get rankings
                vector_names_fn = loc + data_type + "/nnet/spaces/entitynames.txt"
                class_names_fn = property_names_fn

                cluster_amt = deep_size[x] * cluster_multiplier

                """ Begin Methods """
                print(file_name)
                final_fn = ""
                """ CLUSTERING """
                # Choosing the score-type
                if breakoff:
                    score_limit = score_limit

                    final_fn = file_name + str(score_limit) + str(cluster_amt)
                    if add_all_terms:
                        final_fn = final_fn + "AllTerms"
                else:
                    final_fn = final_fn + " SimilarityClustering"
                names_fn = property_names_fn
                if skip_nn is False:
                    if average_ppmi:
                        final_fn = final_fn + "APPMI"

                    if is_identity:
                        final_fn = final_fn + " IT"

                    epochs = epochs
                    final_fn = final_fn + str(epochs)
                    final_fn = final_fn + "FT"
                    final_fn = final_fn + "L0"
                if limit_entities is False:
                    old_vp = vector_path
                    vector_path = get_nnet_vectors_path
                print(file_name)
                svm.createSVM(vector_path, bow_path, property_names_fn, file_name, lowest_count=lowest_amt,
                  highest_count=highest_count, data_type=data_type, get_kappa=kappa,
                  get_f1=False, svm_type=svm_type, getting_directions=True, threads=threads, rewrite_files=rewrite_files,
                              classification=new_classification_task, lowest_amt=lowest_amt, chunk_amt=chunk_amt, chunk_id=chunk_id)


                if chunk_amt > 0:
                    if chunk_id == chunk_amt-1:
                        dt.compileSVMResults(file_name, chunk_amt, data_type)

                    else:
                        if d != len(deep_fns)-1:
                            randomcount = 0
                            while not dt.fileExists(loc+data_type+"/nnet/spaces/"+final_fn+".txt"):
                                randomcount += 1
                            print(randomcount)
                            time.sleep(10)
                        else:
                            print("exit")
                            if d != len(deep_fns)-1:
                                while not dt.fileExists(loc+data_type+"/nnet/spaces/"+deep_fns[d+1]+".txt"):
                                    time.sleep(10)

                if chunk_id == chunk_amt -1 or chunk_amt <= 0:
                    if not kappa:
                        rank.getAllPhraseRankings(directions_fn, vector_path, class_names_fn, vector_names_fn, file_name,
                                      data_type=data_type, rewrite_files=rewrite_files)
                        ndcg.getNDCG(loc + data_type + "/rank/numeric/" + file_name + "ALL.txt", file_name,
                                 data_type=data_type, lowest_count=lowest_amt, rewrite_files=rewrite_files,
                                     highest_count=highest_count, classification=new_classification_task)
                    if kappa is False:
                        scores_fn = loc + data_type + "/ndcg/" + file_name + ".txt"
                        file_name = file_name + "ndcg"
                    else:
                        scores_fn = loc + data_type + "/svm/kappa/" + file_name + ".txt"
                        file_name = file_name + "kappa"

                    """ CLUSTERING """
                    # Choosing the score-type
                    if breakoff:
                        score_limit = score_limit

                        file_name = file_name + str(score_limit) + str(cluster_amt)
                        if add_all_terms:
                            file_name = file_name + "AllTerms"
                    else:
                        file_name = file_name + " SimilarityClustering"
                    file_name = file_name + "MC" + str(min_size) + "MS" + str(min_score)
                    names_fn = property_names_fn

                    if breakoff:
                        similarity_threshold = 0.5
                        amount_to_start = amount_to_start
                        dissimilarity_threshold = 0.9
                        add_all_terms = add_all_terms
                        clusters_fn = loc + data_type + "/cluster/hierarchy_directions/" + file_name + ".txt"
                        cluster_names_fn = loc + data_type + "/cluster/hierarchy_names/" + file_name + ".txt"
                    else:
                        high_threshold = 0.5
                        low_threshold = 0.1
                        clusters_fn = loc + data_type + "/cluster/clusters/" + file_name + ".txt"
                        cluster_names_fn = loc + data_type + "/cluster/names/" + file_name + ".txt"

                    if breakoff:
                        hierarchy.initClustering(vector_path, directions_fn, scores_fn, names_fn, amount_to_start, False,
                             similarity_threshold,  cluster_amt, score_limit, file_name, kappa, dissimilarity_threshold,
                                     add_all_terms=add_all_terms, data_type=data_type, rewrite_files=rewrite_files,
                                                 lowest_amt=lowest_amt, highest_amt=highest_count, classification=new_classification_task,
                                                 min_score=min_score, min_size = min_size, largest_clusters=largest_cluster)
                    else:
                        cluster.getClusters(directions_fn, scores_fn, names_fn, False,  0, 0, file_name, cluster_amt,
                                            high_threshold, low_threshold, data_type, rewrite_files=rewrite_files)

                    """ CLUSTER RANKING """
                    if limit_entities is False:
                        limited_label_fn = loc + data_type + "/classify/"+classification_task+"/available_entities.txt"
                    else:
                        vector_names_fn = loc + data_type + "/classify/"+classification_task+"/available_entities.txt"
                        limited_label_fn = None
                    ranking_fn = loc + data_type + "/rank/numeric/" + file_name + ".txt"


                    csv_name = loc + data_type + "/rules/tree_csv/" + file_name + ".csv"

                    csv_fns_dt[counter] = csv_name
                    counter+=1

                    rank.getAllRankings(clusters_fn, vector_path, cluster_names_fn , vector_names_fn, 0.2, 1, False, file_name,
                                        False, data_type=data_type, rewrite_files=rewrite_files)
                    if skip_nn:
                        file_name = file_name + " " + classification_task

                    if dt_dev:
                        file_name = file_name + " tdev"

                    file_name = file_name + str(max_depth)
                    tree.DecisionTree(ranking_fn, classification_path, label_names_fn, cluster_names_fn, file_name, 10000,
                              max_depth=max_depth, balance="balanced", criterion="entropy", save_details=True, cv_splits=cv_splits, split_to_use=splits,
                              data_type=data_type, csv_fn=csv_name, rewrite_files=rewrite_files, development=dt_dev, limit_entities=limit_entities,
                                      limited_label_fn=limited_label_fn, vector_names_fn=vector_names_fn)

                    tree.DecisionTree(ranking_fn, classification_path, label_names_fn, cluster_names_fn, file_name + "None", 10000,
                                          max_depth=None, balance="balanced", criterion="entropy", save_details=False,
                                      data_type=data_type, csv_fn=csv_name, rewrite_files=rewrite_files,
                                      cv_splits=cv_splits, split_to_use=splits, development=dt_dev, limit_entities=limit_entities,
                                      limited_label_fn=limited_label_fn, vector_names_fn=vector_names_fn)

                    wekatree.DecisionTree(ranking_fn, classification_path, label_names_fn , cluster_names_fn , file_name,
                       save_details=True, data_type=data_type,split_to_use=splits,
                                      limited_label_fn=limited_label_fn,
                       csv_fn=csv_name, cv_splits=cv_splits, limit_entities=limit_entities, vector_names_fn=vector_names_fn)

                    break

                    if use_pruned:
                        clusters_fn = loc+ data_type + "/rules/clusters/" + file_name + ".txt"
                        cluster_names_fn = loc+ data_type + "/rules/names/" + file_name + ".txt"

                    classes = dt.import1dArray(label_names_fn)
                    current_fn = file_name

                    # Use an SVM to classify each of the classes
                    if svm_classify:
                        for c in classes:
                            print(c)
                            file_name = current_fn + c
                            class_c_fn = loc+ data_type + "/rules/clusters/" + file_name + ".txt"
                            class_n_fn = loc+ data_type + "/rules/names/" + file_name + ".txt"
                            rank.getAllRankings(class_c_fn, vector_path, class_n_fn , vector_names_fn, 0.2, 1, False, file_name,
                                                False, data_type=data_type, rewrite_files=rewrite_files)
                            class_rank_fn = loc+ data_type + "/rank/numeric/" + file_name + ".txt"
                            class_p_fn = loc + data_type + "/classify/" +  classification_task + "/class-" + c
                            svm.createSVM(class_rank_fn, class_p_fn, class_n_fn, file_name, lowest_count=lowest_amt,
                                      highest_count=highest_count, data_type=data_type, get_kappa=False,
                                      get_f1=True, single_class=True,svm_type=svm_type, getting_directions=False, threads=1,
                                      rewrite_files=rewrite_files,
                                      classification=classification, lowest_amt=lowest_amt, chunk_amt=chunk_amt,
                                      chunk_id=chunk_id)


                    file_name = current_fn

                    limited_rankings_fn = loc+ data_type + "/rank/numeric/" + file_name + ".txt"
                    """
                    svm.createSVM(limited_rankings_fn, classification_path, class_names_fn, file_name, lowest_count=lowest_amt,
                                  highest_count=highest_count, data_type=data_type, get_kappa=False,
                                  get_f1=True, single_class=True,svm_type=svm_type, getting_directions=False, threads=1,
                                  rewrite_files=rewrite_files,
                                  classification=classification, lowest_amt=lowest_amt, chunk_amt=chunk_amt,
                                  chunk_id=chunk_id)
                    """
                    """ Testing consistency
                    tree.DecisionTree(ranking_fn, classification_path, label_names_fn, cluster_names_fn,
                                      file_name, 10000,
                                      max_depth=max_depth, balance="balanced", criterion="entropy", save_details=False,
                                      cv_splits=cv_splits, split_to_use=splits,
                                      data_type=data_type, csv_fn=csv_name, rewrite_files=True,
                                      development=dt_dev)
                    """

                    # Decision tree

                    if average_ppmi:
                        file_name = file_name + "APPMI"

                    class_path = loc + data_type + "/finetune/" + file_name + ".txt"

                    if average_ppmi:
                        fto.pavPPMIAverage(cluster_names_fn, ranking_fn, file_name, data_type=data_type, rewrite_files=rewrite_files,
                                classification=classification, lowest_amt=lowest_amt, limit_entities=limit_entities,highest_amt=highest_count)
                    else:
                        fto.pavPPMI(cluster_names_fn, ranking_fn, file_name, data_type=data_type, rewrite_files=rewrite_files,
                                classification=classification, lowest_amt=lowest_amt, limit_entities=limit_entities,highest_amt=highest_count)

                    """ FINETUNING """

                    if is_identity:
                        file_name = file_name + " IT" + str(amount_of_finetune)

                    epochs = epochs
                    file_name = file_name + str(epochs)

                    fine_tune_weights_fn = [clusters_fn]

                    batch_size = 200
                    learn_rate = learn_rate
                    identity_swap = False
                    randomize_finetune_weights = False
                    from_ae = True
                    finetune_size = cluster_amt

                    loss = "mse"
                    optimizer_name = "sgd"

                    hidden_layer_size = deep_size[x]

                    past_model_weights_fn = [loc + data_type + "/nnet/weights/" + new_file_names[x] + ".txt"]
                    past_model_bias_fn = [loc + data_type + "/nnet/bias/" + new_file_names[x] + ".txt"]

                    """ DECISION TREES FOR NNET RANKINGS """
                    nnet_ranking_fn = loc + data_type + "/nnet/clusters/" + file_name + "FT.txt"

                    csv_name = loc + data_type + "/rules/tree_csv/" + file_name + ".csv"

                    file_name = file_name + "FT"
                    if  arcca is False:

                        SDA = nnet.NeuralNetwork(noise=0, fine_tune_weights_fn=fine_tune_weights_fn, optimizer_name=optimizer_name,
                                    past_model_bias_fn=past_model_bias_fn, save_outputs=True,
                                    randomize_finetune_weights=randomize_finetune_weights,
                                    vector_path=init_vector_path, hidden_layer_size=hidden_layer_size, class_path=class_path,
                                    identity_swap=identity_swap, amount_of_finetune=amount_of_finetune,
                                    hidden_activation=hidden_activation, output_activation=output_activation, epochs=epochs,
                                    learn_rate=learn_rate, is_identity=is_identity, batch_size=batch_size,
                                    past_model_weights_fn=past_model_weights_fn, loss=loss, rewrite_files=rewrite_files,
                                    file_name=file_name, from_ae=from_ae, finetune_size=finetune_size, data_type=data_type,
                                               get_nnet_vectors_path= get_nnet_vectors_path)

                        new_file_names[x] = file_name

                        tree.DecisionTree(nnet_ranking_fn, classification_path, label_names_fn, cluster_names_fn, file_name, 10000,
                                              max_depth=max_depth, balance="balanced", criterion="entropy", save_details=True,
                                          data_type=data_type, csv_fn=csv_name, rewrite_files=rewrite_files,
                                          cv_splits=cv_splits, split_to_use=splits, development=dt_dev, limit_entities=limit_entities,
                                          limited_label_fn=limited_label_fn, vector_names_fn=vector_names_fn)

                        tree.DecisionTree(nnet_ranking_fn, classification_path, label_names_fn, cluster_names_fn, file_name + "None", 10000,
                                              max_depth=None, balance="balanced", criterion="entropy", save_details=False,
                                          data_type=data_type, csv_fn=csv_name, rewrite_files=rewrite_files,
                                          cv_splits=cv_splits, split_to_use=splits, development=dt_dev, limit_entities=limit_entities,
                                          limited_label_fn=limited_label_fn, vector_names_fn=vector_names_fn)

                        wekatree.DecisionTree(nnet_ranking_fn, classification_path, label_names_fn, cluster_names_fn, file_name,
                                              save_details=True, data_type=data_type,split_to_use=splits,
                                              csv_fn=csv_name, cv_splits=cv_splits, limit_entities=limit_entities,
                                              limited_label_fn=limited_label_fn, vector_names_fn=vector_names_fn)

                        current_fn = file_name
                        #SVM Classification
                        if svm_classify:
                            for c in classes:
                                print(c)
                                file_name = current_fn + c
                                class_c_fn = loc + data_type + "/rules/clusters/" + file_name + ".txt"
                                class_n_fn = loc + data_type + "/rules/names/" + file_name + ".txt"
                                rank.getAllRankings(class_c_fn, vector_path, class_n_fn, vector_names_fn, 0.2, 1, False,
                                                    file_name,
                                                    False, data_type=data_type, rewrite_files=rewrite_files)
                                class_rank_fn = loc+ data_type + "/rank/numeric/" + file_name + ".txt"
                                class_p_fn = loc + data_type + "/classify/" +  classification_task + "/class-" + c
                                svm.createSVM(class_rank_fn, class_p_fn, class_n_fn, file_name, lowest_count=lowest_amt,
                                          highest_count=highest_count, data_type=data_type, get_kappa=False,
                                          get_f1=True, single_class=True,svm_type=svm_type, getting_directions=False, threads=1,
                                          rewrite_files=rewrite_files,
                                          classification=classification, lowest_amt=lowest_amt, chunk_amt=chunk_amt,
                                          chunk_id=chunk_id)
                        file_name = current_fn
                        rank.getAllRankings(clusters_fn, vector_path, cluster_names_fn, vector_names_fn, 0.2, 1, False,
                                            file_name,
                                            False, data_type=data_type, rewrite_files=rewrite_files)
                        csv_name = loc + data_type + "/rules/tree_csv/" + file_name + "TopDT.csv"
                        tree.DecisionTree(nnet_ranking_fn, classification_path, label_names_fn, cluster_names_fn, file_name + "None", 10000,
                                              max_depth=None, balance="balanced", criterion="entropy", save_details=False,
                                          data_type=data_type, csv_fn=csv_name, rewrite_files=rewrite_files,
                                          cv_splits=cv_splits, split_to_use=splits, development=dt_dev, limit_entities=limit_entities,
                                          limited_label_fn=limited_label_fn, vector_names_fn=vector_names_fn)
                        csv_name = loc + data_type + "/rules/tree_csv/" + file_name + "TopDTJ48.csv"
                        wekatree.DecisionTree(ranking_fn, classification_path, label_names_fn, cluster_names_fn, file_name,
                                              save_details=True, data_type=data_type,split_to_use=splits,
                                              csv_fn=csv_name, cv_splits=cv_splits, limit_entities=limit_entities,
                                              limited_label_fn=limited_label_fn, vector_names_fn=vector_names_fn)
                    if limit_entities is False:
                        vector_path = old_vp

                    if len(new_file_names) > 1:
                        init_vector_path = vector_path
            if len(deep_size) > 1:
                init_vector_path = loc + data_type + "/nnet/spaces/" + new_file_names[0] + "L0.txt"
                deep_size = deep_size[1:]
        print("GETTING FNS")
        for a in range(len(csv_fns_dt)):
            csv_fns_dt_a[a].append(csv_fns_dt[a])
        if not skip_nn:
            for a in range(len(csv_fns_nn)):
                csv_fns_nn_a[a].append(csv_fns_nn[a])

    for a in range(len(csv_fns_dt_a)):
        dt.averageCSVs(csv_fns_dt_a[a])
    if not skip_nn:
        for a in range(len(csv_fns_nn_a)):
            dt.averageCSVs(csv_fns_nn_a[a])
    #jvm.stop()


arcca = False
if arcca:
    loc = "/scratch/c1214824/data/"
else:
    loc = "../data/"

"""
data_type = "wines"
classification_task = "types"
file_name = "wines ppmi"
lowest_amt = 50
highest_amt = 10
init_vector_path = loc+data_type+"/nnet/spaces/wines100-"+classification_task+".txt"
"""

data_type = "movies"
classification_task = "ratings"
file_name = "movies pca 100"
lowest_amt = 100
highest_amt = 10
init_vector_path = loc+data_type+"/pca/class-all-100-10-genresd100"
#init_vector_path = loc+data_type+"/nnet/spaces/films200-"+classification_task+".txt"
#file_name = "films200-genres100ndcg0.85200 tdev3004FTL0"
#init_vector_path = loc+data_type+"/nnet/spaces/"+file_name+".txt"
vector_path_replacement = loc+data_type+"/pca/class-all-100-10-genresd100"
"""
data_type = "placetypes"
classification_task = "opencyc"
lowest_amt = 50
highest_amt = 10
#init_vector_path = "../data/"+data_type+"/bow/ppmi/class-all-"+str(lowest_amt)+"-"+str(highest_amt)+"-"+classification_task
#file_name = "placetypes bow"
init_vector_path = "../data/"+data_type+"/nnet/spaces/places100-"+classification_task+".txt"

vector_path_replacement = loc+data_type+"/pca/spaces/class-all-50-10-alld100.txt"

file_name = "places pca 100"
limit_entities = False
if limit_entities:
    get_nnet_vectors_path = None
else:
    get_nnet_vectors_path = loc + data_type +"/pca/class-all-50-10-alld100"
"""
"""
hidden_activation = "tanh"
dropout_noise = 0.6
output_activation = "softmax"
trainer = "adadelta"
loss="categorical_crossentropy"
class_weight = None
deep_size = [100]
ep =2000
lr = 0.01
vector_path_replacement = "films100-ratings"
nnet_dev = False
"""

"""
hidden_activation = "tanh"
dropout_noise = 0.2
output_activation = "softmax"
cutoff_start = 0.2
deep_size = [100]
init_vector_path = loc+data_type+"/bow/ppmi/class-all-"+str(lowest_amt)+"-"+str(highest_amt)+"-"+classification_task
ep =100
lr = 0.1
class_weight = "balanced"
trainer = "rmsprop"
loss="categorical_crossentropy"
class_weight = "balanced"
rewrite_files = True
"""

hidden_activation = "tanh"
dropout_noise = 0.5
output_activation = "sigmoid"
trainer = "adagrad"
loss="binary_crossentropy"
class_weight = None
deep_size = [200]
ep =100
lr = 0.01
rewrite_files = False
nnet_dev = False

limit_entities = False

cutoff_start = 0.2
is_identity = True
amount_of_finetune = 1

min_size = 1

min_score = 0.6
largest_cluster = 2
breakoff = True
score_limit = 0.94
cluster_multiplier =2
kappa = False
dt_dev = True
add_all_terms = False
average_ppmi = False
use_pruned = False
svm_classify = False
rewrite_files = False
epochs=3002
learn_rate=0.001
max_depth = 3

limit_entities = False
get_nnet_vectors_path = loc+data_type+"/pca/class-all-100-10-genresd100"

amount_to_start = 0
skip_nn = True
cross_val = 1


threads=30
chunk_amt = 0
chunk_id = 0
for c in range(chunk_amt):
    chunk_id = c
    variables = [data_type, classification_task, file_name, init_vector_path, hidden_activation,
                                   is_identity, amount_of_finetune, breakoff, kappa, score_limit, rewrite_files,
                                   cluster_multiplier, threads, dropout_noise, learn_rate, epochs, cross_val, ep,
                                   output_activation, cutoff_start, deep_size, classification_task, highest_amt,
                                   lowest_amt, loss, nnet_dev, add_all_terms, average_ppmi, trainer, class_weight,
                                   amount_to_start, chunk_amt, chunk_id, lr, vector_path_replacement, dt_dev, use_pruned, max_depth,
                                   min_score, min_size, limit_entities, svm_classify, get_nnet_vectors_path, arcca, largest_cluster,
                 skip_nn]

    sys.stdout.write("python pipeline.py ")
    variable_string = "python $SRCPATH/pipeline.py "
    filename_variables = ""
    counter = 0
    for v in variables:
        new_v = dt.stripPunctuation(str(v))
        if len(new_v) < 15 and counter > 5:
            filename_variables = filename_variables + str(new_v) + " "
        if type(v) == str:
            v = '"' + v + '"'
        if type(v) == list:
            v = '"' + str(v) + '"'
        sys.stdout.write(str(v) + " ")
        variable_string += str(v) + " "
        counter += 1

    manual_write_cmd_flag = True
    if manual_write_cmd_flag:
        dt.write1dLinux(["#!/bin/bash",
                         "#PBS -l select=1:ncpus=3:mem=8gb",
                         "#PBS -l walltime=05:00:00",
                         "#PBS -N svm",
                         "#PBS -q serial",
                         "#PBS -P PR338",
                         "module load python/3.5.1-comsc",
                         "SRCPATH=/scratch/$USER/src",
                         "WDPATH=/scratch/$USER/$PBS_JOBID",
                         "mkdir -p $WDPATH",
                         "cd $WDPATH",
                         "export PYTHONPATH=$SRCPATH",
                         variable_string], "../data/" + data_type + "/cmds/" + "pipelinesvm" +str(c) + ".sh")

print("")
args = sys.argv[1:]
if len(args) > 0:
    data_type = args[0]
    classification_task = args[1]
    file_name = args[2]
    init_vector_path = args[3]
    hidden_activation = args[4]
    is_identity = args[5]
    amount_of_finetune = args[6]
    breakoff = args[7]
    kappa = args[8]
    score_limit = args[9]
    rewrite_files = args[10]
    cluster_multiplier = args[11]
    threads = args[12]
    dropout_noise = args[13]
    learn_rate = args[14]
    epochs = args[15]
    cross_val = args[16]
    ep = args[17]
    output_activation = args[18]
    cutoff_start = args[19]
    deep_size = args[20]
    classification_task = args[21]
    highest_amt = args[22]
    lowest_amt = args[23]
    loss = args[24]
    nnet_dev = args[25]
    add_all_terms = args[26]
    average_ppmi = args[27]
    trainer = args[28]
    class_weight = args[29]
    amount_to_start = args[30]
    chunk_amt = args[31]
    chunk_id = args[32]
    lr = args[33]
    vector_path_replacement = args[34]
    dt_dev = args[35]
    use_pruned = args[36]
    max_depth = args[37]
    min_score = args[38]
    min_size = args[39]
    limit_entities = args[40]
    svm_classify = args[41]
    get_nnet_vectors_path = args[42]
    arcca = args[43]
    largest_cluster = args[44]
    skip_nn = args[45]




if  __name__ =='__main__':main(data_type, classification_task, file_name, init_vector_path, hidden_activation,
                               is_identity, amount_of_finetune, breakoff, kappa, score_limit, rewrite_files,
                               cluster_multiplier, threads, dropout_noise, learn_rate, epochs, cross_val, ep,
                               output_activation, cutoff_start, deep_size, classification_task, highest_amt,
                               lowest_amt, loss, nnet_dev, add_all_terms, average_ppmi, trainer, class_weight,
                               amount_to_start, chunk_amt, chunk_id, lr, vector_path_replacement, dt_dev, use_pruned, max_depth,
                               min_score, min_size, limit_entities, svm_classify, get_nnet_vectors_path, arcca, loc, largest_cluster, skip_nn)
