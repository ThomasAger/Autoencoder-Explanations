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

def main(data_type, classification_task_a, file_name, init_vector_path, hidden_activation, is_identity_a, amount_of_finetune_a,
         breakoff_a, kappa_a, score_limit_a, rewrite_files, cluster_multiplier_a, threads, dropout_noise, learn_rate_a, epochs_a, cross_val, ep,
         output_activation, cs, deep_size, classification, direction_count, lowest_amt, loss, development, add_all_terms_a,
         average_ppmi_a, optimizer_name, class_weight, amount_to_start_a, chunk_amt, chunk_id, lr, vector_path_replacement, dt_dev,
         use_pruned, max_depth, min_score, min_size, limit_entities_a, svm_classify, get_nnet_vectors_path, arcca, loc, largest_cluster,
         skip_nn, dissim, dissim_amt_a, hp_opt, find_most_similar, use_breakoff_dissim_a, get_all_a, half_ndcg_half_kappa_a,
         sim_t, one_for_all, ft_loss_a, ft_optimizer_a, bag_of_clusters_a, just_output):


    prune_val = 2

    average_csv_fn = file_name

    if isinstance(deep_size, str):
        if not hp_opt:
            dissim_amt_a = [dt.stringToArray(dissim_amt_a)[0]]
            breakoff_a = [dt.stringToArray(breakoff_a)[0]]
            score_limit_a = [dt.stringToArray(score_limit_a)[0]]
            amount_to_start_a = [dt.stringToArray(amount_to_start_a)[0]]
            cluster_multiplier_a = [dt.stringToArray(cluster_multiplier_a)[0]]
            kappa_a = dt.stringToArray(kappa_a)[0]
            classification_task_a = dt.stringToArray(classification_task_a)[0]
            use_breakoff_dissim_a = dt.stringToArray(use_breakoff_dissim_a)[0]
            get_all_a = dt.stringToArray(get_all_a)[0]
            half_ndcg_half_kappa_a = dt.stringToArray(half_ndcg_half_kappa_a)[0]
            learn_rate_a = dt.stringToArray(learn_rate_a)[0]
            ft_loss_a = dt.stringToArray(ft_loss_a)[0]
            ft_optimizer_a = dt.stringToArray(ft_optimizer_a)[0]
            is_identity_a = dt.stringToArray(is_identity_a)[0]
            amount_of_finetune = dt.stringToArray(amount_of_finetune_a)[0]
            epochs_a = dt.stringToArray(epochs_a)[0]
            average_ppmi_a = dt.stringToArray(average_ppmi_a)[0]
            limit_entities_a = dt.stringToArray(limit_entities_a)[0]
            bag_of_clusters_a = dt.stringToArray(bag_of_clusters_a)[0]
            add_all_terms_a = dt.stringToArray(add_all_terms_a)[0]
        else:
            dissim_amt_a = dt.stringToArray(dissim_amt_a)
            breakoff_a = dt.stringToArray(breakoff_a)
            score_limit_a = dt.stringToArray(score_limit_a)
            amount_to_start_a = dt.stringToArray(amount_to_start_a)
            cluster_multiplier_a = dt.stringToArray(cluster_multiplier_a)
            kappa_a = dt.stringToArray(kappa_a)
            classification_task_a = dt.stringToArray(classification_task_a)
            use_breakoff_dissim_a = dt.stringToArray(use_breakoff_dissim_a)
            half_ndcg_half_kappa_a = dt.stringToArray(half_ndcg_half_kappa_a)
            get_all_a = dt.stringToArray(get_all_a)
            learn_rate_a = dt.stringToArray(learn_rate_a)
            ft_loss_a = dt.stringToArray(ft_loss_a)
            ft_optimizer_a = dt.stringToArray(ft_optimizer_a)
            is_identity_a = dt.stringToArray(is_identity_a)
            amount_of_finetune = dt.stringToArray(amount_of_finetune_a)
            epochs_a = dt.stringToArray(epochs_a)
            average_ppmi_a = dt.stringToArray(average_ppmi_a)
            limit_entities_a = dt.stringToArray(limit_entities_a)
            bag_of_clusters_a = dt.stringToArray(bag_of_clusters_a)
            add_all_terms_a = dt.stringToArray(add_all_terms_a)
        ep = int(ep)
        dropout_noise = float(dropout_noise)
        cross_val = int(cross_val)
        lowest_amt = int(lowest_amt)
        threads = int(threads)
        deep_size = dt.stringToArray(deep_size)
        if class_weight != 'None':
            class_weight = dt.stringToArray(class_weight)
        else:
            class_weight = None
        rewrite_files = dt.toBool(rewrite_files)
        development = dt.toBool(development)
        svm_classify = dt.toBool(svm_classify)
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
        if vector_path_replacement == 'None':
            vector_path_replacement = None
        if get_nnet_vectors_path == 'None':
            get_nnet_vectors_path = None
        skip_nn = dt.toBool(skip_nn)
        dissim = float(dissim)
        find_most_similar = dt.toBool(find_most_similar)

    elif not isinstance(deep_size, str) and not hp_opt:
        dissim_amt_a = [dissim_amt_a[0]]
        breakoff_a = [breakoff_a[0]]
        score_limit_a = [score_limit_a[0]]
        amount_to_start_a = [amount_to_start_a[0]]
        cluster_multiplier_a = [cluster_multiplier_a[0]]
        kappa_a = [kappa_a[0]]
        classification_task_a = [classification_task_a[0]]
        use_breakoff_dissim_a = [use_breakoff_dissim_a[0]]
        get_all_a = [get_all_a[0]]
        half_ndcg_half_kappa_a = [half_ndcg_half_kappa_a[0]]
        learn_rate_a = [learn_rate_a[0]]
        ft_loss_a = [ft_loss_a[0]]
        ft_optimizer_a = [ft_optimizer_a[0]]
        is_identity_a = [is_identity_a[0]]
        amount_of_finetune = [amount_of_finetune_a[0]]
        epochs = [epochs_a[0]]
        average_ppmi_a = [average_ppmi_a[0]]
        limit_entities_a = [limit_entities_a[0]]
        bag_of_clusters_a = [bag_of_clusters_a[0]]
        add_all_terms_a = [add_all_terms_a[0]]


    variables_to_execute = []

    for d in dissim_amt_a:
        for b in breakoff_a:
            for s in score_limit_a:
                for a in amount_to_start_a:
                    for c in cluster_multiplier_a:
                        for k in kappa_a:
                            for ct in classification_task_a:
                                for ub in use_breakoff_dissim_a:
                                    for ga in get_all_a:
                                        for hnk in half_ndcg_half_kappa_a:
                                            for l in limit_entities_a:
                                                for bc in bag_of_clusters_a:
                                                    for aa in add_all_terms_a:
                                                        variables_to_execute.append((d, b, s, a, c, k, ct, ub, ga, hnk, l, bc, aa))
    all_csv_fns = []
    for a in  classification_task_a:
        all_csv_fns.append([])
    arrange_name = ""
    for vt in variables_to_execute:
        file_name = average_csv_fn
        dissim_amt = vt[0]
        breakoff = vt[1]
        score_limit = vt[2]
        amount_to_start = vt[3]
        cluster_multiplier = vt[4]
        score_type = vt[5]
        classification_task = vt[6]
        use_breakoff_dissim = vt[7]
        get_all = vt[8]
        half_ndcg_half_kappa = vt[9]
        limit_entities = vt[10]
        bag_of_clusters = vt[11]
        add_all_terms = vt[12]
        class_task_index = 0

        if limit_entities:
            get_nnet_vectors_path = None

        for c in range(len(classification_task_a)):
            if classification_task == classification_task_a[c]:
                class_task_index = c

        """ CLUSTER RANKING """
        vector_names_fn = loc + data_type + "/nnet/spaces/entitynames.txt"
        limited_label_fn = loc + data_type + "/classify/" + classification_task + "/available_entities.txt"

        if one_for_all and not skip_nn:
            classification_names = dt.import1dArray(loc + data_type + "/classify/" + classification_task + "/names.txt")
        else:
            classification_names = ["all"]
        for classification_name in classification_names:
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
                    fn = file_name + "E" + str(ep) + "DS" + str(deep_size) + "DN" + str(dropout_noise) + "CT" + classification_task + \
                                "HA" + str(hidden_activation) + "CV" + str(cv_splits)  +  " S" + str(s)
                    if development:
                        fn = fn  + " Dev"
                    if limit_entities:
                        fn = fn + " LE"
                else:
                    fn = file_name  + "CV" + str(cv_splits)  +  "S" + str(s)
                    if limit_entities:
                        fn = fn + "LE"

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
                    classification_path = loc + data_type + "/classify/" + classification_task + "/class-"+classification_name
                    label_names_fn = loc + data_type + "/classify/" + classification_task + "/names.txt"
                    fine_tune_weights_fn = None
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

                    file_name = file_name + " " + classification_name
                    csv_fns_nn[nn_counter] = loc + data_type + "/nnet/csv/" + file_name + ".csv"
                    nn_counter+=1
                    print("nnet hi", arcca)
                    arrange_name = file_name
                    if not arcca and not skip_nn:
                        print ("nnet hello?")
                        SDA = nnet.NeuralNetwork(noise=0, fine_tune_weights_fn=fine_tune_weights_fn, optimizer_name=optimizer_name,
                            get_scores=get_scores, past_model_bias_fn=past_model_bias_fn, deep_size=deep_size,  cutoff_start=cs,
                            randomize_finetune_weights=randomize_finetune_weights, amount_of_finetune=amount_of_finetune_a[0],
                            vector_path=init_vector_path, hidden_layer_size=hidden_layer_size, class_path=classification_path,
                            identity_swap=identity_swap, dropout_noise=dropout_noise, save_outputs=save_outputs,
                            hidden_activation=hidden_activation, output_activation=output_activation, epochs=ep,
                            learn_rate=lr, is_identity=is_identity_a[0], output_size=output_size, split_to_use=splits, label_names_fn=label_names_fn,
                            batch_size=batch_size, past_model_weights_fn=past_model_weights_fn, loss=loss, cv_splits=cv_splits, csv_fn = file_name,
                            file_name=file_name, from_ae=from_ae, data_type=data_type, rewrite_files=rewrite_files, development=development,
                                                 class_weight=class_weight, get_nnet_vectors_path=get_nnet_vectors_path,
                                                 limit_entities=limit_entities, limited_label_fn=limited_label_fn,
                                                 vector_names_fn=vector_names_fn, classification_name=classification_name)

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

                        breakoff = breakoff
                        score_type = score_type

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
                        class_names_fn = property_names_fn

                        cluster_amt = deep_size[x] * cluster_multiplier

                        """ Begin Methods """
                        print(file_name)
                        """ CLUSTERING """
                        # Choosing the score-type

                        names_fn = property_names_fn

                        print(file_name)
                        svm.createSVM(vector_path, bow_path, property_names_fn, file_name, lowest_count=lowest_amt,
                          highest_count=highest_count, data_type=data_type, get_kappa=score_type,
                          get_f1=False, svm_type=svm_type, getting_directions=True, threads=threads, rewrite_files=rewrite_files,
                                      classification=new_classification_task, lowest_amt=lowest_amt, chunk_amt=chunk_amt, chunk_id=chunk_id)


                        if chunk_amt > 0:
                            if chunk_id == chunk_amt-1:
                                dt.compileSVMResults(file_name, chunk_amt, data_type)

                            else:
                                if d != len(deep_fns)-1:
                                    randomcount = 0
                                    while not False: #NOTE, REWRITE THIS, ADD A TEMP MARKER FOR WHEN THE PROCESS RETURNS HERE, PREVIOUSLY USED EXISTENCE OF FN
                                        randomcount += 1
                                    print(randomcount)
                                    time.sleep(10)
                                else:
                                    print("exit")
                                    if d != len(deep_fns)-1:
                                        while not dt.fileExists(loc+data_type+"/nnet/spaces/"+deep_fns[d+1]+".txt"):
                                            time.sleep(10)

                        if chunk_id == chunk_amt -1 or chunk_amt <= 0:
                            if score_type is not "kappa":
                                rank.getAllPhraseRankings(directions_fn, vector_path, class_names_fn, vector_names_fn, file_name,
                                              data_type=data_type, rewrite_files=rewrite_files)
                                ndcg.getNDCG(loc + data_type + "/rank/numeric/" + file_name + "ALL.txt", file_name,
                                         data_type=data_type, lowest_count=lowest_amt, rewrite_files=rewrite_files,
                                             highest_count=highest_count, classification=new_classification_task)
                            half_ndcg_half_kappa = ""
                            if half_ndcg_half_kappa:
                                scores_fn = loc + data_type + "/ndcg/" + file_name + ".txt"
                                half_ndcg_half_kappa = loc + data_type + "/svm/kappa/" + file_name + ".txt"
                                file_name = file_name + "halfnk"
                            elif score_type is "ndcg":
                                scores_fn = loc + data_type + "/ndcg/" + file_name + ".txt"
                                file_name = file_name + "ndcg"
                            elif score_type is "kappa":
                                scores_fn = loc + data_type + "/svm/kappa/" + file_name + ".txt"
                                file_name = file_name + "kappa"
                            elif score_type is "spearman":
                                scores_fn = loc + data_type + "/svm/spearman/" + file_name + ".txt"
                                file_name = file_name + "spearman"

                            """ CLUSTERING """
                            # Choosing the score-type
                            if breakoff:
                                score_limit = score_limit

                                file_name = file_name + str(score_limit)
                                if get_all:
                                    file_name = file_name + " GA"
                                if add_all_terms:
                                    file_name = file_name + " AllTerms"
                                file_name = file_name + " Breakoff"
                            else:
                                file_name = file_name + " KMeans"

                            if not use_breakoff_dissim and breakoff:
                                dissim = 0
                                dissim_amt = 0
                                cluster_multiplier = 2000000
                            file_name = file_name + " CA" +  str(cluster_amt)
                            file_name = file_name + " MC" + str(min_size) + " MS" + str(min_score)
                            names_fn = property_names_fn
                            file_name = file_name + " ATS" + str(amount_to_start) + " DS" + str(dissim_amt)
                            if breakoff:
                                if find_most_similar:
                                    file_name = file_name + " FMS"
                                similarity_threshold = sim_t
                                amount_to_start = amount_to_start
                                add_all_terms = add_all_terms
                                clusters_fn = loc + data_type + "/cluster/hierarchy_directions/" + file_name + ".txt"
                                cluster_names_fn = loc + data_type + "/cluster/hierarchy_names/" + file_name + ".txt"
                                cluster_dict_fn = cluster_names_fn
                            else:
                                high_threshold = 0.5
                                low_threshold = 0.1
                                clusters_fn = loc + data_type + "/cluster/clusters/" + file_name + ".txt"
                                cluster_names_fn = loc + data_type + "/cluster/names/" + file_name + ".txt"
                                cluster_dict_fn = loc + data_type + "/cluster/dict/" + file_name + ".txt"

                            if breakoff:
                                hierarchy.initClustering(vector_path, directions_fn, scores_fn, names_fn, amount_to_start, False,
                                      cluster_amt, score_limit, file_name, score_type, similarity_threshold,
                                             add_all_terms=add_all_terms, data_type=data_type, rewrite_files=rewrite_files,
                                                         lowest_amt=lowest_amt, highest_amt=highest_count, classification=new_classification_task,
                                                         min_score=min_score, min_size = min_size, largest_clusters=largest_cluster, dissim=dissim,
                                                         dissim_amt=dissim_amt, find_most_similar=find_most_similar, get_all=get_all,
                                                         half_ndcg_half_kappa=half_ndcg_half_kappa)
                            else:
                                cluster.getClusters(directions_fn, scores_fn, names_fn, False, dissim_amt, amount_to_start, file_name, cluster_amt,
                                                    dissim, min_score, data_type, rewrite_files=rewrite_files,
                                                         half_kappa_half_ndcg=half_ndcg_half_kappa)

                            ranking_fn = loc + data_type + "/rank/numeric/" + file_name + ".txt"

                            rank.getAllRankings(clusters_fn, vector_path, cluster_names_fn , vector_names_fn, 0.2, 1, False, file_name,
                                                False, data_type=data_type, rewrite_files=rewrite_files)
                            if skip_nn:
                                file_name = file_name + " " + classification_task

                            if dt_dev:
                                file_name = file_name + " tdev"

                            csv_name = loc + data_type + "/rules/tree_csv/" + file_name + ".csv"

                            csv_fns_dt[counter] = csv_name
                            counter += 1

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
                               save_details=False, data_type=data_type,split_to_use=splits, pruning=2,
                                              limited_label_fn=limited_label_fn, rewrite_files=rewrite_files,
                               csv_fn=csv_name, cv_splits=cv_splits, limit_entities=limit_entities, vector_names_fn=vector_names_fn)
                            if not skip_nn:

                                variables_to_execute = []

                                for d in learn_rate_a:
                                    e = epochs_a[0]
                                    for b in ft_loss_a:
                                        for s in ft_optimizer_a:
                                            for a in is_identity_a:
                                                for c in amount_of_finetune_a:
                                                    for x in average_ppmi_a:
                                                        variables_to_execute.append((d, b, s, a, c, e, x))
                                orig_fn = file_name
                                for v in variables_to_execute:
                                    learn_rate = v[0]
                                    ft_loss = v[1]
                                    ft_optimizer = v[2]
                                    is_identity = v[3]
                                    amount_of_finetune = v[4]
                                    epochs = v[5]
                                    average_ppmi = v[6]

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
                                                      max_dep=max_depth, balance="balanced", criterion="entropy", save_details=False,
                                                      cv_spli=cv_splits, split_to_use=splits,
                                                      data_type=data_type, csv_fn=csv_name, rewrite_files=True,
                                                      development=dt_dev)
                                    """
                                    # Decision tree

                                    if average_ppmi:
                                        file_name = file_name + "APPMI"
                                    elif bag_of_clusters:
                                        file_name = file_name + "BOC"

                                    if bag_of_clusters:
                                        class_path = loc + data_type + "/finetune/boc/" + file_name + ".txt"
                                    else:
                                        class_path = loc + data_type + "/finetune/" + file_name + ".txt"

                                    if average_ppmi:
                                        fto.pavPPMIAverage(cluster_dict_fn, ranking_fn, file_name, data_type=data_type, rewrite_files=rewrite_files,
                                                classification=classification, lowest_amt=lowest_amt, limit_entities=limit_entities, highest_amt=highest_count)
                                    elif bag_of_clusters:
                                        fto.bagOfClustersPavPPMI(cluster_dict_fn, ranking_fn, file_name, data_type=data_type, rewrite_files=rewrite_files,
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

                                    loss = ft_loss
                                    optimizer_name = ft_optimizer

                                    hidden_layer_size = deep_size[x]

                                    past_model_weights_fn = [loc + data_type + "/nnet/weights/" + new_file_names[x] + ".txt"]
                                    past_model_bias_fn = [loc + data_type + "/nnet/bias/" + new_file_names[x] + ".txt"]

                                    """ DECISION TREES FOR NNET RANKINGS """
                                    nnet_ranking_fn = loc + data_type + "/nnet/clusters/" + file_name + "FT.txt"

                                    csv_name = loc + data_type + "/rules/tree_csv/" + file_name + ".csv"

                                    file_name = file_name + "FT"
                                    if arcca is False:

                                        SDA = nnet.NeuralNetwork(noise=0, fine_tune_weights_fn=fine_tune_weights_fn, optimizer_name=optimizer_name,
                                                    past_model_bias_fn=past_model_bias_fn, save_outputs=True,
                                                    randomize_finetune_weights=randomize_finetune_weights,
                                                    vector_path=init_vector_path, hidden_layer_size=hidden_layer_size, class_path=class_path,
                                                    identity_swap=identity_swap, amount_of_finetune=amount_of_finetune,
                                                    hidden_activation=hidden_activation, output_activation=output_activation, epochs=epochs,
                                                    learn_rate=learn_rate, is_identity=is_identity, batch_size=batch_size,
                                                    past_model_weights_fn=past_model_weights_fn, loss=loss, rewrite_files=rewrite_files,
                                                    file_name=file_name, from_ae=from_ae, finetune_size=finetune_size, data_type=data_type,
                                                               get_nnet_vectors_path= get_nnet_vectors_path, limit_entities=True,
                                                 vector_names_fn=vector_names_fn, classification_name=classification_name)

                                        #new_file_names[x] = file_name

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

                                        wekatree.DecisionTree(ranking_fn, classification_path, label_names_fn,
                                                              cluster_names_fn, file_name,
                                                              save_details=True, data_type=data_type,
                                                              split_to_use=splits, pruning=2,
                                                              limited_label_fn=limited_label_fn, rewrite_files=rewrite_files,
                                                              csv_fn=csv_name, cv_splits=cv_splits,
                                                              limit_entities=limit_entities,
                                                              vector_names_fn=vector_names_fn)


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
                                        """
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
                                                              save_details=True, data_type=data_type,split_to_use=splits, rewrite_files=rewrite_files,
                                                              csv_fn=csv_name, cv_splits=cv_splits, limit_entities=limit_entities,
                                                              limited_label_fn=limited_label_fn, vector_names_fn=vector_names_fn)
                                        """
                                    file_name = orig_fn
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
                for n in csv_fns_dt_a[a]:
                    all_csv_fns[class_task_index].append(n)
            if not skip_nn:
                for a in range(len(csv_fns_nn_a)):
                    dt.averageCSVs(csv_fns_nn_a[a])

    for c in range(len(all_csv_fns)):
            dt.arrangeByScore(np.unique(np.asarray(all_csv_fns[c])), classification_task_a[c], "../data/"+data_type+"/rules/tree_csv/"
                          +classification_task_a[c] + " " + arrange_name + str(len(all_csv_fns[0])) + ".csv")
    jvm.stop()

just_output = True
arcca = False
if arcca:
    loc = "/scratch/c1214824/data/"
else:
    loc = "../data/"
"""
data_type = "wines"
classification_task = ["types"]
file_name = "wines pca 100"
lowest_amt = 50
highest_amt = 10

init_vector_path = loc+data_type+"/nnet/spaces/wines100.txt"
vector_path_replacement = loc+data_type+"/nnet/spaces/wines100.txt"
get_nnet_vectors_path = loc+data_type+"/nnet/spaces/wines100.txt"
"""
"""
init_vector_path = loc+data_type+"/pca/class-all-50-10-alld100"
vector_path_replacement = loc+data_type+"/pca/class-all-50-10-alld100"
get_nnet_vectors_path = loc+data_type+"/nnet/spaces/films100-genres.txt"
"""
"""
data_type = "movies"
classification_task = ["genres", "keywords"]
file_name = "f200ge"
lowest_amt = 100
highest_amt = 10
init_vector_path = loc+data_type+"/nnet/spaces/films200-genres.txt"
#init_vector_path = loc+data_type+"/nnet/spaces/films200-"+classification_task+".txt"
#file_name = "films200-genres100ndcg0.85200 tdev3004FTL0"
#init_vector_path = loc+data_type+"/nnet/spaces/"+file_name+".txt"
get_nnet_vectors_path = loc+data_type+"/nnet/spaces/films200-genres.txt"
deep_size = [200]
"""

data_type = "placetypes"
classification_task = ["opencyc"]
lowest_amt = 50
highest_amt = 10
#init_vector_path = "../data/"+data_type+"/bow/ppmi/class-all-"+str(lowest_amt)+"-"+str(highest_amt)+"-"+classification_task
#file_name = "placetypes bow"
init_vector_path = "../data/"+data_type+"/nnet/spaces/places100.txt"

vector_path_replacement = loc+data_type+"/nnet/spaces/places100.txt"
get_nnet_vectors_path = loc + data_type + "/nnet/spaces/places100.txt"
file_name = "places mds 100"

"""
hidden_activation = "tanh"
dropout_noise = 0.5
output_activation = "softmax"
trainer = "adadelta"
loss="categorical_crossentropy"
class_weight = None
lr = 0.01
nnet_dev = False
ep=1400
deep_size = [100]
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
ep =1400
lr = 0.01
rewrite_files = False
learn_rate= [ 0.001]
cutoff_start = 0.2


is_identity = [False]
amount_of_finetune = [1]
ft_loss = ["mse"]
ft_optimizer = ["adagrad"]
min_size = 1

# Set to 0.0 for a janky skip, can set to 1.0 to delete it
sim_t = 1.0#1.0


nnet_dev = False

deep_size = [100]
limit_entities = [False]

min_score = 0.4
largest_cluster = 1
dissim = 0.0
dissim_amt = [400]
find_most_similar = True#False
breakoff = [False, True]
score_limit = [0.8]
amount_to_start = [1000]
cluster_multiplier = [2, 4, 1]#50
score_type = ["ndcg", "kappa"]
use_breakoff_dissim = [False]
get_all = [False]
half_ndcg_half_kappa = [False]
add_all_terms = [True, False]


average_ppmi = [False]i_    
bag_of_clusters = [True, False]


epochs=[300]

"""
sim_t = 0.0#1.0
find_most_similar = False#False
cluster_multiplier = [50]#50
score_limit = [0.0]
"""
hp_opt = True

dt_dev = True
use_pruned = False
svm_classify = False
rewrite_files = False
max_depth = 2

skip_nn = False

cross_val = 1
one_for_all = False

threads=2
chunk_amt = 0
chunk_id = 0
for c in range(chunk_amt):
    chunk_id = c
    variables = [data_type, classification_task, file_name, init_vector_path, hidden_activation,
                                   is_identity, amount_of_finetune, breakoff, score_type, score_limit, rewrite_files,
                                   cluster_multiplier, threads, dropout_noise, learn_rate, epochs, cross_val, ep,
                                   output_activation, cutoff_start, deep_size, classification_task, highest_amt,
                                   lowest_amt, loss, nnet_dev, add_all_terms, average_ppmi, trainer, class_weight,
                                   amount_to_start, chunk_amt, chunk_id, lr, vector_path_replacement, dt_dev, use_pruned, max_depth,
                                   min_score, min_size, limit_entities, svm_classify, get_nnet_vectors_path, arcca, largest_cluster,
                 skip_nn, dissim, dissim_amt, hp_opt, find_most_similar, use_breakoff_dissim, get_all, half_ndcg_half_kappa, sim_t,
                 one_for_all, bag_of_clusters]

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
    score_type = args[8]
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
    dissim = args[46]
    dissim_amt = args[47]
    hp_opt = args[48]
    find_most_similar = args[49]
    get_all = args[50]
    half_ndcg_half_kappa = args[51]
    one_for_all = args[52]
    bag_of_clusters = args[53]


if  __name__ =='__main__':main(data_type, classification_task, file_name, init_vector_path, hidden_activation,
                               is_identity, amount_of_finetune, breakoff, score_type, score_limit, rewrite_files,
                               cluster_multiplier, threads, dropout_noise, learn_rate, epochs, cross_val, ep,
                               output_activation, cutoff_start, deep_size, classification_task, highest_amt,
                               lowest_amt, loss, nnet_dev, add_all_terms, average_ppmi, trainer, class_weight,
                               amount_to_start, chunk_amt, chunk_id, lr, vector_path_replacement, dt_dev, use_pruned, max_depth,
                               min_score, min_size, limit_entities, svm_classify, get_nnet_vectors_path, arcca, loc, largest_cluster,
                               skip_nn, dissim, dissim_amt, hp_opt, find_most_similar, use_breakoff_dissim, get_all,
                               half_ndcg_half_kappa, sim_t, one_for_all, ft_loss, ft_optimizer, bag_of_clusters, just_output)
