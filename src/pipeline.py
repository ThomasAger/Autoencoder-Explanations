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
#import wekatree
from sklearn.model_selection import KFold
from sklearn.preprocessing import MultiLabelBinarizer
#import weka.core.jvm as jvm
import random
import sys
from itertools import product
import time

#jvm.start(max_heap_size="512m")

def main(data_type, classification_task_a, file_name, init_vector_path, hidden_activation, is_identity_a, amount_of_finetune_a,
         breakoff_a, kappa_a, score_limit_a, rewrite_files, cluster_multiplier_a, threads, dropout_noise_a, learn_rate_a, epochs_a, cross_val, ep,
         output_activation, cs, deep_size, classification, direction_count, lowest_amt, loss, development, add_all_terms_a,
         average_ppmi_a, optimizer_name, class_weight, amount_to_start_a, chunk_amt, chunk_id, lr, vector_path_replacement, dt_dev,
         use_pruned, max_depth_a, min_score, min_size, limit_entities_a, svm_classify, get_nnet_vectors_path, arcca, loc, largest_cluster,
         skip_nn, dissim, dissim_amt_a, hp_opt, find_most_similar, use_breakoff_dissim_a, get_all_a, half_ndcg_half_kappa_a,
         sim_t, one_for_all, ft_loss_a, ft_optimizer_a, bag_of_clusters_a, just_output, arrange_name, only_most_similar_a,
         dont_cluster_a, top_dt_clusters_a, by_class_finetune_a, cluster_duplicates_a, repeat_finetune_a, save_results_so_far,
         finetune_ppmi_a, average_nopav_ppmi_a, boc_average_a, identity_activation_a, ppmi_only_a, boc_only_a, pav_only_a,
         multi_label_a ,use_dropout_in_finetune_a, lock_weights_and_redo_a):


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
            only_most_similar_a = dt.stringToArray(only_most_similar_a)[0]
            dont_cluster_a = dt.stringToArray(dont_cluster_a)[0]
            top_dt_clusters_a = dt.stringToArray(top_dt_clusters_a)[0]
            by_class_finetune_a = dt.stringToArray(by_class_finetune_a)[0]
            cluster_duplicates_a = dt.stringToArray(cluster_duplicates_a)[0]
            repeat_finetune_a = dt.stringToArray(repeat_finetune_a)[0]
            finetune_ppmi_a = dt.stringToArray(finetune_ppmi_a)[0]
            average_nopav_ppmi_a = dt.stringToArray(average_nopav_ppmi_a)[0]
            boc_average_a = dt.stringToArray(boc_average_a)[0]
            identity_activation_a = dt.stringToArray(identity_activation_a)[0]
            ppmi_only_a = dt.stringToArray(ppmi_only_a)[0]
            boc_only_a = dt.stringToArray(boc_only_a)[0]
            pav_only_a = dt.stringToArray(pav_only_a)[0]
            max_depth_a = dt.stringToArray(max_depth_a)[0]
            use_dropout_in_finetune_a = dt.stringToArray((use_dropout_in_finetune_a))[0]
            lock_weights_and_redo_a = dt.stringToArray(lock_weights_and_redo_a)[0]
            dropout_noise_a = dt.stringToArray(dropout_noise_a)[0]
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
            only_most_similar_a = dt.stringToArray(only_most_similar_a)
            dont_cluster_a = dt.stringToArray(dont_cluster_a)
            top_dt_clusters_a = dt.stringToArray(top_dt_clusters_a)
            by_class_finetune_a = dt.stringToArray(by_class_finetune_a)
            cluster_duplicates_a = dt.stringToArray(cluster_duplicates_a)
            repeat_finetune_a = dt.stringToArray(repeat_finetune_a)
            finetune_ppmi_a = dt.stringToArray(finetune_ppmi_a)
            average_nopav_ppmi_a = dt.stringToArray(average_nopav_ppmi_a)
            boc_average_a = dt.stringToArray(boc_average_a)
            identity_activation_a = dt.stringToArray(identity_activation_a)
            ppmi_only_a = dt.stringToArray(ppmi_only_a)
            boc_only_a = dt.stringToArray(boc_only_a)
            pav_only_a = dt.stringToArray(pav_only_a)
            max_depth_a = dt.stringToArray(max_depth_a)
            multi_label_a = dt.stringToArray(multi_label_a)
            use_dropout_in_finetune_a = dt.stringToArray(use_dropout_in_finetune_a)
            lock_weights_and_redo_a = dt.stringToArray(lock_weights_and_redo_a)
            dropout_noise_a = dt.stringToArray(dropout_noise_a)

        ep = int(ep)
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
        only_most_similar_a = [only_most_similar_a[0]]
        dont_cluster_a = [dont_cluster_a[0]]
        top_dt_clusters_a = [top_dt_clusters_a[0]]
        by_class_finetune_a = [by_class_finetune_a[0]]
        cluster_duplicates_a = [cluster_duplicates_a[0]]
        repeat_finetune_a = [repeat_finetune_a[0]]
        finetune_ppmi_a = [finetune_ppmi_a[0]]
        average_nopav_ppmi_a = [average_nopav_ppmi_a[0]]
        boc_average_a = [boc_average_a[0]]
        identity_activation_a = [identity_activation_a[0]]
        ppmi_only_a = [ppmi_only_a[0]]
        boc_only_a = [boc_only_a[0]]
        pav_only_a = [pav_only_a[0]]
        max_depth_a = [max_depth_a[0]]
        multi_label_a = [multi_label_a[0]]
        use_dropout_in_finetune_a = [use_dropout_in_finetune_a[0]]
        lock_weights_and_redo_a = [lock_weights_and_redo_a[0]]
        dropout_noise_a = [dropout_noise_a[0]]

    variables_to_execute = list(product(dissim_amt_a,breakoff_a,score_limit_a, amount_to_start_a,cluster_multiplier_a,
                                   kappa_a,classification_task_a,use_breakoff_dissim_a,get_all_a,half_ndcg_half_kappa_a,
                                   limit_entities_a,add_all_terms_a,only_most_similar_a,dont_cluster_a,
                                   top_dt_clusters_a,by_class_finetune_a, cluster_duplicates_a, repeat_finetune_a, max_depth_a,
                                        multi_label_a, dropout_noise_a))
    all_csv_fns = []
    original_fn = []
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
        add_all_terms = vt[11]
        only_most_similar = vt[12]
        dont_cluster = vt[13]
        top_dt_clusters = vt[14]
        by_class_finetune = vt[15]
        cluster_duplicates = vt[16]
        repeat_finetune = vt[17]
        max_depth = vt[18]
        multi_label = vt[19]
        dropout_noise = vt[20]
        class_task_index = 0

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
                                "HA" + str(hidden_activation) + "CV" + str(cv_splits)  +  " S" + str(s) + "OA " + output_activation
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
                    identity_swap = False
                    from_ae = False
                    past_model_weights_fn = None
                    past_model_bias_fn = None
                    randomize_finetune_weights = False
                    output_size = 10
                    randomize_finetune_weights = False
                    corrupt_finetune_weights = False
                    get_scores = True

                    file_name = file_name + " " + classification_name
                    csv_fns_nn[nn_counter] = loc + data_type + "/nnet/csv/" + file_name + ".csv"
                    nn_counter+=1
                    print("nnet hi", arcca)
                    if not arcca and not skip_nn:
                        print ("nnet hello?")
                        SDA = nnet.NeuralNetwork(noise=0, fine_tune_weights_fn=fine_tune_weights_fn, optimizer_name=optimizer_name,
                            get_scores=get_scores, past_model_bias_fn=past_model_bias_fn, deep_size=deep_size,  cutoff_start=cs,
                            randomize_finetune_weights=randomize_finetune_weights, amount_of_finetune=amount_of_finetune_a[0],
                            vector_path=init_vector_path, hidden_layer_size=0, class_path=classification_path,
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
                        if skip_nn == True:
                            deep_size = [amount_of_finetune_a[0][len(amount_of_finetune_a[0])-1]]
                            x = 0

                        if limit_entities is False:
                            new_classification_task = "all"
                        else:
                            new_classification_task = classification_task
                        file_name = new_file_names[x]

                        vector_path = loc + data_type + "/nnet/spaces/" + new_file_names[x] + ".txt"
                        if skip_nn:
                            vector_path = vector_path_replacement
                        """ Begin Filename """

                        breakoff = breakoff
                        score_type = score_type

                        file_name = file_name + str(lowest_amt)

                        """ Begin Parameters """
                        """ SVM """
                        svm_type = "svm"
                        highest_count = direction_count

                        bow_path = loc + data_type + "/bow/binary/phrases/class-all-"+str(lowest_amt)+"-"+str(highest_count)+"-"+new_classification_task
                        property_names_fn = loc + data_type + "/bow/names/" + str(lowest_amt) + "-" +str(highest_count)+"-"+ new_classification_task +".txt"
                        directions_fn = loc + data_type + "/svm/directions/" + file_name + ".txt"


                        """ DIRECTION RANKINGS """
                        # Get rankings
                        class_names_fn = property_names_fn

                        cluster_amt = deep_size[x] * cluster_multiplier

                        if dont_cluster > 0:
                            cluster_amt = dont_cluster
                            amount_to_start = dont_cluster
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
                            dissim_amt = cluster_amt * 2
                            file_name = file_name + " MC" + str(min_size) + " MS" + str(min_score)
                            names_fn = property_names_fn
                            file_name = file_name + " ATS" + str(amount_to_start) + " DS" + str(dissim_amt)


                            if dont_cluster:
                                file_name = file_name + " DC" +str(dont_cluster)
                            if breakoff:
                                if only_most_similar:
                                    file_name = file_name + " OMS"
                                if find_most_similar:
                                    file_name = file_name + " FMS"
                                similarity_threshold = sim_t
                                add_all_terms = add_all_terms
                                clusters_fn = loc + data_type + "/cluster/hierarchy_directions/" + file_name + ".txt"
                                cluster_dict_fn = loc + data_type + "/cluster/hierarchy_names/" + file_name + ".txt"
                                cluster_names_fn = cluster_dict_fn
                            else:
                                high_threshold = 0.5
                                low_threshold = 0.1
                                clusters_fn = loc + data_type + "/cluster/clusters/" + file_name + ".txt"
                                cluster_names_fn = loc + data_type + "/cluster/dict/" + file_name + ".txt"
                                cluster_dict_fn = loc + data_type + "/cluster/dict/" + file_name + ".txt"

                            if breakoff:

                                hierarchy.initClustering(vector_path, directions_fn, scores_fn, names_fn, amount_to_start, False,
                                      cluster_amt, score_limit, file_name, score_type, similarity_threshold,
                                             add_all_terms=add_all_terms, data_type=data_type, rewrite_files=rewrite_files,
                                                         lowest_amt=lowest_amt, highest_amt=highest_count, classification=new_classification_task,
                                                         min_score=min_score, min_size = min_size, largest_clusters=largest_cluster, dissim=dissim,
                                                         dissim_amt=dissim_amt, find_most_similar=find_most_similar, get_all=get_all,
                                                         half_ndcg_half_kappa=half_ndcg_half_kappa, only_most_similar=only_most_similar,
                                                         dont_cluster=dont_cluster)
                                ft_names_fn = "../data/"+data_type+"/cluster/hierarchy_names/" + file_name + ".txt"
                                ft_score_fn = dt.getScores(ft_names_fn, scores_fn, names_fn, file_name, data_type)

                            else:

                                cluster.getClusters(directions_fn, scores_fn, names_fn, False, dissim_amt, amount_to_start, file_name, cluster_amt,
                                                    dissim, min_score, data_type, rewrite_files=rewrite_files,
                                                         half_kappa_half_ndcg=half_ndcg_half_kappa, dont_cluster=dont_cluster)




                            ranking_fn = loc + data_type + "/rank/numeric/" + file_name + ".txt"

                            rank.getAllRankings(clusters_fn, vector_path, cluster_names_fn , vector_names_fn, 0.2, 1, False, file_name,
                                                False, data_type=data_type, rewrite_files=rewrite_files)
                            if skip_nn:
                                file_name = file_name

                            if dt_dev:
                                file_name = file_name + " tdev"


                            csv_name = loc + data_type + "/rules/tree_csv/" + file_name + " " + classification_task + ".csv"

                            csv_fns_dt[counter] = csv_name
                            if cv_splits == 0:
                                all_csv_fns.append(csv_name)
                            else:
                                if splits == 0:
                                    original_fn.append(csv_name)
                            counter += 1

                            if cluster_duplicates:
                                file_name = file_name + " UNIQUE"

                            if multi_label:
                                file_name = file_name + "ML"

                            #file_name = "NMF 200"
                            #ranking_fn = "../data/movies/NMF/all-100-10frob.txt"
                            tree.DecisionTree(ranking_fn, classification_path, label_names_fn, cluster_dict_fn, file_name + " " + classification_task, 10000,
                                      max_depth=max_depth, balance="balanced", criterion="entropy", save_details=True, cv_splits=cv_splits, split_to_use=splits,
                                      data_type=data_type, csv_fn=csv_name, rewrite_files=rewrite_files, development=dt_dev, limit_entities=limit_entities,
                                              limited_label_fn=limited_label_fn, vector_names_fn=vector_names_fn, clusters_fn = clusters_fn,
                                              cluster_duplicates = cluster_duplicates, save_results_so_far=save_results_so_far,
                                              multi_label=multi_label)


                            tree.DecisionTree(ranking_fn, classification_path, label_names_fn, cluster_dict_fn, file_name  + " " + classification_task+ "None", 10000,
                                                  max_depth=None, balance="balanced", criterion="entropy", save_details=True,
                                              data_type=data_type, csv_fn=csv_name, rewrite_files=rewrite_files,
                                              cv_splits=cv_splits, split_to_use=splits, development=dt_dev, limit_entities=limit_entities,
                                              limited_label_fn=limited_label_fn, vector_names_fn=vector_names_fn, clusters_fn = clusters_fn,
                                              cluster_duplicates=cluster_duplicates, save_results_so_far=save_results_so_far,
                                              multi_label=multi_label)

                            """
                            wekatree.DecisionTree(ranking_fn, classification_path, label_names_fn , cluster_dict_fn , file_name,
                               save_details=False, data_type=data_type,split_to_use=splits, pruning=2,
                                              limited_label_fn=limited_label_fn, rewrite_files=rewrite_files,
                               csv_fn=csv_name, cv_splits=cv_splits, limit_entities=limit_entities, vector_names_fn=vector_names_fn, save_results_so_far=save_results_so_far)
                            """
                            variables_to_execute_a = []

                            variables_to_execute_a = list(
                                product(learn_rate_a, ft_loss_a, ft_optimizer_a, is_identity_a,
                                        amount_of_finetune_a,
                                        epochs_a, average_ppmi_a, finetune_ppmi_a, average_nopav_ppmi_a,
                                        boc_average_a, bag_of_clusters_a,
                                        identity_activation_a, ppmi_only_a, boc_only_a, pav_only_a, use_dropout_in_finetune_a,
                                        lock_weights_and_redo_a))
                            orig_fn = file_name
                            for v in variables_to_execute_a:
                                learn_rate = v[0]
                                ft_loss = v[1]
                                ft_optimizer = v[2]
                                is_identity = v[3]
                                amount_of_finetune = v[4]
                                epochs = v[5]
                                average_ppmi = v[6]
                                finetune_ppmi = v[7]
                                average_nopav_ppmi = v[8]
                                boc_average = v[9]
                                bag_of_clusters = v[10]
                                identity_activation = v[11]
                                ppmi_only = v[12]
                                boc_only = v[13]
                                pav_only = v[14]
                                use_dropout_in_finetune = v[15]
                                lock_weights_and_redo = v[16]


                                if top_dt_clusters:
                                    ranking_fn = loc+ data_type + "/rules/rankings/" + file_name + ".txt"
                                    cluster_names_fn = loc+ data_type + "/rules/names/" + file_name + ".txt"
                                    clusters_fn = loc+ data_type + "/rules/clusters/" + file_name + ".txt"
                                    file_name = file_name + " TOPDT"


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
                                                  data_type=data_type, csv_fn=csv_name, rewrite_files=rewrite_files,
                                                  development=dt_dev)
                                """
                                # Decision tree
                                if repeat_finetune > 0:
                                    file_name = file_name + "RPFT" + str(repeat_finetune)


                                file_name = file_name + "FT"
                                for f in range(repeat_finetune+1):
                                    if f > 0:
                                        ranking_fn = nnet_ranking_fn
                                    if average_ppmi:
                                        file_name = file_name + " APPMIFi"
                                        class_path = loc + data_type + "/finetune/" + file_name + ".txt"
                                    elif bag_of_clusters:
                                        file_name = file_name + " BOCFi"
                                        class_path = loc + data_type + "/finetune/boc/" + file_name + ".txt"
                                    elif finetune_ppmi:
                                        file_name = file_name + " PPMI"
                                        class_path = loc + data_type + "/finetune/" + file_name + ".txt"
                                    elif average_nopav_ppmi:
                                        file_name = file_name + " APPMINP"
                                        class_path = loc + data_type + "/finetune/" + file_name + ".txt"
                                    elif boc_average:
                                        file_name = file_name + " BOCPPMI"
                                        class_path = loc + data_type + "/finetune/boc/" + file_name + ".txt"
                                    else:
                                        class_path = loc + data_type + "/finetune/" + file_name + ".txt"
                                    if boc_only:
                                        print("boc only")
                                        class_path = "../data/" + data_type + "/bow/ppmi/" + file_name + ".txt"



                                    if average_ppmi:
                                        fto.pavPPMIAverage(cluster_dict_fn, ranking_fn, file_name, data_type=data_type, rewrite_files=rewrite_files,
                                                classification=classification, lowest_amt=lowest_amt, limit_entities=limit_entities, highest_amt=highest_count)
                                    elif bag_of_clusters:
                                        fto.bagOfClustersPavPPMI(cluster_dict_fn, ranking_fn, file_name, data_type=data_type, rewrite_files=rewrite_files,
                                                    classification=classification, lowest_amt=lowest_amt, limit_entities=limit_entities,highest_amt=highest_count)
                                    elif finetune_ppmi:
                                        fto.PPMIFT(cluster_dict_fn, ranking_fn, file_name, data_type=data_type, rewrite_files=rewrite_files,
                                                    classification=classification, lowest_amt=lowest_amt, limit_entities=limit_entities,highest_amt=highest_count)
                                    elif average_nopav_ppmi:
                                        fto.avgPPMI(cluster_dict_fn, ranking_fn, file_name, data_type=data_type, rewrite_files=rewrite_files,
                                                    classification=classification, lowest_amt=lowest_amt, limit_entities=limit_entities,highest_amt=highest_count)
                                    elif boc_average:
                                        fto.bagOfClusters(cluster_dict_fn, ranking_fn, file_name, data_type=data_type, rewrite_files=rewrite_files,
                                                    classification=classification, lowest_amt=lowest_amt, limit_entities=limit_entities,highest_amt=highest_count)
                                    else:
                                        fto.pavPPMI(cluster_dict_fn, ranking_fn, file_name, data_type=data_type, rewrite_files=rewrite_files,
                                                    classification=classification, lowest_amt=lowest_amt, limit_entities=limit_entities,highest_amt=highest_count)

                                    """ FINETUNING """

                                    if is_identity:
                                        file_name = file_name + " NT" + str(amount_of_finetune) + identity_activation

                                    epochs = epochs
                                    file_name = file_name + str(epochs)
                                    if data_type == "newsgroups":
                                        file_name = file_name + "S6040"

                                    file_name = file_name + "V1.2"

                                    if lock_weights_and_redo:
                                        file_name = file_name + "LOCK"

                                    if dropout_noise > 0.0:
                                        file_name = file_name + " DO" + str(dropout_noise)

                                    fine_tune_weights_fn = [clusters_fn]

                                    batch_size = 200
                                    learn_rate = learn_rate
                                    identity_swap = False
                                    randomize_finetune_weights = False
                                    from_ae = True
                                    finetune_size = int(cluster_amt/2)

                                    loss = ft_loss
                                    optimizer_name = ft_optimizer

                                    hidden_layer_size = finetune_size

                                    past_model_weights_fn = [loc + data_type + "/nnet/weights/" + new_file_names[x] + ".txt"]
                                    past_model_bias_fn = [loc + data_type + "/nnet/bias/" + new_file_names[x] + ".txt"]

                                    """ DECISION TREES FOR NNET RANKINGS """
                                    nnet_ranking_fn = loc + data_type + "/nnet/clusters/" + file_name + ".txt"

                                    if ppmi_only > 0:
                                        file_name = file_name + "ppmionly"
                                        if skip_nn is True:
                                            ppmi_fn = "../data/" + data_type + "/bow/ppmi/top" + str(ppmi_only) + ".txt"
                                            name_fn = "../data/" + data_type + "/bow/names/top" + str(ppmi_only) + ".txt"
                                        cluster.makePPMI(names_fn, scores_fn, ppmi_only, data_type, ppmi_fn, name_fn)
                                        nnet_ranking_fn = ppmi_fn
                                        cluster_dict_fn = name_fn
                                        clusters_fn = ppmi_fn
                                        print("ppmi only")
                                    elif boc_only:
                                        file_name = file_name + "boconlya"
                                        cluster_dict_fn = name_fn
                                        clusters_fn = boc_fn
                                        nnet_ranking_fn = boc_fn
                                    elif pav_only:
                                        print("pav only")
                                        file_name = file_name + "pavonly"
                                        nnet_ranking_fn = class_path
                                        clusters_fn = class_path
                                        cluster_dict_fn = name_fn

                                    csv_name = loc + data_type + "/rules/tree_csv/" + file_name + str(max_depth) + " " + classification_task +   ".csv"
                                    if cv_splits == 0:
                                        all_csv_fns.append(csv_name)
                                    else:
                                        if splits == 0:
                                            original_fn.append(csv_name)
                                    if arcca is False:
                                        if skip_nn:
                                            from_ae = False
                                        if not boc_only and not pav_only and ppmi_only == 0:
                                            print("NNET inc")
                                            #init_vector_path = "../data/newsgroups/bow/ppmi/" + "class-all-30-18836-all"

                                            SDA = nnet.NeuralNetwork(noise=0, fine_tune_weights_fn=fine_tune_weights_fn, optimizer_name=optimizer_name,
                                                    past_model_bias_fn=past_model_bias_fn, save_outputs=True,
                                                    randomize_finetune_weights=randomize_finetune_weights, dropout_noise=dropout_noise,
                                                    vector_path=init_vector_path, hidden_layer_size=hidden_layer_size, class_path=class_path,
                                                    identity_swap=identity_swap, amount_of_finetune=amount_of_finetune,
                                                    hidden_activation=hidden_activation, output_activation=output_activation, epochs=epochs,
                                                    learn_rate=learn_rate, is_identity=is_identity, batch_size=batch_size,
                                                    past_model_weights_fn=past_model_weights_fn, loss=loss, rewrite_files=rewrite_files,
                                                    file_name=file_name, from_ae=from_ae, finetune_size=finetune_size, data_type=data_type,
                                                               get_nnet_vectors_path= get_nnet_vectors_path, limit_entities=True,
                                                 vector_names_fn=vector_names_fn, classification_name=classification_name,
                                                                 identity_activation=identity_activation, lock_weights_and_redo=lock_weights_and_redo)
                                        ft_vector_path = loc + data_type + "/nnet/spaces/" + file_name + "L0.txt"
                                        ft_directions = loc + data_type + "/svm/directions/" + file_name + ".txt"
                                        #new_file_names[x] = file_name
                                        """
                                        svm.createSVM(ft_vector_path, bow_path, property_names_fn, file_name,
                                                      lowest_count=lowest_amt,
                                                      highest_count=highest_count, data_type=data_type,
                                                      get_kappa=score_type,
                                                      get_f1=False, svm_type=svm_type, getting_directions=True,
                                                      threads=threads, rewrite_files=rewrite_files,
                                                      classification=new_classification_task, lowest_amt=lowest_amt,
                                                      chunk_amt=chunk_amt, chunk_id=chunk_id)

                                        rank.getAllPhraseRankings(ft_directions, ft_vector_path, class_names_fn,
                                                                  vector_names_fn, file_name,
                                                                  data_type=data_type,
                                                                  rewrite_files=rewrite_files)

                                        ndcg.getNDCG(loc + data_type + "/rank/numeric/" + file_name + "ALL.txt",
                                                     file_name,
                                                     data_type=data_type, lowest_count=lowest_amt,
                                                     rewrite_files=rewrite_files,
                                                     highest_count=highest_count,
                                                     classification=new_classification_task)
                                        """

                                            #"n100mdsCV1S0 SFT0 allL030kappa23232 Breakoff CA1161650 MC1 MS0.4 ATS2000 DS2323300 OMS FMSFT NTlinear[100] NT[100]100linearS6040V1.1L1"

                                        print("got to trees, who dis?")

                                        tree.DecisionTree(nnet_ranking_fn, classification_path, label_names_fn, cluster_dict_fn, file_name + " " + classification_task, 10000,
                                                              max_depth=max_depth, balance="balanced", criterion="entropy", save_details=True,
                                                          data_type=data_type, csv_fn=csv_name, rewrite_files=rewrite_files,
                                                          cv_splits=cv_splits, split_to_use=splits, development=dt_dev, limit_entities=limit_entities,
                                                          limited_label_fn=limited_label_fn, vector_names_fn=vector_names_fn, clusters_fn=clusters_fn,
                                              cluster_duplicates=cluster_duplicates)

                                        tree.DecisionTree(nnet_ranking_fn, classification_path, label_names_fn, cluster_dict_fn, file_name  + " " + classification_task+ "None", 10000,
                                                              max_depth=None, balance="balanced", criterion="entropy", save_details=False,
                                                          data_type=data_type, csv_fn=csv_name, rewrite_files=rewrite_files,
                                                          cv_splits=cv_splits, split_to_use=splits, development=dt_dev, limit_entities=limit_entities,
                                                          limited_label_fn=limited_label_fn, vector_names_fn=vector_names_fn, clusters_fn=clusters_fn,
                                              cluster_duplicates=cluster_duplicates, multi_label=multi_label)


                                        ft_directions_fn = "../data/"+data_type+"/nnet/weights/" + file_name + "L1.txt"

                                        file_name = file_name + "cluster_ft_dir"

                                        ft_clusters_fn = loc + data_type + "/cluster/clusters/" + file_name + ".txt"
                                        ft_cluster_names_fn = loc + data_type + "/cluster/dict/" + file_name + ".txt"
                                        ft_cluster_dict_fn = loc + data_type + "/cluster/dict/" + file_name + ".txt"

                                        cluster_amt = hidden_layer_size * 2
                                        if breakoff:
                                            cluster.getClusters(ft_directions_fn, ft_score_fn, ft_names_fn, False, dissim_amt,
                                                                amount_to_start,
                                                                file_name, cluster_amt,
                                                                dissim, min_score, data_type, rewrite_files=rewrite_files,
                                                                half_kappa_half_ndcg=half_ndcg_half_kappa,
                                                                dont_cluster=dont_cluster)


                                            ranking_fn = loc + data_type + "/rank/numeric/" + file_name + ".txt"

                                            rank.getAllRankings(ft_clusters_fn, vector_path, ft_cluster_names_fn, vector_names_fn,
                                                                0.2, 1, False, file_name,
                                                                False, data_type=data_type, rewrite_files=rewrite_files)

                                            tree.DecisionTree(ranking_fn, classification_path, label_names_fn,
                                                              ft_cluster_dict_fn, file_name + " " + classification_task,
                                                              10000,
                                                              max_depth=max_depth, balance="balanced",
                                                              criterion="entropy", save_details=True,
                                                              data_type=data_type, csv_fn=csv_name,
                                                              rewrite_files=rewrite_files,
                                                              cv_splits=cv_splits, split_to_use=splits,
                                                              development=dt_dev, limit_entities=limit_entities,
                                                              limited_label_fn=limited_label_fn,
                                                              vector_names_fn=vector_names_fn, clusters_fn=clusters_fn,
                                                              cluster_duplicates=cluster_duplicates)
                                        """
                                        wekatree.DecisionTree(nnet_ranking_fn, classification_path, label_names_fn,
                                                              cluster_dict_fn, file_name,
                                                              save_details=True, data_type=data_type,
                                                              split_to_use=splits, pruning=2,
                                                              limited_label_fn=limited_label_fn, rewrite_files=rewrite_files,
                                                              csv_fn=csv_name, cv_splits=cv_splits,
                                                              limit_entities=limit_entities,
                                                              vector_names_fn=vector_names_fn)
                                        """

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
            """
                print("GETTING FNS")
                for a in range(len(csv_fns_dt)):
                    csv_fns_dt_a[a].append(csv_fns_dt[a])
                if not skip_nn:
                    for a in range(len(csv_fns_nn)):
                        csv_fns_nn_a[a].append(csv_fns_nn[a])

            for a in range(len(csv_fns_dt_a)):
                dt.averageCSVs(csv_fns_dt_a[a])
            """
            #if not skip_nn:
            #    for a in range(len(csv_fns_nn_a)):
            #        dt.averageCSVs(csv_fns_nn_a[a])
    loc ="../data/"+data_type+"/rules/tree_csv/"

    for fn in original_fn:
        avg_fn = fn[:-4] +"AVG.csv"
        fn = fn.split("/")[len(fn.split("/"))-1]
        try:
            fns_to_add = dt.getCSVsToAverage("../data/"+data_type+"/rules/tree_csv/",fn)
        except IndexError:
            fns_to_add = dt.getCSVsToAverage("../data/" + data_type + "/rules/tree_csv/", fn[:-4] + str(max_depth) + ".csv")
        all_csv_fns.append(fns_to_add)

    dt.arrangeByScore(
        np.unique(
            np.asarray(all_csv_fns))
        ,loc + " " + arrange_name + file_name[:50] + " " + classification_task + " " +  str(len(all_csv_fns)) + ".csv")
    #jvm.stop()

print("Begin top of parameters")

just_output = True
arcca = False
if arcca:
    loc = "/scratch/c1214824/data/"
else:
    loc = "../data/"

limit_entities = [False]

"""
data_type = "wines"
classification_task = ["types"]
file_name = "wines pca 100"
lowest_amt = 50
highest_amt = 10

init_vector_path = loc+data_type+"/nnet/spaces/wines100.txt"
vector_path_replacement = loc+data_type+"/nnet/spaces/wines100.txt"
get_nnet_vectors_path = loc+data_type+"/nnet/spaces/wines100.txt"
limit_entities = [True]
"""
"""
init_vector_path = loc+data_type+"/pca/class-all-50-10-alld100"
vector_path_replacement = loc+data_type+"/pca/class-all-50-10-alld100"
get_nnet_vectors_path = loc+data_type+"/nnet/spaces/films100-genres.txt"
"""
"""
data_type = "movies"
classification_task = ["ratings", "keywords", "genres"] #Run keywords as separate process
#arrange_name = arrange_name + classification_task[0]
skip_nn = True
if skip_nn is False:
    file_name = "f200ge"
else:
    file_name = "films200-genres"
lowest_amt = 100
highest_amt = 10
init_vector_path = loc+data_type+"/nnet/spaces/films200-genres.txt"
#init_vector_path = loc+data_type+"/nnet/spaces/films200-"+classification_task+".txt"
#file_name = "films200-genres100ndcg0.85200 tdev3004FTL0"
get_nnet_vectors_path = loc+data_type+"/nnet/spaces/films200-genres.txt"
vector_path_replacement = loc+data_type+"/nnet/spaces/films200-genres.txt"
if classification_task[0] == "us-ratings":
    deep_size = [200]
else:
    deep_size = [200]
"""
"""
data_type = "newsgroups"
classification_task = ["newsgroups"]
#arrange_name = arrange_name + classification_task[0]
skip_nn = True
if skip_nn is False:
    file_name = "n100mds2nnet"
else:
    file_name = "n100mds2"
lowest_amt = 30
highest_amt = 18836
init_vector_path = loc+data_type+"/nnet/spaces/mds.txt"
get_nnet_vectors_path = loc+data_type+"/nnet/spaces/mds.txt"
vector_path_replacement =  loc+data_type+"/nnet/spaces/mds.txt"
#init_vector_path = loc+data_type+"/bow/ppmi/class-all-50-0.95-all"
#get_nnet_vectors_path = loc+data_type+"/bow/ppmi/class-all-50-0.95-all"
#vector_path_replacement = loc+data_type+"/bow/ppmi/class-all-50-0.95-all"
deep_size = [100]

"""
data_type = "placetypes"
classification_task = ["opencyc"]
lowest_amt = 50
highest_amt = 10

places_size = 50
init_vector_path = "../data/"+data_type+"/nnet/spaces/places"+str(places_size)+".txt"
skip_nn = True
if skip_nn is False:
    file_name = "places mds "+str(places_size)
else:
    if places_size == 100:
        file_name = "places NONNET"
    else:
        file_name = "places NONNET"+str(places_size)

vector_path_replacement = loc+data_type+"/nnet/spaces/places"+str(places_size)+".txt"
get_nnet_vectors_path = loc + data_type + "/nnet/spaces/places"+str(places_size)+".txt"
deep_size = [places_size]

if classification_task[0] == "geonames" or classification_task[0] == "foursquare" or classification_task[0] == "newsgroups" :
    hidden_activation = "tanh"
    dropout_noise = [0.0]
    output_activation = "softmax"
    trainer = "adadelta"
    loss="categorical_crossentropy"
    class_weight = None
    lr = 0.01
    nnet_dev = False
    ep=400
else:
    hidden_activation = "tanh"
    dropout_noise = [0.0]
    output_activation = "sigmoid"
    trainer = "adagrad"
    loss="binary_crossentropy"
    class_weight = None
    nnet_dev = False
    if classification_task[0] == "ratings":
        ep = 1400
    elif classification_task[0] == "keywords":
        ep = 1500
    else:
        ep = 600
    lr = 0.01
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
"""
"""
lock_weights_and_redo = [False]

learn_rate= [ 0.001]
cutoff_start = 0.2
use_dropout_in_finetune = [False]

is_identity = [False]
amount_of_finetune = [[places_size] ]
ft_loss = ["mse"]
ft_optimizer = ["adagrad"]
min_size = 1

# Set to 0.0 for a janky skip, can set to 1.0 to delete it
sim_t = 1.0#1.0


min_score = 0.4
largest_cluster = 1
dissim = 0.0
dissim_amt = [2]
breakoff = [False]
score_limit = [0.9] #23232 val to use for all terms
amount_to_start = [2000]
cluster_multiplier = [2]#50 #23233  val to use for all terms
score_type = ["ndcg"]
use_breakoff_dissim = [False]
get_all = [False]
half_ndcg_half_kappa = [False]
add_all_terms = [False]
find_most_similar = True#False
only_most_similar = [True]
dont_cluster = [0]
save_results_so_far = False

ppmi_only = [0] #amt of ppmi to test
boc_only = [False]
pav_only = [False]


average_ppmi = [False]
bag_of_clusters = [True]
finetune_ppmi = [False]
average_nopav_ppmi_a = [False]
boc_average = [ False]
identity_activation = ["linear"]

top_dt_clusters = [False]
top_dt_clusters = [False]
by_class_finetune = [False]
use_pruned = False
cluster_duplicates = [False]
repeat_finetune = [0]


multi_label = [False] #Currently broken

epochs=[300]

"""
sim_t = 0.0#1.0
find_most_similar = False#False
cluster_multiplier = [50]#50
score_limit = [0.0]
"""
hp_opt = True

dt_dev = False
svm_classify = False
rewrite_files = False
max_depth = [3]

cross_val = 5
one_for_all = False

arrange_name = "cluster ratings BCS" + str(max_depth) + str(dt_dev)

threads=1
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
                 one_for_all, bag_of_clusters, arrange_name, only_most_similar, dont_cluster, top_dt_clusters, by_class_finetune,
                 cluster_duplicates, repeat_finetune, save_results_so_far, finetune_ppmi, average_nopav_ppmi_a, boc_average,
                 identity_activation, ppmi_only, boc_only, pav_only, multi_label, use_dropout_in_finetune, lock_weights_and_redo]

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
    arrange_name = args[54]
    only_most_similar = args[55]
    dont_cluster = args[56]
    top_dt_clusters = args[57]
    by_class_finetune = args[58]
    cluster_duplicates = args[59]
    repeat_finetune = args[60]
    save_results_so_far  = args[61]
    finetune_ppmi = args[62]
    average_nopav_ppmi_a = args[63]
    boc_average = args[64]
    identity_activation = args[65]
    ppmi_only = args[66]
    boc_only = args[67]
    pav_only = args[68]
    multi_label = args[69]
    use_dropout_in_finetune = args[70]
    lock_weights_and_redo = args[71]
if  __name__ =='__main__':
    print("begin main")
    for c in classification_task:
        ct_1 = [c]
        main(data_type, ct_1, file_name, init_vector_path, hidden_activation,
                                       is_identity, amount_of_finetune, breakoff, score_type, score_limit, rewrite_files,
                                       cluster_multiplier, threads, dropout_noise, learn_rate, epochs, cross_val, ep,
                                       output_activation, cutoff_start, deep_size, ct_1, highest_amt,
                                       lowest_amt, loss, nnet_dev, add_all_terms, average_ppmi, trainer, class_weight,
                                       amount_to_start, chunk_amt, chunk_id, lr, vector_path_replacement, dt_dev, use_pruned, max_depth,
                                       min_score, min_size, limit_entities, svm_classify, get_nnet_vectors_path, arcca, loc, largest_cluster,
                                       skip_nn, dissim, dissim_amt, hp_opt, find_most_similar, use_breakoff_dissim, get_all,
                                       half_ndcg_half_kappa, sim_t, one_for_all, ft_loss, ft_optimizer, bag_of_clusters, just_output,
                                       arrange_name, only_most_similar, dont_cluster, top_dt_clusters, by_class_finetune, cluster_duplicates,
                                       repeat_finetune, save_results_so_far, finetune_ppmi, average_nopav_ppmi_a, boc_average, identity_activation,
                                       ppmi_only, boc_only, pav_only, multi_label, use_dropout_in_finetune, lock_weights_and_redo)
