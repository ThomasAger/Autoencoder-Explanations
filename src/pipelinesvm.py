# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import numpy as np
import data as dt
import svm
import random
import sys
import time

def main(data_type, classification_task, file_name, init_vector_path, hidden_activation, is_identity, amount_of_finetune,
         breakoff, kappa, score_limit, rewrite_files, cluster_multiplier, threads, dropout_noise, learn_rate, epochs, cross_val, ep,
         output_activation, cs, deep_size, classification, direction_count, lowest_amt, loss, development, add_all_terms,
         average_ppmi, optimizer_name, class_weight, amount_to_start, chunk_amt, chunk_id, arcca, vector_path_replacement):
    print("start pipeline")
    if arcca:
        loc = "/scratch/c1214824/data/"
    else:
        loc = "../data/"

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
            class_weight = float(class_weight)
        else:
            class_weight = None
        rewrite_files = dt.toBool(rewrite_files)
        development = dt.toBool(development)
        add_all_terms = dt.toBool(add_all_terms)
        is_identity = dt.toBool(is_identity)
        average_ppmi = dt.toBool(average_ppmi)
        breakoff = dt.toBool(breakoff)
        kappa = dt.toBool(kappa)
        arcca = dt.toBool(arcca)
        score_limit = float(score_limit)
        amount_of_finetune = int(amount_of_finetune)
        chunk_amt = int(chunk_amt)
        chunk_id = int(chunk_id)

    cv_splits = cross_val
    init_vector_path = init_vector_path
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
        fn = file_name + " E" + str(ep) + " DS" + str(deep_size) + " DN" + str(dropout_noise) + " CT" + classification_task + \
                        " HA" + str(hidden_activation) + " CV" + str(cv_splits)  +  " S" + str(s) + " Dev" + str(development)
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
        random_number = random.random()
        deep_size = deep_size
        rewrite_files = rewrite_files
        print(file_name)
        print("SPLIT", str(splits))

        deep_fns = []
        for s in range(len(deep_size)):
            deep_fns.append(split_fns[splits] + " SFT" + str(s))

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
            lr = 0.01
            fine_tune_weights_fn = None
            ep = ep
            amount_of_finetune = 0
            batch_size = 200
            save_outputs = True
            dropout_noise = dropout_noise
            is_identity = False
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
                file_name = new_file_names[x]
                #file_name = vector_path_replacement
                """ Begin Filename """

                is_identity = is_identity
                breakoff = breakoff
                kappa = kappa

                file_name = file_name + str(lowest_amt)

                """ Begin Parameters """
                """ SVM """
                svm_type = "svm"
                highest_count = direction_count

                #vector_path = loc + data_type + "/nnet/spaces/"+new_file_names[x]+".txt"
                vector_path = vector_path_replacement

                bow_path = loc + data_type + "/bow/binary/phrases/class-all-" + str(lowest_amt) + "-" + str(
                    highest_count) + "-" + "all"
                property_names_fn = loc + data_type + "/bow/names/" + str(lowest_amt) + "-" + str(
                    highest_count) + "-" + "all" + ".txt"
                directions_fn = loc + data_type + "/svm/directions/" + file_name + ".txt"

                # Get rankings
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
                if average_ppmi:
                    final_fn = final_fn + "APPMI"

                if is_identity:
                    final_fn = final_fn + " IT"

                epochs = epochs
                final_fn = final_fn + str(epochs)
                final_fn = final_fn + "FT"
                final_fn = final_fn + "L0"

                svm.createSVM(vector_path, bow_path, property_names_fn, file_name, lowest_count=lowest_amt,
                  highest_count=highest_count, data_type=data_type, get_kappa=kappa,
                  get_f1=False, svm_type=svm_type, getting_directions=True, threads=threads, rewrite_files=rewrite_files,
                              classification="all", lowest_amt=lowest_amt, chunk_amt=chunk_amt, chunk_id=chunk_id, loc=loc)

                if chunk_amt > 0:
                    if chunk_id == chunk_amt-1:
                        print("compiling")
                        dt.compileSVMResults(file_name, chunk_amt, data_type)

                    else:
                        exit()
print("ran first part")
arcca = True
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
init_vector_path = "../data/"+data_type+"/nnet/spaces/wines100trimmed.txt"
"""

data_type = "movies"
classification_task = "ratings"
lowest_amt = 100
highest_amt = 10
#init_vector_path = "../data/"+data_type+"/nnet/spaces/films200-"+classification_task+".txt"
file_name = "f200ge"
init_vector_path = loc+data_type+"/nnet/spaces/films100.txt"
"""
data_type = "placetypes"
classification_task = "foursquare"
file_name = "placetypes ppmi"
lowest_amt = 50t
highest_amt = 10
init_vector_path = "../data/"+data_type+"/nnet/spaces/places100-"+classification_task+".txt"
"""
"""
hidden_activation = "relu"
dropout_noise = 0.5
output_activation = "sigmoid"
cutoff_start = 0.2
ep=200
"""
hidden_activation = "tanh"
dropout_noise = 0.5
output_activation = "sigmoid"
trainer = "adagrad"
loss="binary_crossentropy"
cutoff_start = 0.2
deep_size = [200]
#init_vector_path = "../data/"+data_type+"/bow/ppmi/class-all-"+str(lowest_amt)+"-"+str(highest_amt)+"-"+classification_task
ep =100

class_weight = None
is_identity = True
amount_of_finetune = 1

breakoff = True
score_limit = 0.81
cluster_multiplier = 1
epochs=3000
learn_rate=0.001
kappa = False
development = False
add_all_terms = True
average_ppmi = False
amount_to_start = 1000

cross_val = 1
vector_path_replacement = loc+data_type+"/nnet/spaces/films100.txt"
rewrite_files = False
threads=3
chunk_amt = 20
chunk_id = 0
variables = [data_type, classification_task, file_name, init_vector_path, hidden_activation,
                               is_identity, amount_of_finetune, breakoff, kappa, score_limit, rewrite_files,
                               cluster_multiplier, threads, dropout_noise, learn_rate, epochs, cross_val, ep,
                               output_activation, cutoff_start, deep_size, classification_task, highest_amt,
                               lowest_amt, loss, development, add_all_terms, average_ppmi, trainer, class_weight,
                               amount_to_start, chunk_amt, chunk_id, arcca]
sys.stdout.write("python pipelinesvm.py ")

cmd = "python $SRCPATH/pipelinesvm.py "
count = 0
for v in variables:
    if type(v) == str:
        v = '"' + v + '"'
    if type(v) == list:
        v = '"' + str(v) + '"'
    sys.stdout.write(str(v) + " ")
    if count < len(variables)-2:
        cmd += str(v) + " "
    count += 1

print("living dream")
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
    development = args[25]
    print(development)
    add_all_terms = args[26]
    average_ppmi = args[27]
    trainer = args[28]
    class_weight = args[29]
    amount_to_start = args[30]
    chunk_amt = args[31]
    chunk_id = args[32]
    arcca = args[33]

if len(args) == 0:
    for c in range(chunk_amt):
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
                         cmd + str(c) + " True"], "../data/"+data_type+"/cmds/pipelinesvm"+str(c)+".sh")

print("got to final area")
if  __name__ =='__main__':main(data_type, classification_task, file_name, init_vector_path, hidden_activation,
                               is_identity, amount_of_finetune, breakoff, kappa, score_limit, rewrite_files,
                               cluster_multiplier, threads, dropout_noise, learn_rate, epochs, cross_val, ep,
                               output_activation, cutoff_start, deep_size, classification_task, highest_amt,
                               lowest_amt, loss, development, add_all_terms, average_ppmi, trainer, class_weight,
                               amount_to_start, chunk_amt, chunk_id, arcca, vector_path_replacement)
