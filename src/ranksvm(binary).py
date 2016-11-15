import helper.data as dt
import numpy as np
from sklearn import datasets
import subprocess
import os

file_name = "films100"
output_fn = "../data/movies/ranksvm/svmlight/" +file_name +".dat"
rank_svm_output_fn = "../data/movies/ranksvm/results/" +file_name +".dat"
try:
    file = open(output_fn)
except FileNotFoundError:
    vectors = np.asarray(dt.import2dArray("../data/movies/nnet/spaces/" + file_name + ".txt"))[:3]
    classes = np.asarray(dt.import2dArray("../data/movies/bow/frequency/phrases/class-all-200")).transpose()[:3]
    datasets.dump_svmlight_file(vectors, classes, output_fn, multilabel=True)

print("Running")
# Command with shell expansion
subprocess.call('"../library/ranksvm/svm_rank_learn.exe" -c 20.0 -y 3 "'+output_fn+'" "'+rank_svm_output_fn+'" ', shell=True)