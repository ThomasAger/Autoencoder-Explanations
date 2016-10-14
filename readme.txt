pipeline.py is a pipeline to perform all tasks in order to obtain a space that has been fine-tuned for interpretability.
src contains all of the files required to do this, in the form of command line tools (nnet, svm, rank, cluster, gini, pav, plot)
data contains all of the data produced by each of these tools, with each tool relying on different kinds of data. All data that is numeric has been saved as numpy files, all data that is interpretable has been saved as a csv.
hypothesis is used for hypothesis driven testing
opencsv is used to read/write files
keras is used for neural networks
scikit-learn is used for the svm
the clustering and ranking code uses similarity tasks from scikit-learn, but is otherwise hand-written
