pipeline.py is a pipeline to perform all tasks in order to obtain a space that has been fine-tuned for interpretability.
src contains all of the files required to do this, in the form of command line tools (nnet, svm, rank, cluster, gini, pav, plot)
data contains all of the data produced by each of these tools, with each tool relying on different kinds of data. All data that is numeric has been saved as numpy files, all data that is interpretable has been saved as a csv.
hypothesis is used for hypothesis driven testing
opencsv is used to read/write files
keras is used for neural networks
scikit-learn is used for the svm
the clustering and ranking code uses similarity tasks from scikit-learn, but is otherwise hand-written

#data is for storing en-masse data that is not produced by the pipeline, and does not need to be loaded or saved based on parameter configurations
#data_store is for storing the data with UUID names so that they can be accessed by data_manage easily
#data_manage is where csv files for each of the methods is stored, and every parameter configuration with its associated files and parameters are listed. 
#data_request is where csv files are generated based on user requests, e.g. all of the scores for all of the different data_types

master.csv is for the pipeline, and lists the run_id and the parameters overall for that run

in #data_manage, each csv file corresponds to a method and has the following columns:

run_id, connected to master.csv 
filename_[1,2,...,N], the output files of the method ## All saved as either npy or npz
data_type, the dataset being used
param_[Name,Name,...,Name], the parameters of the method

