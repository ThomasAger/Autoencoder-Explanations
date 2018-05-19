from sklearn.decomposition import NMF, LatentDirichletAllocation
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import data as dt
import numpy as np
import tree
from itertools import product
from sklearn.externals import joblib
import scipy.sparse as sp

def print_top_words(model, feature_names):
    names = []
    for topic_idx, topic in enumerate(model.components_):
        message = ""
        message += " ".join([feature_names[i]
                             for i in topic.argsort()])
        print(message[:100])
        names.append(message)
    print()
    return names

def save_model(model, feature_names, n_top_words):
    print("test")

def LDA(tf, names, components, file_name,   doc_topic_prior, topic_word_prior,  data_type, rewrite_files):
    # Removed model name as it was unused and I manually renamed a bunch of files and was too lazy to do model too
    rep_name = "../data/"+data_type+"/LDA/rep/"+file_name+".txt"
    model_name = "../data/"+data_type+"/LDA/model/"+file_name+".txt"
    names_name = "../data/"+data_type+"/LDA/names/"+file_name+".txt"

    all_names = [rep_name,  names_name]

    if dt.allFnsAlreadyExist(all_names) and not rewrite_files:
        print("Already completed")
        return
    print(len(tf), print(len(tf[0])))

    print("Fitting LDA models with tf features,")
    lda = LatentDirichletAllocation(doc_topic_prior=doc_topic_prior, topic_word_prior=topic_word_prior, n_topics=components )
    t0 = time()
    tf = np.asarray(tf).transpose()
    new_rep = lda.fit_transform(tf)
    print("done in %0.3fs." % (time() - t0))

    print("\nTopics in LDA model:")
    topics = print_top_words(lda, names)
    topics.reverse()
    dt.write1dArray(topics, "../data/" + data_type + "/LDA/names/" + file_name + ".txt")
    dt.write2dArray(new_rep.transpose(), rep_name)
    joblib.dump(lda, model_name)

def main(data_type, class_labels_fn, class_names_fn, ft_names_fn, max_depth, limit_entities,
         limited_label_fn, vector_names_fn, dt_dev, doc_topic_prior, topic_word_prior, n_topics, file_name, final_csv_name,
         high_amt, low_amt, cross_val, rewrite_files, classify, tf_fn):

    print("importing class all")
    tf = np.asarray(sp.load_npz("../data/"+data_type+"/bow/frequency/phrases/"+tf_fn).todense())
    names = dt.import1dArray(ft_names_fn)
    variables_to_execute = list(product(doc_topic_prior, topic_word_prior, n_topics))
    print("executing", len(variables_to_execute), "variations")
    csvs = []
    csv_fns = []
    file_names = []
    for vt in variables_to_execute:
        doc_topic_prior = vt[0]
        topic_word_prior = vt[1]
        n_topics = vt[2]
        file_names.append(file_name + "DTP" + str(doc_topic_prior) + "TWP" + str(topic_word_prior) + "NT" + str(n_topics))
    final_csv_fn = "../data/"+data_type+"/rules/tree_csv/"+ file_name + final_csv_name + ".csv"
    for vt in range(len(variables_to_execute)):
        doc_topic_prior = variables_to_execute[vt][0]
        topic_word_prior = variables_to_execute[vt][1]
        n_topics = variables_to_execute[vt][2]
        file_name = file_names[vt]
        LDA(tf, names, n_topics, file_name,
            doc_topic_prior, topic_word_prior, data_type, rewrite_files)

        dimension_names_fn = "../data/" + data_type + "/LDA/names/" + file_name + ".txt"

        #NMFFrob(dt.import2dArray("../data/"+data_type+"/bow/ppmi/class-all-100-10-all"),  dt.import1dArray("../data/"+data_type+"/bow/names/100.txt"), 200, file_name)

        topic_model_fn = "../data/" + data_type + "/LDA/rep/" + file_name + ".txt"
        cv_fns = []
        og_fn = file_name
        for c in range(cross_val):
            file_name = og_fn + " " + str(cross_val) + "CV " + str(c) + classify + "Dev" + str(dt_dev)
            csv_name = "../data/" + data_type + "/rules/tree_csv/" + file_name + ".csv"
            cv_fns.append(csv_name)

            tree.DecisionTree(topic_model_fn, class_labels_fn, class_names_fn, dimension_names_fn, file_name, 10000,
                              max_depth=1, balance="balanced", criterion="entropy", save_details=False,
                              cv_splits=cross_val,
                              split_to_use=c, data_type=data_type, csv_fn=csv_name, rewrite_files=rewrite_files,
                              development=dt_dev,
                              limit_entities=limit_entities,
                              limited_label_fn=limited_label_fn, vector_names_fn=vector_names_fn,
                              clusters_fn=topic_model_fn,
                              cluster_duplicates=True, save_results_so_far=False)

            tree.DecisionTree(topic_model_fn, class_labels_fn, class_names_fn, dimension_names_fn, file_name, 10000,
                              max_depth=max_depth, balance="balanced", criterion="entropy", save_details=False, cv_splits=cross_val,
                              split_to_use=c,  data_type=data_type, csv_fn=csv_name, rewrite_files=rewrite_files, development=dt_dev,
                              limit_entities=limit_entities,
                              limited_label_fn=limited_label_fn, vector_names_fn=vector_names_fn, clusters_fn=topic_model_fn,
                              cluster_duplicates=True, save_results_so_far=False)

            tree.DecisionTree(topic_model_fn, class_labels_fn, class_names_fn, dimension_names_fn, file_name + "None", 10000,
                              max_depth=None, balance="balanced", criterion="entropy", save_details=False,
                              data_type=data_type, csv_fn=csv_name, rewrite_files=rewrite_files,cv_splits=cross_val,
                              split_to_use=c,  development=dt_dev, limit_entities=limit_entities,
                              limited_label_fn=limited_label_fn, vector_names_fn=vector_names_fn, clusters_fn=topic_model_fn,
                              cluster_duplicates=True, save_results_so_far=False)

        dt.averageCSVs(cv_fns)
        file_name = og_fn + " " + str(cross_val) + "CV " + str(0) + classify + "Dev" + str(dt_dev)
        csvs.append("../data/" + data_type + "/rules/tree_csv/" + file_name + "AVG.csv")
    dt.arrangeByScore(np.unique(np.asarray(csvs)), final_csv_fn)
data_type = "newsgroups"
high_amt = 30
low_amt = 18836

#all-30-18836DTP0.001TWP0.1NT200

#all-100-10DTP0.1TWP0.001NT400 1CV 0genresDevTrueAVG.csv
#all-100-10DTP0.1TWP0.001NT400 1CV 0keywordsDevTrueAVG.csv

#all-100-10DTP0.1TWP0.01NT100 1CV 0ratingsDevTrueAVG.csv



classify = ["newsgroups"]

max_depth = 3
limit_entities = False
dt_dev = False
vector_names_fn = "../data/" + data_type + "/nnet/spaces/entitynames.txt"
feature_names_fn = "../data/" + data_type + "/bow/names/"+str(high_amt)+"-"+str(low_amt)+"-all.txt"
rewrite_files = False
cross_val = 1
tf_fn = "simple_numeric_stopwords_bow 30-0.999-all.npz"

doc_topic_prior = [ 0.01]
topic_word_prior = [ 0.1]
n_topics = [50]
for c in classify:
    file_name = "simple_numeric_stopwords_bow 30-0.999-all.npz"
    final_csv_name = "final" + c + str(dt_dev)
    class_labels_fn = "../data/" + data_type + "/classify/"+c+"/class-all"
    class_names_fn = "../data/" + data_type + "/classify/"+c+"/names.txt"
    limited_label_fn = "../data/" + data_type + "/classify/" + c + "/available_entities.txt"
    if  __name__ =='__main__':main(data_type, class_labels_fn, class_names_fn, feature_names_fn, max_depth, limit_entities,
             limited_label_fn, vector_names_fn, dt_dev, doc_topic_prior, topic_word_prior, n_topics, file_name, final_csv_name,
                                   high_amt, low_amt, cross_val, rewrite_files, c, tf_fn)