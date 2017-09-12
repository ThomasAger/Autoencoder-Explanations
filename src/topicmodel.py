from sklearn.decomposition import NMF, LatentDirichletAllocation
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import data as dt
import numpy as np
import tree
from itertools import product
from sklearn.externals import joblib

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

def LDA(tf, names, components, file_name,   doc_topic_prior, topic_word_prior,  data_type):
    rep_name = "../data/"+data_type+"/LDA/rep/"+file_name+".txt"
    model_name = "../data/"+data_type+"/LDA/model/"+file_name+".txt"
    names_name = "../data/"+data_type+"/LDA/names/"+file_name+".txt"

    all_names = [rep_name, model_name, names_name]

    if dt.allFnsAlreadyExist(all_names):
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
    dt.write2dArray(topics, "../data/" + data_type + "/LDA/names/" + file_name + ".txt")
    dt.write2dArray(new_rep.transpose(), rep_name)
    joblib.dump(lda, model_name)

def main(data_type, class_labels_fn, class_names_fn, feature_names_fn, max_depth, limit_entities,
         limited_label_fn, vector_names_fn, dt_dev, doc_topic_prior, topic_word_prior, n_topics, file_name, final_csv_name,
         high_amt, low_amt):

    print("importing class all")
    tf = dt.import2dArray("../data/"+data_type+"/bow/frequency/phrases/class-all-"+str(high_amt)+"-"+str(low_amt)+"-all")
    names = dt.import1dArray(feature_names_fn)
    variables_to_execute = list(product(doc_topic_prior, topic_word_prior, n_topics))
    print("executing", len(variables_to_execute), "variations")
    csvs = []
    file_names = []
    for vt in variables_to_execute:
        doc_topic_prior = vt[0]
        topic_word_prior = vt[1]
        n_topics = vt[2]
        file_names.append(file_name + "DTP" + str(doc_topic_prior) + "TWP" + str(topic_word_prior) + "NT" + str(n_topics))
    for vt in range(len(variables_to_execute)):
        doc_topic_prior = variables_to_execute[vt][0]
        topic_word_prior = variables_to_execute[vt][1]
        n_topics = variables_to_execute[vt][2]
        file_name = file_names[vt]

        LDA(tf, names, n_topics, file_name,
            doc_topic_prior, topic_word_prior, data_type)

        #NMFFrob(dt.import2dArray("../data/"+data_type+"/bow/ppmi/class-all-100-10-all"),  dt.import1dArray("../data/"+data_type+"/bow/names/100.txt"), 200, file_name)

        topic_model_fn = "../data/" + data_type + "/LDA/rep/" + file_name + ".txt"

        csv_name = "../data/" + data_type + "/rules/tree_csv/" + file_name + ".csv"
        csvs.append(csv_name)

        tree.DecisionTree(topic_model_fn, class_labels_fn, class_names_fn, feature_names_fn, file_name, 10000,
                          max_depth=max_depth, balance="balanced", criterion="entropy", save_details=True, cv_splits=1,
                          split_to_use=0,
                          data_type=data_type, csv_fn=csv_name, rewrite_files=False, development=dt_dev,
                          limit_entities=limit_entities,
                          limited_label_fn=limited_label_fn, vector_names_fn=vector_names_fn, clusters_fn=topic_model_fn,
                          cluster_duplicates=True, save_results_so_far=False)

        tree.DecisionTree(topic_model_fn, class_labels_fn, class_names_fn, feature_names_fn, file_name + "None", 10000,
                          max_depth=None, balance="balanced", criterion="entropy", save_details=False,
                          data_type=data_type, csv_fn=csv_name, rewrite_files=False,
                          cv_splits=1, split_to_use=0, development=dt_dev, limit_entities=limit_entities,
                          limited_label_fn=limited_label_fn, vector_names_fn=vector_names_fn, clusters_fn=topic_model_fn,
                          cluster_duplicates=True, save_results_so_far=False)

    dt.arrangeByScore(np.unique(np.asarray(csvs)), "../data/"+data_type+"/rules/tree_csv/"+file_name + final_csv_name + ".csv")
data_type = "movies"
high_amt = 100
low_amt = 10

classify = ["genres", "keywords", "ratings"]
feature_names_fn = "../data/" + data_type + "/bow/names/"+str(high_amt)+".txt"
max_depth = 3
limit_entities = False
dt_dev = True
vector_names_fn = "../data/" + data_type + "/nnet/spaces/entitynames.txt"



doc_topic_prior = {0.1, 0.01, 0.001}
topic_word_prior = {0.1, 0.01, 0.001}
n_topics = [10,30,50]
for c in classify:
    file_name = "all-" + str(high_amt) + "-" + str(low_amt) + c
    final_csv_name = "recommended params"
    class_labels_fn = "../data/" + data_type + "/classify/"+c+"/class-all"
    class_names_fn = "../data/" + data_type + "/classify/"+c+"/names.txt"
    limited_label_fn = "../data/" + data_type + "/classify/" + c + "/available_entities.txt"
    if  __name__ =='__main__':main(data_type, class_labels_fn, class_names_fn, feature_names_fn, max_depth, limit_entities,
             limited_label_fn, vector_names_fn, dt_dev, doc_topic_prior, topic_word_prior, n_topics, file_name, final_csv_name,
                                   high_amt, low_amt)