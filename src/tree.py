import numpy as np
import helper.data as dt
import pydotplus as pydot
from sklearn import tree
from sklearn.metrics import f1_score, accuracy_score
from inspect import getmembers
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
import jsbeautifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold

class DecisionTree:
    clf = None
    def __init__(self, features_fn, classes_fn,  class_names_fn, cluster_names_fn, filename,
                 training_data,  max_depth=None, balance=None, criterion="entropy", save_details=False, data_type="movies",cv_splits=5,
                 csv_fn="../data/temp/no_csv_provided.csv", cross_validation=False, rewrite_files=False):

        vectors = np.asarray(dt.import2dArray(features_fn))

        labels = np.asarray(dt.import2dArray(classes_fn, "i"))

        vectors = vectors.transpose()
        labels = labels.transpose()

        cluster_names = dt.import1dArray(cluster_names_fn)
        label_names = dt.import1dArray(class_names_fn)
        all_fns = []
        if save_details:
            dot_file_fn = '../data/' + data_type + '/rules/tree_data/' + label_names[0] + " " + filename + '.txt'
            graph_fn = '../data/' + data_type + '/rules/tree_data/' + label_names[0] + " " + filename + '.txt'
            graph_png_fn = '../data/' + data_type + '/rules/tree_images/' + label_names[0] + " " + filename + ".png"
            all_fns.append(dot_file_fn)
            all_fns.append(graph_fn)
            all_fns.append(graph_png_fn)
        file_names = ['ACC ' + filename, 'F1 ' + filename]
        acc_fn = '../data/' + data_type + '/rules/tree_scores/' + file_names[0] + '.scores'
        f1_fn = '../data/' + data_type + '/rules/tree_scores/' + file_names[1] + '.scores'
        all_fns.append(acc_fn)
        all_fns.append(f1_fn)

        if dt.allFnsAlreadyExist(all_fns) and not rewrite_files:
            print("Skipping task", "DecisionTree")
            return
        else:
            print("Running task", "DecisionTree")

        x_train, x_test, y_train, y_test = train_test_split(vectors, labels, test_size=0.3, random_state=0)

        filename += str(max_depth)
        for l in range(len(cluster_names)):
            cluster_names[l] = cluster_names[l].split()[0]
        for l in range(len(label_names)):
            if label_names[l][:6] == "class-":
                label_names[l] = label_names[l][6:]

        scores_array = []
        f1_array = []
        accuracy_array = []
        params  = []

        y_train = y_train.transpose()
        y_test = y_test.transpose()

        for l in range(len(y_train)):
            c_x_train = x_train
            c_y_train = y_train[l]
            c_y_test = y_test[l]
            """
            pipeline = Pipeline([('clf', tree.DecisionTreeClassifier(criterion=criterion, random_state=20000, class_weight=balance))])

            parameters = {
                'clf__max_depth': (3, 5, 10, 25),
                'clf__min_samples_split': (4, 7, 2),
                'clf__min_samples_leaf': (1 ,4, 7),
                'clf__min_weight_fraction_leaf': (0.1, 0.2, 0.4),
                'clf__max_leaf_nodes': (2, 4, 6, None)
            }

            grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='f1')
            grid_search.fit(x_development, y_development)
            print('Best score: %0.3f' % grid_search.best_score_)
            print('Best parameters set:')
            best_parameters = grid_search.best_estimator_.get_params()
            for param_name in sorted(parameters.keys()):
                print( '\t%s: %r' % (param_name, best_parameters[param_name]))

            dt.writeArrayDict1D(best_parameters,"../data/movies/rules/tree_params/" + filename + str(l) +".txt")
            params.append(best_parameters)
            """
            #balanced_x_train, y_train = dt.balanceClasses(x_train, y_train)

            clf = tree.DecisionTreeClassifier( max_depth=max_depth, criterion=criterion, class_weight=balance)

            # Select training data with cross validation
            if cross_validation:
                kf = KFold(n_splits=cv_splits, shuffle=False, random_state=None)
                predicted_labels = cross_val_score(clf, c_x_train, c_y_train, cv=kf, scoring="f1", verbose=1)
                print("Cross val score", predicted_labels)
                index = np.argmax(predicted_labels, 0)
                counter = 0
                for train, test in kf.split(c_x_train):
                    if counter == index:
                        c_x_train = c_x_train[train]
                        c_y_train = c_y_train[train]
                        break
                    else:
                        counter += 1
            clf = clf.fit(c_x_train, c_y_train)
            predicted_test = clf.predict(x_test)
            f1 = f1_score(c_y_test, predicted_test, average="macro")
            accuracy = accuracy_score(c_y_test, predicted_test)
            f1_array.append(f1)
            accuracy_array.append(accuracy)
            scores = [[label_names[l], "f1", f1, "accuracy", accuracy]]
            print(scores)
            class_names = [ label_names[l], "NOT "+label_names[l]]



            # Export a tree for each label predicted by the clf
            if save_details:
                dot_file_fn = '../data/' + data_type + '/rules/tree_data/' + label_names[l] + " " + filename + '.txt'
                graph_fn = '../data/' + data_type + '/rules/tree_data/' + label_names[l] + " " + filename + '.txt'
                graph_png_fn = '../data/' + data_type + '/rules/tree_images/' + label_names[l] + " " + filename + ".png"
                tree.export_graphviz(clf, feature_names=cluster_names, class_names=class_names, out_file=dot_file_fn, max_depth=max_depth)
                rewrite_dot_file = dt.import1dArray(dot_file_fn)
                new_dot_file = []
                for s in rewrite_dot_file:
                    new_string = s
                    if "->" not in s and "digraph" not in s and "node" not in s and "(...)" not in s and "}" not in s:
                        index = s.index("value")
                        new_string = s[:index] + '"] ;'
                    new_dot_file.append(new_string)
                dt.write1dArray(new_dot_file, dot_file_fn)

                graph = pydot.graph_from_dot_file(graph_fn)
                graph.write_png(graph_png_fn)
                self.get_code(clf, cluster_names, class_names, label_names[l]+ " " + filename, data_type)

        accuracy_array = np.asarray(accuracy_array)
        accuracy_average = np.average(accuracy_array)

        f1_array = np.asarray(f1_array)
        f1_average = np.average(f1_array)

        accuracy_array = np.append(accuracy_array, accuracy_average)
        f1_array = np.append(f1_array, f1_average)


        scores = [accuracy_array, f1_array]

        dt.write1dArray(accuracy_array, acc_fn)
        dt.write1dArray(f1_array, f1_fn)

        if dt.file_exists(csv_fn):
            print("File exists, writing to csv")
            dt.write_to_csv(csv_fn, file_names, scores)
        else:
            print("File does not exist, recreating csv")
            key = []
            for l in label_names:
                key.append(l)
            key.append("AVERAGE")
            dt.write_csv(csv_fn, file_names, scores, key)
        """
        dt.write2dArray(params, "../data/movies/rules/tree_paramas/" + filename +".txt")
        """

    def get_code(self, tree, feature_names, class_names, filename, data_type):
        left      = tree.tree_.children_left
        right     = tree.tree_.children_right
        threshold = tree.tree_.threshold
        value = tree.tree_.value

        #print tree.tree_.feature, len(tree.tree_.feature
        # )
        features = []
        for i in tree.tree_.feature:
            if i != -2 or i <= 200:
                features.append(feature_names[i])
        rules_array = []
        def recurse(left, right, threshold, features,  node):
                if (threshold[node] != -2):
                        line = "IF ( " + features[node] + " <= " + str(threshold[node]) + " ) {"
                        rules_array.append(line)
                        if left[node] != -1:
                                recurse (left, right, threshold, features,left[node])
                        line = "} ELSE {"
                        rules_array.append(line)
                        if right[node] != -1:
                                recurse (left, right, threshold, features,right[node])
                        line = "}"
                        rules_array.append(line)
                else:
                        if value[node][0][0] >= value[node][0][1]:
                            line = "return", class_names[0]
                            rules_array.append(line)
                        else:
                            line = "return", class_names[1]
                            rules_array.append(line)
        recurse(left, right, threshold, features, 0)
        dt.write1dArray(rules_array, "../data/" + data_type + "/rules/text_rules/"+filename+".txt")
        cleaned = jsbeautifier.beautify_file("../data/" + data_type + "/rules/text_rules/"+filename+".txt")
        file = open("../data/" + data_type + "/rules/text_rules/"+filename+".txt", "w")
        file.write(cleaned)
        file.close()



def main():
    cluster_to_classify = -1
    max_depth = None
    classify = "types"
    data_type = "wines"
    cross_val = True
    save_details = True
    label_names_fn = "../data/"+data_type+"/classify/"+classify+"/names.txt"
    cluster_labels_fn = "../data/"+data_type+"/classify/"+classify+"/class-All"
    threshold = 0.9
    split = 0.1
    file_name = "ndcg0.9200pavPPMIITsgdmse1000SFT1svm0.9200SFT2svm0.9200"
    criterion = "entropy"
    balance = "balanced"
    cluster_names_fn = "../data/"+data_type+"/cluster/hierarchy_names/"+file_name+".txt"
    cluster_names_fn = "../data/movies/bow/names/200.txt"
    #cluster_names_fn = "../data/movies/cluster/names/" + file_name + ".txt"
    #cluster_vectors_fn = "../data/movies/rank/numeric/" + file_name + "400.txt"
    #file_name = "L3" + file_name + "L3100N0.5InClusterN0.5FTadagradcategorical_crossentropy100Genres100L3L4"

    #vector_fn = "films100svmndcg0.9240pavPPMIN0.5FTRsgdmse1000"
    vector_fn = "wines100trimmedsvmkappa0.9200"
    csv_name = vector_fn#"wines100trimmedsvmkappa0.9200"
    csv_fn = "../data/"+data_type+"/rules/tree_csv/"+csv_name+".csv"
    #vector_fn = "films100"
    #cluster_vectors_fn = "../data/"+data_type+"/cluster/all_directions/" +file_name + ".txt"
    #file_name = file_name + "all_dir"
    #cluster_vectors_fn = "../data/"+data_type+"/nnet/clusters/"+vector_fn+".txt"
    #file_name = file_name + "nnet_rank"
    #cluster_vectors_fn = "../data/"+data_type+"/finetune/"+vector_fn+".txt"
    #file_name = file_name + "finetune_pavppmi"
    #cluster_vectors_fn = "../data/"+data_type+"/nnet/spaces/"+vector_fn+".txt"
    #file_name = file_name + "vector"
    cluster_vectors_fn = "../data/"+data_type+"/rank/numeric/"+vector_fn+".txt"
    file_name = file_name + "ranks"
    file_name = vector_fn + classify + str(max_depth)

    clf = DecisionTree(cluster_vectors_fn, cluster_labels_fn, label_names_fn , cluster_names_fn , file_name, 10000,
                       max_depth, balance=balance, criterion=criterion, save_details=save_details, data_type=data_type,
                       csv_fn=csv_fn, cross_validation=cross_val)

    """
    fn = "films100"
    cluster_names_fn = "../data/movies/cluster/names/"+fn+".txt"
    file_name = fn + "InClusterN0.5FTITSadagradbinary_crossentropy100"
    #file_name = fn + str(cutoff) + str(amt_of_clusters)
    cluster_vectors_fn = "../data/movies/nnet/clusters/" + file_name + ".txt"
    #cluster_vectors_fn = "../data/movies/rank/numeric/" + fn + ".txt"
    #cluster_names_fn = "../data/movies/cluster/hierarchy_names/" + file_name + ".txt"
    clf = DecisionTree(cluster_vectors_fn, cluster_labels_fn, label_names_fn, cluster_names_fn, file_name, 10000,
                       max_depth)

    fn = "films200L325N0.5"
    cluster_names_fn = "../data/movies/cluster/names/"+fn+".txt"
    file_name = fn + "InClusterN0.5FTadagradcategorical_crossentropy100"
    cluster_vectors_fn = "../data/movies/nnet/clusters/" + file_name + ".txt"
    clf = DecisionTree(cluster_vectors_fn, cluster_labels_fn, label_names_fn, cluster_names_fn, file_name, 10000,
                       max_depth)
    fn = "films200L250N0.5"
    cluster_names_fn = "../qdata/movies/cluster/names/"+fn+".txt"
    file_name = fn + "InClusterN0.5FTadagradcategorical_crossentropy100"
    cluster_vectors_fn = "../data/movies/nnet/clusters/" + file_name + ".txt"
    clf = DecisionTree(cluster_vectors_fn, cluster_labels_fn, label_names_fn, cluster_names_fn, file_name, 10000,
                       max_depth)

    fn = "films200L1100N0.5"
    cluster_names_fn = "../data/movies/cluster/names/" + fn + ".txt"
    file_name = fn + "InClusterN0.5FTadagradcategorical_crossentropy100"
    cluster_vectors_fn = "../data/movies/nnet/clusters/" + file_name + ".txt"
    clf = DecisionTree(cluster_vectors_fn, cluster_labels_fn, label_names_fn, cluster_names_fn, file_name, 10000,
                       max_depth)
    """


if  __name__ =='__main__':main()