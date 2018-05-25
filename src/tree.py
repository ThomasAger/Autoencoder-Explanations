import numpy as np
import data as dt
import pydotplus as pydot
import math
from sklearn import tree
from sklearn.metrics import f1_score, accuracy_score
import jsbeautifier
from sklearn.model_selection import KFold
import random
from sklearn.externals import joblib
from sklearn.metrics import precision_recall_fscore_support
import graphviz
import os
class DecisionTree:
    clf = None
    def __init__(self, features_fn, classes_fn,  class_names_fn, cluster_names_fn, filename,
                 training_data,  max_depth=None, balance=None, criterion="entropy", save_details=False, data_type="movies",cv_splits=5,
                 csv_fn="../data/temp/no_csv_provided.csv", rewrite_files=False, split_to_use=-1, development=False,
                 limit_entities=False, limited_label_fn=None, vector_names_fn=None, clusters_fn="", cluster_duplicates=False,
                 save_results_so_far=False, multi_label=False):

        label_names = dt.import1dArray(class_names_fn)

        filename = filename + str(max_depth)

        all_fns = []
        file_names = ['ACC ' + filename, 'F1 ' + filename]
        acc_fn = '../data/' + data_type + '/rules/tree_scores/' + file_names[0] + '.scores'
        prediction_fn = '../data/' + data_type + '/rules/tree_output/' + filename + '.scores'
        f1_fn = '../data/' + data_type + '/rules/tree_scores/' + file_names[1] + '.scores'
        all_top_names_fn = "../data/"+data_type+"/rules/names/" + filename + ".txt"
        all_top_rankings_fn = "../data/"+data_type+"/rules/rankings/" + filename + ".txt"
        all_top_clusters_fn = "../data/"+data_type+"/rules/clusters/" + filename + ".txt"

        fns_name = "../data/" + data_type + "/rules/names/" + filename + label_names[0] + ".txt"
        features_name = "../data/" + data_type + "/rules/rankings/" + filename + label_names[0] + ".txt"
        dt_clusters_name = "../data/" + data_type + "/rules/clusters/" + filename + label_names[0] + ".txt"
        if save_details is False:
            all_fns = [acc_fn, f1_fn, prediction_fn, csv_fn]
        else:
            new_graph_png_fn = '../data/' + data_type + '/rules/tree_images/' + label_names[0] + " " + filename + '.png'
            all_fns = [acc_fn, f1_fn, prediction_fn, csv_fn, new_graph_png_fn]

        if max_depth is not None:
            all_fns.append(all_top_names_fn)
            all_fns.append(all_top_rankings_fn)
            all_fns.append(all_top_clusters_fn)

        if save_details:
            orig_dot_file_fn = '../data/' + data_type + '/rules/tree_data/' + label_names[0] + " " + filename  + 'orig.txt'
           # all_fns.append(orig_dot_file_fn)
            model_name_fn = "../data/" + data_type + "/rules/tree_model/" + label_names[0] + " " + filename + ".model"
            #all_fns.append(model_name_fn)

        if dt.allFnsAlreadyExist(all_fns) and not rewrite_files:
            print("Skipping task", "DecisionTree")
            return
        else:
            print("Running task", "DecisionTree")


        vectors = np.asarray(dt.import2dArray(features_fn))
        if data_type =="sentiment":
            labels = np.asarray(dt.import1dArray(classes_fn, "i"))
        else:
            labels = np.asarray(dt.import2dArray(classes_fn, "i"))

            print("vectors", len(vectors), len(vectors[0]))
            print("labels", len(labels), len(labels[0]))

        if data_type == "sentiment" or len(vectors) != len(labels[0]):
            vectors = vectors.transpose()

        print("vectors", len(vectors), len(vectors[0]))
        cluster_names = dt.import2dArray(cluster_names_fn, "s")
        clusters = dt.import2dArray(clusters_fn, "f")
        original_vectors = vectors

        if "ratings" in classes_fn:
            orig_path = "/".join(classes_fn.split("/")[:-1]) + "/"
            match_ids_fn = orig_path + "matched_ids.txt"
            if os.path.exists(match_ids_fn):
                matched_ids = dt.import1dArray(match_ids_fn, "i")
            else:
                vector_names = dt.import1dArray(vector_names_fn)
                limited_labels = dt.import1dArray(limited_label_fn)
                matched_ids = dt.match_entities(vector_names, limited_labels)
                dt.write1dArray(matched_ids, match_ids_fn)
            vectors = vectors[matched_ids]
            print("vectors",len(vectors))
        print("Past limit entities")


        for l in range(len(label_names)):
            if label_names[l][:6] == "class-":
                label_names[l] = label_names[l][6:]

        scores_array = []
        f1_array = []
        accuracy_array = []
        prec_array = []
        recall_array = []
        params  = []


        if not multi_label and data_type != "sentiment":
            labels = labels.transpose()
            print("labels transposed")
            print("labels", len(labels), len(labels[0]))
        else:
            labels = [labels]

        all_top_clusters = []
        all_top_rankings = []
        all_top_names = []
        all_top_inds = []

        all_y_test = []
        all_predictions = []
        print("At label prediction")
        for l in range(len(labels)):
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

            # Select training data with cross validation


            ac_y_test = []
            ac_y_train = []
            ac_x_train = []
            ac_x_test = []
            ac_y_dev = []
            ac_x_dev = []
            cv_f1 = []
            cv_acc = []
            cv_prec = []
            cv_recall = []
            if cv_splits == 1:
                if data_type != "newsgroups":
                    kf = KFold(n_splits=3, shuffle=False, random_state=None)
            else:
                kf = KFold(n_splits=cv_splits, shuffle=False, random_state=None)
            c = 0
            if data_type != "newsgroups" and data_type != "sentiment":
                for train, test in kf.split(vectors):
                    if split_to_use > -1:
                        if c != split_to_use:
                            c += 1
                            continue
                    ac_y_test.append(labels[l][test])
                    ac_y_train.append(labels[l][train[:int(len(train) * 0.8)]])
                    ac_x_train.append(vectors[train[:int(len(train) * 0.8)]])
                    ac_x_test.append(vectors[test])
                    ac_x_dev.append(vectors[train[int(len(train) * 0.8):len(train)]])
                    ac_y_dev.append(labels[l][train[int(len(train) * 0.8):len(train)]])
                    c += 1
                    if cv_splits == 1:
                        break
            elif data_type == "newsgroups":
                ac_x_train =  [vectors[:int(11314 *0.8)]]
                ac_y_train =  [labels[l][:int(11314 *0.8)]]
                ac_x_test = [vectors[11314:]]
                ac_y_test = [labels[l][11314:]]
                ac_x_dev =  [vectors[int(11314 *0.8):11314]]
                ac_y_dev =  [labels[l][int(11314 *0.8):11314]]
            elif data_type == "sentiment":
                ac_x_train =  [vectors[:int(25000 *0.8)]]
                ac_y_train =  [labels[l][:int(25000 *0.8)]]
                ac_x_test = [vectors[25000:]]
                ac_y_test = [labels[l][25000:]]
                ac_x_dev =  [vectors[int(25000 *0.8):25000]]
                ac_y_dev =  [labels[l][int(25000 *0.8):25000]]
            predictions = []

            if development:
                ac_x_test = ac_x_dev
                ac_y_test = ac_y_dev

            for splits in range(len(ac_y_test)):
                model_name_fn = "../data/" + data_type + "/rules/tree_model/" + label_names[
                    l] + " " + filename + ".model"
                """
                if dt.fileExists(model_name_fn) and not rewrite_files:
                    try:
                        clf = joblib.load(model_name_fn)
                    except KeyError:
                        print(model_name_fn) # If a model is disrupted partway through its processing
                else:
                """
                clf = tree.DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, class_weight=balance)
                clf.fit(ac_x_train[splits], ac_y_train[splits])
                joblib.dump(clf, model_name_fn)
                predictions.append(clf.predict(ac_x_test[splits]))

            ac_y_test = list(ac_y_test)
            predictions = list(predictions)

            for i in range(len(predictions)):
                if len(predictions) == 1:
                    all_y_test.append(ac_y_test[i])
                    all_predictions.append(predictions[i])
                if multi_label:
                    prec, recall, fbeta, score = precision_recall_fscore_support(ac_y_test[i], predictions[i],
                                                                                 average="macro")
                else:
                    prec, recall, fbeta, score = precision_recall_fscore_support(ac_y_test[i], predictions[i],
                                                                                 average="binary")
                cv_prec.append(prec)
                cv_recall.append(recall)
                try:
                    f1 = 2 * ((prec * recall) / (prec + recall))
                except ZeroDivisionError:
                    f1 = 0
                accuracy = accuracy_score(ac_y_test[i], predictions[i])
                cv_acc.append(accuracy)
                if math.isnan(f1):
                    print("NAN", prec, recall)
                    f1 = 0.0
                scores = [[label_names[l], "f1", f1, "accuracy", accuracy]]

                print(scores)
                class_names = ["NOT " + label_names[l], label_names[l]]

                # Export a tree for each label predicted by the clf
                if save_details:
                    orig_dot_file_fn = '../data/' + data_type + '/rules/tree_data/' + label_names[l] + " " + filename  + 'orig.txt'
                    new_dot_file_fn = '../data/' + data_type + '/rules/tree_data/' + label_names[l] + " " + filename  + '.txt'
                    orig_graph_png_fn = '../data/' + data_type + '/rules/tree_images/' + label_names[l] + " " + filename + 'orig.png'
                    new_graph_png_fn = '../data/' + data_type + '/rules/tree_images/' + label_names[l] + " " + filename + '.png'
                    orig_temp_graph_png_fn = '../data/' + data_type + '/rules/tree_temp/' + label_names[l] + " " + filename + 'orig.png'
                    new_temp_graph_png_fn = '../data/' + data_type + '/rules/tree_temp/' + label_names[l] + " " + filename + '.png'
                    output_names = []
                    for c in cluster_names:
                        line = ""
                        counter = 0
                        for i in range(len(c)):
                            line = line + c[i] + " "
                            counter += 1
                            if counter == 8:
                                break
                        output_names.append(line)
                    failed = False
                    try:
                        tree.export_graphviz(clf, feature_names=output_names, class_names=class_names, out_file=orig_dot_file_fn,
                                         max_depth=max_depth, label='all', filled=True, impurity=True, node_ids=True, proportion=True, rounded=True,)
                    except FileNotFoundError:
                        try:
                            orig_dot_file_fn = "//?/" + orig_dot_file_fn
                            tree.export_graphviz(clf, feature_names=output_names, class_names=class_names, out_file=orig_dot_file_fn,
                                         max_depth=max_depth, label='all', filled=True, impurity=True, node_ids=True, proportion=True, rounded=True)

                        except FileNotFoundError:
                            failed = True
                            print("doesnt work fam")
                    if failed == False:
                        rewrite_dot_file = dt.import1dArray(orig_dot_file_fn)
                        new_dot_file = []
                        max = 3
                        min = -3
                        """
                        for f in original_vectors:
                            for n in f:
                                if n > max:
                                    max = n
                                if n < min:
                                    min = n
                        """
                        print(max)
                        print(min)
                        boundary = max - min
                        boundary = boundary / 5
                        bound_1 = 0 - boundary * 2
                        bound_2 = 0 - boundary * 1
                        bound_3 = 0
                        bound_4 = 0 + boundary
                        bound_5 = 0 + boundary * 2
                        for s in rewrite_dot_file:
                            if ":" in s:
                                s = s.split("<=")
                                no_num = s[0]
                                num = s[1]
                                num = num.split()
                                end = " ".join(num[:-1])
                                num_split = num[0].split("\\")
                                num = num_split[0]
                                end = end[len(num):]
                                num = float(num)
                                replacement = ""
                                if num <= bound_2:
                                    replacement = "VERY LOW"
                                elif num <= bound_3:
                                    replacement = "VERY LOW - LOW"
                                elif num <= bound_4:
                                    replacement = "VERY LOW - AVERAGE"
                                elif num <= bound_5:
                                    replacement = "VERY LOW - HIGH"
                                elif num >= bound_5:
                                    replacement = "VERY HIGH"
                                new_string_a = [no_num, replacement, end]
                                new_string = " ".join(new_string_a)
                                new_dot_file.append(new_string)
                                if "]" in new_string:
                                    if '"' not in new_string[len(new_string)-10:]:
                                        for c in range(len(new_string)):
                                            if new_string[c+1] == "]":
                                                new_string = new_string[:c] + '"' + new_string[c:]
                                                break
                            else:
                                new_dot_file.append(s)

                            """
                            new_string = s
                            if "->" not in s and "digraph" not in s and "node" not in s and "(...)" not in s and "}" not in s:
                                index = s.index("value")
                                new_string = s[:index] + '"] ;'
                            new_dot_file.append(new_string)
                            """
                            #new_dot_file.append(s)
                        dt.write1dArray(new_dot_file, new_dot_file_fn)
                        try:
                            orig_graph = pydot.graph_from_dot_file(orig_dot_file_fn)
                            new_graph = pydot.graph_from_dot_file(new_dot_file_fn)
                            orig_graph.write_png(orig_graph_png_fn)
                            new_graph.write_png(new_graph_png_fn)
                            orig_graph.write_png(orig_temp_graph_png_fn)
                            new_graph.write_png(new_temp_graph_png_fn)
                        except FileNotFoundError:
                            orig_graph_png_fn = "//?/" + orig_graph_png_fn
                            try:
                                orig_graph.write_png(orig_graph_png_fn)
                                new_graph_png_fn = "//?/" + new_graph_png_fn
                                new_graph.write_png(new_graph_png_fn)
                            except FileNotFoundError:
                                print("failed graph")

                    self.get_code(clf, output_names, class_names, label_names[l] + " " + filename, data_type)
                    dt_clusters, features, fns, inds = self.getNodesToDepth(clf, original_vectors, cluster_names, clusters)
                    print(filename+label_names[l])
                    fns_name = "../data/"+data_type+"/rules/names/"+filename+label_names[l]+".txt"
                    features_name = "../data/"+data_type+"/rules/rankings/"+filename+label_names[l]+".txt"
                    dt_clusters_name = "../data/"+data_type+"/rules/clusters/"+filename+label_names[l]+".txt"
                    dt.write2dArray(fns, fns_name)
                    dt.write2dArray(features, features_name)
                    dt.write2dArray(dt_clusters, dt_clusters_name)
                    all_top_rankings.extend(features)
                    all_top_clusters.extend(dt_clusters)
                    all_top_names.extend(fns)
                    all_top_inds.extend(inds)
            average_prec = np.average(cv_prec)
            average_recall = np.average(cv_recall)
            f1 = 2 * ((average_prec * average_recall) / (average_prec + average_recall))

            if math.isnan(f1):
                print("NAN", prec, recall)
                f1 = 0.0
            f1_array.append(f1)
            prec_array.append(average_prec)
            recall_array.append(average_recall)
            accuracy_array.append(np.average(np.asarray(cv_acc)))

        print("len clusters", len(all_top_clusters))
        print("len rankings", len(all_top_rankings))
        print("len names", len(all_top_names))

        if len(all_top_clusters) != len(all_top_rankings) or len(all_top_clusters) != len(all_top_names):
            print("stop")

        accuracy_array = np.asarray(accuracy_array)
        accuracy_average = np.average(accuracy_array)

        prec_array = np.asarray(prec_array)
        average_prec = np.average(prec_array)

        recall_array = np.asarray(recall_array)
        average_recall = np.average(recall_array)

        f1_average = 2 * ((average_prec * average_recall) / (average_prec + average_recall))

        if math.isnan(f1_average):
            print("NAN", prec, recall)
            f1_average = 0.0
        all_y_test = np.asarray(all_y_test)
        all_predictions = np.asarray(all_predictions)

        micro_average = f1_score(all_y_test, all_predictions, average="micro")

        accuracy_array = accuracy_array.tolist()

        accuracy_array.append(accuracy_average)
        accuracy_array.append(0.0)

        f1_array.append(f1_average)
        f1_array.append(micro_average)

        scores = [accuracy_array, f1_array]

        dt.write1dArray(accuracy_array, acc_fn)
        dt.write1dArray(f1_array, f1_fn)
        dt.write2dArray(all_predictions, prediction_fn)

        if dt.fileExists(csv_fn):
            print("File exists, writing to csv")
            try:
                dt.write_to_csv(csv_fn, file_names, scores)
            except PermissionError:
                print("CSV FILE WAS OPEN, WRITING TO ANOTHER FILE")
                print("CSV FILE WAS OPEN, WRITING TO ANOTHER FILE")
                print("CSV FILE WAS OPEN, WRITING TO ANOTHER FILE")
                print("CSV FILE WAS OPEN, WRITING TO ANOTHER FILE")
                print("CSV FILE WAS OPEN, WRITING TO ANOTHER FILE")
                print("CSV FILE WAS OPEN, WRITING TO ANOTHER FILE")
                dt.write_to_csv(csv_fn[:len(csv_fn)-4] + str(random.random()) + "FAIL.csv", file_names, scores)
            except ValueError:
                print("File does not exist, recreating csv")
                key = []
                for l in label_names:
                    key.append(l)
                key.append("AVERAGE")
                key.append("MICRO AVERAGE")
                dt.write_csv(csv_fn, file_names, scores, key)
        else:
            print("File does not exist, recreating csv")
            key = []
            for l in label_names:
                key.append(l)
            key.append("AVERAGE")
            key.append("MICRO AVERAGE")
            dt.write_csv(csv_fn, file_names, scores, key)

        if max_depth is not None:
            all_top_names = np.asarray(all_top_names)
            all_top_rankings = np.asarray(all_top_rankings)
            all_top_clusters = np.asarray(all_top_clusters)
            all_top_inds = np.asarray(all_top_inds)

            if cluster_duplicates:
                ind_to_keep = np.unique(all_top_inds, return_index=True)[1]
                all_top_names = all_top_names[ind_to_keep]
                all_top_rankings = all_top_rankings[ind_to_keep]
                all_top_clusters = all_top_clusters[ind_to_keep]

            dt.write2dArray(all_top_names, all_top_names_fn)
            dt.write2dArray(all_top_rankings, all_top_rankings_fn)
            dt.write2dArray(all_top_clusters, all_top_clusters_fn)


    def getNodesToDepth(self, tree, rankings, feature_names, clusters):
        fns = []
        features = []
        rankings = np.asarray(rankings).transpose()
        clusters = np.asarray(clusters)
        dt_clusters = []
        for i in range(len(tree.tree_.feature)):
            if i != -2 or i <= len(clusters):
                id = tree.tree_.feature[i]
                if id >=0:
                    fns.append(feature_names[id])
                    features.append(rankings[id])
                    dt_clusters.append(clusters[id])
        if len(fns) != 1:
            fn_test = np.unique(["".join(map(str, i)) for i in fns], return_index=True)
            fn_ids = fn_test[1]
        else:
            fn_ids = [0]
        final_fns = []
        clusters = list(clusters)
        final_rankings = []
        final_clusters = []
        for i in fn_ids:
            final_fns.append(fns[i])
            final_rankings.append(features[i])
            final_clusters.append(dt_clusters[i])
        return final_clusters, final_rankings, final_fns, fn_ids

    def get_code(self, tree, feature_names, class_names, filename, data_type):
        left      = tree.tree_.children_left
        right     = tree.tree_.children_right
        threshold = tree.tree_.threshold
        value = tree.tree_.value

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
        try:
            file = open("../data/" + data_type + "/rules/text_rules/"+filename+".txt", "w")
            file.write(cleaned)
            file.close()
            file = open("../data/" + data_type + "/rules/tree_temp/"+filename+".txt", "w")
            file.write(cleaned)
            file.close()
        except OSError:
            print("Couldn't save")



def main():
    cluster_to_classify = -1
    max_depth = 3
    classify = "newsgroups"
    data_type = "newsgroups"
    cv_split = 1
    jo = True
    save_details = False
    label_names_fn = "../data/"+data_type+"/classify/"+classify+"/names.txt"
    cluster_labels_fn = "../data/"+data_type+"/classify/"+classify+"/class-All"
    threshold = 0.9
    split = 0.1
    file_name = "newsgroupsss"+classify
    criterion = "entropy"
    balance = "balanced"
    cluster_names_fn = "../data/"+data_type+"/cluster/dict/n100mdsCV1S0 SFT0 allL030kappa23232 Breakoff CA1161650 MC1 MS0.4 ATS2000 DS2323300 OMS FMSFT NTlinear[100] NT[100]100linearS6040V1.1cluster_ft_dir.txt"
    #cluster_names_fn = "../data/movies/bow/names/200.txt"
    #cluster_names_fn = "../data/movies/cluster/names/" + file_name + ".txt"
    #cluster_vectors_fn = "../data/movies/rank/numeric/" + file_name + "400.txt"
    #file_name = "L3" + file_name + "L3100N0.5InClusterN0.5FTadagradcategorical_crossentropy100Genres100L3L4"

    #vector_fn = "films100svmndcg0.9240pavPPMIN0.5FTRsgdmse1000"
    vector_fn = "n100mdsCV1S0 SFT0 allL030kappa23232 Breakoff CA1161650 MC1 MS0.4 ATS2000 DS2323300 OMS FMSFT NTlinear[100] NT[100]100linearS6040V1.1cluster_ft_dir"
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
    cluster_vectors_fn = "../data/"+data_type+"/rank/numeric/n100mdsCV1S0 SFT0 allL030kappa23232 Breakoff CA1161650 MC1 MS0.4 ATS2000 DS2323300 OMS FMSFT NTlinear[100] NT[100]100linearS6040V1.1cluster_ft_dir.txt"
    clusters_fn = cluster_vectors_fn
    file_name = file_name + "ranks"
    #cluster_vectors_fn = "../data/"+data_type+"/nnet/spaces/"+vector_fn+".txt"
    #file_name = file_name + "spaces"
    file_name = vector_fn + classify + str(max_depth)

    clf = DecisionTree(cluster_vectors_fn, cluster_labels_fn, label_names_fn , cluster_names_fn , file_name, 10000,
                       max_depth, balance=balance, criterion=criterion, save_details=save_details, data_type=data_type,
                       csv_fn=csv_fn, cv_splits = cv_split, clusters_fn=cluster_vectors_fn, limit_entities=True,
                       rewrite_files=True)

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