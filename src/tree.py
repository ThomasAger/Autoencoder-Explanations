import numpy as np
import helper.data as dt
import pydotplus as pydot
from sklearn import tree
from sklearn.metrics import f1_score, accuracy_score
from inspect import getmembers
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
import jsbeautifier
class DecisionTree:
    clf = None
    def __init__(self, cluster_vectors_fn, cluster_labels_fn,  label_names_fn, cluster_names_fn, filename, training_data,  max_depth, save_details=False):

        vectors = np.asarray(dt.import2dArray(cluster_vectors_fn)).transpose()
        labels = np.asarray(dt.import2dArray(cluster_labels_fn, "i"))
        cluster_names = dt.import1dArray(cluster_names_fn)
        label_names = dt.import1dArray(label_names_fn)

        x_train = np.asarray(vectors[:training_data])
        x_test = np.asarray(vectors[training_data:training_data+training_data/4])
        x_development = np.asarray(vectors[training_data+training_data/4:])
        for l in range(len(label_names)):
            label_names[l] = label_names[l][6:]

        scores_array = []
        f1_array = []
        accuracy_array = []
        for l in range(len(labels[0])):


            new_labels = [0] * 15000
            for x in range(len(labels)):
                new_labels[x] = labels[x][l]

            y_train = np.asarray(new_labels[:training_data])
            y_test = np.asarray(new_labels[ training_data: int(training_data + training_data / 4) ])
            y_development = np.asarray(new_labels[int( training_data + training_data / 4 ) :])
            """
            pipeline = Pipeline([('clf', tree.DecisionTreeClassifier(criterion='entropy', random_state=20000))])

            parameters = {
                'clf__max_depth': (150, 155, 160),
                'clf__min_samples_split': (1, 2, 3),
                'clf__min_samples_leaf': (1, 2, 3)
            }

            grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='f1')
            grid_search.fit(x_train, y_train)
            print('Best score: %0.3f' % grid_search.best_score_)
            print('Best parameters set:')
            best_parameters = grid_search.best_estimator_.get_params()
            for param_name in sorted(parameters.keys()):
                print( '\t%s: %r' % (param_name, best_parameters[param_name]))

            predictions = grid_search.predict(x_development)
            """
            clf = tree.DecisionTreeClassifier( max_depth=max_depth, class_weight="balanced")
            clf = clf.fit(x_train, y_train)

            y_pred = clf.predict(x_test)
            f1 = f1_score(y_test, y_pred, average='binary')
            accuracy = accuracy_score(y_test, y_pred)
            f1_array.append(f1)
            accuracy_array.append(accuracy)
            scores = [[label_names[l], "f1", f1, "accuracy", accuracy]]
            print(scores)
            class_names = [ label_names[l], "NOT "+label_names[l]]

            # Export a tree for each label predicted by the clf
            if save_details:
                tree.export_graphviz(clf, feature_names=cluster_names, class_names=class_names, out_file='../data/movies/rules/tree_data/'+label_names[l]+ " " + filename+'.txt', max_depth=max_depth)

                rewrite_dot_file = dt.import1dArray('../data/movies/rules/tree_data/'+label_names[l]+ " " + filename+'.txt')
                new_dot_file = []
                for s in rewrite_dot_file:
                    new_string = s
                    if "->" not in s and "digraph" not in s and "node" not in s and "(...)" not in s and "}" not in s:
                        index = s.index("value")
                        new_string = s[:index] + '"] ;'
                    new_dot_file.append(new_string)
                dt.write1dArray(new_dot_file, '../data/movies/rules/tree_data/'+label_names[l]+ " " + filename+'.txt')

                graph = pydot.graph_from_dot_file('../data/movies/rules/tree_data/'+label_names[l]+ " " + filename+'.txt')
                graph.write_png('../data/movies/rules/tree_images/'+label_names[l]+ " " + filename+".png")
                self.get_code(clf, cluster_names, class_names, label_names[l]+ " " + filename)


        dt.write1dArray(accuracy_array, '../data/movies/rules/tree_scores/acc'+filename+'.scores')
        dt.write1dArray(f1_array, '../data/movies/rules/tree_scores/f1' + filename + '.scores')

    def get_code(self, tree, feature_names, class_names, filename):
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
        dt.write1dArray(rules_array, "../data/movies/rules/text_rules/"+filename+".txt")
        cleaned = jsbeautifier.beautify_file("../data/movies/rules/text_rules/"+filename+".txt")
        file = open("../data/movies/rules/text_rules/"+filename+".txt", "w")
        file.write(cleaned)
        file.close()


def main():
    cluster_to_classify = -1
    max_depth = 50
    label_names_fn = "../data/movies/classify/keywords/names.txt"
    cluster_labels_fn = "../data/movies/classify/keywords/class-All"
    fn = "films100L175N0.5"
    cluster_names_fn = "../data/movies/cluster/names/"+fn+".txt"
    cluster_vectors_fn = "../data/movies/rank/numeric/" + fn + ".txt"

    fn = fn + "FT"
    cluster_vectors_fn = "../data/movies/nnet/clusters/" + fn + ".txt"
    clf = DecisionTree(cluster_vectors_fn, cluster_labels_fn, label_names_fn, cluster_names_fn, fn, 10000,
                       max_depth)





if  __name__ =='__main__':main()