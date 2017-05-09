import numpy as np
import data as dt
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
import random
from weka.core.converters import Loader
from weka.classifiers import Classifier


class DecisionTree:
    clf = None
    def __init__(self, features_fn, classes_fn,  class_names_fn, cluster_names_fn, filename,
                   max_depth=None, balance=None, criterion="entropy", save_details=False, data_type="movies",cv_splits=5,
                 csv_fn="../data/temp/no_csv_provided.csv", rewrite_files=True, split_to_use=-1, development=False,
                 limit_entities=False, limited_label_fn=None, vector_names_fn=None):

        vectors = np.asarray(dt.import2dArray(features_fn)).transpose()

        labels = np.asarray(dt.import2dArray(classes_fn, "i"))

        print("vectors", len(vectors), len(vectors[0]))
        print("labels", len(labels), len(labels[0]))
        print("vectors", len(vectors), len(vectors[0]))
        cluster_names = dt.import1dArray(cluster_names_fn)
        label_names = dt.import1dArray(class_names_fn)
        all_fns = []
        file_names = ['ACC ' + filename, 'F1 ' + filename]
        acc_fn = '../data/' + data_type + '/rules/tree_scores/' + file_names[0] + '.scores'
        f1_fn = '../data/' + data_type + '/rules/tree_scores/' + file_names[1] + '.scores'
        all_fns.append(acc_fn)
        all_fns.append(f1_fn)

        if dt.allFnsAlreadyExist(all_fns) and not rewrite_files:
            print("Skipping task", "Weka Tree")
            return
        else:
            print("Running task", "Weka Tree")

        for l in range(len(cluster_names)):
            cluster_names[l] = cluster_names[l].split()[0]

        """
        for l in range(len(label_names)):
            if label_names[l][:6] == "class-":
                label_names[l] = label_names[l][6:]
        """
        f1_array = []
        accuracy_array = []


        labels = labels.transpose()
        print("labels transposed")
        print("labels", len(labels), len(labels[0]))

        if limit_entities is False:
            vector_names = dt.import1dArray(vector_names_fn)
            limited_labels = dt.import1dArray(limited_label_fn)
            vectors = np.asarray(dt.match_entities(vectors, limited_labels, vector_names))

        for l in range(len(labels)):

            if balance:
                new_vectors, new_labels = dt.balanceClasses(vectors, labels[l])
            else:
                new_vectors = vectors
                new_labels = labels[l]
            # Select training data with cross validation


            ac_y_test = []
            ac_y_train = []
            ac_x_train = []
            ac_x_test = []
            ac_y_dev = []
            ac_x_dev = []
            cv_f1 = []
            cv_acc = []
            if cv_splits == 1:
                kf = KFold(n_splits=3, shuffle=False, random_state=None)
            else:
                kf = KFold(n_splits=cv_splits, shuffle=False, random_state=None)
            c = 0
            for train, test in kf.split(new_vectors):
                if split_to_use > -1:
                    if c != split_to_use:
                        c += 1
                        continue
                ac_y_test.append(new_labels[test])
                ac_y_train.append(new_labels[train[int(len(train) * 0.2):]])
                val = int(len(train) * 0.2)
                t_val = train[val:]
                nv_t_val = new_vectors[t_val]
                ac_x_train.append(nv_t_val)
                ac_x_test.append(new_vectors[test])
                ac_x_dev.append(new_vectors[train[:int(len(train) * 0.2)]])
                ac_y_dev.append(new_labels[train[:int(len(train) * 0.2)]])
                c += 1
                if cv_splits == 1:
                    break

            predictions = []

            if development:
                ac_x_test = np.copy(np.asarray(ac_x_dev))
                ac_y_test = np.copy(np.asarray(ac_y_dev))

            train_fn = "../data/" + data_type + "/weka/data/" + filename + "Train.txt"
            test_fn = "../data/" + data_type + "/weka/data/" + filename + "Test.txt"

            for splits in range(len(ac_y_test)):
                # Get the weka predictions
                dt.writeArff(ac_x_train[splits], [ac_y_train[splits]], [label_names[splits]], train_fn, header=True)
                dt.writeArff(ac_x_test[splits], [ac_y_test[splits]], [label_names[splits]], test_fn, header=True)
                predictions.append(self.getWekaPredictions(train_fn+label_names[splits]+".arff",
                                                           test_fn+label_names[splits]+".arff", save_details))


            for i in range(len(predictions)):
                f1 = f1_score(ac_y_test[i], predictions[i], average="binary")
                accuracy = accuracy_score(ac_y_test[i], predictions[i])
                cv_f1.append(f1)
                cv_acc.append(accuracy)
                scores = [[label_names[l], "f1", f1, "accuracy", accuracy]]
                print(scores)


                # Export a tree for each label predicted by the clf, not sure if this is needed...
                if save_details:
                    class_names = [label_names[l], "NOT " + label_names[l]]
                    #self.get_code(clf, cluster_names, class_names, label_names[l] + " " + filename, data_type)
            f1_array.append(np.average(np.asarray(cv_f1)))
            accuracy_array.append(np.average(np.asarray(cv_acc)))

        accuracy_array = np.asarray(accuracy_array)
        accuracy_average = np.average(accuracy_array)

        f1_array = np.asarray(f1_array)
        f1_average = np.average(f1_array)

        accuracy_array = np.append(accuracy_array, accuracy_average)
        f1_array = np.append(f1_array, f1_average)

        scores = [accuracy_array, f1_array]

        dt.write1dArray(accuracy_array, acc_fn)
        dt.write1dArray(f1_array, f1_fn)

        print(csv_fn)
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
        else:
            print("File does not exist, recreating csv")
            key = []
            for l in label_names:
                key.append(l)
            key.append("AVERAGE")
            dt.write_csv(csv_fn, file_names, scores, key)


    def get_code(self, tree, feature_names, class_names, filename, data_type):
        rules_array = []
        dt.write1dArray(rules_array, "../data/" + data_type + "/rules/text_rules/"+filename+".txt")
        # Probably not needed
        cleaned = jsbeautifier.beautify_file("../data/" + data_type + "/rules/text_rules/"+filename+".txt")
        file = open("../data/" + data_type + "/rules/text_rules/"+filename+".txt", "w")
        file.write(cleaned)
        file.close()


    def getWekaPredictions(self, train_fn, test_fn, save_details):
        print("weka")


        loader = Loader(classname="weka.core.converters.ArffLoader")
        train_data = loader.load_file(train_fn)
        train_data.class_is_last()

        cls = Classifier(classname="weka.classifiers.trees.J48")


        cls.build_classifier(train_data)

        y_pred = []

        test_data = loader.load_file(test_fn)
        test_data.class_is_last()

        for index, inst in enumerate(test_data):
            pred = cls.classify_instance(inst)
            dist = cls.distribution_for_instance(inst)
            y_pred.append(pred)

        if save_details:
            print("Totes saving details now")

        return y_pred

def main(cluster_vectors_fn, classes_fn, label_names_fn, cluster_names_fn, file_name, balance,save_details, data_type, csv_fn, cv_splits):

    clf = DecisionTree(cluster_vectors_fn, classes_fn, label_names_fn , cluster_names_fn , file_name,
                       balance=balance, save_details=save_details, data_type=data_type,
                       csv_fn=csv_fn, cv_splits=cv_splits)



data_type = "placetypes"
classes = "foursquare"

file_name = "placetypes mds E2000 DS[100] DN0.6 CTfoursquare HAtanh CV1 S0 DevFalse SFT0L050LETruendcg1200MC1MS0.5"

cluster_vectors_fn = "../data/" + data_type + "/rank/numeric/" + file_name + ".txt"
#cluster_vectors_fn = "../data/" + data_type + "/nnet/clusters/" + file_name + ".txt"
#cluster_vectors_fn = "../data/" + data_type + "/nnet/spaces/" + file_name + ".txt"
cluster_names_fn = "../data/" + data_type + "/cluster/hierarchy_names/" + file_name + ".txt"

cluster_names_fn = "../data/" + data_type + "/cluster/hierarchy_names/"+file_name+".txt"

label_names_fn = "../data/" + data_type + "/classify/"+classes+"/names.txt"
classes_fn = "../data/" + data_type + "/classify/"+classes+"/class-All"
lowest_val = 10000
max_depth = None
balance = True
criterion = None
save_details = True
if balance:
    file_name = file_name + "balance"
file_name = file_name + "J48"
csv_fn = "../data/"+ data_type + "/weka/tree_csv/"+file_name+".csv"
cv_splits = 1
if  __name__ =='__main__':main(cluster_vectors_fn, classes_fn, label_names_fn, cluster_names_fn, file_name,
                              balance,  save_details, data_type, csv_fn, cv_splits)