
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, cohen_kappa_score
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import math
import os.path
from util import io as dt

def classifier_score(true_target, prediction):
    if len(true_target[0]):
        prec, recall, fbeta, score = precision_recall_fscore_support(true_target, prediction,
                                                                     average="macro")
    else:
        prec, recall, fbeta, score = precision_recall_fscore_support(true_target, prediction,
                                                                     average="binary")
    auroc = roc_auc_score(true_target, prediction, average="macro")
    try:
        f1 = get_f1_score(prec, recall)
    except ZeroDivisionError:
        f1 = 0.0
    # For multi-label, must exactly match
    accuracy = accuracy_score(true_target, prediction)
    if math.isnan(f1):
        f1 = 0.0
    if math.isnan(accuracy):
        f1 = 0.0
    if math.isnan(prec):
        f1 = 0.0
    if math.isnan(recall):
        f1 = 0.0
    return f1, prec, recall, accuracy, auroc

def get_f1_score(prec, recall):
    return 2 * ((prec * recall) / (prec + recall))

def svm_score(true_target, prediction):
    prec, recall, fbeta, score = precision_recall_fscore_support(true_target, prediction, average="binary")
    accuracy = accuracy_score(true_target, prediction)
    f1 = get_f1_score(prec, recall)
    kappa = cohen_kappa_score(true_target, prediction)
    return accuracy, f1, kappa

class AverageResults():
    all_prec = []
    all_recall = []
    all_acc = []
    all_auroc = []
    all_f1 = []
    class_names = []
    average_prec = -1
    average_recall = -1
    average_auroc = -1
    average_f1 = -1
    average_acc = -1

    def __init__(self, all_target_values, all_predictions, class_names):
        self.class_names = class_names
        for i in range(len(all_target_values)):
            f1, prec, recall, accuracy, auroc = classifier_score(all_target_values[0], all_predictions[0])
            self.all_prec.append(prec)
            self.all_recall.append(recall)
            self.all_acc.append(accuracy)
            self.all_auroc.append(auroc)
            self.all_f1.append(f1)
        self.average_prec = np.average(self.all_prec)
        self.average_recall = np.average(self.all_recall)
        self.average_auroc = np.average(self.all_auroc)
        self.average_acc = np.average(self.all_acc)
        self.average_f1 = get_f1_score(self.average_prec, self.average_recall)
        if math.isnan(self.average_f1):
            self.average_f1 = 0.0

    def save_csv(self, csv_fn):

        csv_acc = self.all_acc
        csv_f1 = self.all_f1
        csv_auroc = self.all_auroc

        csv_acc.append(self.average_acc)
        csv_f1.append(self.average_f1)
        csv_auroc.append(self.average_auroc)

        scores = [csv_acc, csv_f1, csv_auroc]
        col_names = ["acc", "f1", "auroc"]
        if os.path.exists(csv_fn):
            print("File exists, writing to csv")
            try:
                dt.write_to_csv(csv_fn, col_names, scores)
                return
            except PermissionError:
                print("CSV FILE WAS OPEN, SKIPPING")
                return
            except ValueError:
                print("File does not exist, recreating csv")
        print("File does not exist, recreating csv")
        key = []
        for l in self.class_names:
            key.append(l)
        key.append("AVERAGE")
        key.append("MICRO AVERAGE")
        dt.write_csv(csv_fn, col_names, scores, key)