from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
import numpy as np

#Multi-label fake data
predicted_classes = [[0,1,1,0,1,1,1,0,1,0,1,1,1,0,0,0,0,0,0,0,1,1,1,1],
                     [0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1],
                     [0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1]]

real_classes = [[0,0,1,0,0,1,1,1,1,1,1,0,1,1,0,0,0,0,1,0,1,0,1,0],
                     [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0],
                     [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1]]

# Calculating each class individually

precisions = np.zeros(3)
recalls = np.zeros(3)
f1_betas = np.zeros(3)
f1s = np.zeros(3)
calced_f1s = np.zeros(3)

for i in range(len(predicted_classes)):
    prec, recall, fbeta, score = precision_recall_fscore_support(real_classes[i], predicted_classes[i], average="binary")
    precisions[i] = prec
    recalls[i] = recall
    f1_betas[i] = fbeta
    calced_f1s[i] = 2 * ((prec * recall) / (prec + recall))
    f1s[i] = f1_score(real_classes[i], predicted_classes[i], average="binary")

# Averaging F1 scores of every class

print("If these two are the same, it means that f1_beta = f1_score")
print("stupid boi mean f1 beta", np.average(f1_betas))
print("stupid boi mean f1 score", np.average(f1s))

# Calculating Macro average with multi-label input
# Micro average with multi-label input
print("macro f1 score", f1_score(real_classes, predicted_classes, average="macro"))
print("micro f1 score", f1_score(real_classes, predicted_classes, average="micro"))

# Averaging precision recall and then calculating F1 score for every class
average_prec = np.average(precisions)
average_recall = np.average(recalls)
f1 = 2 * ((average_prec * average_recall) / (average_prec + average_recall))
print("")
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11")
print("f1 calced from average prec + average recall", f1)
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11")
print("")
# Same calculation but averaged afterwards

print("stupid boi mean f1 but calced using my formula", np.average(calced_f1s))

real_classes = np.asarray(real_classes).transpose()
predicted_classes = np.asarray(predicted_classes).transpose()

prec, recall, fbeta, score = precision_recall_fscore_support(real_classes, predicted_classes, average="macro")

f1 = 2 * ((prec * recall) / (prec + recall))
print("")
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11")
print("f1 from formula, multi-label, macro average prec recall fscore support", f1)
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11")
print("")
print("f1 from fbeta, multi-label, macro average prec recall fscore support", fbeta)

prec, recall, fbeta, score = precision_recall_fscore_support(real_classes, predicted_classes, average="micro")

f1 = 2 * ((prec * recall) / (prec + recall))

print("f1 from formula, multi-label, micro average prec recall fscore support", f1)
print("f1 from fbeta, multi-label, micro average prec recall fscore support", fbeta)

