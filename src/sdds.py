
from sklearn import linear_model
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score

def perf_measure(y_actual, y_hat): # Get the true positives etc
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i] == 1 and y_hat[i] == 1:
            TP += 1
        if y_hat[i] == 1 and y_actual[i] == 0:
            FP += 1
        if y_actual[i] == 0 and y_hat[i] == 0:
            TN += 1
        if y_hat[i] == 0 and y_actual[i] == 1:
            FN += 1

    return TP, FP, TN, FN

def runLR(vectors, classes):
    # Default is dual formulation, which is unusual. Balanced balances the classes
    clf = linear_model.LogisticRegression(class_weight="balanced", dual=False)
    clf.fit(vectors, classes)  # All of the vectors and classes. No need for training data.
    direction = clf.coef_.tolist()[0]  # Get the direction
    predicted = clf.predict(vectors)
    predicted = predicted.tolist()  # Convert to list so we can calculate the scores
    f1 = f1_score(classes, predicted)
    kappa_score = cohen_kappa_score(classes, predicted)
    acc = accuracy_score(classes, predicted)
    TP, FP, TN, FN = perf_measure(classes, predicted)  # Get the True positive, etc
    return kappa_score, f1, direction, acc, TP, FP, TN, FN