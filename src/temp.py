import numpy as np

from sklearn.model_selection import KFold


k_fold = KFold(n_splits=3, shuffle=False, random_state=None)
letters = np.asarray(["a", "b", "c", "d", "e"])

for train, test in k_fold.split(letters):
    print(letters[train])
    print(letters[test])