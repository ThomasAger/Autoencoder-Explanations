import numpy as np
def gini(actual, pred, cmpcol=0, sortcol=1):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)


def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)

def test_gini():
    def fequ(a, b):
        return abs(a - b) < 1e-6

    def T(a, p, g, n):
        assert (fequ(gini(a, p), g))
        assert (fequ(gini_normalized(a, p), n))

    print(gini([1,1,1],[40,200,3202]))
test_gini()