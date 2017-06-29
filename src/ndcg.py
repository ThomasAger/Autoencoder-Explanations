import numpy as np
import data as dt
from scipy.stats import spearmanr
def ranking_precision_score(y_true, y_score, k=10):
    """Precision at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    precision @k : float
    """
    unique_y = np.unique(y_true)

    if len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")

    pos_label = unique_y[1]
    n_pos = np.sum(y_true == pos_label)

    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    n_relevant = np.sum(y_true == pos_label)

    # Divide by min(n_pos, k) such that the best achievable score is always 1.0.
    return float(n_relevant) / min(n_pos, k)


def average_precision_score(y_true, y_score, k=10):
    """Average precision at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    average precision @k : float
    """
    unique_y = np.unique(y_true)

    if len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")

    pos_label = unique_y[1]
    n_pos = np.sum(y_true == pos_label)

    order = np.argsort(y_score)[::-1][:min(n_pos, k)]
    y_true = np.asarray(y_true)[order]

    score = 0
    for i in range(len(y_true)):
        if y_true[i] == pos_label:
            # Compute precision up to document i
            # i.e, percentage of relevant documents up to document i.
            prec = 0
            for j in range(0, i + 1):
                if y_true[j] == pos_label:
                    prec += 1.0
            prec /= (i + 1.0)
            score += prec

    if n_pos == 0:
        return 0

    return score / n_pos


def dcg_score(y_true, y_score, k=10, gains="exponential"):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    DCG @k : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10, gains="exponential"):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    NDCG @k : float
    """
    best = dcg_score(y_true, y_true, k, gains)
    actual = dcg_score(y_true, y_score, k, gains)
    return actual / best


# Alternative API.

def dcg_from_ranking(y_true, ranking):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    ranking : array-like, shape = [k]
        Document indices, i.e.,
            ranking[0] is the index of top-ranked document,
            ranking[1] is the index of second-ranked document,
            ...
    k : int
        Rank.
    Returns
    -------
    DCG @k : float
    """
    y_true = np.asarray(y_true)
    ranking = np.asarray(ranking)
    rel = y_true[ranking]
    gains = 2 ** rel - 1
    discounts = np.log2(np.arange(len(ranking)) + 2)
    overall = gains / discounts
    sum = np.sum(overall)
    return sum


def ndcg_from_ranking(y_true, ranking):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    ranking : array-like, shape = [k]
        Document indices, i.e.,
            ranking[0] is the index of top-ranked document,
            ranking[1] is the index of second-ranked document,
            ...
    k : int
        Rank.
    Returns
    -------
    NDCG @k : float
    """
    k = len(ranking)
    best_ranking = np.argsort(y_true)[::-1]
    best = dcg_from_ranking(y_true, best_ranking[:k])
    dcg = dcg_from_ranking(y_true, ranking)
    return dcg / best
import linecache
def getNDCG(rankings_fn, fn, data_type, lowest_count, rewrite_files=False, highest_count = 0, classification = ""):
    ndcg_fn = "../data/" + data_type + "/ndcg/"+fn+".txt"
    spearman_fn = "../data/" + data_type + "/spearman/"+fn+".txt"
    all_fns = [ndcg_fn, spearman_fn]
    if dt.allFnsAlreadyExist(all_fns) and not rewrite_files:
        print("Skipping task", getNDCG.__name__)
        return
    else:
        print("Running task", getNDCG.__name__)
    ppmi_fn = "../data/" + data_type + "/bow/ppmi/class-all-"+str(lowest_count)+"-" + str(highest_count)+"-" +classification
    names = dt.import1dArray("../data/" + data_type + "/bow/names/"+str(lowest_count)+"-" + str(highest_count)+"-" +classification+".txt")
    ndcg_a = []
    spearman_a = []
    map_a = []
    kendall_a = []
    r = 0
    with open(rankings_fn) as rankings, open(ppmi_fn) as ppmi:
        r = 0
        for lr in rankings:
                for lp in ppmi:
                    sorted_indices = np.argsort(list(map(float, lr.strip().split())))[::-1]
                    ppmi_scores = list(map(float, lp.strip().split()))
                    ppmi_indices = np.argsort(np.asarray(ppmi_scores))
                    ndcg = ndcg_from_ranking(ppmi_scores, sorted_indices)
                    ndcg_a.append(ndcg)
                    print("ndcg", ndcg, names[r], r)
                    smr = spearmanr(ppmi_indices, sorted_indices)[1]
                    spearman_a.append(smr)
                    print("spearman", smr, names[r], r)

                    r+=1
                    break
    dt.write1dArray(ndcg_a, ndcg_fn)
    dt.write1dArray(spearman_a, spearman_fn)



class Gini:
    def __init__(self, rankings_fn, ppmi_fn, fn):
        getNDCG(rankings_fn, ppmi_fn, fn)

def main(rankings_fn, ppmi_fn,  fn):
    """
    discrete_labels_fn = "Rankings/films100N0.6H75L1P1.discrete"
    ppmi_fn = "Journal Paper Data/Term Frequency Vectors/class-all"
    phrases_fn = "SVMResults/films100.names"
    phrases_to_check_fn = ""#"RuleType/top1ksorted.txt"
    fn = "films 100, 75 L1"
    """
    getNDCG(rankings_fn, fn)

#getNDCG("../data/movies/rank/numeric/films100ALL.txt","films100")