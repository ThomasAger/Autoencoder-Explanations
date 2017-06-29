import numpy as np
import data as dt
from scipy.stats import spearmanr

def getGini(discrete_labels, ppmi):
    x = []
    for i in range(len(discrete_labels)):
        x.append(int(discrete_labels[i][:-1]))
    y = ppmi

    amts = [0] * 100
    totals = [0] * 100
    for i in range(len(x)):
        amts[x[i]-1] += y[i]
        totals[x[i]-1] += 1
    avgs = []
    for i in range(len(amts)):
        avgs.append(amts[i] / totals[i])

    y = avgs
    x = range(0,100)

    return gini2(y)

def getGiniNotAverage(discrete_labels, ppmi):
    x = []
    for i in range(len(discrete_labels)):
        x.append(int(discrete_labels[i][:-1]))
    y = ppmi

    return gini(y)


def gini(data):
    '''
    Calculates the gini coefficient for a given dataset.
    input:
        data- list of values, either raw counts or frequencies.
              Frequencies MUST sum to 1.0, otherwise will be transformed to frequencies
              If raw counts data will be transformed to frequencies.
    output:
        gini- float, from 0.0 to 1.0 (1.0 most likely never realized since it is
              only achieved in the limit)
    '''

    def _unit_area(height, value, width):
        '''
        Calculates a single bars area.
        Area is composed of two parts:
            The height of the bar up until that point
            The addition from the current value (calculated as a triangle)
        input:
            height: previous bar height or sum of values up to current value
            value: current value
            width: width of individual bar
        output:
            bar_area: area of current bar
        '''
        bar_area = (height * width) + ((value * width) / 2.)
        return bar_area

    #Fair area will always be 0.5 when frequencies are used
    fair_area = 0.5
    #If data does not sum to 1.0 transform to frequencies
    datasum = float(sum(data))
    data = [x/datasum for x in data]
    #Calculate the area under the curve for the current dataset
    data.sort()
    width = 1/float(len(data))
    height, area = 0.0, 0.0
    for value in data:
        area += _unit_area(height, value, width)
        height += value
    #Calculate the gini
    gini = (fair_area-area)/fair_area
    return gini
print(gini([1,1.1,1.2,1.3,1.4]))
def gini2(list_of_values):
    list_of_values.sort()
    height, area = 0, 0
    # Calculate the area under the curve
    for value in list_of_values:
        height += value
        area += height - value / 2.
    # Calculate equality area
    fair_area = height * len(list_of_values) / 2.
    return (fair_area - area) / fair_area

def getGinis(ppmi_fn, phrases_fn,  discrete_labels_fn, fn):
    phrases = dt.import1dArray(phrases_fn)
    new_ppmi = dt.import2dArray(ppmi_fn)
    counter = 0
    ginis = []
    discrete_vectors = dt.import2dArray(discrete_labels_fn, "s")
    discrete_vectors.reverse()
    for discrete_label in discrete_vectors:
        giniscore = getGini(discrete_label, new_ppmi[counter])
        ginis.append(giniscore)
        print(counter, phrases[counter], giniscore)
        counter += 1
    ginis_fn = "../data/movies/gini/" + fn + ".txt"
    dt.write1dArray(ginis, ginis_fn)

class Gini:
    def __init__(self, discrete_labels_fn, ppmi_fn, phrases_fn, phrases_to_check_fn, fn):
        getGinis(ppmi_fn, phrases_fn, phrases_to_check_fn, discrete_labels_fn, fn)

def main(discrete_labels_fn, ppmi_fn, phrases_fn, phrases_to_check_fn, fn):
    """
    discrete_labels_fn = "Rankings/films100N0.6H75L1P1.discrete"
    ppmi_fn = "Journal Paper Data/Term Frequency Vectors/class-all"
    phrases_fn = "SVMResults/films100.names"
    phrases_to_check_fn = ""#"RuleType/top1ksorted.txt"
    fn = "films 100, 75 L1"
    """
    Gini(discrete_labels_fn, ppmi_fn, phrases_fn, phrases_to_check_fn, fn)







#if  __name__ =='__main__':main()