import numpy as np
# From 2d array, change any values above 0 to 1
def toBinary(x):
    for i in range(len(x)):
        for j in range(len(x[i])):
            if x[i][j] > 0:
                x[i][j] = 1
    return x

# Get the document frequencies for a corpus
def getDocumentFrequency(x):
    if len(x) < len(x[0]):
        raise ValueError("The array is reversed. (Amount of words < Amount of documents)")
    doc_freq = np.zeros(len(x))
    for i in range(len(x)): #  Words
        for j in range(len(x[i])): # Documents
            if x[i][j] > 0:
                doc_freq[x] += 1
    return doc_freq

def limitDocumentFrequency(doc_freq, word_by_doc, word_list, min_freq, max_freq):
    limit_freq_ind = np.where(doc_freq > min_freq)
    print("Removed terms below", min_freq, "doc frequency", len(limit_freq_ind), "terms remaining")
    word_list = word_list[limit_freq_ind]
    doc_freq = doc_freq[limit_freq_ind]
    word_by_doc = word_by_doc[limit_freq_ind]

    limit_freq_ind = np.where(doc_freq > (len(doc_freq) - max_freq))
    print("Removed terms above", (len(doc_freq) - max_freq), "doc frequency", len(limit_freq_ind), "terms remaining")
    word_list = word_list[limit_freq_ind]
    doc_freq = doc_freq[limit_freq_ind]
    word_by_doc = word_by_doc[limit_freq_ind]

    return word_list, word_by_doc, doc_freq