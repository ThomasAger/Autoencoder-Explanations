import numpy as np
import spacy
from sklearn.datasets import fetch_20newsgroups

nlp = spacy.load("en")

def tokenizeLowercase(corpus):
    docs = np.empty(len(corpus), dtype=np.object)
    for i in range(len(corpus)):
        sent = np.empty(len(corpus[i]), dtype=np.object)
        for j in range(len(nlp(corpus[i].lower()))):
            sent[j]
    return docs

def main(data_type):
    if data_type == "newsgroups":
        corpus = fetch_20newsgroups(subset='all', shuffle=False, remove=("headers", "footers", "quotes")).data
        tokens = tokenizeLowercase(corpus)
        np.save("../data/raw/newsgroups/corpus.npy", tokens)


if __name__ == '__main__': main("newsgroups")
