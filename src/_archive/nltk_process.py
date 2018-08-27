import numpy as np
from nltk.tokenize.api import StringTokenizer
from nltk.tokenize.api import TokenizerI
from gensim.utils import tokenize
from sklearn.datasets import fetch_20newsgroups
import data as dt
from gensim.corpora import Dictionary
import spacy.attrs
import spacy
nlp = spacy.load("en")

def spacyTokenizeLowercase(corpus):
    tokenized_corpus = np.empty(len(corpus), dtype=np.object)
    text_corpus = np.empty(len(corpus), dtype=np.object)
    for i in range(len(corpus)):
        processed_sent = nlp(corpus[i].lower().replace("\n", " "))
        sent = np.empty(len(processed_sent), dtype=np.object)
        for j in range(len(processed_sent)):
            sent[j] = str(processed_sent[j])
        tokenized_corpus[i] = sent
        text_corpus[i] = " ".join(tokenized_corpus[i])
        if i == 100: break
    return tokenized_corpus, text_corpus

def tokenizeNLTK1(corpus):
    processed_corpus = np.empty(len(corpus), dtype=np.object)
    tokenized_corpus = np.empty(len(corpus), dtype=np.object)
    tokenized_ids = np.empty(len(corpus), dtype=np.object)
    for i in range(len(corpus)):

    return processed_corpus, tokenized_corpus, tokenized_ids

def tokenizeNLTK2(corpus):
    processed_corpus = np.empty(len(corpus), dtype=np.object)
    tokenized_corpus = np.empty(len(corpus), dtype=np.object)
    tokenized_ids = np.empty(len(corpus), dtype=np.object)

    return processed_corpus, tokenized_corpus, tokenized_ids
def tokenizeGensim(corpus):
    processed_corpus = np.empty(len(corpus), dtype=np.object)
    tokenized_corpus = np.empty(len(corpus), dtype=np.object)
    tokenized_ids = np.empty(len(corpus), dtype=np.object)

    return processed_corpus, tokenized_corpus, tokenized_ids
def main(corpus_fn, output_folder):
    corpus = fetch_20newsgroups(subset='all', shuffle=False, remove=("headers", "footers", "quotes")).data
    tokenized_corpus, text_corpus = spacyTokenizeLowercase(corpus)
    np.save("../data/raw/newsgroups/corpus.npy", tokenized_corpus)
    dt.write1dArray(text_corpus, "../data/raw/newsgroups/corpus_processed.txt")


if __name__ == '__main__': main("newsgroups")