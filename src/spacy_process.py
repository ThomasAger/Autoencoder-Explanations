import numpy as np
import spacy
from sklearn.datasets import fetch_20newsgroups
import data as dt
import spacy.attrs

# This is probably good for everything that isn't basic tokenization of your own data.
# Do not use. NLTK is better


nlp = spacy.load("en")

def tokenizeLowercase(corpus):
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

def tokenIds(corpus):
    tokenized_corpus = np.empty(len(corpus), dtype=np.object)
    text_corpus = np.empty(len(corpus), dtype=np.object)
    for i in range(len(corpus)):
        processed_sent = nlp(corpus[i].replace("\n", " "))
        tokenized_corpus[i] = processed_sent.to_array([spacy.attrs.SENT_START])
        text_corpus[i] = " ".join(tokenized_corpus[i])
        if i == 100: break
    return tokenized_corpus, text_corpus

def main(data_type):

    if data_type == "newsgroups":
        corpus = fetch_20newsgroups(subset='all', shuffle=False, remove=("headers", "footers", "quotes")).data
        tokenized_corpus, text_corpus = tokenizeLowercaseSpacy(corpus)
        np.save("../data/raw/newsgroups/corpus.npy", tokenized_corpus)
        dt.write1dArray(text_corpus, "../data/raw/newsgroups/corpus_processed.txt")


if __name__ == '__main__': main("newsgroups")
