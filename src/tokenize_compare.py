import numpy as np
import spacy
from gensim.corpora import Dictionary
from gensim.utils import deaccent
from sklearn.datasets import fetch_20newsgroups
from spacy.vocab import Vocab
import data as dt
import spacy.attrs
from spacy.tokenizer import Tokenizer
from gensim.utils import tokenize
from gensim.models import Phrases
from gensim.models.phrases import Phraser
import string
import re
nlp = spacy.load("en")

# Has batch processing (haven't figured out how to use it yet)
# Retains punctuation e.g. won't -> "won't"
# Has many additional options
# Seems hard to get vocab, tokens (maybe misunderstanding)
# Retains brackets, etc , e.g. ["(i", "think", "so)"]
# Can make vocab from your own corpus, but has different quirks
def spacyTokenize(corpus): # Documentation unclear
    tokenized_corpus = np.empty(len(corpus), dtype=np.object)
    tokenized_ids = np.empty(len(corpus), dtype=np.object)
    processed_corpus = np.empty(len(corpus), dtype=np.object)
    for i in range(len(corpus)):
        corpus[i] = corpus[i].replace("\n", " ")
    #vocab_spacy = Vocab(strings=corpus)
    tokenizer_spacy = Tokenizer(nlp.vocab)
    for i in range(len(corpus)):
        spacy_sent = tokenizer_spacy(corpus[i])
        processed_corpus[i] = spacy_sent.text
        tokenized_corpus[i] = list(spacy_sent)
        for j in range(len(tokenized_corpus[i])):
            tokenized_corpus[i][j] = tokenized_corpus[i][j].text
        tokenized_ids[i] = spacy_sent.to_array([spacy.attrs.ID])
        sd=0
    return processed_corpus, tokenized_corpus, tokenized_ids, [None]

def tokenizeNLTK1(corpus):
    processed_corpus = np.empty(len(corpus), dtype=np.object)
    tokenized_corpus = np.empty(len(corpus), dtype=np.object)
    tokenized_ids = np.empty(len(corpus), dtype=np.object)
    for i in range(len(corpus)):
        i=0
    return processed_corpus, tokenized_corpus, tokenized_ids

def tokenizeNLTK2(corpus):
    processed_corpus = np.empty(len(corpus), dtype=np.object)
    tokenized_corpus = np.empty(len(corpus), dtype=np.object)
    tokenized_ids = np.empty(len(corpus), dtype=np.object)

    return processed_corpus, tokenized_corpus, tokenized_ids


# Removes punctuation "won't be done, dude-man." = ["wont", "be", "done", "dudeman"]
# Lowercase and deaccenting "cYkět" = ["cyket"]
# Converting to ID's requires separate process using those vocabs. More time
# Finds phrases using gensim, e.g. "mayor" "of" "new" "york" -> "mayor" "of" "new_york"
def naiveTokenizer(corpus, phrases=True):
    tokenized_corpus = np.empty(len(corpus), dtype=np.object)
    tokenized_ids = np.empty(len(corpus), dtype=np.object)
    processed_corpus = np.empty(len(corpus), dtype=np.object)
    indexes_to_delete = []
    for i in range(len(corpus)):
        # Remove all punctuation, as well as paragraph chars
        table = str.maketrans(dict.fromkeys(string.punctuation + "\n" + "\r"))
        # Replace all long whitespace "          " with single " ", deaccent, convert to lowercase and then remove whitespace on the edge
        processed_corpus[i] = re.sub(r'\s+', ' ', deaccent(corpus[i].translate(table).lower())).strip()
        # Use gensim to tokenize based on the single whitespace made in the previous lines
        tokenized_corpus[i] = list(tokenize(processed_corpus[i]))
        # Print the documents that have been totally obliterated by the tokenization process to ensure nothing went wrong
        if len(tokenized_corpus[i]) == 0:
            print("DEL", corpus[i])
            indexes_to_delete.append(i)

    processed_corpus = np.delete(processed_corpus, indexes_to_delete)

    dct = Dictionary(tokenized_corpus)
    vocab = dct.token2id
    for i in range(len(tokenized_corpus)):
        ids = np.empty(len(tokenized_corpus[i]), dtype=np.object)
        for t in range(len(tokenized_corpus[i])):
            ids[t] = vocab[tokenized_corpus[i][t]]
        tokenized_ids[i] = ids
    return processed_corpus, tokenized_corpus, tokenized_ids, vocab

def ngrams(tokenized_corpus, grams=2):
    phrases = Phrases(tokenized_corpus)
    bigram = Phraser

def main(corpus_fn, output_folder):
    newsgroups = fetch_20newsgroups(subset='all', shuffle=False, remove=("headers", "footers", "quotes"))
    corpus = newsgroups.data
    classes = newsgroups.target
    processed_corpus, tokenized_corpus, tokenized_ids, vocab = spacyTokenize(corpus)
    np.save("../data/raw/newsgroups/corpus(punct).npy", tokenized_corpus)
    np.save("../data/raw/newsgroups/tokenized_corpus(punct).npy", tokenized_ids)
    np.save("../data/raw/newsgroups/vocab(punct).npy", vocab)
    dt.write1dArray(processed_corpus, "../data/raw/newsgroups/corpus_processed(punct).txt")
    processed_corpus, tokenized_corpus, tokenized_ids, vocab = naiveTokenizer(corpus)
    np.save("../data/raw/newsgroups/corpus.npy", tokenized_corpus)
    np.save("../data/raw/newsgroups/tokenized_corpus.npy", tokenized_ids)
    np.save("../data/raw/newsgroups/vocab.npy", vocab)
    dt.write1dArray(processed_corpus, "../data/raw/newsgroups/corpus_processed.txt")
    np.save("../data/raw/newsgroups/classes.npy", classes)




if __name__ == '__main__': main("newsgroups", "crash")