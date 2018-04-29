import numpy as np
import spacy
from gensim.corpora import Dictionary
from gensim.utils import deaccent
from sklearn.datasets import fetch_20newsgroups
import data as dt
import spacy.attrs
from spacy.tokenizer import Tokenizer
from gensim.utils import tokenize
from gensim.models import Phrases
from gensim.models.phrases import Phraser
import string
import re
import os

nlp = spacy.load("en")


# Has batch processing (haven't figured out how to use it yet)
# Retains punctuation e.g. won't -> "won't"
# Has many additional options
# Seems hard to get vocab, tokens (maybe misunderstanding)
# Retains brackets, etc , e.g. ["(i", "think", "so)"]
# Can make vocab from your own corpus, but has different quirks
def spacyTokenize(corpus):  # Documentation unclear
    tokenized_corpus = np.empty(len(corpus), dtype=np.object)
    tokenized_ids = np.empty(len(corpus), dtype=np.object)
    processed_corpus = np.empty(len(corpus), dtype=np.object)
    for i in range(len(corpus)):
        corpus[i] = corpus[i].replace("\n", " ")
    # vocab_spacy = Vocab(strings=corpus)
    tokenizer_spacy = Tokenizer(nlp.vocab)
    for i in range(len(corpus)):
        spacy_sent = tokenizer_spacy(corpus[i])
        processed_corpus[i] = spacy_sent.text
        tokenized_corpus[i] = list(spacy_sent)
        for j in range(len(tokenized_corpus[i])):
            tokenized_corpus[i][j] = tokenized_corpus[i][j].text
        tokenized_ids[i] = spacy_sent.to_array([spacy.attrs.ID])
        sd = 0
    return processed_corpus, tokenized_corpus, tokenized_ids, [None]


def tokenizeNLTK1(corpus):
    processed_corpus = np.empty(len(corpus), dtype=np.object)
    tokenized_corpus = np.empty(len(corpus), dtype=np.object)
    tokenized_ids = np.empty(len(corpus), dtype=np.object)
    for i in range(len(corpus)):
        i = 0
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
    tokenized_ids = tokensToIds(tokenized_corpus, vocab)

    return processed_corpus, tokenized_corpus, tokenized_ids, vocab


def tokensToIds(tokenized_corpus, vocab):
    tokenized_ids = np.empty(len(tokenized_corpus), dtype=np.object)
    for i in range(len(tokenized_corpus)):
        ids = np.empty(len(tokenized_corpus[i]), dtype=np.object)
        for t in range(len(tokenized_corpus[i])):
            ids[t] = vocab[tokenized_corpus[i][t]]
        tokenized_ids[i] = ids
    return tokenized_ids


def ngrams(tokenized_corpus):  # Increase the gram amount by 1
    gram_corpus = np.empty(len(tokenized_corpus), dtype=np.object)
    processed_corpus = np.empty(len(tokenized_corpus), dtype=np.object)
    phrases = Phrases(tokenized_corpus)
    gram = Phraser(phrases)
    for i in range(len(tokenized_corpus)):
        gram_corpus[i] = gram[tokenized_corpus[i]]
        processed_corpus[i] = " ".join(gram_corpus[i])
    dct = Dictionary(gram_corpus)
    vocab = dct.token2id
    tokenized_ids = tokensToIds(gram_corpus, vocab)
    return processed_corpus, gram_corpus, tokenized_ids, vocab


def bow(tokenized_corpus):
    print("")


def main(data_type, output_folder, grams):
    if data_type == "newsgroups":
        newsgroups = fetch_20newsgroups(subset='all', shuffle=False, remove=("headers", "footers", "quotes"))
        corpus = newsgroups.data
        classes = newsgroups.target
    else:
        newsgroups = fetch_20newsgroups(subset='all', shuffle=False, remove=("headers", "footers", "quotes"))
        corpus = newsgroups.data
        classes = newsgroups.target
    if os.path.exists(output_folder + "vocab(punct).npy") is False:
        processed_corpus, tokenized_corpus, tokenized_ids, vocab = spacyTokenize(corpus)
        np.save(output_folder + "corpus(punct).npy", tokenized_corpus)
        np.save(output_folder + "tokenized_corpus(punct).npy", tokenized_ids)
        np.save(output_folder + "vocab(punct).npy", vocab)
        dt.write1dArray(processed_corpus, output_folder + "corpus_processed(punct).txt")
    if os.path.exists(output_folder + "corpus.npy") is False:
        processed_corpus, tokenized_corpus, tokenized_ids, vocab = naiveTokenizer(corpus)
        np.save(output_folder + "corpus.npy", tokenized_corpus)
        np.save(output_folder + "tokenized_corpus.npy", tokenized_ids)
        np.save(output_folder + "vocab.npy", vocab)
        dt.write1dArray(processed_corpus, output_folder + "corpus_processed.txt")
    if os.path.exists(output_folder + "classes.npy") is False:
        np.save(output_folder + "classes.npy", classes)
    if grams > 0:
        for i in range(2, grams):  # Up to 5-length grams
            if os.path.exists(output_folder + "corpus " + str(i) + "-gram" + ".npy") is False:
                processed_corpus, tokenized_corpus, tokenized_ids, vocab = ngrams(tokenized_corpus)
                np.save(output_folder + "corpus " + str(i) + "-gram" + ".npy", tokenized_corpus)
                np.save(output_folder + "tokenized_corpus " + str(i) + "-gram" + ".npy", tokenized_ids)
                np.save(output_folder + "vocab " + str(i) + "-gram" + ".npy", vocab)
                dt.write1dArray(processed_corpus, output_folder + "corpus_processed " + str(i) + "-gram" + ".txt")
                np.save(output_folder + "classes " + str(i) + "-gram" + ".npy", classes)


if __name__ == '__main__': main("newsgroups", "../data/raw/newsgroups/")