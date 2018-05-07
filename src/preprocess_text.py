
from gensim.corpora import Dictionary
from gensim.utils import deaccent
import data as dt
import spacy.attrs
from spacy.tokenizer import Tokenizer
from gensim.utils import tokenize
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from nltk.corpus import stopwords
import numpy as np
from keras.utils import to_categorical
import spacy
import os
import string
from sklearn.datasets import fetch_20newsgroups
import re
from gensim.matutils import corpus2csc
import scipy.sparse as sp
import data as dt
import MovieTasks as mt
from sklearn.decomposition import PCA
from test_representations import testAll
from collections import defaultdict
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


def tokenizeNLTK1(corpus): # This is prob better than current implementation - look into later
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
def naiveTokenizer(corpus):
    tokenized_corpus = np.empty(len(corpus), dtype=np.object)
    for i in range(len(corpus)):
        tokenized_corpus[i] = list(tokenize(corpus[i]))
    return tokenized_corpus

def getVocab(tokenized_corpus):
    dct = Dictionary(tokenized_corpus)
    vocab = dct.token2id
    return vocab, dct

def doc2bow(tokenized_corpus, dct):
    dct.filter_extremes(no_below=2) # Most occur in at least 2 documents
    bow = [dct.doc2bow(text) for text in tokenized_corpus]
    bow = corpus2csc(bow)
    return bow

def filterBow(tokenized_corpus, dct, no_below, no_above):
    dct.filter_extremes(no_below=no_below, no_above=no_above)
    filtered_bow = [dct.doc2bow(text) for text in tokenized_corpus]
    filtered_bow = corpus2csc(filtered_bow)
    return filtered_bow, list(dct.token2id.keys())

def removeEmpty(processed_corpus, tokenized_corpus, tokenized_ids, classes):
    indexes_to_delete = []
    for i in range(len(processed_corpus)):
        if len(tokenized_corpus[i]) == 0:
            print("DEL", processed_corpus[i])
            indexes_to_delete.append(i)
    processed_corpus = np.delete(processed_corpus, indexes_to_delete)
    tokenized_corpus = np.delete(tokenized_corpus, indexes_to_delete)
    tokenized_ids = np.delete(tokenized_ids, indexes_to_delete)
    classes = np.delete(classes, indexes_to_delete)
    return processed_corpus, tokenized_corpus, tokenized_ids, classes, indexes_to_delete

def preprocess(corpus):
    preprocessed_corpus = np.empty(len(corpus), dtype=np.object)
    table = str.maketrans(dict.fromkeys("\n\r", " "))
    for i in range(len(preprocessed_corpus)):
        preprocessed_corpus[i] = corpus[i].translate(table)
    table = str.maketrans(dict.fromkeys(string.punctuation))
    for i in range(len(corpus)):
        preprocessed_corpus[i] = re.sub(r'\s+', ' ', deaccent(preprocessed_corpus[i].translate(table).lower())).strip()
    return preprocessed_corpus

def removeStopWords(tokenized_corpus):
    new_tokenized_corpus = np.empty(len(tokenized_corpus), dtype=np.object)
    stop_words_corpus = np.empty(len(tokenized_corpus), dtype=np.object)
    stop_words = set(stopwords.words('english'))
    for i in range(len(tokenized_corpus)):
        new_tokenized_corpus[i] = [w for w in tokenized_corpus[i] if w not in stop_words]
        stop_words_corpus[i] = " ".join(new_tokenized_corpus[i])
    return new_tokenized_corpus, stop_words_corpus

def tokensToIds(tokenized_corpus, vocab):
    tokenized_ids = np.empty(len(tokenized_corpus), dtype=np.object)
    for i in range(len(tokenized_corpus)):
        ids = np.empty(len(tokenized_corpus[i]), dtype=np.object)
        for t in range(len(tokenized_corpus[i])):
            ids[t] = vocab[tokenized_corpus[i][t]]
        tokenized_ids[i] = ids
    return tokenized_ids

# This causes OOM error. Need to rework
def ngrams(tokenized_corpus):  # Increase the gram amount by 1
    processed_corpus = np.empty(len(tokenized_corpus), dtype=np.object)
    phrases = Phrases(tokenized_corpus)
    gram = Phraser(phrases)
    for i in range(len(tokenized_corpus)):
        tokenized_corpus[i] = gram[tokenized_corpus[i]]
        processed_corpus[i] = " ".join(tokenized_corpus[i])
    return processed_corpus, tokenized_corpus

def getPCA(tf, depth):
    svd = PCA(n_components=depth, svd_solver="full") # use the scipy algorithm "arpack"
    pos = svd.fit_transform(tf)
    return pos

def averageWV(tokenized_corpus, depth):

    print("")

def averageWVPPMI(tokenized_corpus, ppmi):
    print("")

def main(data_type, output_folder, grams,  no_below, no_above):
    if data_type == "newsgroups":
        newsgroups = fetch_20newsgroups(subset='all', shuffle=False, remove=("headers", "footers", "quotes"))
        corpus = newsgroups.data
        classes = newsgroups.target
    else:
        newsgroups = fetch_20newsgroups(subset='all', shuffle=False, remove=("headers", "footers", "quotes"))
        corpus = newsgroups.data
        classes = newsgroups.target

    file_name = "simple"
    processed_corpus = preprocess(corpus)
    tokenized_corpus = naiveTokenizer(processed_corpus)
    vocab, dct = getVocab(tokenized_corpus)
    #bow = doc2bow(tokenized_corpus, dct, 100, 10)
    tokenized_ids = tokensToIds(tokenized_corpus, vocab)
    processed_corpus, tokenized_corpus, tokenized_ids, classes, remove_ind = removeEmpty(processed_corpus, tokenized_corpus,
                                                                             tokenized_ids, classes)


    np.save(output_folder + file_name + "_remove.npy", remove_ind)
    """
    np.save(output_folder + file_name + "_corpus.npy", tokenized_corpus)
    np.save(output_folder + file_name + "_tokenized_corpus.npy", tokenized_ids)
    np.save(output_folder + file_name + "_vocab.npy", vocab)
    dt.write1dArray(processed_corpus, output_folder + file_name + "_corpus_processed.txt")
    np.save(output_folder + file_name + "_classes.npy", classes)
    np.save(output_folder + file_name + "_classes_categorical.npy", to_categorical(classes))
    sp.save_npz(output_folder + file_name + ".npz", bow)
    dt.write1dArray(word_list, output_folder + file_name + "_words.txt")
    
    if grams > 0:
        for i in range(2, grams):  # Up to 5-length grams
            processed_corpus, tokenized_corpus = ngrams(tokenized_corpus)
            vocab, dct = getVocab(tokenized_corpus)
            bow = doc2bow(tokenized_corpus, dct, 100, 10)
            tokenized_ids = tokensToIds(tokenized_corpus, vocab)
            np.save(output_folder + file_name + "_corpus " + str(i) + "-gram" + ".npy", tokenized_corpus)
            np.save(output_folder + file_name + "_tokenized_corpus " + str(i) + "-gram" + ".npy", tokenized_ids)
            np.save(output_folder + file_name + "_vocab " + str(i) + "-gram" + ".npy", vocab)
            dt.write1dArray(processed_corpus, output_folder + file_name + "_corpus_processed " + str(i) + "-gram" + ".txt")
            sp.save_npz(output_folder + file_name + "_bow " + str(i) + "-gram" + ".npz", bow)
            dt.write1dArray(word_list, output_folder + file_name + "_words.txt")
    """


    file_name += "_stopwords"

    filtered_ppmi_fn = "../data/newsgroups/bow/ppmi/" + file_name + "_ppmi " + str(no_below) + "-" + str(
        no_above) + "-all.npz"
    ppmi_fn = "../data/newsgroups/bow/ppmi/" + file_name + "_ppmi " "2" + "-all.npz"
    bow_fn = "../data/newsgroups/bow/frequency/phrases/" + file_name + "_bow " "2" + "-all.npz"
    filtered_bow_fn = "../data/newsgroups/bow/frequency/phrases/" + file_name + "_bow "  + str(
        no_below) + \
                      "-" + str(no_above) + "-all.npz"

    tokenized_corpus, processed_corpus = removeStopWords(tokenized_corpus)
    vocab, dct = getVocab(tokenized_corpus)
    bow = doc2bow(tokenized_corpus, dct)
    filtered_bow, word_list = filterBow(tokenized_corpus, dct, no_below, no_above)
    tokenized_ids = tokensToIds(tokenized_corpus, vocab)
    processed_corpus, tokenized_corpus, tokenized_ids, classes, remove_ind = removeEmpty(processed_corpus, tokenized_corpus,
                                                                             tokenized_ids, classes)
    np.save(output_folder + file_name + "_remove.npy", remove_ind)

    np.save(output_folder + file_name + "_corpus.npy", tokenized_corpus)
    np.save(output_folder + file_name + "_tokenized_corpus.npy", tokenized_ids)
    np.save(output_folder + file_name + "_vocab.npy", vocab)
    dt.write1dArray(processed_corpus, output_folder + file_name + "_corpus_processed.txt")
    np.save(output_folder + file_name + "_classes.npy", classes)
    np.save(output_folder + file_name + "_classes_categorical.npy", to_categorical(classes))
    #sp.save_npz(output_folder + file_name + "_bow "+ ".npz", bow)
    #dt.write1dArray(word_list, output_folder + file_name + "_words.txt")

    sp.save_npz(bow_fn, bow)
    sp.save_npz(filtered_bow_fn, filtered_bow)

    dt.write1dArray(word_list, "../data/newsgroups/bow/names/" + file_name + "_words " +
                    str(no_below) + "-" + str(no_above) + "-all.txt")
    filtered_bow = filtered_bow.transpose()
    ppmi = mt.convertPPMI(filtered_bow)
    ppmi_sparse = sp.csr_matrix(ppmi).transpose()
    sp.save_npz(filtered_ppmi_fn, ppmi_sparse)
    # Create PCA

    bow = bow.transpose()
    ppmi = mt.convertPPMI(bow)
    ppmi_sparse = sp.csr_matrix(ppmi).transpose()
    sp.save_npz(ppmi_fn, ppmi_sparse)

    if grams > 0:
        for i in range(2, grams+1):  # Up to 5-length grams

            filtered_ppmi_fn = "../data/newsgroups/bow/ppmi/" + file_name + "_ppmi " + str(
                grams) + "-gram" + str(no_below) + "-" + str(
                no_above) + "-all.npz"
            ppmi_fn = "../data/newsgroups/bow/ppmi/" + file_name + "_ppmi " + str(
                grams) + "-gram2" + "-all.npz"
            bow_fn = "../data/newsgroups/bow/frequency/phrases/" + file_name + "_bow " + str(
                grams) + "-gram2" + "-all.npz"
            filtered_bow_fn = "../data/newsgroups/bow/frequency/phrases/" + file_name + "_bow " + str(
                grams) + "-gram" + str(
                no_below) + \
                              "-" + str(no_above) + "-all.npz"

            processed_corpus, tokenized_corpus = ngrams(tokenized_corpus)
            vocab, dct = getVocab(tokenized_corpus)
            bow = doc2bow(tokenized_corpus, dct)
            filtered_bow, word_list = filterBow(tokenized_corpus, dct, no_below, no_above)
            tokenized_ids = tokensToIds(tokenized_corpus, vocab)
            np.save(output_folder + file_name + "_corpus " + str(i) + "-gram" + ".npy", tokenized_corpus)
            np.save(output_folder + file_name + "_tokenized_corpus " + str(i) + "-gram" + ".npy", tokenized_ids)
            np.save(output_folder + file_name + "_vocab " + str(i) + "-gram" + ".npy", vocab)
            dt.write1dArray(processed_corpus, output_folder + file_name + "_corpus_processed " + str(i) + "-gram" + ".txt")

            sp.save_npz(bow_fn, bow)
            sp.save_npz(filtered_bow_fn, filtered_bow)

            dt.write1dArray(word_list, "../data/newsgroups/bow/names/" + file_name + "_words "  + str(i) + "-gram"  +
                            str(no_below) + "-" + str(no_above) + "-all.txt")
            filtered_bow = filtered_bow.transpose()
            ppmi = mt.convertPPMI(filtered_bow)
            ppmi_sparse = sp.csr_matrix(ppmi).transpose()
            sp.save_npz(filtered_ppmi_fn, ppmi_sparse)
            # Create PCA

            bow = bow.transpose()
            ppmi = mt.convertPPMI(bow)
            ppmi_sparse = sp.csr_matrix(ppmi).transpose()
            sp.save_npz(ppmi_fn, ppmi_sparse)
            pca_fn = "../data/newsgroups/nnet/spaces/" + file_name + "_ppmi " + str(grams) + "-gram" + str(
                no_below) + "-" + str(
                no_above) + "-all.npy"

            PCA_ppmi = getPCA(ppmi, 100)
            np.save(pca_fn, PCA_ppmi)

    # Create averaged word vectors
    testAll(["freq_bow", "ppmi_bow", "ppmi_pca"], [ppmi, bow.transpose(), PCA_ppmi], [classes, classes, classes], "newsgroups")
if __name__ == '__main__': main("newsgroups", "../data/raw/newsgroups/", 2, 30, 0.999)