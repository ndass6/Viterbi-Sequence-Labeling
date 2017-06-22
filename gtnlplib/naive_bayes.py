import numpy as np
import sys
from collections import defaultdict,Counter
from gtnlplib import scorer, most_common,preproc
from gtnlplib.constants import OFFSET

def get_corpus_counts(x,y,label):
    """Compute corpus counts of words for all documents with a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label for corpus counts
    :returns: defaultdict of corpus counts
    :rtype: defaultdict

    """
    corpus_counts = defaultdict(float)
    for pos, curr_label in enumerate(y):
        if curr_label == label:
            for word in x[pos]:
                corpus_counts[word] += x[pos][word]
    return corpus_counts
    
def estimate_pxy(x,y,label,smoothing,vocab):
    """Compute smoothed log-probability P(word | label) for a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label
    :param smoothing: additive smoothing amount
    :param vocab: list of words in vocabulary
    :returns: defaultdict of log probabilities per word
    :rtype: defaultdict of log probabilities per word

    """
    log_probabilities = defaultdict(float)
    corpus_counts = get_corpus_counts(x, y, label)
    total = sum(corpus_counts.values())
    for word in vocab:
        log_probabilities[word] = np.log(((corpus_counts[word] if word in corpus_counts else 0) + smoothing) / (total + len(vocab) * smoothing))
    return log_probabilities

def estimate_nb(x,y,smoothing):
    """estimate a naive bayes model

    :param x: list of dictionaries of base feature counts
    :param y: list of labels
    :param smoothing: smoothing constant
    :returns: weights
    :rtype: defaultdict 

    """
    labels = set(y)
    doc_counts = defaultdict(float)
    weights = defaultdict(float)

    vocab = set()
    for base_features in x:
        for word in base_features.keys():
            vocab.add(word)

    for label in y:
        doc_counts[label] += 1


    for label in labels:
        weights[(label, OFFSET)] = np.log(doc_counts[label] / sum(doc_counts.values()))
        log_probabilities = estimate_pxy(x, y, label, smoothing, vocab)
        for word in log_probabilities:
            weights[(label, word)] = log_probabilities[word]

    return weights

def estimate_nb_tagger(counters,smoothing):
    """build a tagger based on the naive bayes classifier, which correctly accounts for the prior P(Y)

    :param counters: dict of word-tag counters, from most_common.get_tag_word_counts
    :param smoothing: value for lidstone smoothing
    :returns: classifier weights
    :rtype: defaultdict

    """
    nb_weights = estimate_nb([counters[tag] for tag in counters.keys()], counters.keys(), smoothing)
    tag_count = defaultdict(float)
    total_count = 0.
    for tag in counters.keys():
        tag_count[tag] = sum(counters[tag].values())
        total_count += sum(counters[tag].values())
    for tag in counters.keys():
        nb_weights[(tag, OFFSET)] = np.log(tag_count[tag] / total_count)
    return nb_weights