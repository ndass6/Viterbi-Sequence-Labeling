from gtnlplib.preproc import conll_seq_generator
from gtnlplib.constants import START_TAG, TRANS, END_TAG, EMIT, OFFSET
from gtnlplib import naive_bayes, most_common
import numpy as np
from collections import defaultdict, Counter

def hmm_features(tokens,curr_tag,prev_tag,m):
    """Feature function for HMM that returns emit and transition features

    :param tokens: list of tokens 
    :param curr_tag: current tag
    :param prev_tag: previous tag
    :param m: index of token to be tagged
    :returns: dict of features and counts
    :rtype: dict

    """
    features = { (curr_tag, prev_tag, TRANS) : 1 }
    if m < len(tokens):
        features[(curr_tag, tokens[m], EMIT)] = 1
    return features

def compute_HMM_weights(trainfile,smoothing):
    """Compute all weights for the HMM

    :param trainfile: training file
    :param smoothing: float for smoothing of both probability distributions
    :returns: defaultdict of weights, list of all possible tags (types)
    :rtype: defaultdict, list

    """
    tag_trans_counts = most_common.get_tag_trans_counts(trainfile)
    all_tags = tag_trans_counts.keys()

    transition_weights = compute_transition_weights(tag_trans_counts, smoothing)
    for tag in all_tags + [END_TAG]:
        transition_weights[(START_TAG, tag, TRANS)] = -np.inf
        transition_weights[(tag, END_TAG, TRANS)] = -np.inf
    counters = most_common.get_tag_word_counts(trainfile)
    nb_weights = naive_bayes.estimate_nb_tagger(counters, smoothing)
    emission_weights = {}
    for nb_weight in nb_weights:
        if not OFFSET in nb_weight:
            emission_weights[(nb_weight[0], nb_weight[1], EMIT)] = nb_weights[nb_weight]
    all_weights = Counter(transition_weights)
    all_weights.update(Counter(emission_weights))

    return defaultdict(float, all_weights), all_tags


def compute_transition_weights(trans_counts, smoothing):
    """Compute the HMM transition weights, given the counts.
    
    This will also affect your computation of the denominator.

    :param trans_counts: counts, generated from most_common.get_tag_trans_counts
    :param smoothing: additive smoothing
    :returns: dict of features [(curr_tag,prev_tag,TRANS)] and weights

    """

    weights = defaultdict(float)
    totals = { tag : sum(trans_counts[tag].values()) for tag in trans_counts.keys() }

    for prev_tag in trans_counts:
        for curr_tag in (trans_counts.keys() + [END_TAG]):
            weights[(curr_tag, prev_tag, TRANS)] = np.log((trans_counts[prev_tag][curr_tag] + smoothing) / (totals[prev_tag] + len(trans_counts) * smoothing))

    for tag in trans_counts:
        weights[START_TAG, tag, TRANS] = -np.inf
    return weights