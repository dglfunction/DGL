from __future__ import print_function
from __future__ import division

import numpy as np

def zero_mean_labels(labels_one_hot):
    return labels_one_hot - labels_one_hot.mean(axis=1,keepdims=True)

def balanced_subsample(y, size=None):

    subsample = []

    if size is None:
        n_smp = y.value_counts().min()
    else:
        n_smp = int(size / len(y.value_counts().index))

    for label in y.value_counts().index:
        samples = y[y == label].index.values
        index_range = range(samples.shape[0])
        indexes = np.random.choice(index_range, size=n_smp, replace=False)
        subsample += samples[indexes].tolist()

    return subsample