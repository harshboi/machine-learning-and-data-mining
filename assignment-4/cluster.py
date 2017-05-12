#!/usr/bin/env python2
################################################
# CS434 Machine Learning and Data Mining       #
# Assignment 4: K-means and HAC                #
# Nathan Brahmstadt and Jordan Crane           #
################################################

import math
from math import log
import numpy as np
import random
from collections import namedtuple
from operator import attrgetter

class Data:
    def __init__(self, filename=None):
        file = open(filename)
        self.features = self._build_data_arrays(file)
        file.close()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]

    def _build_data_arrays(self, file):
        features = self._build_data_lists(file)
        return np.array(features, dtype=float)

    def _build_data_lists(self, file):
        features = []
        for line in file:
            features.append(line.split(','))
        return features

class Kmeans:
    def __init__(self, data, k=2):
        self.data = data
        self.k = k
        self.centers = []

    def initialize_centers(self):
        indexes = np.random.choice(len(self.data), self.k, replace=False)
        for i in indexes:
            self.centers.append(self.data[i])

def main():
    data = Data("data/data.txt")
    k_means = Kmeans(data)
    k_means.initialize_centers()

if __name__ == '__main__':main()
