#!/usr/bin/env python2
################################################
# CS434 Machine Learning and Data Mining       #
# Assignment 4: K-means and HAC                #
# Nathan Brahmstadt and Jordan Crane           #
################################################

import math
from math import log
import sys
import numpy as np
import random
from collections import namedtuple
from operator import attrgetter

class Cluster:
    def __init__(self, center, points=[]):
        self.center = center
        self.points = points

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
        self.clusters = []
        self._initialize_clusters()

    def cluster(self):
        converged = False
        while not converged:
            for x in self.data:
                self._assign(x)
            self._update()

    def _assign(self, x):
        min_d = sys.maxint
        for cluster in self.clusters:
            print min_d
            d = self._distance(cluster.center, x)
            if d < min_d:
                min_d = d
                choice = cluster
        choice.points.append(x)

    def _update(self):
        for i, cluster in enumerate(self.clusters):
            new_center = 1 / len(cluster.points) * np.sum(cluster.points)
            cluster.center = new_center

    def _distance(self, a, b):
        return np.sum(np.absolute(a - b))
        return np.sum(a*b)/(np.sum(np.square(a))*np.sum(np.square(b)))

    def _initialize_clusters(self):
        indexes = np.random.choice(len(self.data), self.k, replace=False)
        print indexes
        for i in indexes:
            print self.data[i]
            self.clusters.append(Cluster(self.data[i], []))

def main():
    data = Data("data/data.txt")
    k_means = Kmeans(data)
    k_means.cluster()

if __name__ == '__main__':main()
