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

Cluster = namedtuple('Cluster', 'center points')

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
        min_d = 999999
        for cluster in self.clusters:
            d = self._distance(cluster.center, x)
            if d < min_d:
                min_d = d
                choice = cluster
        choice.points.extend([x])

    def _update(self):
        for i, cluster in enumerate(self.clusters):
            print len(cluster.points)
            #FIXME All points are going to the first cluster for some reason
            new_center = 1 / len(cluster.points) * np.sum(cluster.points)
            new_cluster = Cluster(new_center, [])
            del self.clusters[i]
            self.clusters.append(new_cluster)

    def _distance(self, a, b):
        return np.sum(np.absolute(a - b))

    def _initialize_clusters(self):
        indexes = np.random.choice(len(self.data), self.k, replace=False)
        for i in indexes:
            self.clusters.append(Cluster(self.data[i], []))

def main():
    data = Data("data/data.txt")
    k_means = Kmeans(data)
    k_means.cluster()

if __name__ == '__main__':main()
