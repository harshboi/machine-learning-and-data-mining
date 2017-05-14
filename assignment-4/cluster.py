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
    def update_center(self):
        self.center = np.mean(self.points, axis=0)
        del self.points[:]
    def similarity(self, x):
        #Euclidian Distance
        return np.sum(np.square(self.center - x))
        #Manhattan Distance
        #return np.sum(np.absolute(self.center - x))
        #Cosine Similarity
        #return np.sum(self.center*x)/(np.sum(np.square(self.center))*np.sum(np.square(x)))

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
        self._data = data
        self._k = k
        self._clusters = []
        self._previous_cluster_sizes = []
        self._initialize_clusters()

    def cluster(self):
        converged = False
        while not converged:
            for x in self._data:
                self._assign_point(x)
            if self._check_convergence():
                converged = True
                print "Converged!"
            else:
                self._update_centers()

    def _assign_point(self, x):
        min_error = sys.maxint
        for cluster in self._clusters:
            error = cluster.similarity(x)
            if error < min_error:
                min_error = error
                choice = cluster
        choice.points.append(x)

    def _update_centers(self):
        del self._previous_cluster_sizes[:]
        for cluster in self._clusters:
            self._previous_cluster_sizes.append(len(cluster.points))
            cluster.update_center()

    def _initialize_clusters(self):
        indexes = np.random.choice(len(self._data), self._k, replace=False)
        for i in indexes:
            self._clusters.append(Cluster(self._data[i], []))

    def _check_convergence(self):
        cluster_sizes = []
        for cluster in self._clusters:
            cluster_sizes.append(len(cluster.points))
        print cluster_sizes
        return cluster_sizes == self._previous_cluster_sizes

def main():
    data = Data("data/data.txt")
    k_means = Kmeans(data)
    k_means.cluster()

if __name__ == '__main__':main()
