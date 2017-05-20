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

class CsvPrinter:
    def __init__(self, filename, labels=[], delimiter=','):
        if labels:
            self.file = open(filename, 'w')
            self.columns = len(labels)
            labels.append('\n')
            self.delimiter = delimiter
            self.file.write(self.delimiter.join(labels))
        else:
            print "CsvPrinter: Please provide column labels"
            exit()

    def writerow(self, data):
        if len(data) == self.columns:
            data.append('\n')
            self.file.write(self.delimiter.join([str(x) for x in data]))
        else:
            print "CsvPrinter: Data length should match number of labels"
            exit()

    def close(self):
        self.file.close()

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

    def calculate_sse(self):
        return np.sum(np.square(self.points - self.center))

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

    def calculate_sse(self):
        sse = 0
        for cluster in self._clusters:
            sse += cluster.calculate_sse()
        return sse

    def reset(self):
        del self._clusters[:]
        del self._previous_cluster_sizes[:]
        self._initialize_clusters()

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

class hac_cluster:
    
    def __init__(self, data, id):
        self.id = id   
        self.points = []
        self.points.append(data)
        self.neighbor_id = None
        self.distance_to_neighbor = None
        
    
        
        
def calculate_distance(p1, p2):
    sum = 0
    for i in range(len(p1)):
        sum += (p2[i] - p1[i])**2
    return sum**0.5
        
def update_neighbors_single_link(clusters, removed_id = None):
    #print "Update Neighbors"
    #go through every pair of clusters
    for i, c1 in enumerate(clusters):
        if removed_id == None:
            for j, c2 in enumerate(clusters[i+1:]):
                #go through every point in those 2 clusters
              
                for p1 in c1.points:
                    for p2 in c2.points:
                     
                        distance = calculate_distance(p1, p2)
                    
                        if c1.distance_to_neighbor == None or c1.distance_to_neighbor > distance:
                            c1.distance_to_neighbor = distance
                            c1.neighbor_id = c2.id
                    
                            
                        if c2.distance_to_neighbor == None or c2.distance_to_neighbor > distance:
                            c2.distance_to_neighbor = distance
                            c2.neighbor_id = c1.id
                            
            
        else:

            if removed_id == c1.neighbor_id:
                
                c1.distance_to_neighbor = None
                c1.neighbor_id = None
                
                for j, c2 in enumerate(clusters[i+1:]):
                #go through every point in those 2 clusters
              
                    for p1 in c1.points:
                        for p2 in c2.points:
                         
                            distance = calculate_distance(p1, p2)
                        
                            if c1.distance_to_neighbor == None or c1.distance_to_neighbor > distance:
                                c1.distance_to_neighbor = distance
                                c1.neighbor_id = c2.id
                                
                            if c2.distance_to_neighbor == None or c2.distance_to_neighbor > distance:
                                c2.distance_to_neighbor = distance
                                c2.neighbor_id = c1.id
                
                    
                        
def merge_clusters(clusters):
   # print "Merge"
    smallest_distance = None
    cluster_index = None
    for i, c in enumerate(clusters):
        if smallest_distance == None or smallest_distance < c.distance_to_neighbor:
            smallest_distance = c.distance_to_neighbor
            cluster_index = i
    
    cluster_to_remove = clusters[cluster_index].neighbor_id
  
    for i, c in enumerate(clusters):
        if c.id == cluster_to_remove:
            cluster_index_to_remove = i
          
    new_points = []
    


    for x in clusters[cluster_index_to_remove].points:
        new_points.append(x)
    for x in clusters[cluster_index].points:
        new_points.append(x)    
       
    clusters[cluster_index].points = new_points
    removed_c_id = clusters[cluster_index_to_remove].id
    
    #Points have been unioned...we need to remove all references to the old cluster
    
    clusters.pop(cluster_index_to_remove)
  
    update_neighbors_single_link(clusters, removed_id=removed_c_id)

    return clusters
    
def main():
    data = Data("data/data_reduced.txt")
    #k_means_clustering(data)
    hac(data[:100])

def k_means_clustering(data):
    labels = 'k T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 AVG'.split()
    csv = CsvPrinter("reports/k_means.csv", labels)
    for k in range(2, 11):
        k_means = Kmeans(data, k=k)
        sses = []
        for i in range(10):
            k_means.reset()
            k_means.cluster()
            sses.append(k_means.calculate_sse())
        avg = float(sum(sses))/len(sses)
        sses.insert(0, k)
        sses.append(avg)
        csv.writerow(sses)
    csv.close()

def hac(data):
    clusters = []
    for i, x in enumerate(data):
        clusters.append(hac_cluster(x, i))
    #print len(clusters[0].points) 
    update_neighbors_single_link(clusters)
    while len(clusters) > 10:
        print "# of clusters"
        print len(clusters)      
        clusters = merge_clusters(clusters)
    for c in clusters:
        print c.points
  
    
if __name__ == '__main__':main()
