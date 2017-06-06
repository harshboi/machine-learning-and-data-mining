#!/usr/bin/env python2
################################################
# CS434 Machine Learning and Data Mining       #
# Niave Bayes Attempt                          #
# Nathan Brahmstadt and Jordan Crane           #
################################################

import numpy as np
from gensim.models import word2vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s, level=logging.INFO)



class QuestionPair:
    def __init__(self, string):
        string = string.split(',')
        self.q1 = string[0]
        self.q2 = string[1]
    
        self.is_duplicate = int(string[2][0])
  
  
def main():
    train_file = open('outputs/sentences.csv') 
    
    model = word2vec.load('outputs/first_model')
    
    print "Starting"
    #Go back to start
    positives = []
    negatives = []
    for line in train_file:
     
        qs = QuestionPair(line)
        q1_words = qs.q1.split(' ') 
        q2_words = qs.q2.split(' ')
  
        score1 = model.score(q1_words)
        score2 = model.score(q2_words)
        if qs.is_duplicate:
            positives.append(abs(score1-score2))
        else:
            negatives.append(abs(score1-score2))
    print "Results:"
    print np.mean(positives)
    print np.mean(negatives)
        
    #print "Training"
   # model = word2vec.Word2Vec(sentences, workers=4, size = 300, min_count = 2, sample = 0.001)
   # model.init_sims(replace=True)
   # model.save('outputs/first_model')
   
   
    
    
main()   
