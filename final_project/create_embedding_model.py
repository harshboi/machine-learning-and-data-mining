#!/usr/bin/env python2
################################################
# CS434 Machine Learning and Data Mining       #
# Niave Bayes Attempt                          #
# Nathan Brahmstadt and Jordan Crane           #
################################################

import numpy as np
from gensim.models import word2vec
import codecs
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



class QuestionPair:
    def __init__(self, string):
        string = string.split(',')
        self.q1 = string[0]
        self.q2 = string[1]
    
        #self.is_duplicate = int(string[2][0])
  
  
def main():
    training_file = codecs.open('outputs/parsed_train_data.csv','r','utf-8')
    testing_file = codecs.open('outputs/parsed_test_data.csv','r','utf-8')
    
    total_words = 0

    
    print("Preparing Model...")
    #Go back to start
    sentences = []
    for line in training_file:
     
        qs = QuestionPair(line)
        q1_words = qs.q1.split(' ') 
        q2_words = qs.q2.split(' ')
  

        sentences.append(q1_words)
        sentences.append(q2_words)
    for line in testing_file:
     
        qs = QuestionPair(line)
        q1_words = qs.q1.split(' ') 
        q2_words = qs.q2.split(' ')
  

        sentences.append(q1_words)
        sentences.append(q2_words)
    print("Training")
    model = word2vec.Word2Vec(sentences, workers=4, size = 300, min_count = 2, sample = 0.001)
    model.init_sims(replace=True)
    model.save('outputs/quora_model')
    
    
main()   
