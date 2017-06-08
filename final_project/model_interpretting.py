#!/usr/bin/env python2
################################################
# CS434 Machine Learning and Data Mining       #
# Niave Bayes Attempt                          #
# Nathan Brahmstadt and Jordan Crane           #
################################################

import numpy as np
from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)




class QuestionPair:
    def __init__(self, string):
        string = string.split(',')
        self.q1 = string[0]
        self.q2 = string[1]
        self.is_duplicate = int(string[2])
  
def remove_from_list(array, value):
    return [x for x in array if x != value]
    
def main():
    train_file = open('outputs/parsed_questions.csv') 
    output_file = open('outputs/similarities.csv','w+')
    #Link to google news trained binary: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
    print("Loading pre-trained model...")
    model = KeyedVectors.load_word2vec_format('models/GoogleNews-vectors-negative300.bin', binary=True)
    print("Processing Training Data...")
    for line in train_file:
  
        qs = QuestionPair(line)
        q1_words = qs.q1.split(' ') 
        q2_words = qs.q2.split(' ')
        
        for w in q1_words:
            if not w in model.vocab:
                q1_words = remove_from_list(q1_words,w)
                
        for w in q2_words:
            if not w in model.vocab:
                q2_words = remove_from_list(q2_words,w)
                
        try:
            score = model.n_similarity(q1_words, q2_words)
        except:
            score = 0
        output_file.write(str(score)+',\n')
       
    
main()   
