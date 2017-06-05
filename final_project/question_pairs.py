#!/usr/bin/env python2
################################################
# CS434 Machine Learning and Data Mining       #
# Niave Bayes Attempt                          #
# Nathan Brahmstadt and Jordan Crane           #
################################################

import sys
import numpy as np
import random
from collections import namedtuple
from operator import attrgetter
import statistics as stats


class QuestionPair:
    def __init__(self, string):
        string = string.split(',')
        self.q1 = string[0]
        self.q2 = string[1]
        try:
            self.is_duplicate = int(string[2][0])
        except:
          
            print string
            print string[2]
    
def main():
    train_file = open('data/sentences.csv')
    word_file = open('words.txt', 'w')  
    wrongs_file = open('wrongs.txt', 'w')
    word_dict = dict()
    print "Counting"
    total_words = 0
    
    for line in train_file:
        qs = QuestionPair(line)
        words_in_question = qs.q1.split(' ') + qs.q2.split(' ')
        for current_word in words_in_question: 
            if not current_word == '':
                total_words += 1
                if current_word in word_dict:                 
                    word_dict[current_word] += 1                       
                else:
                    word_dict[current_word] = 1
    
    print "Writing File"
    for w in word_dict:
            word_file.write(w + "," + str(word_dict[w]) + '\n')
            
    scores = []
    print "Learning..."
    #Go back to start
    train_file.seek(0)
    for line in train_file:
     
        qs = QuestionPair(line)
        q1_words = qs.q1.split(' ') 
        q2_words = qs.q2.split(' ')
       
        common_words = []
        different_words = []
        
        score = 0
        
        all_words = q1_words + q2_words
       
        for w in all_words:
            if not (w == '\n' or w == ''):
                if w in q1_words and w in q2_words:
                    common_words.append(w)
                else:
                    different_words.append(w)           
        
        for current_word in common_words:            
             
            score -= float(total_words) / word_dict[current_word]
          
        for current_word in different_words:
     
            score += float(total_words) / word_dict[current_word]
            
        #difference = (q1_score-q2_score)**2
    
        scores.append([score,qs.is_duplicate,qs.q1,qs.q2])
      
        
    pos_average = 0
    pos_count = 0
    neg_average = 0
    neg_count = 0
    
    positives = []
    negatives = []
    
    for score in scores:     
        if score[1] == 1:
            pos_count += 1
            pos_average += score[0]
            positives.append(score[0])
        else:
            neg_count += 1
            neg_average += score[0]
            negatives.append(score[0])
           
    pos_average = pos_average / pos_count   
    neg_average = neg_average / neg_count
    print "Difference score for positives:"
    print pos_average
    print "Difference score for negatives:"
    print neg_average
    print "STD"
    print stats.stdev(positives)
    pos_point = stats.median(positives)

    print stats.stdev(negatives)
    neg_point = stats.median(negatives)
    decision_point = neg_average - pos_average
    
    right = 0
    wrong = 0
    abstains = 0
    
    
    for score in scores:
        if score[0] > neg_point:
            predict = 0
        elif score[0] < pos_point:
            predict = 1
        else:
            predict = None
            abstains += 1
        if not predict == None:
            if predict == score[1]:
                right += 1
            else:
                wrong += 1
                wrongs_file.write(str(score) + '\n')
                
    print "Accuracy of what I said:"        
    print float(right)/(right+wrong)
    print "Percent abstained:"
    print float(abstains)/(abstains+right+wrong)
main()   
