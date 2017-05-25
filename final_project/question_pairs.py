#!/usr/bin/env python2
################################################
# CS434 Machine Learning and Data Mining       #
# Assignment 4: K-means and HAC                #
# Nathan Brahmstadt and Jordan Crane           #
################################################

import sys
import numpy as np
import random
from collections import namedtuple
from operator import attrgetter


class QuestionPair:
    def __init__(self, string):
        string = string.split('","')
        #for i, l in enumerate(string):
            #string[i] = l[1:-1]
            #print l[1:-1]
        self.id = string[0]
        self.qid1 = string[1]
        self.qid2 = string[2]
        self.q1 = regularize(string[3])
        self.q2 = regularize(string[4])
        self.is_duplicate = string[5]

class Word:
    def __init__(self, string):
        self.string = string
        self.count = 1
        
def regularize(string):
    string = string.lower()
    slist = list(string)
    punctuation = ['?', '/', '-', '.', '[', ']', '(', ')', '"', ',']
    for i,c in enumerate(slist):
        if c in punctuation:
            slist[i] = ' '
    #print "".join(slist)
    return "".join(slist) 
    
def main():
    train_file = open('data/train_reduced.csv')
    word_file = open('words.txt', 'w')
    data = []
    print "Parsing"
    for i,line in enumerate(train_file):
        if not i == 0:
            data.append(QuestionPair(line))
        
    words = []
    print "Counting"
    for qs in data:
        words_in_question = qs.q1.split(' ') + qs.q2.split(' ')
        for current_word in words_in_question:
            found_match_flag = False
            if not current_word == '':
                for word in words:
                    if word.string == current_word:
                        word.count += 1
                        found_match_flag = True
                if found_match_flag == False:
                    words.append(Word(current_word))
    print "writing..."  
    total_words = 0
    for w in words:
        if not w.string == '':
            word_file.write(w.string + "," + str(w.count) + '\n')
            total_words += w.count
    scores = []
    for qs in data:
        q1_words = qs.q1.split(' ') 
        q2_words = qs.q2.split(' ')
       
        common_words = []
        different_words = []
        
        score = 0
        
        all_words = q1_words + q2_words
       
        for w in all_words:
            if not w == ' ':
                if w in q1_words and w in q2_words:
                    common_words.append(w)
                else:
                    different_words.append(w)           
        
        for current_word in common_words:
            for w in words:
                if w.string == current_word:
                    count = w.count
            score -= total_words / count
            
        for current_word in q2_words:
            for w in words:
                if w.string == current_word:
                    count = w.count
            score += total_words / count
        #difference = (q1_score-q2_score)**2
        scores.append([score,qs.is_duplicate])
    pos_average = 0
    pos_count = 0
    neg_average = 0
    neg_count = 0
    for score in scores:
        try:
            score[1] = int(score[1][0])
        except:
            print score
        if score[1] == 1:
            pos_count += 1
            pos_average += score[0]
        else:
            neg_count += 1
            neg_average += score[0]
            
    pos_average = pos_average / pos_count
    neg_average = neg_average / neg_count
    print "Difference score for positives:"
    print pos_average
    print "Difference score for negatives:"
    print neg_average
    
    decision_point = neg_average - pos_average
    
    right = 0
    wrong = 0
    for score in scores:
        if score[0] > decision_point:
            predict = 0
        else:
            predict = 1
        if predict == score[1]:
            right += 1
        else:
            wrong += 1
            
    print float(right)/(right+wrong)
main()   
