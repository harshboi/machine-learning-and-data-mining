#!/usr/bin/env python2
################################################
# CS434 Machine Learning and Data Mining       #
# Niave Bayes Attempt                          #
# Nathan Brahmstadt and Jordan Crane           #
################################################

import numpy as np
import statistics as stats
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")


class QuestionPair:
    def __init__(self, string):
        string = string.split(',')
        self.q1 = string[0]
        self.q2 = string[1]
    
        self.is_duplicate = int(string[2][0])
  
    
def main():
    train_file = open('outputs/sentences.csv')
    word_file = open('outputs/words.txt', 'w')  
    wrongs_file = open('outputs/wrongs.csv', 'w')
    features_file = open('outputs/features.csv', 'w')
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
    duplicates = 0
    samples = 0
    common_word_dict = dict()
    different_word_dict = dict()
    for line in train_file:
     
        qs = QuestionPair(line)
        q1_words = qs.q1.split(' ') 
        q2_words = qs.q2.split(' ')
        if qs.is_duplicate:
            duplicates += 1
        samples += 1
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
            if current_word in common_word_dict:
                common_word_dict[current_word][0] += 1
            else:
                common_word_dict[current_word] = [1,0]
            if qs.is_duplicate:
                    common_word_dict[current_word][1] += 1
                    
        for current_word in different_words:  
            if current_word in different_word_dict:
                different_word_dict[current_word][0] += 1
            else:
                different_word_dict[current_word] = [1,0]
            if qs.is_duplicate:
                    different_word_dict[current_word][1] += 1
  
    percent_duplicates = float(duplicates)/samples  
        
    print "Training"
    guesses = []
    
    train_file.seek(0)
    

    
    for line in train_file:
   
        qs = QuestionPair(line)
        q1_words = qs.q1.split(' ') 
        q2_words = qs.q2.split(' ')
        
        common_words = []
        different_words = []
        probability_match = 1
        probability_not_match = 1
        
        all_words = q1_words + q2_words
        
        for w in all_words:
            if not (w == '\n' or w == ''):
                if w in q1_words and w in q2_words:
                    common_words.append(w)
                else:
                    different_words.append(w)   
                    
        #Based on cool derivation of bayes here: https://en.wikipedia.org/wiki/Naive_Bayes_spam_filtering            
        for word in common_words:
      
            probability_match *= (float)(common_word_dict[word][1])/(common_word_dict[word][0])
            probability_not_match *= (1 - ((float)(common_word_dict[word][1])/(common_word_dict[word][0])))
            
        guess_same = float(probability_match)/(probability_match+probability_not_match)    
        
        for word in different_words:
      
            probability_match *= (float)(different_word_dict[word][1])/(different_word_dict[word][0])
            probability_not_match *= (1 - ((float)(different_word_dict[word][1])/(different_word_dict[word][0])))    
        
        guess_different = float(probability_match)/(probability_match+probability_not_match)
       
        guesses.append((guess_same, guess_different, qs.is_duplicate))
        
    print "SVM time..."
    
    features = []   
    classes = []
    for g in guesses:
        features.append((g[0],g[1]))
        classes.append(g[2])
    
    model = SVC(max_iter = 1000000, kernel='linear', C = 1.0)
    model.fit(features, classes)
    
            
    print "Accuracy: " + str(model.score(features, classes))   
    #Plotting code from https://pythonprogramming.net/linear-svc-example-scikit-learn-svm-python/
    np_features = np.array(features)
  
    w = model.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(0,1, num = 50)
    yy = a * xx - model.intercept_[0] / w[1]
    h0 = plt.plot(xx, yy, 'k-', label="non weighted div")
    plt.scatter(np_features[:,0],np_features[:,1], c = classes)  
    plt.legend()
    plt.axis([0, 1, 0, 1])
    plt.show()    

        
    
main()   
