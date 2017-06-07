#!/usr/bin/env python3
################################################
# CS434 Machine Learning and Data Mining       #
# Niave Bayes on Common and Different Words    #
# Nathan Brahmstadt and Jordan Crane           #
################################################

import numpy as np
import statistics as stats
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")


class QuestionPair:
    def __init__(self, string):
        string = string.split(',')
        self.q1 = string[0]
        self.q2 = string[1]
        self.label = int(string[2])           
        
def load_data_from_file(file):
    qs = []
    for line in file:
        qs.append(QuestionPair(line))
   
    return qs    

def remove_from_list(array, value):
    return [x for x in array if x != value]
            
    
def create_common_word_data(qs):

    common_words = []
    
    for q in qs:
        common_words.append([get_common_words(q.q1, q.q2), q.label])

    return common_words   
    
def create_different_word_data(qs):

    different_words = []
    
    for q in qs:
        different_words.append([get_different_words(q.q1, q.q2), q.label])

    return different_words 
    
def get_common_words(string1, string2):

        to_remove = []
        
        q1_words = string1.split(' ')
        q2_words = string2.split(' ')
        
        q1_words = remove_from_list(q1_words, '')
        q2_words = remove_from_list(q2_words, '')
        
        for w in q1_words:
            if w not in q2_words:
                to_remove.append(w)
        for w in q2_words:
            if w not in q1_words:
                to_remove.append(w)
                
        combined_words = q1_words + q2_words
        
        for w in to_remove:
            combined_words = remove_from_list(combined_words,w)

        
        return combined_words

def get_different_words(string1, string2):

        to_remove = []
        
        q1_words = string1.split(' ')
        q2_words = string2.split(' ')
        
        q1_words = remove_from_list(q1_words, '')
        q2_words = remove_from_list(q2_words, '')
        
        for w in q1_words:
            if w in q2_words:
                to_remove.append(w)
        for w in q2_words:
            if w in q1_words:
                to_remove.append(w)
                
        combined_words = q1_words + q2_words
        
        for w in to_remove:
            combined_words = remove_from_list(combined_words,w)

        return combined_words
        

def bag_of_words_counter(data):
    word_dict = dict()
    for d in data:
        label = d[1]
        for word in d[0]:
            if word not in word_dict:
                word_dict[word] = [0,0]
  
            word_dict[word][label] += 1
    return word_dict    
 
def class_counter(qs):
    totals = [0, 0]
    for q in qs:
        totals[q.label] += 1
    return totals

def common_words_test(q1, q2, count_dict, class_totals):

    words = get_common_words(q1, q2)
    return naive_bayes(words, count_dict, class_totals)
    
    
def different_words_test(q1, q2, count_dict, class_totals):

    words = get_different_words(q1, q2)
    return naive_bayes(words, count_dict, class_totals)
    
def naive_bayes(words, count_dict, class_totals):

    label_0_prob = float(class_totals[0]) / (class_totals[0] + class_totals[1])
    label_1_prob = float(class_totals[1]) / (class_totals[0] + class_totals[1])
    
    for word in words:
        label_0_prob *= count_dict[word][0] / (count_dict[word][0] + count_dict[word][1])
        label_1_prob *= count_dict[word][1] / (count_dict[word][0] + count_dict[word][1])
    
    alpha = 1.0 / (label_0_prob + label_1_prob)
    if label_0_prob >= label_1_prob:
        p = label_0_prob
        label = 0
    else:
        p = label_1_prob
        label = 1
    return (p*alpha, label)

    
def create_features(qs, common_words_dict, different_words_dict, class_totals):

    features = []
    classes = []
    
    for q in qs:
        
        p0, label0 = common_words_test(q.q1, q.q2, common_words_dict, class_totals)
        
        p1, label1 = different_words_test(q.q1, q.q2, different_words_dict, class_totals)
        
        classes.append(q.label)
        
        p0 -= 0.5
        p1 -= 0.5
        
        if label0 == 0:
            p0 *= -1
        if label1 == 0:
            p1 *= -1
            
        features.append([p0, p1])
        
    return features, classes
    
def main():

    training_file = open('outputs/parsed_questions.csv') 
    features_file = open('outputs/features.csv', 'w')
    
    print('Loading Data in from file...')
    qs = load_data_from_file(training_file)
    
    print('Finding common words...')
    common_words = create_common_word_data(qs)
    
    print('Finding different words...')
    different_words = create_different_word_data(qs)
    
    print('Counting Words...')
    common_words_dict = bag_of_words_counter(common_words)
    different_words_dict = bag_of_words_counter(different_words)
    class_totals = class_counter(qs)
    
    print('Testing...')
    (features, classes) = create_features(qs, common_words_dict, different_words_dict, class_totals)
    
    print('SVM...')
    model = LinearSVC()
    model.fit(features, classes)
    print("Accuracy: " + str(model.score(features, classes)))
    
    
    #Plotting code from https://pythonprogramming.net/linear-svc-example-scikit-learn-svm-python/
    print('Plotting...')
    
    w = model.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-0.5,0.5, num = 50)
    yy = a * xx - model.intercept_[0] / w[1]
    h0 = plt.plot(xx, yy, 'k-', label="Decision Line")
    
    np_features = np.array(features)
    plt.scatter(np_features[:,0],np_features[:,1], c = classes)  
    plt.grid(True)
    plt.legend()
    plt.axis([-0.6, 0.6, -0.6, 0.6])
    plt.show()    
    
        
    
main()   
