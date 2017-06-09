#!/usr/bin/env python3
################################################
# CS434 Machine Learning and Data Mining       #
# Niave Bayes on Common and Different Words    #
# Nathan Brahmstadt and Jordan Crane           #
################################################

import random
import numpy as np
import statistics as stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style
from math import log
import codecs
style.use("ggplot")

class QuestionPair:
    def __init__(self, string):
        string = string.split(',')
        self.q1 = string[0]
        self.q2 = string[1]
        try:
            self.label = int(string[2])
        except:
            pass

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
        if word not in count_dict:
            count_dict[word] = [0,0]
        label_0_prob *= (count_dict[word][0]+1) / (count_dict[word][0] + count_dict[word][1] + 1)
        label_1_prob *= (count_dict[word][1]+1) / (count_dict[word][0] + count_dict[word][1] + 1)

    alpha = 1.0 / (label_0_prob + label_1_prob)
    if label_0_prob >= label_1_prob:
        p = label_0_prob
        label = 0
    else:
        p = label_1_prob
        label = 1
    return (p*alpha, label)


def double_bayes_train(qs, common_words_dict, different_words_dict, class_totals):

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

def tfidf_compute(qs, common_words_dict, different_words_dict):

    scores = []

    for q in qs:
        score = 0
        common_words = get_common_words(q.q1, q.q2)
        different_words = get_different_words(q.q1, q.q2)
       
            
        for current_word in common_words:            
            current_word_count = 0
            if current_word in common_words_dict:
                current_word_count += common_words_dict[current_word][0] + common_words_dict[current_word][1]
            if current_word in different_words_dict:
                current_word_count += different_words_dict[current_word][0] + different_words_dict[current_word][1]
            score -= log(float(len(qs)) / (current_word_count+1))
          
        for current_word in different_words:
            current_word_count = 0
            if current_word in different_words_dict:
                current_word_count += different_words_dict[current_word][0] + different_words_dict[current_word][1]
            if current_word in common_words_dict:
                current_word_count += common_words_dict[current_word][0] + common_words_dict[current_word][1]
            score += log(float(len(qs)) / (current_word_count+1))
        
        scores.append(score)
    return scores    
    
def double_bayes_test(qs, common_words_dict, different_words_dict, class_totals):

    features = []


    for q in qs:

        p0, label0 = common_words_test(q.q1, q.q2, common_words_dict, class_totals)

        p1, label1 = different_words_test(q.q1, q.q2, different_words_dict, class_totals)

        p0 -= 0.5
        p1 -= 0.5

        if label0 == 0:
            p0 *= -1
        if label1 == 0:
            p1 *= -1

        features.append([p0, p1])

    return features

def create_word_counts(qs):    
    
    common_words = create_common_word_data(qs)


    different_words = create_different_word_data(qs)


    common_words_dict = bag_of_words_counter(common_words)
    different_words_dict = bag_of_words_counter(different_words)
    class_totals = class_counter(qs)
 
    return common_words_dict, different_words_dict, class_totals
    
    
def training_features(training_file, train_data_similarity_file):

    print('Loading in Training Data')
    qs = load_data_from_file(training_file)
    
    print('Analyzing word counts for models...')
    common_words_dict, different_words_dict, class_totals = create_word_counts(qs)
    
    print('tf-idf')
    tfidf_features = tfidf_compute(qs, common_words_dict, different_words_dict)
    
    print('Naive Bayes Training')
    (bayes_features, classes) = double_bayes_train(qs, common_words_dict, different_words_dict, class_totals)
    
    print('Adding embedded similarities as a feature')
    similarity_features = []
    for line in train_data_similarity_file:
        similarity_features.append(float(line.split(',')[0]))
        
    for i,x in enumerate(bayes_features):
        bayes_features[i].append(similarity_features[i])
        bayes_features[i].append(tfidf_features[i])
        
    return qs, bayes_features, common_words_dict, different_words_dict, class_totals, classes

def training_logistic_regression(features, classes):
    model = LogisticRegression()
    model.fit(features, classes)
    training_results = model.predict_proba(features)
    print("Accuracy: " + str(model.score(features, classes)))
    print("Log Loss: " + str(log_loss(classes, training_results)))
    return model, training_results
    
def generate_wrong_file(wrong_file, model, features, qs, training_results, classes):
    guesses = model.predict(features)
    wrong_file.write('q1, q2, label, common_bayes, different_bayes, tfidf, logistic model result\n')
    for i,guess in enumerate(guesses):
        if guess != classes[i]:
            wrong_file.write(qs[i].q1 + ',' + qs[i].q2 + ',' + str(qs[i].label)
                    + ',' + str(features[i][0]) + ',' + str(features[i][1]) + ',' + str(features[i][3]) + ','
                    + str(training_results[i][1]) + '\n')

def generate_test_results_file(testing_file, common_words_dict, different_words_dict, class_totals, model, test_data_similarity_file, test_results_file):

    qs = load_data_from_file(testing_file)
    
    bayes_features = double_bayes_test(qs, common_words_dict, different_words_dict, class_totals)
    
    print("test tfidf")
    tfidf_features = tfidf_compute(qs, common_words_dict, different_words_dict)
    
    print('Adding in similarity scores for additional feature on test data')
    similarity_features = []
    for line in test_data_similarity_file:
        similarity_features.append(float(line.split(',')[0]))
        
    for i,x in enumerate(bayes_features):
        bayes_features[i].append(similarity_features[i])
        bayes_features[i].append(tfidf_features[i])
        
    test_results = model.predict_proba(bayes_features)
    
    for i,guess in enumerate(test_results):
        test_results_file.write(str(i)+','+str(guess[1])+'\n')    
        
def main():
    #All our files
    training_file = codecs.open('outputs/parsed_train_data.csv','r','utf-8')
    testing_file = codecs.open('outputs/parsed_test_data.csv','r','utf-8')
    
    test_results_file = codecs.open('outputs/test_data_results.csv','w+','utf-8')
    
    train_data_similarity_file = codecs.open('outputs/train_data_embedded_similarity.csv','r','utf-8')
    test_data_similarity_file = codecs.open('outputs/test_data_embedded_similarity.csv','r','utf-8')
    
    wrong_file = codecs.open('outputs/wrong.csv','w+','utf-8',)
    
    #Put expected header on test data
    test_results_file.write('test_id,is_duplicate\n')
  
    print('Training...')
    qs, features, common_words_dict, different_words_dict, class_totals, classes = training_features(training_file, train_data_similarity_file)
    
    print('Logistic Regression on all features...')
    model, training_results = training_logistic_regression(features, classes)

    print('Generating file with samples we get wrong...')
    generate_wrong_file(wrong_file, model, features, qs, training_results, classes)

    print('Test Data...')
    generate_test_results_file(testing_file, common_words_dict, different_words_dict, class_totals, model, test_data_similarity_file, test_results_file)
    



main()
