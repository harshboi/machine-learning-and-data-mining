#!/usr/bin/env python3

########################################
## import packages
########################################
import os
import re
import csv
import codecs
#import itertools
import numpy as np
#import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation

#import keras
#from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
#from keras.layers import Input, LSTM, Dense
#from keras.models import Model
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.model_selection import test train split

TRAIN_DATA_FILE = 'data/train.csv'
OUTPUT_DATA_FILE = 'outputs/parsed_questions.csv'

########################################
## process texts in datasets
########################################
print('Processing text dataset')

# The function "text_to_wordlist" is modified from
# https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text
def text_to_wordlist(text, remove_stopwords=False, remove_punctuation=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.

    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        try:
            stops = set(stopwords.words("english"))
        except:
            nltk.download('stopwords')
            stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]

    # Optionally, remove punctuation
    if remove_punctuation:
        text = [w for w in text if not w in punctuation]

    text = " ".join(text)

    # Clean the text
    #text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r"\be g\b", " eg ", text)
    text = re.sub(r"\bb g\b", " bg ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r"\b9 11\b", "911", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"\bj k\b", " jk ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r"\busa\b", " america ", text)
    text = re.sub(r"\bu s\b", " america ", text)
    text = re.sub(r"\bus\b", " america ", text)
    text = re.sub(r"\bu k\b", " england ", text)
    text = re.sub(r"\buk\b", " england ", text)
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r"\bdms\b", "direct messages ", text)
    text = re.sub(r"demonitization", "demonetization", text)
    text = re.sub(r"\bkms\b", " kilometers ", text)
    text = re.sub(r"\0rs ", " rs ", text)
    text = re.sub(r"programing", "programming", text)
    text = re.sub(r"bestfriend", "best friend", text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    # Return a list of words
    return(text)

#Import data
#train = pd.read_csv("data/train.csv")

texts_1 = []
texts_2 = []
label_list = []

output_file = open(OUTPUT_DATA_FILE,'w+')

with codecs.open(TRAIN_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        texts_1.append(text_to_wordlist(values[3], remove_stopwords=True,
            remove_punctuation=True))
        texts_2.append(text_to_wordlist(values[4], remove_stopwords=True,
            remove_punctuation=True))
        label_list.append(int(values[5]))
        output_file.write(texts_1[-1] + "," + texts_2[-1] + "," + str(label_list[-1]) + ",\n")

#questions_1 = np.array(texts_1)
#questions_2 = np.array(texts_2)
#labels = np.array(label_list)
print('Found %s texts in train.csv' % len(texts_1))

#Sanitize data
#for i in range(10):
#    texts_1.append(text_to_wordlist(train.question1[i]))
#    texts_2.append(text_to_wordlist(train.question2[i]))
#    label_list.append(int(train.is_duplicate[i]))
#
#questions_1 = np.array(texts_1)
#questions_2 = np.array(texts_2)
#labels = np.array(label_list)
#print('Found %s texts in train.csv' % len(texts_1))
#
##Create Vocab
#count_vectorizer = CountVectorizer(max_features=10000-1).fit(
#        itertools.chain(questions_1, questions_2))
#other_index = len(count_vectorizer.vocabulary_)

#Prep data

