#!/usr/bin/env python3

########################################
## import packages
########################################
import os
import re
import csv
import codecs
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

TRAIN_DATA_FILE = 'data/train.csv'
OUTPUT_TRAIN_DATA_FILE = 'outputs/parsed_train_data.csv'

TEST_DATA_FILE = 'data/test.csv'
OUTPUT_TEST_DATA_FILE = 'outputs/parsed_test_data.csv'

########################################
## process texts in datasets
########################################

# The function "text_to_wordlist" is modified from
# https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text
def text_to_wordlist(text, remove_stopwords=False, remove_punctuation=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.

    # Convert words to lower case
    text = text.lower()

    # Optionally, remove punctuation
    if remove_punctuation:
        text = re.sub(r"[,.;@#!&$]+\ *", " ", text)

    # Optionally, remove stop words
    if remove_stopwords:
        try:
            stops = set(stopwords.words("english"))
        except:
            nltk.download('stopwords')
            stops = set(stopwords.words("english"))
        text = text.split()
        text = [w for w in text if not w in stops]
        text = " ".join(text)

    # Clean the text
    #text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"[()]", "", text)
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

print('Processing training dataset')
i = 0
with codecs.open(OUTPUT_TRAIN_DATA_FILE,'w','utf-8') as output_train_file:
    with codecs.open(TRAIN_DATA_FILE, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        for values in reader:
            i += 1
            output_train_file.write(
                    "".join([
                        text_to_wordlist(values[3], remove_stopwords=True,
                            remove_punctuation=True), ",",
                        text_to_wordlist(values[4], remove_stopwords=True,
                            remove_punctuation=True), ",",
                        values[5], "\n"
                        ])
                    )

print('Found ', i, ' texts in train.csv')

print('Processing testing dataset')
i = 0
with codecs.open(OUTPUT_TEST_DATA_FILE,'w','utf-8') as output_test_file:
    with codecs.open(TEST_DATA_FILE, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        for values in reader:
            i += 1
            output_test_file.write(
                    "".join([
                        text_to_wordlist(values[1], remove_stopwords=True,
                            remove_punctuation=True),",",
                        text_to_wordlist(values[2], remove_stopwords=True,
                            remove_punctuation=True), "\n"
                        ])
                    )

print('Found ', i, ' texts in test.csv')
