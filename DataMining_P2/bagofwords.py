#!/bin/python3

import json
import _pickle as pickle
import pprint
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

twenty_news = fetch_20newsgroups(subset='all', shuffle=True, random_state=42, remove=('headers', 'footers', 'quotes'))


vectorizer_stop = CountVectorizer(twenty_news, min_df = 2, max_df = 0.5)

#twenty_news_vec = vectorizer_stop.fit(twenty_news.data)
twenty_news_trans = vectorizer_stop.fit_transform(twenty_news.data)
#vectorizer_stop.fit_transform(twenty_news.data)

#print(vectorizer_stop.stop_words_)

word_list = vectorizer_stop.get_feature_names()
count_list = twenty_news_trans.sum(axis=0)

stop_words = vectorizer_stop.stop_words_

word_bag = [(word, count_list[0, idx]) for word, idx in vectorizer_stop.vocabulary_.items()]

#stop_words = str(stop_words)

print(len(word_bag))


# print(word_bag)

# with open('BagofWords.txt', 'w') as file:
# 	for word in word_bag:
# 		print(word, file=file)
# 		print('\n')

# with open('eliminate.txt', 'w') as file:
# 	for word in stop_words:
# 		print(word, file=file)
# 		print('\n')