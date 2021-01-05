#!/bin/python3

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import sklearn.datasets as skd
import numpy as np
import pandas as pd
import glob
import time

news_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers'), shuffle=True, random_state=42)
news_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers'), shuffle=True, random_state=42)

train_path = '20_newsgroups_Train'
test_path = '20_newsgroups_Test'

start = time.time()

#news_train = skd.load_files(train_path, encoding= 'ISO-8859-1')
#news_test = skd.load_files(test_path, encoding= 'ISO-8859-1')

print("Loaded 20newsgroups training and testing sets.")
# filenames_train = glob.glob(train_path + '/*/*')
# dataframes_train = [pd.read_csv(f, header=None, error_bad_lines=False, engine='python', dtype=str) for f in filenames_train]
# train_set = pd.concat(dataframes_train)
# train_set.fillna('0')

# filenames_test = glob.glob(test_path + "/*/*")
# dataframes_test = [pd.read_csv(f, header=None, error_bad_lines=False, engine='python', dtype=str) for f in filenames_test]
# test_set = pd.concat(dataframes_test)
# train_set.fillna('0')


print("Testing and training sets are now being vectorized")
vectorizer = CountVectorizer(min_df = 2, max_df = 0.5, lowercase=False)

vectors_train = vectorizer.fit_transform(news_train.data)
vectors_test = vectorizer.transform(news_test.data)

print("Fitting to neural network classifier.")
mlp = MLPClassifier()

mlp.fit(vectors_train, news_train.target)


print("Predicting test set.")
pred = mlp.predict(vectors_test)


print("\nClassification report:\n" + classification_report(news_test.target, pred))

print ("Runtime: ")
print(time.time() - start)
