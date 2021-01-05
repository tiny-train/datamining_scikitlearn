#!/bin/python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


#-----------------------------------DATA PREPARATION-----------------------------------
training_set = pd.read_csv('credit_trainset', header=None)
test_set = pd.read_csv('credit_testset', header=None)

#encoding the training set
categorical_feature_mask = training_set.dtypes==object
categorical_cols = training_set.columns[categorical_feature_mask].tolist()

le = LabelEncoder()
training_set[categorical_cols] = training_set[categorical_cols].apply(lambda col: le.fit_transform(col))


#encoding the test set
categorical_feature_mask = test_set.dtypes==object
categorical_cols = test_set.columns[categorical_feature_mask].tolist()

le = LabelEncoder()
test_set[categorical_cols] = test_set[categorical_cols].apply(lambda col: le.fit_transform(col))


l = (len(training_set.columns) - 1)

x = training_set.drop([l], axis=1)

y = training_set[l]


l = (len(training_set.columns) - 1)

x_test = test_set.drop([l], axis=1)

print(x)

y_test = test_set[l]



#-----------------------------------MODEL GENERATION AND PREDICTION-----------------------------------
# Decision tree 
dt = DecisionTreeClassifier() 
# Performing training 
dt.fit(x, y) 

#prediciton of test set class attribute
pred = dt.predict(x_test) 


#-----------------------------------COMPARISON AND OUTPUT-----------------------------------
#ytestar = y_test.to_numpy() 

true_values = []
y_test_rows = y_test.shape[0]

for i in range(0, (y_test_rows)):
	if np.array_equal(y_test[i], pred[i]):
		true_values.append(1)
	else:
		true_values.append(0)


#print("For this set, 1 is <=50K, and 0 is >50K")
#for i in range(0, y_test_rows):
#	print("ID = " + str(i) + " predicted = " + str(pred[i]) + " true = " + str(y_test[i]) + " accuracy = " + str(true_values[i])) 


print(classification_report(y_test, pred))