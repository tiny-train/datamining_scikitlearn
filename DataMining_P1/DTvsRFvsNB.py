#!/bin/python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import time


def DTvsRFvsNB(training_file, test_file):
	print("The following is the prediction results for decision tree classification: ")
	decision_tree(training_file, test_file)
	print("\n\nThe following is the prediction results for random forest classification: ")
	random_forest(training_file, test_file)
	print("\n\nThe following is the prediction results for gaussian naive bayes classification: ")
	naive_bayes(training_file, test_file)




def decision_tree(training_file, test_file):

	start = time.time()
	#-----------------------------------DATA PREPARATION-----------------------------------
	training_set = pd.read_csv(training_file, header=None)
	test_set = pd.read_csv(test_file, header=None)


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


	
	for i in range(0, y_test_rows):
		print("ID = " + str(i) + " predicted = " + str(pred[i]) + " true = " + str(y_test[i]) + " accuracy = " + str(true_values[i])) 

	print("\nClassification report:\n" + classification_report(y_test, pred))

	print ("Runtime: ")
	print(time.time() - start)




def random_forest(training_file, test_file):

	start = time.time()
	#-----------------------------------DATA PREPARATION-----------------------------------
	training_set = pd.read_csv(training_file, header=None)
	test_set = pd.read_csv(test_file, header=None)

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

	y_test = test_set[l]



	#-----------------------------------MODEL GENERATION AND PREDICTION-----------------------------------
	# Decision tree 
	rf = RandomForestClassifier(max_depth=2, random_state=0)
	# Performing training 
	rf.fit(x, y) 

	#prediciton of test set class attribute
	pred = rf.predict(x_test) 


	#-----------------------------------COMPARISON AND OUTPUT-----------------------------------
	#ytestar = y_test.to_numpy() 

	true_values = []
	y_test_rows = y_test.shape[0]

	for i in range(0, (y_test_rows)):
		if np.array_equal(y_test[i], pred[i]):
			true_values.append(1)
		else:
			true_values.append(0)


	
	for i in range(0, y_test_rows):
		print("ID = " + str(i) + " predicted = " + str(pred[i]) + " true = " + str(y_test[i]) + " accuracy = " + str(true_values[i])) 

	print("\nClassification report:\n" + classification_report(y_test, pred))

	print ("Runtime: ")
	print(time.time() - start)


def naive_bayes(training_file, test_file):

	start = time.time()
	#-----------------------------------DATA PREPARATION-----------------------------------
	training_set = pd.read_csv(training_file, header=None)
	test_set = pd.read_csv(test_file, header=None)


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

	y_test = test_set[l]



	#-----------------------------------MODEL GENERATION AND PREDICTION-----------------------------------
	# Decision tree 
	gb = GaussianNB()
	# Performing training 
	gb.fit(x, y);

	#prediciton of test set class attribute
	pred = gb.predict(x_test) 


	#-----------------------------------COMPARISON AND OUTPUT-----------------------------------
	#ytestar = y_test.to_numpy() 

	true_values = []
	y_test_rows = y_test.shape[0]

	for i in range(0, (y_test_rows)):
		if np.array_equal(y_test[i], pred[i]):
			true_values.append(1)
		else:
			true_values.append(0)


	for i in range(0, y_test_rows):
		print("ID = " + str(i) + " predicted = " + str(pred[i]) + " true = " + str(y_test[i]) + " accuracy = " + str(true_values[i])) 

	print("\nClassification report:\n" + classification_report(y_test, pred))

	print ("Runtime: ")
	print(time.time() - start)
	

def main():
	print("The following is the results using the Census dataset:\n")
	DTvsRFvsNB('census_trainset', 'census_testset')
	print("\n\n")
	print("The following is the results using the Credit dataset:\n")
	DTvsRFvsNB('credit_trainset', 'credit_testset')
	print("The following is the results using the missing data handled Credit dataset:\n")
	DTvsRFvsNB('Task6_credit_trainset', 'Task6_credit_testset')  

if __name__== "__main__":
  main()

