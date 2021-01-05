#!/bin/python3
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder

#read lines into list and count number of lines 
training_file = input("Enter training set file name: ")
test_file = input("Enter training set file name: ")

training_set = pd.read_csv(training_file, header=None)
test_set = pd.read_csv(test_file, header=None)

l = (len(training_set.columns) - 1)

#encoding the training set
train_types_int = training_set.dtypes==int
train_types_float = training_set.dtypes==float
train_cols_int = training_set.columns[train_types_int].tolist()
train_cols_float = training_set.columns[train_types_float].tolist()

test_types_int = test_set.dtypes==int
test_types_float = test_set.dtypes==float
test_cols_int = test_set.columns[test_types_int].tolist()
test_cols_float = test_set.columns[test_types_float].tolist()

training_cols = train_cols_int + train_cols_float
testing_cols = test_cols_int + test_cols_float



training_set[training_cols] = training_set[training_cols].replace(0, np.NaN)
training_set.dropna(inplace=True)

test_set[testing_cols] = test_set[testing_cols].replace(0, np.NaN)
test_set.dropna(inplace=True)


oname1 = input("Enter the name of the training data file you want to output: ")
training_set.to_csv(oname1, header=None, index=None, sep=',', mode='a')

oname2 = input("Enter the name of the training data file you want to output: ")
test_set.to_csv(oname2, header=None, index=None, sep=',', mode='a')

print("Missing data handled!")