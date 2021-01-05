#!/bin/python3

import random
from math import ceil


#read lines into list and count number of lines 
fname = input("Enter file name: ")
count = 0.0

with open(fname, 'r') as f:
	file_data = []
	for line in f:
		count += 1
		file_data.append(line)
f.close()


training_length = int(count * 0.8)
test_length = int(count * 0.2)

#split = 0.8
#split_index = ceil(len(file_data) * split)

training_data = random.sample(file_data, training_length)
test_data = random.sample(file_data, test_length)

print(len(training_data))
print(len(test_data))


oname1 = input("Enter the name of the training data file you want to output: ")
with open(oname1, "w+") as f:
	for i in range(0, training_length):
		f.write(training_data[i])
f.close()

oname2 = input("Enter the name of the test data file you want to output: ")
with open(oname2, "w+") as f:
	for i in range(0, test_length):
		f.write(test_data[i])
f.close()

print("Test and training datasets generated!")