#!/bin/python3

import pandas as pd
import numpy as np
import time
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score 


def kmeans_dm(input_data, cluster_no):
	start = time.time()

	dataset = pd.read_csv(input_data, sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
	tf_idf_vectorizer = TfidfVectorizer(stop_words = 'english',  max_features = 20000)


	km = KMeans(n_clusters=cluster_no, random_state=0)
	km.fit(dataset)

	labels = km.labels_ 

	dbi = davies_bouldin_score(dataset, labels)
	si = silhouette_score(dataset, labels)

	print ("Runtime: ")
	print(time.time() - start)

	return dbi, si


#---------------------------------------------------------------------------------------


def kmedoids_dm(input_data, cluster_no):
	start = time.time()

	dataset = pd.read_csv(input_data, sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
	tf_idf_vectorizer = TfidfVectorizer(stop_words = 'english',  max_features = 20000)


	kmed = KMedoids(n_clusters=cluster_no, random_state=0)
	
	kmed.fit(dataset)

	labels = kmed.labels_ 

	dbi = davies_bouldin_score(dataset, labels)
	si = silhouette_score(dataset, labels)

	print ("Runtime: ")
	print(time.time() - start)

	return dbi, si



#---------------------------------------------------------------------------------------
#uses agglomerative clustering

def hierarchicalclustering(input_data, cluster_no):
	start = time.time()

	dataset = pd.read_csv(input_data, sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
	tf_idf_vectorizer = TfidfVectorizer(stop_words = 'english',  max_features = 20000)


	ac = AgglomerativeClustering(n_clusters=cluster_no)
	
	ac.fit(dataset)

	labels = ac.labels_ 

	dbi = davies_bouldin_score(dataset, labels)
	si = silhouette_score(dataset, labels)

	print ("Runtime: ")
	print(time.time() - start)

	return dbi, si



#---------------------------------------------------------------------------------------

def graph(input_data):
	cluster_no = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
	dbi_scores_km = []
	si_scores_km = []

	dbi_scores_kmed = []
	si_scores_kmed = []

	dbi_scores_ac = []
	si_scores_ac = []

	for i in cluster_no:
		dbi, si = kmeans_dm(input_data, i)
		dbi_scores_km.append(dbi)
		si_scores_km.append(si)

		dbi, si = kmedoids_dm(input_data, i)
		dbi_scores_kmed.append(dbi)
		si_scores_kmed.append(si)

		dbi, si = hierarchicalclustering(input_data, i)
		dbi_scores_ac.append(dbi)
		si_scores_ac.append(si)

	print("DBI and SI for k-means")
	print(dbi_scores_km)
	print(si_scores_km)
	print("\n")

	print("DBI and SI for k-medoids")
	print(dbi_scores_kmed)
	print(si_scores_kmed)
	print("\n")

	print("DBI and SI for hierarchicalclustering")
	print(dbi_scores_ac)
	print(si_scores_ac)
	print("\n")



	plt.figure(figsize = (9, 3))

	plt.subplot(211)
	plt.plot(cluster_no, dbi_scores_km, 'b', cluster_no, dbi_scores_kmed, 'r', cluster_no, dbi_scores_ac, 'g')
	plt.xlabel('Cluster Number')
	plt.ylabel('David Bouldin Index')
	plt.subplot(212)
	plt.plot(cluster_no, si_scores_km, 'b', cluster_no, si_scores_kmed, 'r', cluster_no, si_scores_ac, 'g')
	plt.xlabel('Cluster Number')
	plt.ylabel('Silhouette Index')
	plt.show()


#---------------------------------------------------------------------------------------

def run_clusters(input_data, cluster_no):
	kmeans_dm(input_data, cluster_no)
	kmedoids_dm(input_data, cluster_no)
	hierarchicalclustering(input_data, cluster_no)


#---------------------------------------------------------------------------------------



def main():
	graph('AllBooks_baseline_DTM_Unlabelled.csv')




if __name__== "__main__":
  main()
