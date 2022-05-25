""" 
Name: sags-nlp
File: pipepline.py
-----------------------------
Author: Sagun Shakya
Date:   23/5/2022, 4:32:47 pm
Description:
    Pipeline for generating an extractive summary of a piece of text article using unsupervised means.
"""
import numpy as np

from summary.kmeans import KMeansClustering

x1 = np.array([3, 1, 1, 2, 1, 6, 6, 6, 5, 6, 7, 8, 9, 8, 9, 9, 8])
x2 = np.array([5, 4, 5, 6, 5, 8, 6, 7, 6, 7, 1, 2, 1, 2, 3, 2, 3])

sentence_vectors = np.array(list(zip(x1, x2))).reshape(len(x1), 2)

kmeans = KMeansClustering(sentence_vectors, max_iter=300)

kmeans.perform_elbow(1,10, plot_elbow=True)