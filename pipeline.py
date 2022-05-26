""" 
Name: sags-nlp
File: pipepline.py
-----------------------------
Author: Sagun Shakya
Date:   23/5/2022, 4:32:47 pm
Description:
    Pipeline for generating an extractive summary of a piece of text article using unsupervised means.
"""
from matplotlib import pyplot as plt
import numpy as np

from summary.kmeans import KMeansClustering
from utilities.utils import plot_scatter_cluster
from summary.parser import FileParser

def cluster_sample():
    x1 = np.array([3, 1, 1, 2, 1, 6, 6, 6, 5, 6, 7, 8, 9, 8, 9, 9, 8, 2, 5.5, 5.2, 5.9, 2.3, 4])
    x2 = np.array([5, 4, 5, 6, 5, 8, 6, 7, 6, 7, 1, 2, 1, 2, 3, 2, 3, 5.5, 7.5, 7.2, 7.9, 5.9, 2])

    sentence_vectors = np.array(list(zip(x1, x2))).reshape(len(x1), 2)

    kmeans = KMeansClustering(sentence_vectors, max_iter=300)

    K, distortions = kmeans.perform_elbow(1,10, plot_elbow=True)

    k_optimal, distortion_optimal = kmeans.get_optimal_k(K, distortions)

    def get_summary_sentence_ids(k_optimal: int):
        closest, kmeansModel = kmeans._clustering(n_clusters = k_optimal, return_closest = True)
        closest = sorted(closest)           # IDs sorted by scores.
        return closest

    closest = get_summary_sentence_ids(k_optimal)

    print(closest)

    kmeansModel = kmeans._clustering(3, return_closest = False)
    labels = kmeansModel.labels_

    plot_scatter_cluster(sentence_vectors, labels, save_location = kmeans.save_dir, closest_id = closest)

def read_clean_file():
    parser = FileParser()
    parser.filepath = r'D:\ML_projects\sags-nlp\test\sample\25022.txt'

    text_init = parser.read_file_from_text_file()
    text_uncleaned, text_cleaned = parser.text_preprocessor(text_init)
    print("Parsed Text: \n", text_uncleaned)
    print()
    print("Cleaned Text:\n", text_cleaned)

if __name__ == "__main__":
    read_clean_file()