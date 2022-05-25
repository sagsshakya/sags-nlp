""" 
Name: sags-nlp
File: kmeans.py
-----------------------------
Author: Sagun Shakya
Date:   23/5/2022, 4:32:47 pm
Description:
    Python class for clustering the sentence vectors based on K-Means.
    Applies Elbow method to determine the optimal number of clusters as well.
"""

from os import makedirs, path
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.cluster.vq import vq
import numpy as np
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings(action = 'ignore')

class KMeansClustering:
    def __init__(self, sentence_vectors, max_iter: int = 300, save_dir: str = r'images') -> None:
        self.sentence_vectors = sentence_vectors            # Shape: (num_sentences, num_features)
        self.max_iter = max_iter
        self.num_sentences = sentence_vectors.shape[0]
        self.k_optimal = 5                                  # Find a better way of selecting this.
        
        makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

    def _clustering(self, n_clusters: int, return_closest: bool = False):
        '''
        Performs KMeans Clustering on the sentence vectors.

        Parameters:
        sentence_vectors -- Array containing the n - dimensional sentence embeddings.
        n_clusters -- Number of clusters (k).
        max_iter = Max. number of iterations to be performed.
        return_labels -- bool; whether to return the labels of the sentences as per the clustering model.

        Returns:
        labels of the sentences and the indices of those sentence vectors which are 
        the closest to their corresponding cluster centroid.
        '''
        kmeans = KMeans(n_clusters, max_iter = self.max_iter)
        kmeans.fit(self.sentence_vectors)

        if return_closest:
            # Return the indices of those sentence vectors which are the closest to their corresponding cluster centroid.
            centroids = kmeans.cluster_centers_             # N-dimensional array with your centroids.
            points = self.sentence_vectors                       # N-dimensional array with your data points.
            closest, distances = vq(centroids, points)
            return closest
        else:
            return kmeans

    def _plot_elbow(self, K: int, distortions: list):
        '''
        Plots the elbow curve.
        '''
        plt.style.use('classic')
        plt.plot(K, distortions, color = 'steelblue', marker = 'X')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method showing the optimal k')
        plt.savefig(path.join(self.save_dir, 'kmeans_elbow.png'), dpi = 300)

    def get_optimal_k(self, lower_bound: int = 1, upper_bound: int = 10, plot_elbow: bool = False):
        distortions = []
        K = range(lower_bound, upper_bound + 1)
        for kk in K:
            kmeansModel = self._clustering(n_clusters = kk, return_closest = False)
            distortion = sum(np.min(cdist(self.sentence_vectors, kmeansModel.cluster_centers_, 'euclidean'), axis=1)) / self.num_sentences
            distortions.append(distortion)

        if plot_elbow:
            assert len(K) == len(distortions)
            self._plot_elbow(K, distortions)

        return K, distortions

    def get_summary_sentence_ids(self):
        closest = self._clustering(n_clusters = self.k_optimal, return_closest = True)
        closest = sorted(closest)           # IDs sorted by scores.
        return closest