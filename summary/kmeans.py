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
from typing import List, Tuple
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
        
        makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

    def _clustering(self, n_clusters: int, return_closest: bool = False):
        '''
        Performs KMeans Clustering on the sentence vectors.

        Parameters:
        n_clusters -- Number of clusters (k).
        return_closest -- Whether to return the indices of the samples which are closest to their cluster centroids.

        Returns:
        labels of the sentences and the indices of those sentence vectors which are 
        the closest to their corresponding cluster centroid.
        '''
        kmeans = KMeans(n_clusters, max_iter = self.max_iter)
        kmeans.fit(self.sentence_vectors)

        if return_closest:
            # Return the indices of those sentence vectors which are the closest to their corresponding cluster centroid.
            centroids = kmeans.cluster_centers_             # N-dimensional array with your centroids.
            points = self.sentence_vectors                  # N-dimensional array with your data points.
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
        plt.plot((K[0], K[-1]), (distortions[0], distortions[-1]), color = 'tomato', alpha = 0.6, linewidth = 1.2, linestyle = 'dotted')
        
        # Plot the optimal value of K.
        k_optimal, distortion_optimal = self.get_optimal_k(K, distortions)
        plt.scatter([k_optimal], [distortion_optimal], marker = 'D', color = 'red', s = 50)
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Intra-cluster Variability')
        plt.xticks(K)
        plt.savefig(path.join(self.save_dir, 'kmeans_elbow.png'), dpi = 300)

    def _dist_point_line(self, point: Tuple[float], coefficients: Tuple[float]) -> float:
        """
        Calculates the perpendicular distance between a 2D point and a line. 

        Args:
            point (Tuple[float]): co-ordinates of the point. e.g (1.2, 6.5)
            coefficients (Tuple[float]): coefficients of the straight line equation. e.g 3x + 4y + 5 = 0 --> (3,4,5)

        Returns:
            float: Perpendicular distance between a 2D point and a line.
        """        
        x,y = point
        a,b,c = coefficients
        numerator = (a * x) + (b * y) + c
        denominator = np.sqrt(a**2 + b**2)
        return abs(numerator / denominator)

    def perform_elbow(self, lower_bound: int = 1, upper_bound: int = 10, plot_elbow: bool = False) -> Tuple[List[int], List[float]]:
        """
        Performs Elbow method and plots the Elbow Curve for a given range of experimental values of cluster numbers.

        Args:
            lower_bound (int, optional): Lower bound of Cluster number. Defaults to 1.
            upper_bound (int, optional): Upper bound of Cluster number. Defaults to 10.
            plot_elbow (bool, optional): Whether to create/save the elbow plot. Defaults to False. If True, it saves the plot in the provided "saved_dir".

        Returns:
            Tuple[List[int], List[float]]: (experimental values of k, intra cluster variability for each k)
        """        
        distortions = []
        K = range(lower_bound, upper_bound + 1)
        for kk in K:
            kmeansModel = self._clustering(n_clusters = kk, return_closest = False)
            #distortion = sum(np.min(cdist(self.sentence_vectors, kmeansModel.cluster_centers_, 'euclidean'), axis=1)) / self.num_sentences
            distortions.append(kmeansModel.inertia_)

        if plot_elbow:
            assert len(K) == len(distortions)
            self._plot_elbow(K, distortions)

        return K, distortions

    def get_optimal_k(self, K: List[int], distortions: List[float]) -> tuple:
        """
        Calculates the optimal value of number of clusters i.e k 
        by maximizing the perpendicular distance between the points in the Elbow Curve and the straight line joining 
        the extreme ends of the Elbow Curve (denoted by red dotted line in the plot).

        The perpendicular distance between a point (x0, y0) and a straight line of the form ax + by + c = 0 is given by:
            r = abs(a * x0 + b * y0 + c) / sqrt(a**2 + b**2)

        Args:
            K (List[int]): List of experimental cluster numbers.
            distortions (List[float]): Intra cluster variability for each cluster number.

        Returns:
            tuple: (optimal value of k, distortion corresponding to this k)
        """       
        # Obtain coefficients of the straight line. 
        a = distortions[0] - distortions[-1]
        b = K[-1] - K[0]
        c1 = K[0] * distortions[-1]
        c2 = K[-1] * distortions[0]
        c = c1  -c2

        # Calculate the perpn distances of all points of elbow curve with respect to the straight line ax + by + c = 0.
        distances = []
        for kk, dist in zip(K, distortions):
            point = (kk, dist)
            coeff = (a, b, c)
            distances.append(self._dist_point_line(point, coeff))

        # Get the index of the maximum value of the perpn distance.
        index_max = max(range(len(distances)), key=distances.__getitem__)

        k_optimal = K[index_max]
        distortion_optimal = distortions[index_max]

        return k_optimal, distortion_optimal