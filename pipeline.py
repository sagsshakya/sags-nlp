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

from utilities.utils import plot_scatter_cluster, project_to_2d
from summary.kmeans import KMeansClustering
from summary.sentence_embedding import SentenceEmbedding
from summary.parser import FileParser

def read_clean_file():
    parser = FileParser()
    parser.filepath = r'D:\ML_projects\sags-nlp\test\sample\25022.txt'

    text_init = parser.read_file_from_text_file()
    text_uncleaned, text_cleaned = parser.text_preprocessor(text_init)

    print("Total number of sentences: ", len(text_uncleaned))
    
    return text_cleaned, text_uncleaned

def generate_summary(text_cleaned, text_uncleaned):
    num_sentences = len(text_uncleaned)
    assert len(text_cleaned) == num_sentences
    
    upper_bound = int(num_sentences / 3)

    embedder = SentenceEmbedding(model_name = 'google/muril-base-cased', max_seq_len = 300)
    sentence_vectors = embedder(text_cleaned)

    kmeans = KMeansClustering(sentence_vectors, max_iter=300)

    K, distortions = kmeans.perform_elbow(lower_bound = 3, upper_bound = upper_bound, plot_elbow=True)
    print(f'\nElbow Curve saved as /{kmeans.save_dir}/kmeans_elbow.png...\n')

    k_optimal, distortion_optimal = kmeans.get_optimal_k(K, distortions)

    print(f"\nOptimal Number of clusters : {k_optimal}")
    print(f"Intra-cluster variability : {distortion_optimal: 5.3f}\n")

    closest, kmeansModel = kmeans._clustering(n_clusters = k_optimal, return_closest = True)
    closest = sorted(closest)           # IDs sorted by scores.
    
    print(f"Number of sentences in the summary : {len(closest)}")
    result = [(ID, text_uncleaned[ID]) for ID in closest]

    print("\nGenerating Scatter Plot... \n")
    labels = kmeansModel.labels_
    sentence_vectors_projected = project_to_2d(sentence_vectors, output_dim = 2)
    plot_scatter_cluster(sentence_vectors_projected, labels, save_location = kmeans.save_dir, closest_id = closest)

    return result


if __name__ == "__main__":
    text_cleaned, text_uncleaned = read_clean_file()
    summary = generate_summary(text_cleaned, text_uncleaned) 
    print(f'\n{"-"*50}\nSummary\n')
    for ii in summary:
        print(ii)
        print()