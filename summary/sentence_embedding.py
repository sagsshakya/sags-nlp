""" 
Name: sags-nlp
File: sentence_embedding.py
-----------------------------
Author: Sagun Shakya
Data:   26/5/2022, 2:20:29 pm
Description:
    Python class to compute the 768 dimensional sentence embedding for the given list of sentences using Sentence Transformer.
    Reference: https://www.sbert.net/examples/applications/computing-embeddings/README.html
"""

# Necessary imports.
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from warnings import filterwarnings
filterwarnings(action = 'ignore')

class SentenceEmbedding:
    def __init__(self, 
                model_name: str = 'google/muril-base-cased', 
                max_seq_len: int = 100, 
                device: str = "cpu"):
        
        self.model_name = model_name
        self.max_seq_len = max_seq_len
        self.device = device

        # Initialize the model.     
        self.model = SentenceTransformer(model_name_or_path = self.model_name, device = self.device)
        self.model.max_seq_length = 100

    def __call__(self, sentences: List[str]) -> np.array:
        """
        Gets the 768 dimensional sentence embedding for the given list of sentences using Sentence Transformer.
        Reference: https://www.sbert.net/examples/applications/computing-embeddings/README.html

        Args:
            sentences (List[str]): List of sentences to encode.

        Returns:
            np.array: Numpy array of encoded sentences. Shape --> (number of sentences, 768)
        """        
        assert len(sentences) > 0, "No sentences were found."
        embeddings = self.model.encode(sentences, show_progress_bar = True, convert_to_numpy = True)    # Shape --> (num_sentences, 768 : embedding dimension)
        return embeddings