""" 
Name: sags-nlp
File: utils.py
-----------------------------
Author: Sagun Shakya
Data:   25/5/2022, 2:02:29 pm
Description:
    Helper Functions for the project.
"""

from os import makedirs
import os
from random import randint
import re
from string import ascii_lowercase, ascii_uppercase, punctuation

from matplotlib import pyplot as plt
import numpy as np


def write_id_to_file(filepath: str):
    """
    Writes a text at the end of a text file.
    If the file does not exist, it will create a new file and writes the text on the first line.

    Args:
        filepath (str): Full path to the TXT file.
    """    

    with open(filepath, 'r+', encoding='utf8') as ff:

        # Data already present.
        data = [x.rstrip() for x in ff.readlines()]
        
        # Keep on generating new id until a unique ID is found.
        flag = True
        while flag:
            text_id = str(randint(100000, 999999))
            if text_id not in data:
                data.append(text_id)
                flag = False
        
        # Write the list.
        ff.seek(0)
        ff.truncate()
        for item in data:
            ff.write(f'{item}\n')

        print(f'ID : {text_id} stored at {filepath}.\n')
        return text_id

def plot_scatter_cluster(sentence_vec, labels, closest_id, save_location = r'images'):
    '''sentence_vec --> 2d version of the original sentence_vectors.'''
    
    with plt.ioff():
        plot1 = plt.figure(1)
        plt.style.use('classic')

        unique_labels = np.unique(labels)
        colors = ['steelblue', 'green', 'gray']
        
        for clr, ll in zip(colors, unique_labels):
            sent_vec_selected = sentence_vec[labels == ll]
            x = sent_vec_selected[:, 0]
            y = sent_vec_selected[:, -1]

            x0 = np.mean(x)
            y0 = np.mean(y)

            plt.scatter(x, y, marker = 'D', s = 50, color = clr, alpha = 0.7)
            plt.scatter([x0], [y0], marker = 'X', s = 60, color = 'red')

        closest_points = sentence_vec[closest_id]
        plt.scatter(closest_points[:, 0], closest_points[:, -1], color = 'black', alpha = 0.7, marker = 'D', s = 50)
        plt.xlabel('Principle Component 1')
        plt.ylabel('Principle Component 2')
        makedirs(save_location, exist_ok = True)
        plt.savefig(os.path.join(save_location, 'scatter_plot.png'), dpi = 300)


def remove_unnecessary_characters(sentence):    
    temp = sentence.replace('[^a-zA-Z-]', '')  # removing puctuations.
    #temp = temp.replace('[a-zA-Z.]', '')    # removing english letters and digits.
    
    nepali_digits = ''.join([chr(2406 + ii) for ii in range(10)])
    english_digits = ''.join([chr(48 + ii) for ii in range(10)])
    #english_alphabets = ascii_uppercase + ascii_lowercase
    other_punctuations = '।‘’' + '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~' + chr(8211)
    
    to_remove = punctuation + nepali_digits + english_digits + other_punctuations + chr(8226)     
    temp = temp.translate(str.maketrans('', '', to_remove))

    return temp

def generate_word_tokens(sentence):
    return sentence.split()

