""" 
Name: sags-nlp
File: utils.py
-----------------------------
Author: Sagun Shakya
Data:   25/5/2022, 2:02:29 pm
Description:
    Helper Functions for the project.
"""

from random import randint


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