""" 
Name: sags-nlp
File: parser.py
-----------------------------
Author: Sagun Shakya
Data:   26/5/2022, 2:32:29 pm
Description:
    Python class to compute the 768 dimensional sentence embedding for the given list of sentences using Sentence Transformer.
    Reference: https://www.sbert.net/examples/applications/computing-embeddings/README.html
"""
# Necessary imports.
import re
from typing import List
import emoji

# Local Modules.
from utilities.utils import remove_unnecessary_characters

class FileParser:
    def __init__(self):
        self.filepath = r'D:\ML_projects\sags-nlp\test\sample\25022.txt'

    def read_file_from_text_file(self):
        '''
        Reads the text present in the .txt file and returns the output in string format.
        
        Parameters:
        filename -- Full path of the text file.
        
        Output:
        String format text.
        
        '''
        with open(file = self.filepath, encoding = 'utf8') as ff:
            text = ff.readlines()
            text = self._first_stage_cleaner(text)
            return text        

    def text_preprocessor(self, paragraph: str) -> List[str]:
        '''
        Text cleaning pipeline. Takes in helper functions from utilities/utils.py.

        Pipeline:
            - Reads Text file as string. Does some pre-processing (remove bad characters, etc.)
            - Separates the sentences of the paragraph using delimiters : '[।l?|]'.
            - Removes unnecessary characters like punctuation, eng + nep digits, etc.
        '''
        uncleaned = self._sent_tokenize(paragraph)
        cleaned = [remove_unnecessary_characters(SENTENCE).strip() for SENTENCE in uncleaned]
        return uncleaned, cleaned

    def _first_stage_cleaner(self, text: str) -> str:
        """
        Minor cleanup before dumping into pre-processing pipeline.

        Args:
            text (str): Chunk of raw text to be summarixed.

        Returns:
            str: First stage cleaned stringed text.
        """        

        text = [tt for tt in text if len(tt) > 5]                       # Removing sentences with less than 5 characters.

        text = ' '.join(text)
        text = text.replace('-', '')
        text = text.replace(u'\xa0', ' ').encode('utf-8').decode()      # Removing the \xa0 character.
        text = text.replace(u'\xad', ' ').encode('utf-8').decode()      # Removing the \xad character.
        text = text.replace(u'\u200c', ' ').encode('utf-8').decode()    # Removing the \u200c character.
        text = text.replace(u'\u200d', ' ').encode('utf-8').decode()    # Removing the \u200d character.
        text = text.replace('\n', '')                                   # Removing the \n character.

        # Remove hyperlinks, hashtags and mentions.
        pattern = r'(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)|https?:\/\/\S*'
        text = re.sub(pattern, "", text)
    
        # Remove emojis.
        text = emoji.replace_emoji(text, "")

        return text

    def _sent_tokenize(self, paragraph: str) -> List[str]:
        """
        Tokenizes a paragraph to list of sentences using delimiters: { ।,l,?,|}

        Args:
            paragraph (str): Collection of sentences. Since it is Deanagiri, it is usually delimited by '।'.

        Returns:
            List[str]: List of sentences.
        """        
        temp =  re.split('[।l?|]', paragraph)
        return [SENT.strip() for SENT in temp if len(SENT.strip()) > 0]



