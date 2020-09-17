import spacy
from spacy.matcher import Matcher 
from spacy.tokens import Span
from spacy import displacy
import pandas as pd 
import json
import csv

# Open CSV 
def open_dr_words_csv(filepath):
    '''
    DESCRIPTION
    -------------
    Forms a list of the items in the 0th position of each row in the CSV file. 

    PARAMETERS
    -------------
    filepath: str  
        Location of file of CSV file 

    RETURNS
    -------------
    dr_words: list 
        The list contains the doctor's words only 
    '''

    if type(filepath) != str:
        raise TypeError('Filepath must be a string')

    dr_words = []
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            # Skip the first row in the list, which is a header 
            if row[0].split(',')[0] == 'WORD':
                pass
            else:
                dr_words.append(row[0].split(',')[0])
    return dr_words

def convert_list_to_str(original_list, seperator=' '):
    ''' 
        Converts a list to a single string by joining the items in the original list with the 
        separator value (default separator is ' ', i.e. a space).
        Returns the concatenated string. 
    '''
    return seperator.join(original_list)

# Intro patterns to search text for

intro_pattern1 = [  
                    {'LOWER': 'my'},
                    {'LOWER': 'name'},
                    {'LOWER': 'is'}
                ]

intro_pattern2 = [
                    {'LOWER': {'IN': ['hello', 'hi']}}
                ]

# 'I am' will capture 'I'm' etc as we apply nlp to the doc 
intro_pattern3 = [
                    {'LOWER': 'i'},
                    {'LOWER': 'am'}
                ]

intro_pattern4 = [
                    {'LOWER': 'how'},
                    {'LOWER': 'are'},
                    {'LOWER': 'you'}
                ]

intro_pattern5 = [
                    {'LOWER': 'we'},
                    {'LOWER': 'can'},
                    {'LOWER': 'support'},
                    {'LOWER': 'you'}
                ]

intro_pattern6 = [
                    {'LOWER': 'we'},
                    {'LOWER': 'can'},
                    {'LOWER': 'help'},
                    {'LOWER': 'you'}
                ]


intro_pattern7 = [
                    {'LOWER': 'I'},
                    {'LOWER': 'want'},
                    {'LOWER': 'to'},
                    {'LOWER': 'talk'}
                ]

intro_pattern7 = [
                    {'LOWER': "I'm"},
                    {'LOWER': 'just'},
                    {'LOWER': 'wondering'},
                    {'LOWER': 'if'},
                    {'LOWER': 'your'},
                    {'LOWER': 'okay'},
                    {'LOWER': 'to'},
                    {'LOWER': 'discuss'}
                ]

# Matcher - match expressions related to introduction i.e. 'setting'
nlp = spacy.load('en_core_web_sm')

# Instantiate matcher object 
matcher = Matcher(nlp.vocab, validate=True)
matcher.add('INTRO_PATTERN', None, intro_pattern1, intro_pattern2, intro_pattern3, intro_pattern4, intro_pattern5)

test = open_dr_words_csv('../dcapp/75d88_dr_speech.csv')
fullstr = convert_list_to_str(test)
# print(fullstr)
doc = nlp(fullstr)
# print(doc)

for match_id, start, end in matcher(doc):
    print(doc[start-2: end+2])
