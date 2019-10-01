"""****************************************************************************"""

"""SCRIPT FOR TEXT PREPROCESSING"""


"""Steps involved in preprocessing
1. Noise Removal.
i.   Removing HTML tags.
ii.  Replacing contractions.

2. Tokenization.

3. Normalisation.
i.   Removing non-ASCII characters.
ii.  Converting text corpus into lowercase.    
iii. Removing punctuations.
iv.  Replacing numbers with words.
v.   Removing STOP words.

4. Lemmatization."""

"""****************************************************************************"""

"""****************************************************************************"""

"""Importing Required Libraries"""

import re, string, unicodedata
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import contractions
import inflect
import pandas as pd
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer

"""****************************************************************************"""



"""****************************************************************************"""

"""Noise Removal"""

def remove_html_tags(text):
    """Remove html tags from a text"""
    #clean = re.compile('<.*?>')
    clean = re.compile('<[^<]+?>')
    return re.sub(clean, '', text)

def replace_contractions(text):
    """Replace contractions in text"""
    return contractions.fix(text)

"""Perform Tokenization"""

"""Perform Normalization"""

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    #words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    return words

"""Performing Lemmatization"""

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

"""Function to perform preprocessing"""

def text_preprocessing(text):
    text = replace_contractions(text)
    words = nltk.word_tokenize(text)
    words = normalize(words)
    processed_text = lemmatize_verbs(words)
    return processed_text

"""****************************************************************************"""    


"""****************************************************************************"""

"""Reading Data as Input"""

Data = pd.read_csv('C:/Users/priya/Desktop/GeneToVec/Gene2Vec (Private GitHub Repository)/Data collection/Dataset/Annotated/positive.csv')
Temp1 = Data['Abstract']

"""Removing HTML Tags"""

for i in range(len(Temp1)):
    Temp2 = []
    #Temp2 = remove_html_tags(Temp1[i])
    #text = text + Temp2
    Temp2.append(remove_html_tags(Temp1[i]))
    
"""Converting list into string for further preprocessing"""    
text = ""
text = text.join(Temp2)
corpus = text_preprocessing(text)