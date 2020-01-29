import numpy as np
import multiprocessing as mp
import pandas as pd

import string
import spacy
from nltk.tokenize import word_tokenize
from normalise import normalise

nlp = spacy.load('en_core_web_sm')
    
def my_tokenizer(text, variety = "BrE", user_abbrevs={}):
    normalized_text = _normalise(text, variety, user_abbrevs)
    doc = nlp(normalized_text)
    removed_punct = _remove_punct(doc)
    remove_stop_words = _remove_stop_words(removed_punct)
    remove_by_pos = _remove_by_pos(remove_stop_words)
    return _lemmatize(remove_by_pos)

def _normalise(text, variety, user_abbrevs):
        # some issues in normalise package
    try:
        return ' '.join(normalise(text, variety=variety, user_abbrevs= user_abbrevs, verbose=False, lowercase = True))
    except:
        return text

def _remove_punct(doc):
    return [t for t in doc if t.text not in string.punctuation]

def _remove_stop_words(doc):
    return [t for t in doc if not t.is_stop]

def _lemmatize(doc):
    return ([t.lemma_ for t in doc])
    
def _remove_by_pos(doc):
    return([t for t in doc if (t.pos_ == 'PROPN' or t.pos_ == 'NOUN')])