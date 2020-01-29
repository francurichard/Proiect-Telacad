import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy 

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import LatentDirichletAllocation

from tokenizer import my_tokenizer, _normalise

from topic_analyzer import TopicAnalyzer

df = pd.read_csv('task2.csv', delimiter=',', nrows = 1000)

bodys = df['body'].values

topic_analyzer = TopicAnalyzer()
topic_analyzer.train(bodys)

print(topic_analyzer.predict(text=bodys[0]))

