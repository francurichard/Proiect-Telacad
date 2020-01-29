import numpy as np
from tokenizer import my_tokenizer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class TopicAnalyzer():

    def train(self, X, *_):
        self.data = X
        self.tf_idf_vectorizer = TfidfVectorizer(tokenizer = my_tokenizer)
        self.tf_idf_vectorizer.fit(self.data)
    
    def predict(self, text):
        response = self.tf_idf_vectorizer.transform([text])
        feature_array = np.array(self.tf_idf_vectorizer.get_feature_names())
        tfidf_sorting = np.argsort(response.toarray()).flatten()[::-1]
        n = 3
        top_n_feature = feature_array[tfidf_sorting][:n]
        return(top_n_feature)