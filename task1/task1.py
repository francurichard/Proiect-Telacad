import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

def get_data(filename):
    df = pd.read_csv(filename, delimiter=',')
    labels = df['label'].values
    tweets = df['tweet'].values

    shuffle_stratified = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2)

    for train_index, test_index in shuffle_stratified.split(tweets, labels):
        tweets_train, tweets_test = tweets[train_index], tweets[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]

    return tweets_train, labels_train, tweets_test, labels_test


X_train, y_train, X_test, y_test = get_data('tweets.csv')

# classifier with MultinomialNB
tweets_clf = Pipeline([
    ('vect', CountVectorizer(ngram_range = (1, 2))),
    ('tfidf', TfidfTransformer(use_idf = True)),
    ('clf', MultinomialNB(alpha = 0.01)),
])

# Parameter tuning using GridSearch

# parameteres = {
#     'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
#     'tfidf__use_idf': (True, False),
#     'clf__alpha': (0.1, 0.01, 0.001)
# }

# gs_classification=GridSearchCV(tweets_clf, parameteres, cv = 5)
# gs_classification.fit(X_train, y_train)

# predicted = gs_classification.predict(X_test)

#classifier with SGDClassifier(SVM)
# tweets_clf = Pipeline([
#     ('vect', CountVectorizer()),
#     ('tfidf', TfidfTransformer()),
#     ('clf', SGDClassifier(alpha = 0.01)),
# ])

# for param_name in sorted(parameteres.keys()):
#     print("%s: %r" % (param_name, gs_classification.best_params_[param_name]))

tweets_clf.fit(X_train, y_train)
predicted = tweets_clf.predict(X_test)

#use metrics to evaluate the model
print(metrics.classification_report(y_test, predicted, target_names= ['0', '1']))