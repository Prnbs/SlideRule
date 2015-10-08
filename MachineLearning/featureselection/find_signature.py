#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)


### the words (features) and authors (labels), already largely processed
### these files should have been created from the previous (Lesson 10) mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )



### test_size is the percentage of events assigned to the test set (remainder go into training)
### feature matrices changed to dense representations for compatibility with classifier
### functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english')

features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()

### a classic way to overfit is to use a small number
### of data points and a large number of features
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]


### your code goes here
from sklearn import tree
dtree = tree.DecisionTreeClassifier()
dtree.fit(features_train, labels_train)
pred = dtree.predict(features_test)

from sklearn.metrics import accuracy_score
print accuracy_score(labels_test, pred)

most_important_feature = max(dtree.feature_importances_)
most_imp_feat_index = [i for i , j in enumerate(dtree.feature_importances_) if j == most_important_feature]

#after removing the top 2 most powerful words
most_important_feature_list = [elem for elem in dtree.feature_importances_ if elem > 0.2]
print "list of importants ",most_important_feature_list

print  most_important_feature, most_imp_feat_index

idf = vectorizer.idf_

names_idf =  zip(vectorizer.get_feature_names(), idf)
powerful_word = vectorizer.get_feature_names()
print powerful_word[most_imp_feat_index[0]]


print most_important_feature


