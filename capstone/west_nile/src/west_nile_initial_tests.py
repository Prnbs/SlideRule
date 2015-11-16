__author__ = 'psinha4'

import pandas as pd
from sklearn import cross_validation
from time import time
import numpy as np

traps = pd.read_csv('../input/train_weather_spray.csv', verbose=True)#, parse_dates=['Date'])
# traps = pd.read_csv('../input/train_weather_spray.csv')
# extract label
labels = traps['WnvPresent']
# drop label from features
traps.drop('WnvPresent', 1, inplace=True)
# split by 60 40
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(traps, labels, test_size=0.4,random_state=0)

# features_train.to_csv('../input/feat_train.csv')

# DECISION TREE STARTS
# from sklearn import tree
#
# dtree = tree.DecisionTreeClassifier(criterion='gini')
# dtree.fit(features_train, labels_train)
# pred = dtree.predict(features_test)
#
# print features_train.head(2)
#
# from sklearn.metrics import accuracy_score
# print accuracy_score(labels_test, pred)
#
#
# print dtree.feature_importances_
# DECISION TREE ENDS

# SVM Starts
from sklearn.svm import SVC

clf = SVC(kernel='linear', C=10000.0)
t0 = time()
clf.fit(features_train, labels_train)
print "time to train:", round(time()-t0), "s"
t1 = time()
pred = clf.predict(features_test)
print "time to predict:", round(time()-t1), "s"
from sklearn.metrics import accuracy_score
accuracy_svc = accuracy_score(labels_test, pred)
print accuracy_svc

print clf.coef_
# SVM Ends

#Random forest classifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, verbose=1)
t0 = time()
clf.fit(features_train, labels_train)
print "time to train:", round(time()-t0), "s"
t1 = time()
pred = clf.predict(features_test)
print "time to predict:", round(time()-t1), "s"
from sklearn.metrics import accuracy_score
accuracy_rfc = accuracy_score(labels_test, pred)
print accuracy_rfc

print clf.feature_importances_

print np.array(accuracy_svc) - np.array(accuracy_rfc)