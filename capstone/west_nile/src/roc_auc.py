__author__ = 'psinha4'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split

traps = pd.read_csv('../input/train_weather_spray.csv', verbose=True)#, parse_dates=['Date'])

# extract label
labels = traps['WnvPresent']
# drop label from features
traps.drop('WnvPresent', 1, inplace=True)


# shuffle and split training and test sets
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(traps, labels, test_size=0.4,random_state=0)

# Learn to predict each class against the other
from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=100)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

fpr, tpr, _ = roc_curve(labels_test, pred)
roc_auc = auc(fpr, tpr)

from sklearn.metrics import accuracy_score
print accuracy_score(labels_test, pred)
# Compute micro-average ROC curve and ROC area



##############################################################################
# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


##############################################################################