__author__ = 'psinha4'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split

traps = pd.read_csv('../input/train_weather_spray.csv', verbose=True)#, parse_dates=['Date'])

# extract label
labels = traps['WnvPresent']
# drop label from features
traps.drop('WnvPresent', 1, inplace=True)

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# clf = RandomForestClassifier(n_estimators=100, verbose=1)
clf = GaussianNB()
# clf = SVC(kernel='linear', C=1000.0)
# clf = tree.DecisionTreeClassifier(criterion='gini',min_samples_split=100)
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

# Compute micro-average ROC curve and ROC area

kf = KFold(len(traps), n_folds=10, shuffle=True)
plt.figure()
for train_indices, test_indices in kf:

    features_train = traps.iloc[train_indices]
    features_test  = traps.iloc[test_indices]
    labels_train   = [labels[ii] for ii in train_indices]
    labels_test    = [labels[ii] for ii in test_indices]

    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)

    fpr, tpr, _ = roc_curve(labels_test, pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)

    # print features_train.head(2)

    from sklearn.metrics import accuracy_score
    print accuracy_score(labels_test, pred)
    print accuracy_score(labels_test, pred)



##############################################################################
# Plot of a ROC curve for a specific class
from matplotlib.font_manager import FontProperties

fontP = FontProperties()
fontP.set_size('small')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic GaussianNB')
plt.legend(loc="lower right", prop = fontP)
# plt.show()
plt.savefig('../input/GaussianNB.jpg')


##############################################################################