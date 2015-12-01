__author__ = 'prnbs'

from sklearn.cross_validation import KFold
from sklearn import tree
import pandas as pd
import numpy as np

traps = pd.read_csv('../input/train_weather_spray.csv', verbose=True)#, parse_dates=['Date'])

# extract label
labels = traps['WnvPresent']
# drop label from features
traps.drop('WnvPresent', 1, inplace=True)

kf = KFold(len(traps), n_folds=10, shuffle=True)

dtree = tree.DecisionTreeClassifier(criterion='gini',min_samples_split=100)

accuracy = []

for train_indices, test_indices in kf:

    features_train = traps.iloc[train_indices]
    features_test  = traps.iloc[test_indices]
    labels_train   = [labels[ii] for ii in train_indices]
    labels_test    = [labels[ii] for ii in test_indices]

    dtree.fit(features_train, labels_train)
    pred = dtree.predict(features_test)

    # print features_train.head(2)

    from sklearn.metrics import accuracy_score
    print accuracy_score(labels_test, pred)
    accuracy.append(accuracy_score(labels_test, pred))


    # print dtree.feature_importances_

accuracy_np = np.array(accuracy)
print "Avg = ",accuracy_np.mean()
