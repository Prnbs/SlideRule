__author__ = 'psinha4'

import seaborn as sns
import pandas as pd
from sklearn import cross_validation

# traps = pd.read_csv('../input/train_weather_clean.csv', parse_dates=['Date'])
traps = pd.read_csv('../input/train_weather_clean.csv')

# extract label
labels = traps['WnvPresent']
# drop label from features
traps.drop('WnvPresent', 1, inplace=True)
# split by 60 40
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(traps, labels, test_size=0.4,random_state=0)

print features_train.head()

from sklearn import tree

dtree = tree.DecisionTreeClassifier()
dtree.fit(features_train, labels_train)
pred = dtree.predict(features_test)

from sklearn.metrics import accuracy_score
print accuracy_score(labels_test, pred)

# most_important_feature = max(dtree.feature_importances_)
# most_imp_feat_index = [i for i , j in enumerate(dtree.feature_importances_) if j == most_important_feature]


print dtree.feature_importances_

# from sklearn import linear_model
# regression = linear_model.Lasso()
# regression.fit(features_train, labels_train)
# print regression.coef_