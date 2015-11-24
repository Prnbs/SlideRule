__author__ = 'psinha4'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split

from matplotlib.font_manager import FontProperties

def plot_roc_auc(clf, plt, traps, name):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Compute micro-average ROC curve and ROC area

    kf = KFold(len(traps), n_folds=10, shuffle=True)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    for train_indices, test_indices in kf:

        features_train = traps.iloc[train_indices]
        features_test  = traps.iloc[test_indices]
        labels_train   = [labels[ii] for ii in train_indices]
        labels_test    = [labels[ii] for ii in test_indices]

        clf.fit(features_train, labels_train)
        pred = clf.predict(features_test)

        fpr, tpr, _ = roc_curve(labels_test, pred)

        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0

        roc_auc = auc(fpr, tpr)
        label = 'ROC for ' + name +' (area = %0.2f)'
        plt.plot(fpr, tpr, label=label % roc_auc)

        # print features_train.head(2)

        from sklearn.metrics import accuracy_score
        print accuracy_score(labels_test, pred)
        print accuracy_score(labels_test, pred)
        return plt

    mean_tpr /= len(kf)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

if __name__ == '__main__':
    traps = pd.read_csv('../input/train_weather_spray.csv', verbose=True)#, parse_dates=['Date'])

# extract label
    labels = traps['WnvPresent']
    # drop label from features
    traps.drop('WnvPresent', 1, inplace=True)

    from sklearn import tree
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from scipy import interp
    clf_rf = RandomForestClassifier(n_estimators=100, verbose=1)
    clf_nb = GaussianNB()
    clf_svc = SVC(kernel='linear', C=1000.0)
    clf_dt = tree.DecisionTreeClassifier(criterion='gini',min_samples_split=100)
    # Compute ROC curve and ROC area for each class

    plt = plot_roc_auc(clf_dt, plt, traps, 'D Tree')
    plt = plot_roc_auc(clf_nb, plt, traps, 'G NB')
    plt = plot_roc_auc(clf_rf, plt, traps, 'R Forest')
    # plot_roc_auc(clf_svc, plt, traps)

    fontP = FontProperties()
    fontP.set_size('small')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right", prop = fontP)
    # plt.show()
    plt.savefig('../plots/No_precip.jpg')
