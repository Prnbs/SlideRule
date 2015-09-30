#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.svm import SVC
clf = SVC(kernel='rbf', C=10000.0)

#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

t0 = time()
clf.fit(features_train, labels_train)
print "time to train:", round(time()-t0), "s"
t1 = time()
pred = clf.predict(features_test)
print "time to predict:", round(time()-t1), "s"
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, pred)
print accuracy
#10th element
print pred[10]
#26th element
print pred[26]
#50
print pred[50]
print sum(pred)
#########################################################


