__author__ = 'prnbs'

import tensorflow as tf
import pandas as pd
import numpy as np


def actual_file_name(file_name, day):
    index = len(file_name) - 4
    actual_name = file_name[:index] + str(day) + ".csv"
    return actual_name


def split_train(in_file, days):
    in_file = actual_file_name(in_file, days)
    traps = pd.read_csv(in_file)
    # extract label
    labels = traps['WnvPresent']
    # drop label from features
    traps.drop('WnvPresent', 1, inplace=True)
    labels_one_hot = create_numpy_nd_array(labels)
    print labels_one_hot.shape
    print traps.shape
    return traps, labels_one_hot


def create_numpy_nd_array(list_of_values):
    output = np.full((len(list_of_values),2), 0)
    for i, item in enumerate(list_of_values):
        output[i, item] = 1
    # print output
    return output


def train_nn(train_step, in_file, day, sess):
    in_file = actual_file_name(in_file, day)
    traps = pd.read_csv(in_file)
    # extract label
    labels = traps['WnvPresent']
    # drop label from features
    traps.drop('WnvPresent', 1, inplace=True)
    from sklearn.cross_validation import StratifiedKFold
    kf = StratifiedKFold(labels, n_folds=10, shuffle=True, random_state =0)
    for train_indices, test_indices in kf:
        features_train = traps.iloc[train_indices]
        # features_test  = traps.iloc[test_indices]
        labels_train   = [labels[ii] for ii in train_indices]
        # labels_test    = [labels[ii] for ii in test_indices]

        labels_one_hot = create_numpy_nd_array(labels_train)
        sess.run(train_step, feed_dict={x:features_train, y_:labels_one_hot})
    return train_step


if __name__ == '__main__':
    sess = tf.Session()

    # create x of dimension Any x 28
    x = tf.placeholder("float", shape=[None, 27])
    # create y of dimension Any x 2
    y_ = tf.placeholder("float", shape=[None, 2])

    # w is 28 x 2 since there are 28 inputs and 2 outputs
    w = tf.Variable(tf.zeros([27, 2]))
    # b is of dimension 2 since there are 2 outputs
    b = tf.Variable(tf.zeros([2]))

    # initialize al the variables
    sess.run(tf.initialize_all_variables())

    # implement regression model
    y = tf.nn.softmax(tf.matmul(x, w) + b)

    # set cost function as cross entropy
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))

    # train the model using gradient descent
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    in_trap_clustered_file =  '../output/train_weather_spray_clustered_traps.csv'
    train_step = train_nn(train_step, in_trap_clustered_file, 14, sess)

    correct_prediction = tf.equal(tf.arg_max(y,1), tf.arg_max(y_,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    train, labels = split_train(in_trap_clustered_file, 14)

    print(sess.run(accuracy, feed_dict={x: train, y_:labels}))

