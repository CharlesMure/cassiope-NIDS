#! usr/bin/python3

# -*- coding: utf8 -*-

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

import numpy

import scipy
import scipy.io.arff

from sklearn.preprocessing import LabelBinarizer

def arff_to_ndarray(path_to_arff_file):
    """
    Converts content of .arff file to numpy matrix.
    :param path_to_arff_file:
    :return: numpy.ndarray matrix for feature values, vector with labels/classes.
    """

    # Load as numpy objects.
    data, meta = scipy.io.arff.loadarff(path_to_arff_file)

    # Extract labels.
    labels = data[meta.names()[-1]]

    # Discard last column (labels).
    data = data[meta.names()[:-1]]

    # Use view(numpy.float) to convert elements from numpy.void to numpy.float. Use -1 to let numpy infer the shape.
    data = data.view().reshape(data.shape)

    return data, labels

def generate_model(shape):
    # define the model
    model = keras.models.Sequential()

    model.add(Dense(82, input_dim=shape, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(41, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(20, activation='softmax'))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))
    model.add(Dropout(0.4))
    model.add(Dense(2, activation='softmax'))
    print(model.summary())

    return model

encoder = LabelBinarizer()

x_train, train_label = arff_to_ndarray("KDDTrain+_20Percent.arff")
x_train = numpy.array(x_train)
y_train = encoder.fit_transform(train_label)
y_train = numpy.array(y_train)

x_test, test_label = arff_to_ndarray("KDDTest+.arff")
x_test = numpy.array(x_test)
y_test = encoder.fit_transform(test_label)
y_test = numpy.array(y_test)

model = generate_model(41)

model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))

model.fit(x_train, y_train, epochs=10)
