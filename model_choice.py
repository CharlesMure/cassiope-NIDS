#! usr/bin/python3

# -*- coding: utf8 -*-

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

import numpy as np

x_train = np.random.random((1000, 13)) #nb de dataset , iteration de ces datasets
y_train = keras.utils.to_categorical(np.random.randint(2, size=(1000, 1)), num_classes=2)
x_test = np.random.random((100, 13))
y_test = keras.utils.to_categorical(np.random.randint(2, size=(100, 1)), num_classes=2)

model = Sequential()

model.add(Dense(26, activation='softmax', input_dim=13))
model.add(Dropout(0.5))
model.add(Dense(13, activation='softmax'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

keras.utils.plot_model(model, show_shapes=True, rankdir="LR")

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100, batch_size=20)
score = model.evaluate(x_test, y_test, batch_size=20)

keras.utils.plot_model(model, show_shapes=True)
