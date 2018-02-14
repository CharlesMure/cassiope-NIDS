#! usr/bin/python3

# -*- coding: utf8 -*-

import keras
import csv
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

import numpy as np


from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def transform(x):
    return {
        'back': 'back',
        'buffer_overflow': 'buffer_overflow',
        'ftp_write': 'ftp_write',
        'guess_passwd': 'guess_passwd',
        'imap': 'imap',
        'ipsweep': 'ipsweep',
        'land': 'land',
        'loadmodule': 'loadmodule',
        'multihop': 'multihop',
        'neptune': 'neptune',
        'nmap': 'nmap',
        'perl': 'perl',
        'phf': 'phf',
        'pod': 'pod',
        'portsweep': 'portsweep',
        'rootkit': 'rootkit',
        'satan': 'satan',
        'smurf': 'smurf',
        'spy': 'spy',
        'teardrop': 'teardrop',
        'warezclient': 'warezclient',
        'warezmaster': 'warezmaster',
        'normal': 'normal',
    }.get(x, 'unknown') 

def generate_model(shape):
    # define the model
    model = keras.models.Sequential()

    model.add(Dense(82, input_dim=shape, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(82, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(41, activation='softmax'))
    model.add(Dropout(0.2))
    model.add(Dense(24, activation='softmax'))
    print(model.summary())

    return model

encoder = LabelBinarizer()
encoder2 = LabelEncoder()

filereader = csv.reader(open("Data/KDDTrain+.txt"), delimiter=",")
data = np.array(list(filereader))

x_train = data[:,np.arange(0,41)]
x_train[:,1] = encoder2.fit_transform(x_train[:,1])
x_train[:,2] = encoder2.fit_transform(x_train[:,2])
x_train[:,3] = encoder2.fit_transform(x_train[:,3])

train_label = data[:,41]
#y_train = encoder.fit_transform(train_label)
train_label = [transform(attacktype) for attacktype in train_label]

filereaderTest = csv.reader(open("Data/KDDTest+.txt"), delimiter=",")
dataTest = np.array(list(filereaderTest))



x_test = dataTest[:,np.arange(0,41)]
x_test[:,1] = encoder2.fit_transform(x_test[:,1])
x_test[:,2] = encoder2.fit_transform(x_test[:,2])
x_test[:,3] = encoder2.fit_transform(x_test[:,3])

test_label = dataTest[:,41]
test_label = [transform(attacktype) for attacktype in test_label]

label = np.concatenate((train_label,test_label))
#print(np.shape(label))

y = encoder.fit_transform(label)
#print(label)
trainlen = np.size(train_label)
testlen = np.size(test_label)
y_train = y[np.arange(0,trainlen)]
y_test = y[np.arange(trainlen,trainlen+testlen)]

## Generation d'un dataset de validation
seed = 7
np.random.seed(seed)
x_validation, x_test, y_validation, y_test = train_test_split(x_test, y_test, test_size=0.75, random_state=seed)

print(np.shape(y_train))
print(np.shape(y_test))
print(np.shape(x_train))
print(np.shape(x_test))
 
model = generate_model(41)

model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.SGD(lr=0.005, momentum=0.9, nesterov=True),metrics=['accuracy'])

model.fit(x_train, y_train,validation_data=(x_validation,y_validation), epochs=30, batch_size=15)

score = model.evaluate(x_test, y_test, verbose=0)

print("precision on test data:", score[1]*100, "%")
