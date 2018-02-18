#! usr/bin/python3

# -*- coding: utf8 -*-

import keras
import csv
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.layers import Dense, Dropout, Activation, Embedding, BatchNormalization
from keras.optimizers import SGD

import numpy as np


from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def transform(x):
    return {
        'back': 'dos',
        'buffer_overflow': 'u2r',
        'ftp_write': 'r2l',
        'guess_passwd': 'r2l',
        'imap': 'r2l',
        'ipsweep': 'probe',
        'land': 'dos',
        'loadmodule': 'u2r',
        'multihop': 'r2l',
        'neptune': 'dos',
        'nmap': 'probe',
        'perl': 'u2r',
        'phf': 'r2l',
        'pod': 'dos',
        'portsweep': 'probe',
        'rootkit': 'u2r',
        'satan': 'probe',
        'smurf': 'dos',
        'spy': 'r2l',
        'teardrop': 'dos',
        'warezclient': 'r2l',
        'warezmaster': 'r2l',
        'normal': 'normal',
    }.get(x, 'unknown') 

def generate_model(shape):
    # define the model
    model = keras.models.Sequential()

    model.add(Dense(82, input_dim=shape, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(164, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(82, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(41, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(6, activation='softmax'))
    print(model.summary())

    return model


def generate_cnn_model(shape):

    model = Sequential()

    return model


def preporcess(data, dataTest):
    scaler = MinMaxScaler()
    encoder = LabelBinarizer()
    encoder2 = LabelEncoder()

    x_train = data[:,np.arange(0,41)]
    x_train[:,1] = encoder2.fit_transform(x_train[:,1])
    x_train[:,2] = encoder2.fit_transform(x_train[:,2])
    x_train[:,3] = encoder2.fit_transform(x_train[:,3])

    scaler.fit(x_train)
    x_train = scaler.transform(x_train)

    train_label = data[:,41]
    #y_train = encoder.fit_transform(train_label)
    train_label = [transform(attacktype) for attacktype in train_label]

    x_test = dataTest[:,np.arange(0,41)]
    x_test[:,1] = encoder2.fit_transform(x_test[:,1])
    x_test[:,2] = encoder2.fit_transform(x_test[:,2])
    x_test[:,3] = encoder2.fit_transform(x_test[:,3])

    scaler.fit(x_test)
    x_test = scaler.transform(x_test)

    test_label = dataTest[:,41]
    test_label = [transform(attacktype) for attacktype in test_label]

    # Join dataset => meme encoding pour train et test
    label = np.concatenate((train_label,test_label))
    y = encoder.fit_transform(label)

    trainlen = np.size(train_label)
    testlen = np.size(test_label)
    y_train = y[np.arange(0,trainlen)]
    y_test = y[np.arange(trainlen,trainlen+testlen)]

    ## Generation d'un dataset de validation
    seed = 9
    np.random.seed(seed)
    x_validation, x_test, y_validation, y_test = train_test_split(x_test, y_test, test_size=0.70, random_state=seed)

    return x_train, y_train, x_validation, y_validation, x_test, y_test


def eval(model,x_test,y_test):
    score = model.evaluate(x_test, y_test, verbose=1)
    print("loss on test data:", score[0])
    print("accuracy on test data:", score[1]*100, "%")




def main():

    input_list = ["1. MLP", "2. CNN1D"]

    filereader = csv.reader(open("Data/KDDTrain+.txt"), delimiter=",")
    data = np.array(list(filereader))

    filereaderTest = csv.reader(open("Data/KDDTest+.txt"), delimiter=",")
    dataTest = np.array(list(filereaderTest))

    x_train, y_train, x_validation, y_validation, x_test, y_test = preporcess(data,dataTest)


    P1 = input("Choisissez le reseau: " + ' '.join(input_list) + "\n")

    if (P1 == '1'):
         ### MLP model ###
        model = generate_model(41)
        model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.SGD(lr=0.002,decay=1e-6, momentum=0.1, nesterov=True),metrics=['accuracy'])
    
    elif (P1 == '2'):
        ### CNN 1D model ##
        model = generate_cnn_model(41)

        # initiate RMSprop optimizer
        opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
        model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

    model.fit(x_train, y_train,validation_data=(x_validation,y_validation), epochs=15, batch_size=6,  callbacks=[TensorBoard(log_dir='/tmp/neuralnet')])

    eval(model, x_test, y_test)


if __name__ == "__main__":
    main()








