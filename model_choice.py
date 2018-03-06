#! usr/bin/python3

# -*- coding: utf8 -*-

import keras
import csv
from keras.callbacks import TensorBoard
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from keras.models import Sequential
from keras.layers import Conv2D, GlobalAveragePooling1D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Embedding, BatchNormalization, Flatten
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier

import numpy as np


from sklearn.preprocessing import LabelBinarizer, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV


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
        'apache2': 'dos',
        'mailbomb': 'dos',
        'processtable': 'dos',
        'mscan': 'probe',
        'saint': 'probe',
        'sendmail': 'r2l',
        'named': 'r2l',
        'snmpgetattack': 'r2l',
        'snmpguess': 'r2l',
        'xlock': 'r2l',
        'xsnoop': 'r2l',
        'worm': 'r2l',
        'httptunnel': 'u2r',
        'ps': 'u2r',
        'sqlattack': 'u2r',
        'xterm': 'u2r',
        'normal': 'normal'
    }.get(x, 'unknown')


def generate_model(shape):
    # define the model
    model = keras.models.Sequential()

    model.add(Dense(82, input_dim=shape, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(164, activation='relu'))
    model.add(Dense(164, activation='relu'))
    model.add(Dense(82, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(41, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(6, activation='softmax'))

    return model


def generate_cnn_model(shape):
    '''
        Model from a reasearch paper
        https://www.researchgate.net/publication/319717354_A_Few-shot_Deep_Learning_Approach_for_Improved_Intrusion_Detection
    '''
    model = Sequential()
    model.add(Conv2D(64, (3, 1), activation='relu', input_shape=(shape, 1, 1)))
    model.add(Conv2D(64, (3, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Conv2D(128, (3, 1), activation='relu'))
    model.add(Conv2D(128, (3, 1), activation='relu', padding="same"))
    model.add(Conv2D(128, (3, 1), activation='relu', padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Conv2D(256, (3, 1), activation='relu', padding="same"))
    model.add(Conv2D(256, (3, 1), activation='relu', padding="same"))
    model.add(Conv2D(256, (3, 1), activation='relu', padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(6, activation='softmax'))
    return model


def create_model(optimizer='rmsprop', init='glorot_uniform'):

    model = keras.models.Sequential()

    model.add(Dense(82, input_dim=41, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(164, kernel_initializer=init, activation='relu'))
    model.add(Dense(164, kernel_initializer=init, activation='relu'))
    model.add(Dense(82, kernel_initializer=init, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(41, kernel_initializer=init, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(20, kernel_initializer=init, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(6, kernel_initializer=init, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])
    return model


def preporcess(data, dataTest, CNN=False):
    scaler = MinMaxScaler()
    encoder = LabelBinarizer()
    encoder2 = LabelEncoder()

    x_train = data[:, np.arange(0, 41)]
    x_train[:, 1] = encoder2.fit_transform(x_train[:, 1])
    x_train[:, 2] = encoder2.fit_transform(x_train[:, 2])
    x_train[:, 3] = encoder2.fit_transform(x_train[:, 3])

    scaler.fit(x_train)
    x_train = scaler.transform(x_train)

    train_label = data[:, 41]
    # y_train = encoder.fit_transform(train_label)
    train_label = [transform(attacktype) for attacktype in train_label]

    x_test = dataTest[:, np.arange(0, 41)]
    x_test[:, 1] = encoder2.fit_transform(x_test[:, 1])
    x_test[:, 2] = encoder2.fit_transform(x_test[:, 2])
    x_test[:, 3] = encoder2.fit_transform(x_test[:, 3])

    scaler.fit(x_test)
    x_test = scaler.transform(x_test)

    test_label = dataTest[:, 41]
    test_label = [transform(attacktype) for attacktype in test_label]

    # Join dataset => meme encoding pour train et test
    label = np.concatenate((train_label, test_label))
    y = encoder.fit_transform(label)

    trainlen = np.size(train_label)
    testlen = np.size(test_label)
    y_train = y[np.arange(0, trainlen)]
    y_test = y[np.arange(trainlen, trainlen+testlen)]

    if(CNN):
        x_final_train = []
        x_final_test = []
        for x in x_train:
            sample = x.reshape([41, 1, 1])
            x_final_train.append(sample)
        x_train = np.array(x_final_train)

        for x in x_test:
            sample = x.reshape([41, 1, 1])
            x_final_test.append(sample)
        x_test = np.array(x_final_test)

        print(np.shape(x_test))

    # Generation d'un dataset de validation
    seed = 9
    np.random.seed(seed)
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_test, y_test, test_size=0.70, random_state=seed)

    return x_train, y_train, x_validation, y_validation, x_test, y_test


def eval(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=1)
    print("loss on test data:", score[0])
    print("accuracy on test data:", score[1]*100, "%")


def main():

    input_list = ["1. MLP", "2. CNN2D", "3. Sklearn Classifier"]

    filereader = csv.reader(open("Data/KDDTrain+.txt"), delimiter=",")
    data = np.array(list(filereader))

    filereaderTest = csv.reader(open("Data/KDDTest+.txt"), delimiter=",")
    dataTest = np.array(list(filereaderTest))

    P1 = input("Choisissez le reseau: " + ' '.join(input_list) + "\n")
    CNN = False
    if(P1 == '2' ):
        CNN = True

    x_train, y_train, x_validation, y_validation, x_test, y_test = preporcess(
        data, dataTest, CNN)

    if (P1 == '1'):
         ### MLP model ###
        model = generate_model(41)
        model.compile(loss='categorical_crossentropy',optimizer=optimizers.Adam(lr=0.001), metrics=['accuracy'])

    elif (P1 == '2'):
        ### CNN 2D model ##
        model = generate_cnn_model(41)

        # initiate RMSprop optimizer
        opt = optimizers.Adam(lr=0.0005)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt, metrics=['accuracy'])
    elif (P1 == '3'):

        model = KerasClassifier(build_fn=create_model, verbose=0)

        # grid search epochs, batch size and optimizer
        opt = ['rmsprop', 'adam']
        init = ['glorot_uniform', 'normal', 'uniform']
        epochs = [5,10,20]
        batches = [20,50,100]
    
        param_grid = dict(optimizer=opt, epochs=epochs,
                  batch_size=batches, init=init)
        grid = GridSearchCV(estimator=model, param_grid=param_grid)
        grid_result = grid.fit(x_train, y_train, validation_data=(x_validation, y_validation))

        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))

    if (P1 != '3'):
        model.fit(x_train, y_train,validation_data=(x_validation,y_validation), epochs=1, batch_size=100,  callbacks=[TensorBoard(log_dir='/tmp/neuralnet')])

        eval(model, x_test, y_test)


if __name__ == "__main__":
    main()








