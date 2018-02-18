#! usr/bin/python3

# -*- coding: utf8 -*-

import keras
import csv
from keras.callbacks import TensorBoard
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from keras.models import Sequential
from keras.layers import Conv2D, GlobalAveragePooling1D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Embedding, BatchNormalization, Flatten
from keras.optimizers import SGD

import h5py
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
    model.add(Conv2D(64,(3,1),activation='relu',input_shape=(shape,1,1)))
    model.add(Conv2D(64,(3,1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Conv2D(128,(3,1),activation='relu'))
    model.add(Conv2D(128,(3,1),activation='relu',padding="same"))
    model.add(Conv2D(128,(3,1),activation='relu',padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Conv2D(256,(3,1),activation='relu',padding="same"))
    model.add(Conv2D(256,(3,1),activation='relu',padding="same"))
    model.add(Conv2D(256,(3,1),activation='relu',padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(124, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))
    return model


def preporcess(data, dataTest, CNN=False):
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
    
    ## Generation d'un dataset de validation
    seed = 9
    np.random.seed(seed)
    x_validation, x_test, y_validation, y_test = train_test_split(x_test, y_test, test_size=0.70, random_state=seed)

    return x_train, y_train, x_validation, y_validation, x_test, y_test


def CapsNet(input_shape, n_class, routings):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=64, kernel_size=(3,1), strides=1, padding='valid', activation='relu', name='conv1')(x)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=(3,1), strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                             name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])

    return train_model


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


def eval(model,x_test,y_test):
    score = model.evaluate(x_test, y_test, verbose=1)
    print("loss on test data:", score[0])
    print("accuracy on test data:", score[1]*100, "%")




def main():

    input_list = ["1. MLP", "2. CNN2D", "3. CapsNet"]

    filereader = csv.reader(open("Data/KDDTrain+.txt"), delimiter=",")
    data = np.array(list(filereader))

    filereaderTest = csv.reader(open("Data/KDDTest+.txt"), delimiter=",")
    dataTest = np.array(list(filereaderTest))

    P1 = input("Choisissez le reseau: " + ' '.join(input_list) + "\n")
    CNN = False
    if(P1=='2' or P1=='3'): 
        CNN = True

    x_train, y_train, x_validation, y_validation, x_test, y_test = preporcess(data,dataTest,CNN)

    if (P1 == '1'):
         ### MLP model ###
        model = generate_model(41)
        model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=0.001),metrics=['accuracy'])
    
    elif (P1 == '2'):
        ### CNN 1D model ##
        model = generate_cnn_model(41)

        # initiate RMSprop optimizer
        opt =optimizers.Adam(lr=0.001)
        model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
    elif (P1 =='3'):
        # compile the model
        # define model
        model = CapsNet(input_shape=(41,1,1), n_class=len(np.unique(np.argmax(y_train, 1))),routings=3)
        model.compile(optimizer=optimizers.Adam(lr=0.001),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., 0.392],
                  metrics={'capsnet': 'accuracy'})

    print(model.summary())

    model.fit(x_train, y_train,validation_data=(x_validation,y_validation), epochs=20, batch_size=50,  callbacks=[TensorBoard(log_dir='/tmp/neuralnet')])

    eval(model, x_test, y_test)

    model.save_weights("./result/trained_model.h5")
    print("Trained model saved to ./result/trained_model.h5")


if __name__ == "__main__":
    main()








