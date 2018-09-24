#! usr/bin/python3


import keras
import csv
import h5py
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from keras.models import Sequential, model_from_json
from keras.layers import Conv2D, GlobalAveragePooling1D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Embedding, BatchNormalization, Flatten
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler

import numpy as np

from sklearn.preprocessing import LabelBinarizer, MinMaxScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import classification_report

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
    model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(20, kernel_initializer='normal',
                    activation='relu', name='output'))
    model.add(Dense(10, kernel_initializer='normal', activation='softmax'))
    return model

def preporcess(data, dataTest):

    scaler = MinMaxScaler()
    encoder = LabelBinarizer()
    encoder2 = LabelEncoder()

    # Isolate only the selected feature. See ReadMe for more details
    x_train = data[:, [1,6,7,8,9,10,11,12,13,27,28,32,33,34,35,36]]
    x_test = dataTest[:, [1,6,7,8,9,10,11,12,13,27,28,32,33,34,35,36]]

    # Normalize features
    x_train = x_train.astype(float)
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = x_test.astype(float)
    scaler.fit(x_test)
    x_test = scaler.transform(x_test)

    # Retrieves the label from the dataset 
    train_label = data[:, 43]
    test_label = dataTest[:, 43]

    # Encode the attacks type. fit() generate a dictionnary text=>value
    # [6:'Fuzzers',4:'Backdoor',1:'DoS',2:'Exploits',3:'Generic',5:'Reconnaissance',7:'Normal',8:'Shellcode',9:'Worms']
    encoder2.fit(train_label)
    y_train = encoder2.transform(train_label)
    y_test = encoder2.transform(test_label)

    # Transform to binary
    encoder.fit(y_train)
    y_train = encoder.transform(y_train)
    y_test = encoder.transform(y_test)

    # hack: CNN works with 3D inputs in Keras, so change vector of size x to [1,1,x] tab
    x_final_train = []
    x_final_test = []
    size = np.size(x_train,axis=1)
    for x in x_train:
        sample = x.reshape([size, 1, 1])
        x_final_train.append(sample)
    x_train = np.array(x_final_train)

    for x in x_test:
        sample = x.reshape([size, 1, 1])
        x_final_test.append(sample)
    x_test = np.array(x_final_test)


    # Split the dataset into test and validation sets
    seed = 9
    np.random.seed(seed)
    x_validation, x_test_nn, y_validation, y_test_nn = train_test_split(
        x_test, y_test, test_size=0.80, random_state=seed)

    return x_train, y_train, x_validation, y_validation, x_test, y_test


def eval(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=1)
    print("loss on test data:", score[0])
    print("accuracy on test data:", score[1]*100, "%")


def main():

    # Open the datasets for train and test
    filereader = csv.reader(open("Data/UNSW-NB15/UNSW_NB15_training-set.csv"), delimiter=",")
    data = np.array(list(filereader))
    filereaderTest = csv.reader(open("Data/UNSW-NB15/UNSW_NB15_testing-set.csv"), delimiter=",")
    dataTest = np.array(list(filereaderTest))

    x_train, y_train, x_validation, y_validation, x_test, y_test = preporcess(
        data, dataTest)

    # Declare the model
    model = generate_cnn_model(np.size(x_train,axis=1))

    # Compile the network
    opt = optimizers.Adam()
    model.compile(loss='categorical_crossentropy',
                optimizer=opt, metrics=['categorical_accuracy'])

    # Early stopper declaration for the traning
    stopper = EarlyStopping(monitor='val_binary_accuracy', patience=3, mode='auto')

    # Learning
    model.fit(x_train, y_train, epochs=1, batch_size=50, validation_data=(x_validation, y_validation), callbacks=[stopper])

    # Evaluate the performance of the model
    eval(model, x_test, y_test)

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")


if __name__ == "__main__":
    main()
