
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
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler, LabelEncoder

import numpy as np
import time
import os
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

def main():

    # Set the filename and open the file
    filename = 'dataset.log'
    file = open(filename, 'r')

    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    filereader = csv.reader(open("Data/UNSW-NB15/UNSW_NB15_training-set.csv"), delimiter=",")
    data = np.array(list(filereader))
    
    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    while 1:
        where = file.tell()
        line = file.readline()
        if not line:
            time.sleep(1)
            file.seek(where)
        else:
            line = line[:-1]
            dataline = np.array(line.split(","))
            scaler = MinMaxScaler()

            x_train = data[:, [1,6,7,8,9,10,11,12,13,27,28,32,33,34,35,36]]
            x_train = x_train.astype(float)
            scaler.fit(x_train)

            print(dataline)
            dataline.astype(float)
            dataline = dataline.reshape([1,16])
            dataline = scaler.transform(dataline)

            dataline = dataline.reshape([1,16, 1,1])
            print(type(loaded_model.predict(dataline)))
            print("prediction: ")
            print(tabulate(loaded_model.predict(dataline),headers=['Analysis','Backdoor','DoS','Exploits','Fuzzers','Generic','Normal','Reconnaissance','Shellcode','Worms']))
            #print(loaded_model.predict(dataline))
            npi=loaded_model.predict(dataline)
            with open('predictions.dat', 'a') as predictValues:
              np.savetxt(predictValues, npi, delimiter=",")
            predictValues.close()

if __name__ == "__main__":
    main()
