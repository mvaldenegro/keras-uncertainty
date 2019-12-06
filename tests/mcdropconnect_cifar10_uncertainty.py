import keras_uncertainty
from keras_uncertainty.utils import numpy_negative_log_likelihood, numpy_entropy
from keras_uncertainty.layers import DropConnectConv2D, DropConnectDense
from keras_uncertainty.models import MCDropoutClassifier

import keras
import keras.backend as K
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from keras.models import Model, Sequential
from keras.datasets import cifar10
from keras.utils import to_categorical

import numpy as np

from sklearn.metrics import accuracy_score

def network():
    model = Sequential()
    model.add(DropConnectConv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(DropConnectConv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(DropConnectConv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(DropConnectConv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(DropConnectConv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(DropConnectConv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(DropConnectConv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(DropConnectConv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(DropConnectConv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(DropConnectDense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(DropConnectDense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])    
    
    return model

if __name__ == "__main__":
    model = network()
    model.summary() 

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model.fit(x_train, y_train, batch_size=32, epochs=30, validation_data=(x_test, y_test))

    model.save("dropconnect_cifar10_vgg-custom.hdf5")

    mcd = MCDropoutClassifier(model)

    for samples in range(1, 20):
        y_pred = mcd.predict(x_test, num_samples=samples)

        class_preds = np.argmax(y_pred, axis=1)
        class_true = np.argmax(y_test, axis=1)
        acc = accuracy_score(class_true, class_preds)
        nll = numpy_negative_log_likelihood(y_test, y_pred)
        entropy = np.mean(numpy_entropy(y_pred))

        print("{} samples - Accuracy {:.3f} NLL {:.3f} Entropy {:.3f}".format(samples, acc, nll, entropy))