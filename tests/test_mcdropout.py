import keras
import numpy as np

from keras.layers import Dense, Dropout
from keras.models import Sequential

from keras_uncertainty.models import MCDropoutModel

def test_mcdropout():
    model = Sequential()
    model.add(Dense(10, input_shape=(1,), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    mc_model = MCDropoutModel(model)

    inp = np.array([[3]])
    mean, std = mc_model.predict(inp)

    print(mean, std)

