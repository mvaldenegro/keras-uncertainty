import keras
import numpy as np

from keras.layers import Dense, Dropout
from keras.models import Sequential

from keras_uncertainty.models import MCDropoutClassifier, MCDropoutRegressor

def test_mcdropout_classifier():
    model = Sequential()
    model.add(Dense(10, input_shape=(1,), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation="softmax"))

    mc_model = MCDropoutClassifier(model)

    inp = np.array([[2]])
    probs = mc_model.predict(inp)[0]

    print("One-sample prediction: {}".format(probs))

    inp = np.array([[3], [4], [5]])
    probs = mc_model.predict(inp)[0]

    print("Multi-sample prediction: {}".format(probs))

def test_mcdropout_regressor():
    model = Sequential()
    model.add(Dense(10, input_shape=(1,), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    mc_model = MCDropoutRegressor(model)

    inp = np.array([[2]])
    probs = mc_model.predict(inp)[0]

    print("One-sample prediction: {}".format(probs))

    inp = np.array([[3], [4], [5]])
    probs = mc_model.predict(inp)[0]

    print("Multi-sample prediction: {}".format(probs))

if __name__ == "__main__":
    test_mcdropout_classifier()
    test_mcdropout_regressor()
     