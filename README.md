# keras-uncertainty
Utilities and Models to perform Uncertainty Quantification on Keras.

Keras-Uncertainty is a high-level API to perform uncertainty quantification of machine learning models built with Keras.
Many real-world applications not only require a model to make a prediction, but also to provide a confidence value
that indicates how trustworthy is the prediction, which can be integrated into the decision making progress.

Typical research in machine learning applications (Computer Vision, NLP, etc) usually does not consider ways to produce well behaved
uncertainty estimates, and machine learning methods can be used to extract or include uncertainty information into the model.

## Installation

Clone this repository and then install using setuptools:

```
git clone git@github.com:mvaldenegro/keras-uncertainty.git
cd keras-uncertainty
python3 setup.py install --user
```

## Features

- Entropy and Negative Log-Likelihood metrics.
- Calibration plots for classification.

## Current Implementations

- Monte Carlo Dropout (MC-Dropout)
- Deep Ensembles

