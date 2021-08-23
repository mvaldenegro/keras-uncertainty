# keras-uncertainty
Utilities and Models to perform Uncertainty Quantification on Keras.

Keras-Uncertainty is a high-level API to perform uncertainty quantification of machine learning models built with Keras.
Many real-world applications not only require a model to make a prediction, but also to provide a confidence value
that indicates how trustworthy is the prediction, which can be integrated into the decision making progress.

Typical research in machine learning applications (Computer Vision, NLP, etc) usually does not consider ways to produce well behaved
uncertainty estimates, and machine learning methods can be used to extract or include uncertainty information into the model.

![Regression example](https://raw.githubusercontent.com/mvaldenegro/keras-uncertainty/master/examples/deepensemble-x-pow-3.png)

## Installation

You can easily install with pip, using the following command:

```
pip install --user git+https://github.com/mvaldenegro/keras-uncertainty.git
```

## Features

- Entropy and Negative Log-Likelihood metrics.
- Calibration plots for classification.
- Accuracy vs Confidence plot for classification.

## Current Implementations

- Monte Carlo Dropout (MC-Dropout)
- Monte Carlo DropConnect (MC-DropConnect)
- Deep Ensembles
- DUQ (Upcoming).
