# keras-uncertainty
Utilities and Models to perform Uncertainty Quantification on Keras.

Keras-Uncertainty is a high-level API to perform uncertainty quantification of machine learning models built with Keras.
Many real-world applications not only require a model to make a prediction, but also to provide a confidence value
that indicates how trustworthy is the prediction, which can be integrated into the decision making progress.

Typical research in machine learning applications (Computer Vision, NLP, etc) usually does not consider ways to produce well behaved
uncertainty estimates, and machine learning methods can be used to extract or include uncertainty information into the model.

## Classification Uncertainty
![Classification Comparison](https://raw.githubusercontent.com/mvaldenegro/keras-uncertainty/master/examples/images/uncertainty-two-moons.png)

## Regression Uncertainty
![Regression example](https://raw.githubusercontent.com/mvaldenegro/keras-uncertainty/master/examples/images/uncertainty-toy-regression.png)

## Installation

You can easily install with pip, using the following command:

```
pip install --user git+https://github.com/mvaldenegro/keras-uncertainty.git
```

## Features

- Entropy and Negative Log-Likelihood metrics.
- Calibration plots for classification.
- Accuracy vs Confidence plot for classification.

## Currently Implemented Methods

| Method            | Classification     | Regression        |
|-------------------|--------------------|-------------------|
| Ensembles         | :heavy_check_mark: | :heavy_check_mark:|
| MC-Dropout        | :heavy_check_mark: | :heavy_check_mark:|
| MC-Dropout        | :heavy_check_mark: | :heavy_check_mark:|
| Direct UQ         | :heavy_check_mark: | :x:               |
| Bayes by Backprop | :heavy_check_mark: | :heavy_check_mark:|
| Flipout           | :heavy_check_mark: | :heavy_check_mark:|
| Gradient          | :heavy_check_mark: | :x:               |
