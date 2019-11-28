import keras

from keras.layers import Dense, Conv1D, Conv2D, Conv3D
import keras.backend as K

class DropConnectDense(Dense):
    def __init__(self, *args, **kwargs):
        self.prob = kwargs.pop('prob', 0.5)

        if 0. < self.prob < 1.:
            self.uses_learning_phase = True

        if "drop_bias" in kwargs:
            self.drop_bias = kwargs.pop("drop_bias")
        else:
            self.drop_bias = False

        super(DropConnectDense, self).__init__(*args, **kwargs)

    def call(self, x, mask=None):
        if 0. < self.prob < 1.:
            self.kernel = K.in_train_phase(K.dropout(self.kernel, self.prob), self.kernel)

            if self.drop_bias:
                self.bias = K.in_train_phase(K.dropout(self.bias, self.prob), self.bias)

        # Same as original
        output = K.dot(x, self.kernel)

        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')

        if self.activation is not None:
            output = self.activation(output)
            
        return output


class DropConnectConv2D(Conv2D):
    def __init__(self, *args, **kwargs):
        self.prob = kwargs.pop('prob', 0.5)

        if 0. < self.prob < 1.:
            self.uses_learning_phase = True

        if "drop_bias" in kwargs:
            self.drop_bias = kwargs.pop("drop_bias")
        else:
            self.drop_bias = False

        if "drop_shape" in kwargs:
            self.drop_shape = kwargs.pop("drop_shape")
        else:
            self.drop_shape = None

        super(DropConnectConv2D, self).__init__(*args, **kwargs)

    def call(self, inputs):
        if 0. < self.prob < 1.:
            self.kernel = K.in_train_phase(K.dropout(self.kernel, self.prob, self.drop_shape), self.kernel)

            if self.drop_bias:
                self.bias = K.in_train_phase(K.dropout(self.bias, self.prob, self.drop_shape), self.bias)

        outputs = K.conv2d(
                inputs,
                self.kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)
        
        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs