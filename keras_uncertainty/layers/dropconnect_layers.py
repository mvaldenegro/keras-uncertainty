import keras

from keras.layers import Dense, Conv1D, Conv2D, Conv3D
import keras.backend as K

class DropConnect:
    def __init__(self, **kwargs):
        self.prob = kwargs.pop('prob', 0.5)
        self.drop_bias = kwargs.pop("drop_bias", False)
        self.drop_noise_shape = kwargs.pop("drop_noise_shape", None)

    @property
    def needs_drop(self):
        return 0.0 < self.prob < 1.0

    def get_config(self):
        config = {
            "prob": self.prob,
            "drop_bias": self.drop_bias,
            "drop_noise_shape": self.drop_noise_shape
        }

        return config


class DropConnectDense(DropConnect, Dense):
    def __init__(self, *args, **kwargs):        
        DropConnect.__init__(self, **kwargs)
        Dense.__init__(self, *args, **kwargs)

        if self.needs_drop:
            self.uses_learning_phase = True

    def call(self, x, mask=None):
        if self.needs_drop:
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

    def get_config(self):
        config_dc = DropConnect.get_config(self)
        config_base = Dense.get_config(self)

        return dict(list(config_dc.items()) + list(config_base.items()))


class DropConnectConv2D(DropConnect, Conv2D):
    def __init__(self, *args, **kwargs):        
        DropConnect.__init__(self, **kwargs)
        Conv2D.__init__(self, *args, **kwargs)

        if self.needs_drop:
            self.uses_learning_phase = True

    def call(self, inputs):
        if self.needs_drop:
            self.kernel = K.in_train_phase(K.dropout(self.kernel, self.prob, self.drop_noise_shape), self.kernel)

            if self.drop_bias:
                self.bias = K.in_train_phase(K.dropout(self.bias, self.prob, self.drop_noise_shape), self.bias)

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

    def get_config(self):
        config_dc = DropConnect.get_config(self)
        config_base = Conv2D.get_config(self)

        return dict(list(config_dc.items()) + list(config_base.items()))