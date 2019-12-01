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

    def build(self, input_shape):
        Dense.build(self, input_shape)

        if self.needs_drop:
            self.kernel = K.in_train_phase(K.dropout(self.kernel, self.prob, self.drop_noise_shape), self.kernel)

            if self.drop_bias:
                self.bias = K.in_train_phase(K.dropout(self.bias, self.prob, self.drop_noise_shape), self.bias)

    def get_config(self):
        config_dc = DropConnect.get_config(self)
        config_base = Dense.get_config(self)

        return dict(list(config_dc.items()) + list(config_base.items()))

class DropConnectConv1D(DropConnect, Conv1D):
    def __init__(self, *args, **kwargs):        
        DropConnect.__init__(self, **kwargs)
        Conv1D.__init__(self, *args, **kwargs)

        if self.needs_drop:
            self.uses_learning_phase = True

    def build(self, input_shape):
        Conv1D.build(self, input_shape)

        if self.needs_drop:
            self.kernel = K.in_train_phase(K.dropout(self.kernel, self.prob, self.drop_noise_shape), self.kernel)

            if self.drop_bias:
                self.bias = K.in_train_phase(K.dropout(self.bias, self.prob, self.drop_noise_shape), self.bias)

    def get_config(self):
        config_dc = DropConnect.get_config(self)
        config_base = Conv1D.get_config(self)

        return dict(list(config_dc.items()) + list(config_base.items()))

class DropConnectConv2D(DropConnect, Conv2D):
    def __init__(self, *args, **kwargs):        
        DropConnect.__init__(self, **kwargs)
        Conv2D.__init__(self, *args, **kwargs)

        if self.needs_drop:
            self.uses_learning_phase = True

    def build(self, input_shape):
        Conv2D.build(self, input_shape)

        if self.needs_drop:
            self.kernel = K.in_train_phase(K.dropout(self.kernel, self.prob, self.drop_noise_shape), self.kernel)

            if self.drop_bias:
                self.bias = K.in_train_phase(K.dropout(self.bias, self.prob, self.drop_noise_shape), self.bias)

    def get_config(self):
        config_dc = DropConnect.get_config(self)
        config_base = Conv2D.get_config(self)

        return dict(list(config_dc.items()) + list(config_base.items()))

class DropConnectConv3D(DropConnect, Conv3D):
    def __init__(self, *args, **kwargs):        
        DropConnect.__init__(self, **kwargs)
        Conv3D.__init__(self, *args, **kwargs)

        if self.needs_drop:
            self.uses_learning_phase = True

    def build(self, input_shape):
        Conv3D.build(self, input_shape)

        if self.needs_drop:
            self.kernel = K.in_train_phase(K.dropout(self.kernel, self.prob, self.drop_noise_shape), self.kernel)

            if self.drop_bias:
                self.bias = K.in_train_phase(K.dropout(self.bias, self.prob, self.drop_noise_shape), self.bias)

    def get_config(self):
        config_dc = DropConnect.get_config(self)
        config_base = Conv3D.get_config(self)

        return dict(list(config_dc.items()) + list(config_base.items()))