import keras

from keras.layers import Dense, Conv1D, Conv2D, Conv3D
import keras.backend as K

class DropConnect:
    def __init__(self, prob=0.5, drop_bias=False, drop_noise_shape=None):
        self.prob = prob
        self.drop_bias = drop_bias
        self.drop_noise_shape = drop_noise_shape

    @property
    def needs_drop(self):
        return 0.0 < self.prob < 1.0

    def replace_tensor(self, tensor_train, tensor_test):
        if self.uses_learning_phase:
            return K.in_train_phase(tensor_train, tensor_test)
        else:
            return tensor_train

    def get_config(self):
        config = {
            "prob": self.prob,
            "drop_bias": self.drop_bias,
            "drop_noise_shape": self.drop_noise_shape
        }

        return config

class DropConnectDense(DropConnect, Dense):
    def __init__(self, units, prob=0.5, drop_bias=False, drop_noise_shape=None, use_learning_phase = False, **kwargs):
        DropConnect.__init__(self, prob=prob, drop_bias=drop_bias, drop_noise_shape=drop_noise_shape)
        Dense.__init__(self, units, **kwargs)

        if self.needs_drop:
            self.uses_learning_phase = use_learning_phase

    def build(self, input_shape):
        Dense.build(self, input_shape)

        if self.needs_drop:
            dc_kernel = K.dropout(self.kernel, self.prob, self.drop_noise_shape)
            self.kernel = self.replace_tensor(dc_kernel, self.kernel)

            if self.drop_bias:
                dc_bias = K.dropout(self.bias, self.prob, self.drop_noise_shape)
                self.bias = self.replace_tensor(dc_bias, self.bias)

    def get_config(self):
        config_dc = DropConnect.get_config(self)
        config_base = Dense.get_config(self)

        return dict(list(config_dc.items()) + list(config_base.items()))

class DropConnectConv1D(DropConnect, Conv1D):
    def __init__(self, filters, kernel_size, prob=0.5, drop_bias=False, drop_noise_shape=None, use_learning_phase = False, **kwargs):        
        DropConnect.__init__(self, prob=prob, drop_bias=drop_bias, drop_noise_shape=drop_noise_shape)
        Conv1D.__init__(self, filters, kernel_size, **kwargs)

        if self.needs_drop:
            self.uses_learning_phase = use_learning_phase

    def build(self, input_shape):
        Conv1D.build(self, input_shape)

        if self.needs_drop:
            dc_kernel = K.dropout(self.kernel, self.prob, self.drop_noise_shape)
            self.kernel = self.replace_tensor(dc_kernel, self.kernel)

            if self.drop_bias:
                dc_bias = K.dropout(self.bias, self.prob, self.drop_noise_shape)
                self.bias = self.replace_tensor(dc_bias, self.bias)

    def get_config(self):
        config_dc = DropConnect.get_config(self)
        config_base = Conv1D.get_config(self)

        return dict(list(config_dc.items()) + list(config_base.items()))

class DropConnectConv2D(DropConnect, Conv2D):
    def __init__(self, filters, kernel_size, prob=0.5, drop_bias=False, drop_noise_shape=None, use_learning_phase = False, **kwargs):        
        DropConnect.__init__(self, prob=prob, drop_bias=drop_bias, drop_noise_shape=drop_noise_shape)
        Conv2D.__init__(self, filters, kernel_size, **kwargs)

        if self.needs_drop:
            self.uses_learning_phase = use_learning_phase

    def build(self, input_shape):
        Conv2D.build(self, input_shape)

        if self.needs_drop:
            dc_kernel = K.dropout(self.kernel, self.prob, self.drop_noise_shape)
            self.kernel = self.replace_tensor(dc_kernel, self.kernel)

            if self.drop_bias:
                dc_bias = K.dropout(self.bias, self.prob, self.drop_noise_shape)
                self.bias = self.replace_tensor(dc_bias, self.bias)

    def get_config(self):
        config_dc = DropConnect.get_config(self)
        config_base = Conv2D.get_config(self)

        return dict(list(config_dc.items()) + list(config_base.items()))

class DropConnectConv3D(DropConnect, Conv3D):
    def __init__(self, filters, kernel_size, prob=0.5, drop_bias=False, drop_noise_shape=None, use_learning_phase = False, **kwargs):
        DropConnect.__init__(self, prob=prob, drop_bias=drop_bias, drop_noise_shape=drop_noise_shape)
        Conv3D.__init__(self, filters, kernel_size, **kwargs)

        if self.needs_drop:
            self.uses_learning_phase = use_learning_phase

    def build(self, input_shape):
        Conv3D.build(self, input_shape)

        if self.needs_drop:
            dc_kernel = K.dropout(self.kernel, self.prob, self.drop_noise_shape)
            self.kernel = self.replace_tensor(dc_kernel, self.kernel)

            if self.drop_bias:
                dc_bias = K.dropout(self.bias, self.prob, self.drop_noise_shape)
                self.bias = self.replace_tensor(dc_bias, self.bias)

    def get_config(self):
        config_dc = DropConnect.get_config(self)
        config_base = Conv3D.get_config(self)

        return dict(list(config_dc.items()) + list(config_base.items()))