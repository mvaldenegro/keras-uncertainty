import keras_uncertainty.backend as K

Dense = K.layers.Dense
Conv1D = K.layers.Conv1D
Conv2D = K.layers.Conv2D
Conv3D = K.layers.Conv3D

class DropConnect:
    def __init__(self, prob=0.5, drop_bias=False, noise_shape=None):
        self.prob = prob
        self.drop_bias = drop_bias
        self.noise_shape = noise_shape
        self.kernel_noise_shape = None
        self.bias_noise_shape = None

    @property
    def needs_drop(self):
        return 0.0 < self.prob < 1.0

    def sample(self, tensor, dropit=True, noise_shape=None):
        if dropit:
            return K.dropout(tensor, self.prob, noise_shape)

        return tensor

    def replace_tensor(self, tensor_train, tensor_test):
        if self.uses_learning_phase:
            return K.in_train_phase(tensor_train, tensor_test)
        else:
            return tensor_train

    def get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape

        symbolic_shape = K.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape
                       for axis, shape in enumerate(self.noise_shape)]

        return tuple(noise_shape)

    def get_config(self):
        config = {
            "prob": self.prob,
            "drop_bias": self.drop_bias,
            "noise_shape": self.noise_shape
        }

        return config

class DropConnectDense(DropConnect, Dense):
    def __init__(self, units, prob=0.5, drop_bias=False, noise_shape=None, use_learning_phase = False, **kwargs):
        DropConnect.__init__(self, prob=prob, drop_bias=drop_bias, noise_shape=noise_shape)
        Dense.__init__(self, units, **kwargs)

        if self.needs_drop:
            self.uses_learning_phase = use_learning_phase

    def call(self, inputs, **kwargs):
        kernel_sample = self.sample(self.kernel)
        bias_sample = self.sample(self.bias, dropit=self.drop_bias)

        outputs = K.dot(inputs, kernel_sample)
        
        if self.use_bias:
            outputs += bias_sample

        # This always produces stochastic outputs
        return self.activation(outputs)

    def get_config(self):
        config_dc = DropConnect.get_config(self)
        config_base = Dense.get_config(self)

        return dict(list(config_dc.items()) + list(config_base.items()))

class DropConnectConv1D(DropConnect, Conv1D):
    def __init__(self, filters, kernel_size, prob=0.5, drop_bias=False, noise_shape=None, use_learning_phase = False, **kwargs):        
        DropConnect.__init__(self, prob=prob, drop_bias=drop_bias, noise_shape=noise_shape)
        Conv1D.__init__(self, filters, kernel_size, **kwargs)

        if self.needs_drop:
            self.uses_learning_phase = use_learning_phase

    def build(self, input_shape):
        Conv1D.build(self, input_shape)

        if self.needs_drop:
            dc_kernel = K.dropout(self.kernel, self.prob, self.noise_shape)
            self.kernel = self.replace_tensor(dc_kernel, self.kernel)

            if self.drop_bias:
                dc_bias = K.dropout(self.bias, self.prob, self.noise_shape)
                self.bias = self.replace_tensor(dc_bias, self.bias)

    def get_config(self):
        config_dc = DropConnect.get_config(self)
        config_base = Conv1D.get_config(self)

        return dict(list(config_dc.items()) + list(config_base.items()))

class DropConnectConv2D(DropConnect, Conv2D):
    def __init__(self, filters, kernel_size, prob=0.5, drop_bias=False, noise_shape=None, use_learning_phase = False, **kwargs):        
        DropConnect.__init__(self, prob=prob, drop_bias=drop_bias, noise_shape=noise_shape)
        Conv2D.__init__(self, filters, kernel_size, **kwargs)

        if self.needs_drop:
            self.uses_learning_phase = use_learning_phase

    def build(self, input_shape):
        Conv2D.build(self, input_shape)

        if self.needs_drop:
            dc_kernel = K.dropout(self.kernel, self.prob, self.noise_shape)
            self.kernel = self.replace_tensor(dc_kernel, self.kernel)

            if self.drop_bias:
                dc_bias = K.dropout(self.bias, self.prob, self.noise_shape)
                self.bias = self.replace_tensor(dc_bias, self.bias)

    def get_config(self):
        config_dc = DropConnect.get_config(self)
        config_base = Conv2D.get_config(self)

        return dict(list(config_dc.items()) + list(config_base.items()))

class DropConnectConv3D(DropConnect, Conv3D):
    def __init__(self, filters, kernel_size, prob=0.5, drop_bias=False, noise_shape=None, use_learning_phase = False, **kwargs):
        DropConnect.__init__(self, prob=prob, drop_bias=drop_bias, noise_shape=noise_shape)
        Conv3D.__init__(self, filters, kernel_size, **kwargs)

        if self.needs_drop:
            self.uses_learning_phase = use_learning_phase

    def build(self, input_shape):
        Conv3D.build(self, input_shape)

        if self.needs_drop:
            dc_kernel = K.dropout(self.kernel, self.prob, self.noise_shape)
            self.kernel = self.replace_tensor(dc_kernel, self.kernel)

            if self.drop_bias:
                dc_bias = K.dropout(self.bias, self.prob, self.noise_shape)
                self.bias = self.replace_tensor(dc_bias, self.bias)

    def get_config(self):
        config_dc = DropConnect.get_config(self)
        config_base = Conv3D.get_config(self)

        return dict(list(config_dc.items()) + list(config_base.items()))