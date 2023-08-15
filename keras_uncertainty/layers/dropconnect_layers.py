import keras_uncertainty.backend as K

Dense = K.layers.Dense
Layer = K.layers.Layer
activations = K.activations
conv_utils = K.conv_utils

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

class DropConnectConvND(DropConnect, Layer):
    def __init__(self, rank, filters, kernel_size, strides=1, padding="valid", dilation_rate=(1, 1, 1), activation="linear", prob=0.5, drop_bias=False, noise_shape=None, **kwargs):
        DropConnect.__init__(self, prob=prob, drop_bias=drop_bias, noise_shape=noise_shape)
        Layer.__init__(**kwargs)

        self.rank = rank
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, self.rank, "kernel_size")
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.activation = activations.get(activation)

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.filters,)

    def build(self, input_shape):
        input_channels = input_shape[-1]
        kernel_shape = self.kernel_size + (input_channels, self.filters)
        self.kernel = self.add_weight(name="kernel_mean", shape=kernel_shape, initializer="glorot_uniform")

        bias_shape = (self.filters, )
        self.bias = self.add_weight(name="bias_mean", shape=bias_shape, initializer="glorot_uniform")

    def conv(self, inputs, kernel):
        conv_dict = {
            1: K.conv1d,
            2: K.conv2d,
            3: K.conv3d
        }

        return conv_dict[self.rank](inputs, kernel, strides=self.strides, padding=self.padding, data_format="channels_last", dilation_rate=self.dilation_rate)

    def call(self, inputs):
        kernel_sample = self.sample(self.kernel)
        output = self.conv(inputs, kernel_sample)

        if self.use_bias:
            bias_sample = self.sample(self.bias, dropit=self.drop_bias)
            output = K.bias_add(output, bias_sample, data_format="channels_last")

        if self.activation is not None:
            output = self.activation(output)

        return output

    def get_config(self):
        config = DropConnect.get_config(self)
        config.update(Layer.get_config(self))

        config["rank"] = self.rank
        config["filters"] = self.filters
        config["kernel_size"] = self.kernel_size
        config["strides"] = self.strides
        config["padding"] = self.padding
        config["dilation_rate"] = self.dilation_rate
        config["activation"] = self.activation.__name__

        return config

class DropConnectConv1D(DropConnectConvND):
    def __init__(self, filters, kernel_size, strides=1, padding="valid", dilation_rate=(1, ), activation="linear", prob=0.5, drop_bias=False, noise_shape=None, **kwargs):        
        DropConnectConvND.__init__(self, rank=1, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate, activation=activation, 
                                   prob=prob, drop_bias=drop_bias, noise_shape=noise_shape, **kwargs)

class DropConnectConv2D(DropConnectConvND):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding="valid", dilation_rate=(1, 1), activation="linear", prob=0.5, drop_bias=False, noise_shape=None, **kwargs):        
        DropConnectConvND.__init__(self, rank=2, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate, activation=activation, 
                                   prob=prob, drop_bias=drop_bias, noise_shape=noise_shape, **kwargs)

class DropConnectConv3D(DropConnectConvND):
    def __init__(self, filters, kernel_size, strides=(1, 1, 1), padding="valid", dilation_rate=(1, 1, 1), activation="linear", prob=0.5, drop_bias=False, noise_shape=None, **kwargs):        
        DropConnectConvND.__init__(self, rank=3, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate, activation=activation, 
                                   prob=prob, drop_bias=drop_bias, noise_shape=noise_shape, **kwargs)