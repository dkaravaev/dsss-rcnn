import numpy as np

import keras.backend as K

from keras.utils  import conv_utils
from keras.layers import Layer, InputSpec, MaxPool1D


class Downsample1D(Layer):
    def __init__(self, pool_size=2, **kwargs):
        super(Downsample1D, self).__init__(**kwargs)
        self.pool_size = 2

    def compute_output_shape(self, input_shape):
        shape = (1, input_shape[-1], 1)
        length = conv_utils.conv_output_length(
            input_shape[1],
            shape[1],
            padding='causal',
            stride=self.pool_size,
            dilation=1
        )

        return (input_shape[0], length, input_shape[2])

    def build(self, input_shape):
        shape = (1, 1, 1)

        self.one = K.ones(shape=shape)
        self.one._keras_shape = shape

        self.ishape = input_shape
        self.oshape = self.compute_output_shape(input_shape)

    def call(self, inpt):
        C = [inpt[:, :, c] for c in range(self.ishape[-1])]
        
        cshape = self.ishape[:-1] + (1, )
        for c in range(len(C)):
            C[c] = C[c].reshape(cshape)
            C[c]._keras_shape = cshape

        return K.concatenate(
            [K.conv1d(C[c], self.one, strides=self.pool_size)
             for c in range(len(C))], axis=-1)


class Upsample1D(Layer):
    def __init__(self, size=2, **kwargs):
        super(Upsample1D, self).__init__(**kwargs)
        self.size  = size
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.size * input_shape[1], input_shape[2])
    
    def build(self, input_shape):
        shape = (input_shape[0], input_shape[1], (self.size - 1) * input_shape[2])
        z = np.zeros(shape=shape)
        
        self.Z = K.constant(z)
        self.Z._keras_shape = shape
        
    def call(self, inpt):
        out_shape = self.compute_output_shape(inpt.shape)
        R = K.concatenate((inpt, self.Z), axis=-1)
        return R.reshape(out_shape)


class ModulusMaximaPool1D(MaxPool1D):
    def __init__(self, pool_size=2, **kwargs):
        super(MaxPool1D, self).__init__(pool_size, None,
                                        'valid', 'channels_last',
                                        **kwargs)

    def _pooling_function(self, inputs, pool_size, strides,
                          padding, data_format):
        E = K.sum(inputs ** 2, axis=-1, keepdims=True)
        x = K.concatenate([inputs, K.sqrt(E)], axis=-1)
        output = K.pool2d(x, pool_size, strides, padding, data_format, pool_mode='max')
        return output[:, :, :, :-1]


class EnergyMaximaPool1D(MaxPool1D):
    def __init__(self, pool_size=2, **kwargs):
        super(MaxPool1D, self).__init__(pool_size, None,
                                        'valid', 'channels_last',
                                        **kwargs)

    def _pooling_function(self, inputs, pool_size, strides,
                          padding, data_format):
        E = K.sum(inputs ** 2, axis=-1, keepdims=True)
        x = K.concatenate([inputs, E], axis=-1)
        output = K.pool2d(x, pool_size, strides, padding, data_format, pool_mode='max')
        return output[:, :, :, :-1]
