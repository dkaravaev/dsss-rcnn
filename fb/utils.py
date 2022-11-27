import numpy as np

import keras.backend as K
import theano.tensor as T

from keras import initializers, regularizers
from keras.layers import Layer, InputSpec


def fresponse(h, num):
    w = np.linspace(-np.pi, np.pi, num=num, endpoint=True)
    H = np.zeros(shape=(w.shape[0], ), dtype=np.complex128)
    for n in range(h.shape[0]):
        H += h[n] * np.exp(1j * w * n)
    return H

def plot_fresponse(irs):    
    for ir in irs:
        plt.plot(w, 20 * np.log10(np.abs(fresponse(w, ir))))
    plt.show()


def get_realpart(x):
    image_format = K.image_data_format()
    ndim = K.ndim(x)
    input_shape = K.shape(x)

    if (image_format == 'channels_first' and ndim != 3) or ndim == 2:
        input_dim = input_shape[1] // 2
        return x[:, :input_dim]

    input_dim = input_shape[-1] // 2
    if ndim == 3:
        return x[:, :, :input_dim]
    elif ndim == 4:
        return x[:, :, :, :input_dim]
    elif ndim == 5:
        return x[:, :, :, :, :input_dim]


def get_imagpart(x):
    image_format = K.image_data_format()
    ndim = K.ndim(x)
    input_shape = K.shape(x)

    if (image_format == 'channels_first' and ndim != 3) or ndim == 2:
        input_dim = input_shape[1] // 2
        return x[:, input_dim:]

    input_dim = input_shape[-1] // 2
    if ndim == 3:
        return x[:, :, input_dim:]
    elif ndim == 4:
        return x[:, :, :, input_dim:]
    elif ndim == 5:
        return x[:, :, :, :, input_dim:]


def complex_prod_const(x, y):
    xshape = K.int_shape(x)
    yshape = K.int_shape(y)

    cx = xshape[-1] // 2
    cy = yshape[-1] // 2

    xr, xi = x[:, :, : cx], x[:, :, cx:]
    yr, yi = y[:, :, : cy], y[:, :, cy:]

    if cx == 1:
        xr = T.patternbroadcast(xr, (False, False, True))
        xi = T.patternbroadcast(xi, (False, False, True))

    if cy == 1:
        yr = T.patternbroadcast(yr, (True, True, True))
        yi = T.patternbroadcast(yi, (True, True, True))

    return K.concatenate([xr * yr - xi * yi, xi * yr + xr * yi], axis=-1)

def complex_prod(x, y):
    xshape = K.int_shape(x)
    yshape = K.int_shape(y)

    cx = xshape[-1] // 2
    cy = yshape[-1] // 2

    xr, xi = x[:, :, : cx], x[:, :, cx:]
    yr, yi = y[:, :, : cy], y[:, :, cy:]

    if cx == 1:
        xr = T.patternbroadcast(xr, (False, False, True))
        xi = T.patternbroadcast(xi, (False, False, True))

    if cy == 1:
        yr = T.patternbroadcast(yr, (False, False, True))
        yi = T.patternbroadcast(yi, (False, False, True))

    return K.concatenate([xr * yr - xi * yi, xi * yr + xr * yi], axis=-1)

def lf_channel(signal):
    shape = K.shape(signal)
    return K.concatenate([K.reshape(signal[:, :, 0], (shape[0], shape[1], 1)),
                          K.reshape(signal[:, :, 2], (shape[0], shape[1], 1))], axis=-1)


def hf_channel(signal):
    shape = K.shape(signal)
    return K.concatenate([K.reshape(signal[:, :, 1], (shape[0], shape[1], 1)),
                          K.reshape(signal[:, :, 3], (shape[0], shape[1], 1))], axis=-1)


def freq_split(signal):
    return lf_channel(signal), hf_channel(signal)


def concatenate_two(signal0, signal1):
    real0 = get_realpart(signal0)
    imag0 = get_imagpart(signal0)

    real1 = get_realpart(signal1)
    imag1 = get_imagpart(signal1)

    return K.concatenate([real0, real1, imag0, imag1], axis=-1)


def concatenate_many(signals):
    result = concatenate_two(signals[0], signals[1])
    for i in range(2, len(signals)):
        result = concatenate_two(result, signals[i])
    return result


def sum_channels(signal):
    shape = K.shape(signal)
    
    c = shape[-1] // 2
    
    real = K.sum(signal[:, :, :c], axis=-1)
    imag = K.sum(signal[:, :, c:], axis=-1)

    real = K.reshape(real, shape=(shape[0], shape[1], 1))
    imag = K.reshape(imag, shape=(shape[0], shape[1], 1))

    return K.concatenate([real, imag], axis=-1)

def complex_sum(x, y):
    xshape = K.int_shape(x)
    yshape = K.int_shape(y)



    xr, xi = x[:, :, : cx], x[:, :, cx:]
    yr, yi = y[:, :, : cy], y[:, :, cy:]

    if cx == 1:
        xr = T.patternbroadcast(xr, (False, False, True))
        xi = T.patternbroadcast(xi, (False, False, True))

    if cy == 1:
        yr = T.patternbroadcast(yr, (False, False, True))
        yi = T.patternbroadcast(yi, (False, False, True))

    return K.concatenate([xr + yr, xi + yi], axis=-1)


class FactorLayer(Layer):
    def __init__(self, regularizer, initializer='glorot_normal', **kwargs):
        super(FactorLayer, self).__init__(**kwargs)

        self.regularizer = regularizers.get(regularizer)
        self.initializer = initializers.get(initializer)
        
    def build(self, input_shape):
        channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        channels = input_shape[channel_axis] // 2
        if channels != 2:
            raise ValueError('Only two channel inputs are supported!')

        self.beta = [
            self.add_weight(
                shape=(1, 1, 2), name='beta_lf',
                initializer=self.initializer,
                regularizer=self.regularizer
            ),
            self.add_weight(
                shape=(1, 1, 2), name='beta_hf',
                initializer=self.initializer,
                regularizer=self.regularizer
            )
        ]
        self.beta[0]._keras_shape = (1, 1, 2)
        self.beta[1]._keras_shape = (1, 1, 2)

        self.build = True

    def call(self, S):
        S0, S1 = freq_split(S)
        return concatenate_two(complex_prod_const(S0, self.beta[0]), complex_prod_const(S1, self.beta[1]))