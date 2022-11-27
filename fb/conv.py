import scipy.signal

import numpy as np

import keras.backend as K
import theano.tensor as T

from fb.utils import *

from complexnn.conv import ComplexConv1D

from keras import initializers, regularizers
from keras.layers import Layer, InputSpec
from keras.utils import conv_utils


class ConvSimple(ComplexConv1D):
    def __init__(self, kernel_size, kernel_regularizer, filters=2, dilation_rate=1):
        super(ConvSimple, self).__init__(padding='causal', filters=filters, 
                                         kernel_size=kernel_size, 
                                         dilation_rate=dilation_rate, 
                                         kernel_regularizer=kernel_regularizer,
                                         kernel_initializer='complex',
                                         use_bias=False)

class Conv(Layer):
    @staticmethod
    def perform_complex(S, F, strides=1, padding='causal', dilation_rate=1):
        ishape = S._keras_shape
        fshape = F._keras_shape

        f = fshape[-1] // 2
        
        F_real = F[:, :, : f]
        F_imag = F[:, :, f :    ]

        F_real._keras_shape = fshape[:-1] + (f, )
        F_imag._keras_shape = fshape[:-1] + (f, )

        F_m_real    = K.concatenate([F_real, -F_imag], axis=-2)
        F_m_imag    = K.concatenate([F_imag,  F_real], axis=-2)
        F_m_complex = K.concatenate([F_m_real, F_m_imag], axis=-1)
        F_m_complex._keras_shape = (fshape[0], ishape[-1], 2 * f)

        return K.conv1d(S, F_m_complex, strides=strides, padding=padding, 
                        dilation_rate=dilation_rate, data_format='channels_last')

    @staticmethod
    def perform_real(S, F, strides=1, padding='causal', dilation_rate=1):
        fs = S._keras_shape[-1]

        freqs = [S[:, :, f] for f in range(fs)]
        for f in range(fs):
            freqs[f]._keras_shape = S._keras_shape[:-1] + (1, )
            freqs[f] = freqs[f].reshape(freqs[f]._keras_shape)

        channels = [K.conv1d(freqs[f], F, strides=strides, padding=padding,
                             dilation_rate=dilation_rate, data_format='channels_last')
                             for f in range(fs)]
        return K.concatenate(channels, axis=-1)

    @staticmethod
    def genfir(ir, type):
        w = 0 if type == 'LF' else np.pi
        n = np.arange(0, taps)

        return ir.reshape(taps, 1) * np.exp(1j * w * n)

    def __init__(self, kernel_initializer, kernel_regularizer, **kwargs):
        super(Conv, self).__init__(**kwargs)
        
        self.rank = 1

        self.filters = 2

        self.padding = conv_utils.normalize_padding('causal')

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

    def test_ir(self, taps): 
        ir = scipy.signal.firwin(
                numtaps=taps, cutoff=0.1, pass_zero="lowpass", 
                window='kaiser', width=0.001
             ).astype('float32').reshape(taps, 1, 1)
        kir = K.variable(ir)
        kir._keras_shape = (taps, 1, 1)
        return kir

    def add_window(self, kernel_size, name):
        win = self.add_weight(
            shape=(kernel_size, 1, 1), name=name,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer
        )
        win._keras_shape = (kernel_size, 1, 1)
        return T.addbroadcast(win, 1, 2)
    
    def add_c_window(self, kernel_size, name):
        win = self.add_weight(
            shape=(kernel_size, 1, 2), name=name,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer
        )
        win._keras_shape = (kernel_size, 1, 2)
        return win

    def convolve_single(self, S, win):
        return self.perform_real(S, win, dilation_rate=self.dilation_rate,
                                 padding=self.padding)

    def convolve_multiple(self, S, win):
        return self.perform_complex(S, win, dilation_rate=self.dilation_rate,
                                    padding=self.padding)

# ----------------------------------------------------------------------------------------------------
# Freq shifting class
# ----------------------------------------------------------------------------------------------------
class ConvShift(Conv):
    def build(self, input_shape):
        w = np.linspace(0, 2 * np.pi, self.filters, endpoint=False)
        n = np.arange(0, input_shape[1])
        E = np.zeros(shape=(input_shape[0], input_shape[1], self.filters * 2), dtype=np.float32)

        m = self.filters
        for k in range(m):
            e = np.exp(1j * w[k] * n)
            E[:, :,     k] = np.real(e)
            E[:, :, m + k] = np.zeros_like(np.imag(e))

        self.E = K.constant(E)
        self.E._keras_shape = E.shape


# ----------------------------------------------------------------------------------------------------
#  Direct filtering class
# ----------------------------------------------------------------------------------------------------
class ConvDirect(Conv):
    def build(self, input_shape):
        w = np.linspace(0, 2 * np.pi, self.filters, endpoint=False)
        n = np.arange(0, self.kernel_size)
        E = np.zeros(shape=(self.kernel_size, 1, self.filters * 2), dtype=np.float32)

        m = self.filters
        for k in range(self.filters):
            e = np.exp(1j * w[k] * n)
            E[:, 0, k]     = np.real(e)
            E[:, 0, m + k] = np.zeros_like(np.imag(e))

        self.E = K.constant(E)
        self.E._keras_shape = E.shape


# ----------------------------------------------------------------------------------------------------
#  Window shared class
# ----------------------------------------------------------------------------------------------------
class ConvShared(Conv):
    def build(self, input_shape):
        #self.win = self.test_ir(self.kernel_size)
        self.win = self.add_window(self.kernel_size, name='win_s')
        self.win._keras_shape = (self.kernel_size, 1, 1)

    def firs(self):
        ir = self.get_weights( )[0]
        return [self.genfir(ir, 'LF'), self.genfir(ir, 'HF')]


# ----------------------------------------------------------------------------------------------------
#  Separated windows class
# ----------------------------------------------------------------------------------------------------
class ConvSeparate(Conv):
    def build(self, input_shape):
        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size, self.kernel_size)

        self.win = [
            #self.test_ir(self.kernel_size[0]),
            #self.test_ir(self.kernel_size[1])
            self.add_window(self.kernel_size[0], name='win_lf'),
            self.add_window(self.kernel_size[1], name='win_hf')
        ]
        self.win[0]._keras_shape = (self.kernel_size[0], 1, 1)
        self.win[1]._keras_shape = (self.kernel_size[1], 1, 1)

    def firs(self):
        return [self.genfir(self.get_weights( )[0], 'LF'), 
                self.genfir(self.get_weights( )[1], 'HF')]