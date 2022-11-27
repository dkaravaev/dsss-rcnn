import scipy.signal

import numpy as np

import keras.backend as K
import theano.tensor as T

from fb.utils import *
from fb.conv  import *

from keras import initializers, regularizers
from keras.layers import Layer, InputSpec
from keras.utils import conv_utils


class Sum(Layer):
    def __init__(self, kernel_regularizer, **kwargs):
        super(Sum, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
         return (input_shape[0], input_shape[1], 2)
        
    def call(self, S):
        return sum_channels(S)
    
class WeightedSum(ComplexConv1D):
    def __init__(self, kernel_regularizer, kernel_initializer='complex', **kwargs):
        super(WeightedSum, self).__init__(
            kernel_regularizer=kernel_regularizer, 
            kernel_initializer=kernel_initializer,
            activation='linear',
            kernel_size=1,
            filters=1,
            use_bias=False,
            **kwargs
        )

# ----------------------------------------------------------------------------------------------------
# Base class
# ----------------------------------------------------------------------------------------------------
class SynthesisBase(Conv):
    def __init__(self, kernel_size, kernel_initializer='glorot_normal', 
                 kernel_regularizer='l2', **kwargs):
        super(SynthesisBase, self).__init__(kernel_initializer, kernel_regularizer, **kwargs)

        self.kernel_size   = kernel_size

        self.filters       = 2
        self.strides       = 1
        self.dilation_rate = 1

    def compute_output_shape(self, input_shape): 
        return (input_shape[0], input_shape[1], 2)

    def build(self, input_shape):
        channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        channels = input_shape[channel_axis] // 2
        if self.filters != channels:
            raise ValueError('Number of input channels is not equals to number of defined '
                             'filters: filters/channels - {0}/{1}.' % self.filters, channels)

        self.input_spec = InputSpec(ndim=self.rank + 2, axes={channel_axis: input_shape[channel_axis]})
        self.build = True


# ----------------------------------------------------------------------------------------------------
#  Shift + Shared
# ----------------------------------------------------------------------------------------------------
class SynthesisSHSH(SynthesisBase, ConvShift, ConvShared):
    def build(self, input_shape):
        SynthesisBase.build(self, input_shape)
        ConvShift.build(self, input_shape)
        ConvShared.build(self, input_shape)

    def call(self, S):
        S._keras_shape = self.E._keras_shape

        C = self.convolve_single(S, self.win)
        C._keras_shape = S._keras_shape

        return sum_channels(complex_prod(C, self.E))


# ----------------------------------------------------------------------------------------------------
#  Shift + Separate
# ----------------------------------------------------------------------------------------------------
class SynthesisSHSE(SynthesisBase, ConvShift, ConvSeparate):
    def build(self, input_shape):
        SynthesisBase.build(self, input_shape)
        ConvShift.build(self, input_shape)
        ConvSeparate.build(self, input_shape)

    def call(self, S):
        S._keras_shape = self.E._keras_shape

        Sl, Sh = freq_split(S)
        Sl._keras_shape = S._keras_shape[:-1] + (2, )
        Sh._keras_shape = S._keras_shape[:-1] + (2, ) 

        C0 = self.convolve_single(Sl, self.win[0])
        C1 = self.convolve_single(Sh, self.win[1])
        C0._keras_shape = Sl._keras_shape
        C1._keras_shape = Sh._keras_shape

        C = concatenate_two(C0, C1)
        C._keras_shape = self.E._keras_shape

        return sum_channels(complex_prod(C, self.E))


# ----------------------------------------------------------------------------------------------------
#  Direct + Shared
# ----------------------------------------------------------------------------------------------------
class SynthesisDISH(SynthesisBase, ConvDirect, ConvShared):
    def build(self, input_shape):
        SynthesisBase.build(self, input_shape)
        ConvShared.build(self, input_shape)

        w = [0, np.pi]
        n = np.arange(0, self.kernel_size)
        E = np.zeros(shape=(self.kernel_size, self.filters, 2))

        m = self.filters
        for k in range(self.filters):
            e = np.exp(1j * w[k] * n)
            E[:, k, 0] = np.real(e)
            E[:, k, 1] = np.imag(e)

        self.E = K.constant(E)
        self.E._keras_shape = E.shape

    def call(self, S):
        F = self.win * self.E
        F._keras_shape = self.E._keras_shape
        return self.convolve_multiple(S, F)

# ----------------------------------------------------------------------------------------------------
#  Direct + Separate
# ----------------------------------------------------------------------------------------------------
class SynthesisDISE(SynthesisBase, ConvDirect, ConvSeparate):
    def build(self, input_shape):
        SynthesisBase.build(self, input_shape)
        ConvSeparate.build(self, input_shape)

        w = [0, np.pi]
        n = [
            np.arange(0, self.kernel_size[0]),
            np.arange(0, self.kernel_size[1])
        ]
        E = [
            np.zeros(shape=(self.kernel_size[0], 1, 2), dtype=np.float32),
            np.zeros(shape=(self.kernel_size[1], 1, 2), dtype=np.float32)
        ]
        for k in range(self.filters):
            e = np.exp(1j * w[k] * n[k])
            E[k][:, 0, 0] = np.real(e)
            E[k][:, 0, 1] = np.imag(e)

        self.E = [
            K.constant(E[0]), K.constant(E[1]) 
        ]
        self.E[0]._keras_shape = E[0].shape
        self.E[1]._keras_shape = E[1].shape

    def call(self, S):
        F = [self.win[k] * self.E[k] for k in range(self.filters)]
        F[0]._keras_shape = self.E[0]._keras_shape
        F[1]._keras_shape = self.E[1]._keras_shape

        Sl, Sh = freq_split(S)
        Sl._keras_shape = S._keras_shape[:-1] + (2, )
        Sh._keras_shape = S._keras_shape[:-1] + (2, )

        return sum_channels(concatenate_two(self.convolve_multiple(Sl, F[0]),
                                            self.convolve_multiple(Sh, F[1])))

class SynthesisCDISE(SynthesisBase, ConvDirect, ConvSeparate):
    def build(self, input_shape):
        SynthesisBase.build(self, input_shape)
        
        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size, self.kernel_size)
            
        self.win = [
            self.add_c_window(self.kernel_size[0], name='win_lf'),
            self.add_c_window(self.kernel_size[1], name='win_hf')
        ]
        self.win[0]._keras_shape = (self.kernel_size[0], 1, 2)
        self.win[1]._keras_shape = (self.kernel_size[1], 1, 2)

        w = [0, np.pi]
        n = [
            np.arange(0, self.kernel_size[0]),
            np.arange(0, self.kernel_size[1])
        ]
        E = [
            np.zeros(shape=(self.kernel_size[0], 1, 1), dtype=np.float32),
            np.zeros(shape=(self.kernel_size[1], 1, 1), dtype=np.float32)
        ]
        for k in range(self.filters):
            e = np.exp(1j * w[k] * n[k])
            E[k][:, 0, 0] = np.real(e)

        self.E = [
            K.constant(E[0]), K.constant(E[1]) 
        ]
        self.E[0] = T.addbroadcast(self.E[0], 1, 2)
        self.E[1] = T.addbroadcast(self.E[1], 1, 2)
        
        self.E[0]._keras_shape = E[0].shape
        self.E[1]._keras_shape = E[1].shape

    def call(self, S):
        F = [self.win[k] * self.E[k] for k in range(self.filters)]
        F[0]._keras_shape = self.win[0]._keras_shape
        F[1]._keras_shape = self.win[1]._keras_shape

        Sl, Sh = freq_split(S)
        Sl._keras_shape = S._keras_shape[:-1] + (2, )
        Sh._keras_shape = S._keras_shape[:-1] + (2, )

        return sum_channels(concatenate_two(self.convolve_multiple(Sl, F[0]),
                                            self.convolve_multiple(Sh, F[1])))