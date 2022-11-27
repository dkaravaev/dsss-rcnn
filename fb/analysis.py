import scipy.signal

import numpy as np

import keras.backend as K
import theano.tensor as T

from fb.utils import *
from fb.conv  import *

from keras import initializers, regularizers
from keras.layers import Layer, InputSpec
from keras.utils import conv_utils


# ----------------------------------------------------------------------------------------------------
# Base class
# ----------------------------------------------------------------------------------------------------
class AnalysisBase(Conv):
    def __init__(self, kernel_size, dilation_rate=1,
                 kernel_initializer='glorot_normal', 
                 kernel_regularizer='l2', **kwargs):
        super(AnalysisBase, self).__init__(kernel_initializer, kernel_regularizer, **kwargs)

        self.kernel_size   = kernel_size
        self.dilation_rate = dilation_rate

    def compute_output_shape(self, input_shape): 
        return (input_shape[0], input_shape[1], self.filters * 2)

    def build(self, input_shape):
        channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        channels = input_shape[channel_axis] // 2
        if channels != 1:
            raise ValueError('Only single channel inputs are supported!')


        self.input_spec = InputSpec(ndim=self.rank + 2, axes={channel_axis: 2})
        self.build = True

# ----------------------------------------------------------------------------------------------------
#  Shift + Shared
# ----------------------------------------------------------------------------------------------------
class AnalysisSHSH(AnalysisBase, ConvShift, ConvShared):
    def build(self, input_shape):
        AnalysisBase.build(self, input_shape)
        ConvShift.build(self, input_shape)
        ConvShared.build(self, input_shape)

    def call(self, S):
        S = complex_prod(S, self.E)
        S._keras_shape = self.E._keras_shape

        return self.convolve_single(S, self.win)


# ----------------------------------------------------------------------------------------------------
#  Shift + Separate
# ----------------------------------------------------------------------------------------------------
class AnalysisSHSE(AnalysisBase, ConvShift, ConvSeparate):
    def build(self, input_shape):
        AnalysisBase.build(self, input_shape)
        ConvShift.build(self, input_shape)
        ConvSeparate.build(self, input_shape)

    def call(self, S):
        S = complex_prod(S, self.E)
        S._keras_shape = self.E._keras_shape
        
        Sl, Sh = freq_split(S)
        Sl._keras_shape = S._keras_shape[:-1] + (2, )
        Sh._keras_shape = S._keras_shape[:-1] + (2, ) 

        return concatenate_two(self.convolve_single(Sl, self.win[0]),
                               self.convolve_single(Sh, self.win[1]))


# ----------------------------------------------------------------------------------------------------
#  Direct + Shared
# ----------------------------------------------------------------------------------------------------
class AnalysisDISH(AnalysisBase, ConvDirect, ConvShared):
    def build(self, input_shape):
        AnalysisBase.build(self, input_shape)
        ConvDirect.build(self, input_shape)
        ConvShared.build(self, input_shape)

    def call(self, S):
        F = self.win * self.E
        F._keras_shape = self.E._keras_shape

        return self.convolve_multiple(S, F)


# ----------------------------------------------------------------------------------------------------
#  Direct + Separate
# ----------------------------------------------------------------------------------------------------
class AnalysisDISE(AnalysisBase, ConvSeparate):
    def build(self, input_shape):
        AnalysisBase.build(self, input_shape)
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
        m = self.filters
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

        return concatenate_two(self.convolve_multiple(S, F[0]),
                               self.convolve_multiple(S, F[1]))
    
class AnalysisCDISE(AnalysisBase, ConvSeparate):
    def build(self, input_shape):
        AnalysisBase.build(self, input_shape)
        
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
        m = self.filters
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
        F = [self.E[k] * self.win[k] for k in range(self.filters)]
        F[0]._keras_shape = self.win[0]._keras_shape
        F[1]._keras_shape = self.win[1]._keras_shape

        return concatenate_two(self.convolve_multiple(S, F[0]),
                               self.convolve_multiple(S, F[1]))