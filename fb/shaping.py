import scipy.signal

import numpy as np

import keras.backend as K
import theano.tensor as T

from fb.utils import *

from keras import initializers, regularizers
from keras.layers import Layer, InputSpec
from keras.utils import conv_utils



class LFChannel(Layer):
    def call(self, inputs):
        return lf_channel(inputs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], 2)


class HFChannel(Layer):
    def call(self, inputs):
        return hf_channel(inputs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], 2)


class ComplexConcatenate(Layer):
    def call(self, inputs):
        return concatenate_many(inputs)

    def compute_output_shape(self, input_shapes):
        shape = list(input_shapes[0])
        for i in range(1, len(input_shapes)):
            shape[-1] += input_shapes[i][-1]
        return tuple(shape)
