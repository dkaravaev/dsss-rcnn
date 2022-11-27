#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Authors: Dmitriy Serdyuk, Olexa Bilaniuk, Chiheb Trabelsi

import keras.backend as K
import theano.tensor as T
from keras.layers import Layer, Lambda

#
# GetReal/GetImag Lambda layer Implementation
#


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


def get_abs(x):
    real = get_realpart(x)
    imag = get_imagpart(x)

    return K.sqrt(real * real + imag * imag)


def getpart_output_shape(input_shape):
    returned_shape = list(input_shape[:])
    image_format = K.image_data_format()
    ndim = len(returned_shape)

    if (image_format == 'channels_first' and ndim != 3) or ndim == 2:
        axis = 1
    else:
        axis = -1

    returned_shape[axis] = returned_shape[axis] // 2

    return tuple(returned_shape)


class GetReal(Layer):
    def call(self, inputs):
        return get_realpart(inputs)
    def compute_output_shape(self, input_shape):
        return getpart_output_shape(input_shape)
class GetImag(Layer):
    def call(self, inputs):
        return get_imagpart(inputs)
    def compute_output_shape(self, input_shape):
        return getpart_output_shape(input_shape)
class GetAbs(Layer):
    def call(self, inputs):
        return get_abs(inputs)
    def compute_output_shape(self, input_shape):
        return getpart_output_shape(input_shape)

def complex_mpl(x, y):
    xshape = K.int_shape(x)
    yshape = K.int_shape(y)
        
    cx = xshape[-1] // 2
    cy = yshape[-1] // 2
    
    xr, xi = x[:, :, : cx], x[:, :, cx : ]
    yr, yi = y[:, :, : cy], y[:, :, cy : ]

    if cx == 1:
        xr = T.patternbroadcast(xr, (False, False, True))
        xi = T.patternbroadcast(xi, (False, False, True))

    if cy == 1:
        yr = T.patternbroadcast(yr, (False, False, True))
        yi = T.patternbroadcast(yi, (False, False, True))

    return K.concatenate([xr * yr - xi * yi, xi * yr + xr * yi], axis=-1)


def get_channel0(signal):
    shape = K.shape(signal) 
    return K.concatenate([K.reshape(signal[:, :, 0], (shape[0], shape[1], 1)), 
                          K.reshape(signal[:, :, shape[-1] // 2], (shape[0], shape[1], 1))], axis=-1)

def get_channels(signal):
    shape = K.shape(signal)
    return K.concatenate([signal[:, :, 1 : shape[-1] // 2], 
                          signal[:, :, shape[-1] // 2 + 1 : ]], axis=-1)

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

class ComplexConcatenate(Layer):
    def call(self, inputs):
        return concatenate_many(inputs)
    
    def compute_output_shape(self, input_shapes):
        shape = list(input_shapes[0])
        for i in range(1, len(input_shapes)):
            shape[-1] += input_shapes[i][-1]
        return tuple(shape)

class GetChannel0(Layer):
    def call(self, inputs):
        return get_channel0(inputs)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], 2)
    
class GetChannels(Layer):
    def call(self, inputs):
        return get_channels(inputs)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2] - 2)
