#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Authors: Olexa Bilaniuk
#
# What this module includes by default:
from complexnn 		       import bn, conv, dense, fft, init, norm, pool

from complexnn.bn          import ComplexBatchNormalization as ComplexBN
from complexnn.conv        import (ComplexConv,
                                   ComplexConv1D,
                                   ComplexConv2D,
                                   ComplexConv3D,
                                   WeightNorm_Conv)
from complexnn.dense       import ComplexDense
from complexnn.fft         import fft, ifft, fft2, ifft2, FFT, IFFT, FFT2, IFFT2
from complexnn.init        import (ComplexIndependentFilters, IndependentFilters,
                                   ComplexInit, SqrtInit)
from complexnn.norm        import LayerNormalization, ComplexLayerNorm
from complexnn.pool        import SpectralPooling1D, SpectralPooling2D
from complexnn.utils       import (get_realpart, get_imagpart, getpart_output_shape,
                                   GetImag, GetReal, GetAbs, ComplexConcatenate, GetChannel0, GetChannels)
from complexnn.activations import modReLU, zReLU, CReLU

keras_custom_objects = {
    'ComplexConv1D': ComplexConv1D,
    'modReLU': modReLU,
    'CReLU': CReLU,
    'ComplexConcatenate': ComplexConcatenate,
    'ComplexBatchNormalization' : ComplexBN,
    'ComplexBN' : ComplexBN,
    'GetChannel0': GetChannel0,
    'GetChannels': GetChannels
}