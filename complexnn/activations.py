import keras.backend as K
import theano.tensor as T

from keras.layers import Layer, Lambda, ReLU
from keras import initializers
from keras import regularizers

from complexnn.utils import *


class modReLU(Layer):
    def __init__(self, b_initializer='zeros',
                 b_regularizer=None,
                 **kwargs):
        super(modReLU, self).__init__(**kwargs)
        self.b = None
        
        self.b_initializer = initializers.get(b_initializer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.broadcast = (True, True, False)
    
    def build(self, input_shape):
        shape = (1, 1, getpart_output_shape(input_shape)[-1])
        if self.b is None:
            self.b = self.add_weight(shape=shape, name='b', initializer=self.b_initializer, 
                                     regularizer=self.b_regularizer)
        else:
            self.b = T.as_tensor_variable(self.b.reshape(shape))
        
    def call(self, inputs):
        mod = get_abs(inputs)
        
        if K.backend() == 'theano':
            s = mod + K.pattern_broadcast(self.b, self.broadcast)
        else:
            s = mod + self.b
        
        rs = K.relu(s)
        real = rs * get_realpart(inputs) / mod
        imag = rs * get_imagpart(inputs) / mod
        
        return K.concatenate((real, imag), axis=-1)

 #   def get_config(self):
 #       return {
 #           'b_initializer': self.b_initializer,
 #           'b_regularizer': self.b_regularizer
 #       }


class zReLU(Layer):
    def __init__(self, **kwargs):
        super(zReLU, self).__init__(**kwargs)
        self.zeros = None

    def build(self, input_shape):
        self.zeros = T.zeros(shape=(input_shape[1], input_shape[2] // 2))

    def call(self, inputs):
        real = get_realpart(inputs)
        imag = get_imagpart(inputs)
        
        cond = T.and_(real >= 0, imag >= 0)
        x = T.where(cond, real, self.zeros)
        y = T.where(cond, imag, self.zeros)
        return K.concatenate((x, y), axis=-1)

class Linear(Layer):
    def call(self, inputs):
        return inputs


CReLU = ReLU
