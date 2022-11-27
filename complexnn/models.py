import numpy as np
import matplotlib.pyplot as plt

from keras import models
from keras import layers

from complexnn import keras_custom_objects

from complexnn.conv         import ComplexConv1D, ConvFB
from complexnn.activations  import modReLU
from complexnn.bn           import ComplexBatchNormalization
from complexnn.utils        import *


def fresponse(w, h) :
    H = np.zeros(shape=(w.shape[0], ), dtype=np.complex128)
    for n in range(h.shape[0]):
        H += h[n] * np.exp(1j * w * n)
    return H

def plot_fresponse(h):
    w = np.linspace(-np.pi, np.pi, 3000)
    plt.plot(w, np.abs(fresponse(w, h)))
    plt.show()


class WCANN:
    @staticmethod
    def block(signal, filters, dilation_rate, kernel_size, 
              kernel_regularizer, activation, **kwargs):
        FilterLayer = ConvFB(filters=filters, 
                             kernel_size=kernel_size, 
                             kernel_regularizer=kernel_regularizer, 
                             dilation_rate=dilation_rate)
        channels = FilterLayer(signal)
        
        channels = ComplexBatchNormalization()(channels)

        channels = activation(**kwargs)(channels)

        return GetChannel0()(channels), GetChannels()(channels), FilterLayer.name

    @property
    def decomposition_level(self):
        return self.model.layers[-2].output_shape[-1] // 2

    def __init__(self):
        self.model = None
        self.names = None
        
    def compile(self, optimizer, loss):
        self.model.compile(optimizer, loss, metrics=[loss])

    def train(self, X_train, y_train, X_test, y_test, epochs=20, batch_size=100):
        return self.model.fit(X_train, y_train, batch_size=batch_size,
                              epochs=epochs, verbose=True, validation_data=(X_test, y_test))

    def evaluate(self, X, y):
        return self.model.evaluate(X, y)

    def predict(self, X, use_history=True):
        p = self.model.predict(X)
        if use_history:
            return np.vstack((X[0], p[:, -1, :]))
        return p[:, -1, :]

    def recursive_predict(self, series, steps=100, use_history=True):
        ret = series
        for step in range(steps):
            p = self.model.predict(ret[np.newaxis, step:])
            if step == 0 and not use_history:
                ret = np.vstack((ret[0], p[0]))
            else:
                ret = np.vstack((ret, p[0, -1, :]))
        return ret

    def load(self, config, weights):
        with open(config, 'r') as f:
            structure = f.read()
        self.model = models.model_from_json(structure, custom_objects=keras_custom_objects)
        self.model.load_weights(weights)

    def save(self, config, weights):
        with open(config, 'w') as f:
            f.write(self.model.to_json())
        self.model.save_weights(weights)

    def impulse_responses(self, layer):
        fs = self.model.get_layer(name=layer).get_weights( )[0].squeeze()
        result = []

        fnum = fs.shape[-1] // 2

        if len(fs.shape) > 2:
            f_real = fs[:, :, : fnum]
            f_imag = fs[:, :, fnum :]
            for i in range(fnum):
                for j in range(fnum):
                    result.append(fs[:, i, j] + 1j * fs[:, i, fnum + j])
        else:
            f_real = fs[:, : fnum]
            f_imag = fs[:, fnum :]
            for i in range(fnum):
                result.append(fs[:, i] + 1j * fs[:, fnum + i])

        return result

    def decompose(self, signal, **kwargs):
        pass

    def build(self, N, depth, filters=2, taps=4, regularizer=None, 
              activation=modReLU, **kwargs):
        self.activation = activation

        self.depth = depth
        if type(filters) != list:
            self.filters = depth * [filters]
        else:
            self.filters = filters

        if type(taps) != list:
            self.taps = depth * [taps]
        else:
            self.taps = taps

        self.regularizer = regularizer

        signal = layers.Input(shape=(N, 2))

        pred = ComplexConv1D(filters=1, kernel_size=1, kernel_regularizer=self.regularizer, 
                             kernel_initializer='complex', activation='linear',
                             use_bias=False)(self.decompose(signal, **kwargs))

        self.model = models.Model(inputs=signal, outputs=pred)


class WaveletNet(WCANN):
    def decompose(self, signal, **kwargs):
        self.names = []

        nodes = []
        ch0, channels, name = WCANN.block(signal, filters=self.filters[0], dilation_rate=1, 
                                          kernel_size=self.taps[0], kernel_regularizer=self.regularizer,
                                          activation=self.activation, **kwargs)
        nodes.append(channels)
        self.names.append(name)

        for i in range(1, self.depth):
            ch0, channels, name = WCANN.block(ch0, filters=self.filters[i], dilation_rate=2, 
                                              kernel_size=self.taps[i], kernel_regularizer=self.regularizer, 
                                              activation=self.activation, **kwargs)
            nodes.append(channels)
            self.names.append(name)

        nodes.append(ch0)

        return ComplexConcatenate()(nodes)


class PacketNet(WCANN):
    class Filters:
        name  = ''
        left  = None
        right = None

        def __init__(self, name, left, right):
            self.name  = name
            self.left  = left
            self.right = right

    def build(self, N, depth, taps=4, regularizer=None, activation=modReLU, **kwargs):
        return super(PacketNet, self).build(N, depth, 2, taps, regularizer, activation, **kwargs)

    def __build_tree(self, signal, depth, **kwargs):
        if depth == 0:
            return [], None

        i = self.depth - depth
        ch0, ch1, name = WCANN.block(signal, filters=2, dilation_rate=2, 
                                     kernel_size=self.taps[i], kernel_regularizer=self.regularizer, 
                                     activation=self.activation, **kwargs)
        if depth == 1:
            return [ch0, ch1], PacketNet.Filters(name, None, None)
        else:
            left_decomp,  left_names  = self.__build_tree(ch0, depth - 1, **kwargs)
            right_decomp, right_names = self.__build_tree(ch1, depth - 1, **kwargs)

            return left_decomp + right_decomp, PacketNet.Filters(name, left_names, right_names)

    def decompose(self, signal, **kwargs):
        ch0, ch1, name = WCANN.block(signal, filters=2, dilation_rate=1, 
                                     kernel_size=self.taps[0], kernel_regularizer=self.regularizer,
                                     activation=self.activation, **kwargs)
        
        left_decomp,  left_names  = self.__build_tree(ch0, self.depth - 1, **kwargs)
        right_decomp, right_names = self.__build_tree(ch1, self.depth - 1, **kwargs)

        self.names = PacketNet.Filters(name, left_names, right_names)

        return ComplexConcatenate()(left_decomp + right_decomp)
        
