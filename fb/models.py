import numpy as np

import scipy.signal as ss

from keras import models
from keras import layers

from complexnn import keras_custom_objects

from complexnn.activations  import modReLU
from complexnn.conv         import ComplexConv1D
from complexnn.utils        import *

from datasets.utils  import *

from fb.conv      import *
from fb.analysis  import *
from fb.synthesis import *
from fb.resample  import *
from fb.shaping   import *
from fb.bn        import ComplexProperBatchNormalization as ComplexBN


class Block:
    def __init__(self, regularizer, downsample='strided', use_factor=False):
        self.regularizer = regularizer
        self.activate = None 
        self.use_factor = use_factor

        if downsample not in ['strided', 'dilated']:
            raise ValueError("Unknown downsample type! Allowed: ['strided', 'dilated']")
        
        self.downsample = downsample
        
    def set_activation(self, layer, **kwargs):
        self.activate = layer
        self.ackwargs = kwargs
    def set_analysis(self, layer):
        self.afilter  = layer 

    def set_synthesis(self, layer):
        self.sfilter  = layer

    def analysis_step(self, signal, taps, downfactor):
        afilter = None
        if self.downsample == 'dilated':
            afilter = self.afilter(
                kernel_size=taps, dilation_rate=1,
                kernel_regularizer=self.regularizer
            )
        else:
            afilter = self.afilter(
                kernel_size=taps, kernel_regularizer=self.regularizer
            )
        
        channels = afilter(signal)
        if self.use_factor:
            channels = FactorLayer(regularizer=self.regularizer)(channels)
        if self.downsample == 'strided':
            channels = Downsample1D(pool_size=2)(channels)
        #channels = ComplexBN(center=False)(channels)
        channels = self.activate(**self.ackwargs)(channels)
        
        lf, hf = LFChannel()(channels), HFChannel()(channels)

        return lf, hf, afilter.name

    def synthesis_step(self, channels, taps):
        sfilter = self.sfilter(kernel_size=taps, kernel_regularizer=self.regularizer)

        concat = ComplexConcatenate()(channels)
        
        signal = Upsample1D(size=2)(concat)
        if self.use_factor:
            signal = FactorLayer(regularizer=self.regularizer)(signal)
        signal = sfilter(signal)
        signal = ComplexBN(center=False)(signal)
        signal = self.activate(**self.ackwargs)(signal)

        return signal, sfilter.name


class WaveletNet:
    def __init__(self, N, B, taps, regularizer, comp='symmetric', use_factor=False, L=None):
        self.N = N
        self.B = B

        self.L = int(np.floor(np.log2(N)))
        if L is not None:
            self.L = np.min([self.L, L])

        if not isinstance(taps, list):
            self.taps = [taps] * L
        else:
            self.taps = taps
        if len(self.taps) != self.L:
            raise ValueError("List of filter taps must have the same length"
                             "as decomposition level - {}" % self.L)

        if comp not in ['symmetric', 'simple']:
            raise ValueError("Unknown decompostion type! Allowed: ['symmetric', 'simple']")
        
        downsample = 'dilated' if comp == 'simple' else 'strided'
        self.downfactor = [1] + [2] * (self.L - 1)
        
        self.comp = comp
        
        self.regularizer = regularizer
        
        self.block = Block(regularizer, use_factor=use_factor, downsample=downsample)
        self.model = None

    def set_activation(self, layer, **kwargs):
        self.block.set_activation(layer, **kwargs)

    def set_analysis(self, layer):
        self.block.set_analysis(layer)

    def set_synthesis(self, layer):
        self.block.set_synthesis(layer)

    def build(self):
        pass

    def compile(self, optimizer, loss, metrics=[]):
        self.model.compile(optimizer, loss, metrics=[loss] + metrics)

    def train(self, X_train, y_train, X_test, y_test, epochs=20, batch_size=100):
        return self.model.fit(X_train, y_train, batch_size=batch_size,
                              epochs=epochs, verbose=True, validation_data=(X_test, y_test))

    def evaluate(self, X, y):
        return self.model.evaluate(X, y)

    def predict(self, X, mode='full'):
        p = self.model.predict(X, batch_size=self.B)
        ret = tocomplex(np.vstack((p[0, :, :], p[1:, -1, :])))
        
        if mode == 'valid':
            return ret[self.taps[0] * self.taps[-1] - 1: ]
        
        return ret

    def layer(self, name):
        return self.model.get_layer(name)

    def load(self, config, weights):
        with open(config, 'r') as f:
            structure = f.read()
        self.model = models.model_from_json(structure, custom_objects=keras_custom_objects)
        self.model.load_weights(weights)

    def save(self, config, weights):
        with open(config, 'w') as f:
            f.write(self.model.to_json())
        self.model.save_weights(weights)
    
    def comp_simple(self, channels):
        signal = self.block.sfilter(
            kernel_regularizer=self.regularizer
        )(ComplexConcatenate()(channels))
        return signal
    
    def summary(self):
        return self.model.summary()

    def generate_harms(self, num=2048):
        W = np.linspace(-np.pi, np.pi, num=num, endpoint=True)
        E = np.zeros((W.shape[0], self.N, 2), dtype=np.float32)
        n = np.arange(0, self.N)

        for i, w in enumerate(W):
            E[i] = tochanneled(np.exp(1j * w * n))
        
        return E
        
    @staticmethod
    def compute_afr(F):
        ret = np.zeros(shape=(F.shape[0], ))
        for f in range(F.shape[0]):
            A = np.abs(tocomplex(F[f, 200:]))
            ret[f] = A[np.argmax(np.abs(A))]

        return ret
    
    @property
    def total_afr(self):
        E = self.generate_harms()
        F = self.model.predict(E, batch_size=self.B)

        return self.compute_afr(F)

    def run_block(self, i, S):
        C = self.model.get_layer("conv_simple_" + str(i))
        B = self.model.get_layer("complex_proper_batch_normalization_"  + str(i))
        L = self.model.get_layer("lf_channel_"  + str(i))
        H = self.model.get_layer("hf_channel_"  + str(i))

        I = K.placeholder(shape=S.shape)

        F = B(C(I))
        F0, F1 = L(F), H(F)

        block_function = K.function([I, K.learning_phase()], [F0, F1])
        return block_function([S, 0])
    
    def block_afr(self, i):
        F0, F1 = self.run_block(i, self.generate_harms())

        return self.compute_afr(F0), self.compute_afr(F1)

    def IR(self, taps=None):
        if taps is None:
            taps = self.taps[-1] * self.taps[0]
 
        D = np.zeros(shape=(self.B, self.N, 2))
        D[:, 0, 0] = 1.0
        return tocomplex(self.model.predict(D, batch_size=self.B)[0, :taps, :])
    

class DummyNet(WaveletNet):
    def build(self):
        input = layers.Input(shape=(self.N, 2), batch_shape=(self.B, self.N, 2))
        output = ConvSimple(kernel_size=self.taps[0], filters=1, kernel_regularizer=self.regularizer)(input)
        for taps in self.taps[1:]:
            output = ConvSimple(kernel_size=taps, filters=1, kernel_regularizer=self.regularizer)(output)
        self.model = models.Model(inputs=input, outputs=output)        


class DyadicNet(WaveletNet):
    def build(self):
        self.anames = []
        self.snames = []

        nodes = []

        input = layers.Input(shape=(self.N, 2), batch_shape=(self.B, self.N, 2))

        # Analysis
        signal = input
        for i in range(self.L):
            signal, channels, name = self.block.analysis_step(signal, self.taps[i], self.downfactor[i])
            nodes.append(channels)
            self.anames.append(name)
        nodes.append(signal)
        
        # Synthesis
        output = None
        if self.comp == 'simple':
            output = self.comp_simple(nodes)
        else:    
            nodes.reverse( )
            for i in range(self.L):
                signal, name = self.block.synthesis_step(nodes[:2], self.taps[-i - 1])
                nodes.pop(0), nodes.pop(0)
                nodes.insert(0, signal)
                self.snames.append(name)
            output = nodes[0]

        self.model = models.Model(inputs=input, outputs=output)


class PacketNet(WaveletNet):
    class Filters:
        name  = ''
        left  = None
        right = None

        def __init__(self, name, left, right):
            self.name  = name
            self.left  = left
            self.right = right

    def analysis_step(self, signal, l):
        lf, hf, name = self.block.analysis_step(signal, self.taps[l - 1], self.downfactor[l - 1])
        if l == self.L:
            return [lf, hf]#, PacketNet.Filters(name, None, None)

        lfs = self.analysis_step(lf, l + 1)
        hfs = self.analysis_step(hf, l + 1)

        return lfs + hfs#, PacketNet.Filters(name, lnames, rnames)

    def synthesis_step(self, nodes):
        l = self.L
        while len(nodes) != 1:
            n = len(nodes) // 2
            next = []

            for i in range(n):
                signal, name = self.block.synthesis_step(nodes[:2], self.taps[l - 1])
                nodes.pop(0), nodes.pop(0)
                
                next.append(signal)
                #self.sfilters.append(name)

            nodes = next
            l -= 1
                
        return nodes[0]


    def build(self):
        input  = layers.Input(shape=(self.N, 2), batch_shape=(self.B, self.N, 2))
        channels = self.analysis_step(input, 1)
        output = self.comp_simple(channels) if self.comp == 'simple' else self.synthesis_step(channels)
        
        self.model = models.Model(inputs=input, outputs=output)
