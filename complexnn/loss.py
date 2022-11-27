import numpy as np

import scipy.signal as ss

import keras.backend as K

import theano.tensor as T
import theano.tensor.fft as TF
import theano.ifelse as TI

from keras import losses

    
def cmse(y_true, y_pred):
    return K.mean(K.square(y_pred[:, :, 0] - y_true[:, :, 0]) +
                  K.square(y_pred[:, :, 1] - y_true[:, :, 1]), axis=1)


def cmae(y_true, y_pred):
    return K.mean(K.sqrt(K.square(y_pred[:, :, 0] - y_true[:, :, 0]) +
                         K.square(y_pred[:, :, 1] - y_true[:, :, 1])), axis=1)


class FourierMSE:
    @staticmethod
    def fft(x):    
        z = K.permute_dimensions(x, (2, 0, 1))
        z = K.reshape(z, (x.shape[0] * x.shape[2], x.shape[1]))

        B      = z.shape[0] // 2
        L      = z.shape[1]
        C      = T.as_tensor_variable(np.asarray([[[1,-1]]], dtype=T.config.floatX))

        Zr, Zi = TF.rfft(z[:B], norm="ortho"), TF.rfft(z[B:], norm="ortho")
        isOdd  = T.eq(L%2, 1)
        Zr     = TI.ifelse(isOdd, T.concatenate([Zr, C * Zr[:,1:, ][:,::-1]], axis=1),
                                  T.concatenate([Zr, C * Zr[:,1:-1][:,::-1]], axis=1))
        Zi     = TI.ifelse(isOdd, T.concatenate([Zi, C * Zi[:,1:  ][:,::-1]], axis=1),
                                  T.concatenate([Zi, C * Zi[:,1:-1][:,::-1]], axis=1))
        Zi = (C * Zi)[:, :, ::-1]

        return Zr + Zi
    
    def __init__(self, N, window='hanning', **kwargs):
        self.__name__ = 'FourierMSE'
        
        win = ss.get_window(window, N, **kwargs)

        shape = (1, N, 1)
        self.win = K.constant(win, shape=shape)
        self.win = T.patternbroadcast(self.win, (True, False, True))
    
    def spectrum(self, x):
        return self.fft(self.win * x)
    
    def __call__(self, y_true, y_pred):
        return cmse(self.spectrum(y_true), self.spectrum(y_pred))
    
    

class WelchMSE: 
    @staticmethod
    def shift(x):
        return T.concatenate([x[x.shape[0] // 2:], x[:x.shape[0] // 2]])
    
    def __init__(self, N, window='hanning', **kwargs):
        self.__name__ = 'WelchMSE'
        
        win = ss.get_window(window, N, **kwargs)

        shape = (1, N, 1)
        self.win = K.constant(win.reshape(shape), shape=shape)
        self.win = K.pattern_broadcast(self.win, (True, False, True))
        
    def psd(self, x):
        X = FourierMSE.fft(x * self.win)
        return K.mean((X[:, :, 0] ** 2) + (X[:, :, 1] ** 2), axis=0)
    
    def __call__(self, y_true, y_pred):
        return losses.mse(self.psd(y_true), self.psd(y_pred))


    
class WelchSMSE(WelchMSE):
    @staticmethod
    def compute_snapshot(s, N, window='hanning', **kwargs):
        win = ss.get_window(window, N, **kwargs)
        
        n = s.shape[0] // N
        S = win * s[:n * N].reshape(n, N)
        
        F = np.fft.fft(S, norm='ortho', axis=-1)
        return np.mean(np.abs(F) ** 2, axis=0)
    
    def __init__(self, snasphot, window='hanning', **kwargs):
        super(WelchSMSE, self).__init__(snasphot.shape[0], window, **kwargs)
        self.__name__ = 'WelchSMSE'
        self.y_true = K.constant(snasphot)
    
    def __call__(self, y_true, y_pred):
        return losses.mse(self.y_true, self.psd(y_pred))
    
class WelchKL(WelchSMSE):    
    def __init__(self, snapshot, window='hanning', **kwargs):
        super(WelchKL, self).__init__(snapshot, window, **kwargs)
        self.__name__ = 'WelchKL'
        self.St = K.constant(snapshot)
        self.Pt = K.constant(snapshot / np.sum(snapshot))
    
    def __call__(self, y_true, y_pred):
        Pp = self.psd(y_pred) * self.St
        Pp /= K.sum(Pp)
        return K.sum(self.Pt * K.log(self.Pt / Pp))
    
class Weltropy: 
    @staticmethod
    def median(x):
        mid = x.shape[0] // 2
        return T.mean(T.sort(x)[mid - 1 : mid + 1])
    
    def __init__(self, N, R=1, window='hanning', **kwargs):
        self.__name__ = 'weltropy'
        self.backend = WelchMSE(N, window, **kwargs)
        
        Np = N - int(R * N)
        self.sides = (Np // 2, N - Np // 2)
        
    def __call__(self, y_true, y_pred):
        Pp = self.backend.psd(y_pred)
        Pt = self.backend.psd(y_true)
        
        mp, mt = self.median(Pp), self.median(Pt)
        
        Pp = WelchMSE.shift(Pp)[self.sides[0]:self.sides[1]]
        Pp /= K.sum(Pp)
        
        return K.sum(Pp * K.log(Pp)) + 0.5 * (mp - mt) ** 2
    
    
class CrossWelchMSE: 
    def __init__(self, N, window='hanning', **kwargs):
        self.__name__ = 'CrossWelchMSE'
        
        win = ss.get_window(window, N, **kwargs)

        shape = (1, N, 1)
        self.win = K.constant(win, shape=shape)
        self.win = T.patternbroadcast(self.win, (True, False, True))
        
    def compute(self, x, y):
        X = FourierMSE.fft(self.win * x)
        Y = FourierMSE.fft(y)
        R = X[:, :, 0] * Y[:, :, 0] + X[:, :, 1] * Y[:, :, 1]
        I = X[:, :, 1] * Y[:, :, 0] - X[:, :, 0] * Y[:, :, 1]
        
        R = R.reshape(R.shape + (1, ))
        I = R.reshape(I.shape + (1, ))
        
        return K.mean(K.concatenate([R, I], axis=-1), axis=0)
    
    def __call__(self, y_true, y_pred):
        return losses.mse(self.psd(y_true), self.psd(y_pred))