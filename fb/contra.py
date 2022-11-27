import tqdm

import scipy.signal

import numpy as np


class Adapt:
    def __init__(self, M):
        self.M = M
        self.w = np.zeros(shape=(self.M, ), dtype=np.complex128)

    def predict(self, x):
        y = np.hstack((np.zeros(shape=(self.M - 1), dtype=x.dtype), x))
        return np.convolve(y, self.w, mode="valid")
    
    def reject(self, x, d):
        return d - self.predict(x)

    
class RLMS(Adapt):
    @staticmethod
    def herm(x):
        return x.conj().T 
    
    def __init__(self, M, lmd, delta=0.001):
        Adapt.__init__(self, M)
        self.P = 1 / delta * np.eye(M, dtype=np.complex128)
        
        self.gamma = (1 / lmd)
        
    def fit(self, x, d, verbose=False):
        xh = np.pad(x, (self.M, 0), 'constant', constant_values=0)
        
        w = self.w.reshape(self.M, 1)
        steps = x.shape[0]
        for n in tqdm.tqdm(range(steps)):
            u = xh[n : n + self.M]
            u = u.reshape(u.shape[0], 1)
            # k
            kappa = self.gamma * np.dot(self.P, u)
            k = kappa / (1 + np.dot(self.herm(u), kappa))
            # e
            e = d[n] - np.dot(self.herm(w), u)
            # w
            w += np.conj(e) * k
 
            # P
            self.P = self.gamma * self.P - self.gamma * np.dot(np.outer(k, self.herm(u)), self.P)
        self.w = w.squeeze()
            
class FBLMS(Adapt):
    @staticmethod
    def FFT(x):
        return np.fft.fft(x)

    @staticmethod
    def IFFT(x):
        return np.fft.ifft(x)
    
    def __init__(self, M, alpha, gamma, delta, CsdB=np.inf, verbose=False):
        Adapt.__init__(self, M) 

        self.verbose = verbose

        self.Cs = 10 ** (-CsdB / 10)
        
        self.M = M
        self.alpha = alpha
        self.gamma = gamma
        
        self.W = self.FFT(np.pad(self.w, (0, M), 'constant', constant_values=0))
        
        self.e   = np.zeros(shape=(2 * M, ), dtype=np.complex128)
        self.phi = np.zeros(shape=(2 * M, ), dtype=np.complex128)
        
        self.P   = np.zeros(shape=(2 * M, )) + delta
            
        self.mses = None
    
    def fit(self, x, d, verbose=False):
        xh = np.pad(x, (self.M, 0), 'constant', constant_values=0)
        k = xh.shape[0] // self.M
        
        self.mses = []
        for i in range(2, k):
            # Filtering
            u = xh[(i - 1) * self.M : (i + 1) * self.M]
            
            U = self.FFT(u)
            y = self.IFFT(U * self.W.T)[self.M : ]
            # Signal power-per-bin estimation
            self.P = self.gamma * self.P + (1 - self.gamma) * np.abs(U) ** 2
            
            # Error estimation
            self.e[ : self.M] = d[(i - 1) * self.M : i * self.M] - y
            E = self.FFT(self.e)
            # Weight update:
            # 1. Gradient estimation:
            self.phi[ : self.M] = self.IFFT((np.conj(U) * E) / self.P)[self.M : ]
            # 2. Gradient descent in frequency domain
            self.W += self.alpha * self.FFT(self.phi)
            
            # MSE estimation:
            mse = np.mean(np.abs(self.e) ** 2)
            s2p = np.mean(np.abs(d[(i - 1) * self.M : i * self.M]) ** 2)
            
            self.mses.append(mse)
            if mse <= self.Cs * s2p:
                break
                
        self.mses = np.asarray(self.mses)
        self.w = self.IFFT(self.W)[ : self.M]
    
    @property
    def MSE(self):
        return self.mses

    def clear(self):
        self.w = np.zeros(shape=(self.M, ))
        self.W = self.FFT(np.pad(self.w, (0, self.M), 'constant', const))