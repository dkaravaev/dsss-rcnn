import numpy as np
import scipy.signal as ss

from . import utils
from . import signal


class Infr:
    def __init__(self, SIR, Fs, W, Fc, name):
        self.SIR   = SIR
        self.Fs    = Fs
        self.Fc    = Fc
        self.W     = W
        self.name  = name
        
        self.signal = None

    def generate(self, S):
        I = self.signal.generate(S.shape[0])
        return self.amplitude(S, I) * I
    
    def amplitude(self, S, I):
        return np.sqrt(utils.energy(S) / (self.SIR * utils.energy(I)))
    
    def __str__(self):
        suffix = "_Fs" + str(self.Fs) + "_W"   + str(self.W) + \
                 "_Fc" + str(self.Fc) + "_SIR" + str(self.SIRdB)
        return self.name + suffix

    def dataset(self, S, N):
        Si = S + self.generate(S)

        Sx, Sy = utils.tochanneled(Si), utils.tochanneled(S)
        X, y = [ ], [ ]

        M = S.shape[0] - N
        for i in range(M):
            X.append(Sx[i : i + N])
            y.append(Sy[i : i + N])

        return np.asarray(X), np.asarray(y), Si
    

class NoiseInfr(Infr):
    def __init__(self, SIR, Fs, W, Fc, M=0):
        super(NoiseInfr, self).__init__(SIR, Fs, W, Fc, "Noise")
        self.signal = signal.NoiseSignal(Fs, W, Fc)
    
    @property
    def ir(self):
        return self.signal.shift(self.signal.w)

class LFMInfr(Infr):
    def __init__(self, SIR, Fs, W, Fc, M=0):
        super(LFMInfr, self).__init__(SIR, Fs, W, Fc, "LFM")
        self.signal = signal.LFMSignal(Fs, W, Fc)  

class PulsedLFMInfr(Infr):
    def __init__(self, SIR, Fs, W, Fc, M):
        super(PulsedLFMInfr, self).__init__(SIR, Fs, W, Fc, "PulsedLFM")
        self.signal = signal.PulsedLFMSignal(Fs=Fs, W=W, M=M, Fc=Fc)
        
    
class InfrGenerator:    
    def __init__(self, infrs, SIRdB, W, Fs, **kwargs):
        self.m = len(infrs)
        w = np.random.rand(self.m)
        self.SIRs = utils.db2mag(SIRdB) * (w / np.sum(w))
        
        self.W   = self.bands(W)
        self.Fc  = self.freqs(W, Fs)

        self.infrs = [infr(SIR=self.SIRs[i], Fs=Fs, W=self.W[i], Fc=self.Fc[i], **kwargs) 
                      for i, infr in enumerate(infrs)]
    
    def SNIR(self, S, N):
        Ps = utils.energy(S)
        Pn = utils.energy(N)
        
        SIR = utils.db2mag(np.sum(self.SIRs))
        
        Pi = Ps / SIR
        
        return 10 * np.log10(Ps / (Pn + Pi))
    
    
    def bands(self, W):
        return W * np.random.rand(self.m) 
    
    def freqs(self, W, Fs):
        bounds = (-Fs / 4 + W, Fs / 4 - W)
        return bounds[0] + np.random.rand(self.m) * (bounds[1] - bounds[0])
    
    def generate(self, S):
        return np.sum([infr.generate(S) for infr in self.infrs], axis=0)
    
    def names(self):
        return [str(infr) for infr in self.infrs]
    
    @property
    def ir(self):
        ret = self.infrs[0].ir
        #for i in range(1, len(self.infrs)):
        #    ret += self.infrs[i].ir
        return ret
    
    def dataset(self, S, N):
        Si = S + self.generate(S)

        Sx, Sy = utils.tochanneled(Si), utils.tochanneled(S)
        X, y = [ ], [ ]

        M = S.shape[0] - N
        for i in range(M):
            X.append(Sx[i : i + N])
            y.append(Sy[i : i + N])

        return np.asarray(X), np.asarray(y), Si