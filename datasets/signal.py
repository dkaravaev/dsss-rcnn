import numpy as np

import scipy.signal as ss

from commpy.filters import *
from commpy.modulation import PSKModem


def _chirp_phase(t, f0, t1, f1, method='linear', vertex_zero=True):
    t = np.asarray(t)
    f0 = float(f0)
    t1 = float(t1)
    f1 = float(f1)
    if method in ['linear', 'lin', 'li']:
        beta = (f1 - f0) / t1
        phase = 2 * np.pi * (f0 * t + 0.5 * beta * t * t)

    elif method in ['quadratic', 'quad', 'q']:
        beta = (f1 - f0) / (t1 ** 2)
        if vertex_zero:
            phase = 2 * np.pi * (f0 * t + beta * t ** 3 / 3)
        else:
            phase = 2 * np.pi * (f1 * t + beta * ((t1 - t) ** 3 - t1 ** 3) / 3)

    elif method in ['logarithmic', 'log', 'lo']:
        if f0 * f1 <= 0.0:
            raise ValueError("For a logarithmic chirp, f0 and f1 must be "
                             "nonzero and have the same sign.")
        if f0 == f1:
            phase = 2 * pi * f0 * t
        else:
            beta = t1 / np.log(f1 / f0)
            phase = 2 * np.pi * beta * f0 * (np.power(f1 / f0, t / t1) - 1.0)

    elif method in ['hyperbolic', 'hyp']:
        if f0 == 0 or f1 == 0:
            raise ValueError("For a hyperbolic chirp, f0 and f1 must be "
                             "nonzero.")
        if f0 == f1:
            phase = 2 * np.pi * f0 * t
        else:
            sing = -f1 * t1 / (f0 - f1)
            phase = 2 * np.pi * (-sing * f0) * np.log(np.abs(1 - t/sing))

    else:
        raise ValueError("method must be 'linear', 'quadratic', 'logarithmic',"
                " or 'hyperbolic', but a value of %r was given." % method)

    return phase

def complex_chirp(t, f0, t1, f1, method='linear', phi=0, vertex_zero=True):
    phase = _chirp_phase(t, f0, t1, f1, method, vertex_zero)
    phi *= np.pi / 180
    return np.exp(1j * (phase + phi))


class Signal:
    def __init__(self, Fs, W, Fc):
        self.Fs = Fs
        self.Fc = Fc
        self.W  = W
        
    def generate(self, N):
        pass

    def shift(self, S):
        t = np.linspace(0, S.shape[0] / self.Fs, S.shape[0])
        return np.exp(1j * 2 * np.pi * self.Fc * t) * S

    @staticmethod
    def convolve(S, h):
        Z = np.zeros_like(h)
        return ss.oaconvolve(np.hstack((Z[1:], S)), h, mode='valid')[h.shape[0] - 1:]


class NoiseSignal(Signal):
    FIR_TAPS = 2**16 + 1

    def __init__(self, Fs, W, Fc=0):
        super(NoiseSignal, self).__init__(Fs, W, Fc)
        
        if W != Fs:
            Wr = W / Fs
            self.w = ss.firwin(numtaps=self.FIR_TAPS, cutoff=Wr, pass_zero="lowpass", window='kaiser', width=0.01 * Wr)
        else:
            self.w = np.ones(1)

    def generate(self, N):
        L = N + self.FIR_TAPS - 1
        n = np.random.randn(L) + 1j * np.random.randn(L)
        return self.shift(self.convolve(n, self.w))


class LFMSignal(Signal):
    def __init__(self, Fs, W, Fc=0):
        super(LFMSignal, self).__init__(Fs, W, Fc)
        
        self.band = (Fc - W // 2, Fc + W // 2)
            
    def generate(self, N):
        t = np.linspace(0, N / self.Fs, N)
        return complex_chirp(t, self.band[0], t[-1], self.band[1])
    

class PulsedLFMSignal(Signal):
    def __init__(self, Fs, W, M, Fc=0):
        super(PulsedLFMSignal, self).__init__(Fs, W, Fc)
        t = np.linspace(0, M / self.Fs, M)
        self.band = (Fc - W // 2, Fc + W // 2)
        Sp = complex_chirp(t, self.band[0], t[-1], self.band[1])
        Sn = complex_chirp(t, self.band[1], t[-1], self.band[0])
        self.S = np.hstack((Sp, Sn[1:-1]))

    def generate(self, N):
        times = np.ceil(N / self.S.shape[0])
        return np.tile(self.S, int(times))[:N]

class QPSK:
    def modulate(self, bits):
        I = 2 * bits[0::2] - 1
        Q = 2 * bits[1::2] - 1

        return 1 / np.sqrt(2) * (I + 1j * Q)

    def demodulate(self, symbols):
        A = np.asarray([-1 - 1j, -1 + 1j, 1 - 1j, 1 + 1j])
        bits = np.zeros(shape=(0, ))
        for symbol in symbols:
            scalar = np.real(symbol) * np.real(A) + np.imag(symbol) * np.imag(A)
            i = np.argmax(scalar)
            bits = np.hstack((bits, np.asarray([np.real(A[i]), np.imag(A[i])])))

        return (bits + 1) / 2
"""    
class DSSS(Signal):
    @staticmethod
    def generate_msr(length):
        nbits= int(np.ceil(np.log2(length)))
        pn, _ = ss.max_len_seq(nbits=nbits, length=length)
        return 2.0 * pn - 1.0

    @staticmethod
    def generate_bits(length):
        return np.random.randint(0, 2, length)

    def __init__(self, Fs, W, Lc=2048, alpha=0.5, taps=32, Fc=0):
        super(DSSS, self).__init__(Fs, W, Fc)

        self.Tp = int(Fs / W)
        self.Lc = Lc

        self.Td = self.Lc * self.Tp
        self.Ts = self.Tp / self.Fs
        
        self.qpsk = QPSK()

        _, self.f = rrcosfilter(taps, alpha=alpha, Ts=self.Ts, Fs=self.Fs)

        self.pn = np.repeat(self.generate_msr(self.Lc), self.Tp)
 
    def generate_qpsk(self, N):
        length = int(np.ceil((N + self.f.shape[0]) / self.Td))
        
        self.bits = self.generate_bits(length * 2)
        self.symbols = self.qpsk.modulate(self.bits)
        
        return np.repeat(self.symbols, self.Td)

    def generate(self, N):
        data = self.generate_qpsk(N)
        
        m = data.shape[0] // self.pn.shape[0]
        pn = self.pn
        for i in range(m):
            pn = np.hstack((pn, self.pn))
        
        pn = pn[:data.shape[0]]
        return ss.oaconvolve(data * pn, self.f, mode="valid")
"""
