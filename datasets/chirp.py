import scipy.signal

import numpy as np

from .        import utils
from .generic import Generic


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


class SimpleChirp(Generic):
    def __init__(self, M, N, Fmin, Fmax, Fs, method='linear', SNRdB=-1):
        super(SimpleChirp, self).__init__(M, N, SNRdB)

        self.T = (M + N + 1) / Fs
        self.t = np.linspace(0, self.T, M + N + 1)
        self.S = complex_chirp(self.t, Fmin, self.T, Fmax, method)


class RepitedChirp(Generic):
    def __init__(self, M, N, K, Fmin, Fmax, Fs, method='linear', SNRdB=-1):
        super(RepitedChirp, self).__init__(M, N, SNRdB)

        self.T = K / Fs
        self.t = np.linspace(0, self.T, K)
        s = complex_chirp(self.t, Fmin, self.T, Fmax, method)
        s = np.tile(s, self.M // K)
        self.S = np.hstack((s, s[ : self.M % K + N + 1]))


class ContinuousChirp(Generic):
    def __init__(self, M, N, K, Fmin, Fmax, Fs, method='linear', SNRdB=-1):
        super(ContinuousChirp, self).__init__(M, N, SNRdB)

        self.T = K / Fs
        self.t = np.linspace(0, self.T, K)
        s = np.hstack((complex_chirp(self.t, Fmin, self.T, Fmax, method), 
                       complex_chirp(self.t, Fmax, self.T, Fmin, method)))
        s = np.tile(s, self.M // (2 * K))
        self.S = np.hstack((s, s[ : self.M % (2 * K) + N + 1]))




