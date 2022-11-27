import numpy as np

from . import utils


class RayleighFading:
    def __init__(self, M, Fmax, Fs):
        self.M  = M 
        self.wm = 2 * np.pi * Fmax / Fs

    def factors(self, N):
        ret = np.zeros(shape=(N, ), dtype=np.complex128)
        
        A = np.random.randn(self.M)
        alpha = 2 * np.pi * np.random.rand(self.M) - np.pi
        phi   = 2 * np.pi * np.random.rand(self.M) - np.pi

        n = np.arange(0, N)

        for m in range(self.M):
            ret += A[m] * np.exp(1j * (self.wm * n * np.cos(alpha[m]) + phi[m]))

        return ret

    def generate(self, S):
        return self.factors(S.shape[0]) * S

    def dataset(self, S, N):
        Si = S + self.generate(S)

        Sx, Sy = utils.tochanneled(Si), utils.tochanneled(S)
        X, y = [ ], [ ]

        M = S.shape[0] - N
        for i in range(M):
            X.append(Sx[i : i + N])
            y.append(Sy[i : i + N])

        return np.asarray(X), np.asarray(y), S