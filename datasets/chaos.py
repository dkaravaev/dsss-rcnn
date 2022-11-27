import scipy.signal

import numpy as np

from math      import gcd

from .         import utils
from .generic  import Generic


class IkedaMap(Generic):
    def __init__(self, M, N, z0=0.5 + 0 * 1j, A=1, B=0.9, K=-6, C=0.4, SNRdB=-1):
        super(IkedaMap, self).__init__(M, N, SNRdB)

        self.S = z0 + np.zeros(shape=(M + N + 1, ), dtype=np.complex64)
        for n in range(1, M + N + 1):
            t = C + K / (np.absolute(self.S[n - 1]) ** 2 + 1)
            self.S[n] = A + B * self.S[n - 1] * np.exp(1j * t)


class ZadoffChu(Generic):
    def __init__(self, M, N, u, q=0, SNRdB=-1):
        super(ZadoffChu, self).__init__(M, N, SNRdB)

        if gcd(N, u) != 1:
            raise ValueError("N and u must be mutually prime!")

        cf = self.N % 2

        self.S = np.zeros(shape=(M + N + 1, ), dtype=np.complex64)
        for n in range(M + N + 1):
            self.S[n] = np.exp(1j * np.pi * u * n * (n + cf + 2 * q) / N)


class SymmetricChaos(Generic):
    def __init__(self, M, N, z0=0.5 + 0.5 * 1j, m=3, alpha=-1, gamma=0.5, ld=2.202, SNRdB=-1):
        super(SymmetricChaos, self).__init__(M, N, SNRdB)

        self.S = z0 + np.zeros(shape=(M + N + 1, ), dtype=np.complex64)
        for n in range(1, M + N + 1):
            self.S[n] = gamma * (np.conj(self.S[n - 1])) ** (m - 1) + \
                        self.S[n - 1] * (ld + alpha * self.S[n - 1] * np.conj(self.S[n - 1]))
