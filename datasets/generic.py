import numpy as np


from . import utils


class Generic:
    def __init__(self, M, N, SNRdB=-1):
        self.M = M
        self.N = N

        self.sigma = 0
        if SNRdB >= 0:
            self.sigma = 10 ** (-SNRdB / 20.0)

        self.S = np.zeros(shape=(self.N + self.M + 1, ), dtype=np.complex64)

    def generate(self):
        D = utils.tochanneled(self.S)
        X, y = [ ], [ ]

        for i in range(self.M):
            n = self.sigma * (np.random.randn(self.N) + 1j * np.random.randn(self.N))
            X.append(D[i     : i + self.N] + utils.tochanneled(n))
            y.append(D[i + 1 : i + self.N + 1])

        return np.asarray(X), np.asarray(y)

    @property
    def signal(self):
        return self.S

    @property
    def signal_n(self):
        n = self.sigma * (np.random.randn(self.S.shape[0]) + 1j * np.random.randn(self.S.shape[0]))
        return self.S + n