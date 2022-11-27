import scipy.signal

import numpy as np

from .        import utils
from .generic import Generic


class Sin(Generic):
	def __init__(self, M, N, Fc, Fs, SNRdB=-1):
		super(Sin, self).__init__(M, N, SNRdB)

		self.T = (M + N + 1) / Fs
		self.t = np.linspace(0, self.T, M + N + 1, endpoint=True)
		self.S = np.exp(1j * 2 * np.pi * Fc * self.t)