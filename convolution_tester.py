import numpy as np


def convolution_tester():
    a = 10
    N = np.ones(a)
    C = np.array(a, dtype=complex)

    for c in range(C.size-1):
        C[c] = c * np.random.random() + complex(c) * np.random.random()
        N[c] = c * np.random.random() + complex(c) * np.random.random()

    def conv(delta_n, dcf):
        delta_n = np.fft.fftshift(np.fft.fft(delta_n))
        nc = dcf * delta_n
        nc = np.fft.ifft(np.fft.ifftshift(nc))
        return nc

    print(conv(N,C))