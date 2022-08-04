import numpy as np
import matplotlib.pyplot as plt

def test():
    N = 500
    a = np.array([[np.sin(2*np.pi*x/N) for x in range(N)] for y in range(N)]).reshape(N,N)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    im = ax.imshow(a, aspect='auto', cmap='viridis')
    fig.colorbar(im)

    a_hat = np.fft.fft2(a)
    a_hat = np.real(a_hat * np.conj(a_hat))

    fig, ax = plt.subplots(nrows=1, ncols=1)
    im = ax.imshow(a_hat, aspect='auto', cmap='viridis')
    fig.colorbar(im)

    min = np.min(a_hat)
    max = np.max(a_hat)

    print(min)
    print(max)