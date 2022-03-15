import numpy as np
import matplotlib.pyplot as plt

def test_convolution():
    # Parameters
    N = 2**10
    L = 100
    a = 0.1

    # Define domains
    dx = 2*L/N
    x = np.linspace(-L,L,N+1)
    k_fft_norm = 2*np.pi/(N*dx)
    f_fft_norm = N/(2*L)
    k = k_fft_norm * np.linspace(-N/2,N/2,N+1)

    # FFT solution
    f = np.exp(-a*x*x)
    fhat = np.fft.fftshift(np.abs(np.fft.fft(f)))

    # Exact solution
    f_ex = f_fft_norm * np.sqrt(np.pi/a)*np.exp(-0.25*k*k/a)

    # Plotting
    plt.plot(k,f_ex,k,fhat)
    plt.show()

def half_interval_test_convolution():
    # Parameters
    N = 2 ** 10
    L = 10
    a = 100
    Gamma = 1
    kappa = 1

    def dcf(k, Gamma, kappa):
        return 4 * np.pi * Gamma / (k ** 2 + kappa ** 2)

    # Define domains
    dx = L / N
    x = np.linspace(0, L, N + 1)
    f_fft_norm = 1 / dx

    # FFT solution
    f = np.sqrt(a / np.pi) * np.exp(-a * (x - (L/2)) ** 2)
    fhat = np.fft.fftshift(np.fft.fft(f))

    # Exact solution
    #f_ex = f_fft_norm * np.sqrt(np.pi / a) * np.exp(-0.25 * k * k / a)

    conv = fhat * dcf(k,Gamma, kappa)
    conv = np.fft.ifft(np.fft.ifftshift(conv))

    conv_ex = 2 * np.pi * Gamma / kappa * np.exp(-kappa * np.abs(x-(L/2)))

    # Plotting
    # plt.plot(k,f_ex, label="exact")
    # plt.plot(k,fhat, label="fft")
    plt.plot(x,f, label="f(x)")
    plt.plot(x,conv, label="convolution")
    plt.plot(x,conv_ex, label="conv_ex")
    plt.legend()
    plt.show()

