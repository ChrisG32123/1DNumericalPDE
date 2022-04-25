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
    k_fft_norm = 2 * np.pi / (N * dx)
    k = k_fft_norm * np.linspace(-N / 2, N / 2 - 1, N)

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

def meyer_convolution():
    # Parameters
    N = 2 ** 10
    L = 10
    a = 100
    b = 50
    Gamma = 1
    kappa = 1

    def dcf(k, Gamma, kappa):
        return 4 * np.pi * Gamma / (k ** 2 + kappa ** 2)

    # Define domains
    dx = L / N
    x = np.linspace(0, L - dx, N)
    f_fft_norm = 1 / dx
    k_fft_norm = 2 * np.pi / (N * dx)
    k = k_fft_norm * np.linspace(-N / 2, N / 2 - 1, N)

    # Find c_hat(k) = 4pi integral 0 to infinity c(r)r^2 sinc(kr) dr
    # Parameters
    Nr = int(1e2)
    c_hat = np.zeros(N)
    beta = 1  # TODO: = 1 / (kB*T) (Nondimensionalized?)

    for ii in range(int(N/2 + 1)):
        rmax = 100  # TODO: Change per loop
        r = np.linspace(0, rmax, Nr)
        r[0] = 1
        u = Gamma / r * np.exp(-r)
        # c = np.exp(-beta*u) - 1
        # c[0] = -1
        r[0] = 0
        b = 1
        c = np.exp(-b * r ** 2)
        c_hat[ii] = 4 * np.pi * np.trapz((r**2)*c*np.sinc(k[ii]*r/np. pi), r)  # np.sinc(x) = sinc(pi*x)/(pi*x)

    for ii in range(int(N/2 - 1)):
        c_hat[N-1-ii] = c_hat[ii]

    c_hat_ex = (np.pi / b) ** (3/2) * np.exp(- k ** 2 / (4*b))

    # FFT solution 1
    f = np.sqrt(a / np.pi) * np.exp(-a * (x - (L/2)) ** 2)
    fhat = np.fft.fftshift(np.fft.fft(f))

    conv_ex = np.pi / b * np.exp(-b * (x - L/2) ** 2)

    conv = c_hat * fhat
    conv = np.fft.ifft(np.fft.ifftshift(conv))

    # Exact solution
    #f_ex = f_fft_norm * np.sqrt(np.pi / a) * np.exp(-0.25 * k * k / a)

    # conv = fhat * ghat
    # conv = np.fft.ifft(np.fft.ifftshift(conv))
    #
    # conv_ex = 2 * np.pi * Gamma / kappa * np.exp(-kappa * np.abs(x-(L/2)))

    # Plotting
    # plt.plot(k,f_ex, label="exact")
    # plt.plot(k,fhat, label="fft")
    plt.plot(x,f, label="f(x)")
    plt.plot(x,conv, label="convolution")
    plt.plot(x,conv_ex,label = "conv_ex")
    # plt.plot(x,conv_ex, label="conv_ex")
    plt.legend()

    plt.figure()
    plt.plot(k,np.abs(c_hat), label="c_hat")
    plt.plot(k,c_hat_ex, label="c_hat_ex")
    # plt.loglog(k[int(N/2 + 1):int(N+1)], c_hat[int(N/2 + 1):int(N+1)])
    plt.legend()

    plt.show()

    #
    # plt.figure()
    # plt.plot(r,c, label="c(x)")
    # plt.legend()
    # plt.show()

def meyer_convolution_1():
    # Parameters
    N = 2 ** 10
    L = 10
    a = 100
    b = 50

    # Define domains
    dx = L / N
    x = np.linspace(0, L - dx, N)
    f_fft_norm = 1 / dx
    k_fft_norm = 2 * np.pi / (N * dx)
    k = k_fft_norm * np.linspace(-N / 2, N / 2 - 1, N)

    # Parameters
    Nr = int(1e3)
    rmax = 100  # TODO: Change per loop
    r = np.linspace(0, rmax, Nr)

    # FFT Function 1
    f = np.exp(-b * r ** 2)
    fhat = np.fft.fftshift(np.fft.fft(f))
    f_hat_ex = (np.pi / b) ** (3/2) * np.exp(- k ** 2 / (4*b))

    # FFT Function 2
    g = np.sqrt(a / np.pi) * np.exp(-a * (x - (L/2)) ** 2)
    ghat = np.fft.fftshift(np.fft.fft(g))
    g_hat_ex = np.exp(- k ** 2 / (4 * a))

    conv = f_hat_ex * ghat
    conv = np.fft.ifft(np.fft.ifftshift(conv))

    # conv_fft = f_hat_ex * g_hat_ex
    # conv_fft = np.fft.ifft(np.fft.ifftshift(conv_fft))

    normalization = np.sqrt(2)/3
    conv_ex = (np.pi/b) * np.sqrt((a+b)/b) * np.exp(-(a+b) * (normalization**2) * (x - L / 2) ** 2)
    conv_ex = conv_ex * normalization

    # Plotting
    plt.plot(x,conv, label="Code Convolution = conv")
    # plt.plot(x,conv_fft, label="FFT Convolution = conv_fft")
    plt.plot(x,conv_ex,label = "Exact Convolution = conv_ex")
    plt.legend()
    plt.show()

def meyer_convolution_2():
    # Parameters
    N = 2 ** 10
    L = 10
    a = 100
    b = 50

    # Define domains
    dx = L / N
    x = np.linspace(0, L - dx, N)
    f_fft_norm = 1 / dx
    k_fft_norm = 2 * np.pi / (N * dx)
    k = k_fft_norm * np.linspace(-N / 2, N / 2 - 1, N)

    # Parameters
    Nr = int(1e3)
    rmax = 100  # TODO: Change per loop
    r = np.linspace(0, rmax, Nr)

    # FFT Function 1
    f = np.exp(-b * r ** 2)
    fhat = np.fft.fftshift(np.fft.fft(f))
    f_hat_ex = (np.pi / b) ** (3/2) * np.exp(- k ** 2 / (4*b))

    # FFT Function 2
    g = np.sqrt(a / np.pi) * np.exp(-a * (x - (L/2)) ** 2)
    ghat = np.fft.fftshift(np.fft.fft(g))
    g_hat_ex = np.exp(- k ** 2 / (4 * a))

    conv = f_hat_ex * ghat
    conv = np.fft.ifft(np.fft.ifftshift(conv))

    # conv_fft = f_hat_ex * g_hat_ex
    # conv_fft = np.fft.ifft(np.fft.ifftshift(conv_fft))

    normalization = np.sqrt(2)/3
    conv_ex = (np.pi/b) * np.sqrt((a+b)/b) * np.exp(-(a+b) * (normalization**2) * (x - L / 2) ** 2)
    conv_ex = conv_ex * normalization

    # Plotting
    plt.plot(x,conv, label="Code Convolution = conv")
    # plt.plot(x,conv_fft, label="FFT Convolution = conv_fft")
    plt.plot(x,conv_ex,label = "Exact Convolution = conv_ex")
    plt.legend()
    plt.show()