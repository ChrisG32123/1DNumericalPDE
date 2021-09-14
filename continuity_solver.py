import numpy as np
import matplotlib.pyplot as plt

def continuity():
    # Numerical parameters
    N = int(1e3)  # grid points
    T = int(1e3)  # time steps

    # Domain
    L = 10  # system size
    x = np.linspace(0, L, N)
    dx = x[2] - x[1]
    dt = 1e-3  # step size

    # Initial condition
    n_0 = np.exp(-10 * (x - L / 2) ** 2)


    # Choose flux function
    def F(x, n):
        return n * (2 + np.sin(2 * np.pi * x / 10))


    # MAIN LOOP
    n = n_0
    nTot = np.zeros(T)
    for tt in range(T):
        # Lax-Friedrichs time-stepping
        # Central differencing in space
        F_0 = F(x, n)
        Fplus = np.roll(F_0, -1)
        Fminus = np.roll(F_0, 1)
        nplus = np.roll(n, -1)
        nminus = np.roll(n, 1)
        n = 0.5 * (nplus + nminus) - 0.5 * dt * (Fplus - Fminus) / dx

        # Measure integral over n(x,t)
        nTot[tt] = np.sum(n) * dx

    # Plot n(x,0) and n(x,t)
    plt.figure()
    plt.plot(x, n_0, label="n_0")
    plt.plot(x, n, label="n")

    # Plot integral of n(x,t) over time
    # plt.plot(nTot, label = "nTot")

    plt.title("Continuity")
    plt.legend()
    plt.show(block=True)
    plt.interactive(False)