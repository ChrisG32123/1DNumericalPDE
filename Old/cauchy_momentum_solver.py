import numpy as np
import matplotlib.pyplot as plt

def cauchy():
    # Numerical parameters
    N = int(1e3)  # grid points
    T = int(1e3)  # time steps

    # Domain
    L = 10  # system size
    x = np.linspace(0, L, N)
    dx = x[2] - x[1]
    dt = 1e-3  # step size

    # Initial condition
    k = 2 * np.pi / L
    n_0 = 3/(4 * np.pi) + 1e-4 * np.sin(k * x)
    u_0 = np.exp(-10*(x-L/2)**2)
    phi_0 = np.ones(T) #TODO

    # Choose flux function
    def F(x,n):
        return n*(2 + np.sin(2*np.pi*x/10))

    # MAIN LOOP
    n = n_0
    nTot = np.zeros(T)
    u = u_0
    uTot = np.zeros(T)
    phi = phi_0
    phiTot = np.zeros(T)
    gamma = 1 # input("Coulomb Coupling Parameter:")
    for tt in range(T):
        # Lax-Friedrichs time-stepping
        # Central differencing in space
        F_0 = F(x, n)
        Fplus = np.roll(F_0, -1)
        Fminus = np.roll(F_0, 1)
        nplus = np.roll(n, -1)
        nminus = np.roll(n, 1)
        uplus = np.roll(u, -1)
        uminus = np.roll(u, 1)
        phiplus = np.roll(phi, -1)
        phiminus = np.roll(phi, 1)
        n = 0.5 * (nplus + nminus) - 0.5 * dt * (Fplus - Fminus) / dx
        u = 0.5 * (uplus + uminus) - (dt / dx) * (-uplus ** 2 / 2 + uminus ** 2 / 2 + np.log(nplus / nminus) + gamma * (phiplus - phiminus))

        # Measure integral over n(x,t), u(x,t), phi(x,t)
        nTot[tt] = np.sum(n) * dx
        uTot[tt] = np.sum(u) * dx
        phiTot[tt] = np.sum(phi) * dx

    # Plot n(x,0) and n(x,t)
    plt.figure()
    plt.plot(x, n_0, label="rho_0")
    plt.plot(x, n, label="n")

    # Plot integral of n(x,t) over time
    # plt.plot(nTot, label = "nTot")

    # Plot u(x,0) and u(x,t)
    plt.plot(x, u_0, label="u_0")
    plt.plot(x, u, label="u")

    # Plot integral of u(x,t) over time
    # plt.plot(uTot, label = "uTot")

    plt.title("Continuity")
    plt.legend()
    plt.show(block=True)
    plt.interactive(False)