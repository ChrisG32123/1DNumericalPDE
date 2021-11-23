import matplotlib.pyplot as plt
import numpy as np


def tester():
    # TODO: Changes Made
    #  U_IC +=
    #  dt for instabilities?
    #  shock variable names
    #  shock IC values (no change. good values)

    #TODO: Changes to make
    # step function to sin
    # space time plots
    #

    N = int(1e2)  # grid points
    T = int(1e4)  # time steps
    L = 10
    x = np.linspace(0, L - L / N, N)
    x3 = np.linspace(-L, 2 * L - L / N, 3 * N)
    dx = x[2] - x[1]
    dt = 1e-4  # step size
    n_0 = 3 / (4 * np.pi)  # Mean Density
    Gamma_0 = 100
    kappa_0 = .1

    n_IC = n_0 * np.ones(N)
    u_IC = np.zeros(N)
    for ii in range(N):
        n_IC[ii] = n_IC[ii] + .2 * np.sin(2 * np.pi * ii * dx / L)
        u_IC[ii] = u_IC[ii] + np.sin(2 * np.pi * ii * dx / L)

    # Memory Allocation
    n = np.zeros(N)

    u = np.zeros(N)

    phi = np.zeros(N)
    phi_coefs = np.ones(N)

    # Setting Initial Conditions
    n = n_IC
    # nTot = np.zeros(N)

    u = u_IC
    # uTot = np.zeros(N)
    # phiTot = np.zeros(N)

    # Choose flux functions
    def f_n(n, u):
        return n * u

    def f_u(n, u, v):
        return .5 * u * u + np.log(n) + Gamma_0 * v

    # Solve
    for tt in range(T):
        # Phi
        # Define f(x)
        f = 3 - 4 * np.pi * dx * dx * n
        f = f - np.mean(f)

        # First sweep
        phi_coefs[0] = -0.5
        f[0] = -0.5 * f[0]

        for ii in range(1, N - 1):
            phi_coefs[ii] = -1 / (2 + phi_coefs[ii - 1])
            f[ii] = (f[ii - 1] - f[ii]) / (2 + phi_coefs[ii - 1])

        # Second sweep
        phi[0] = f[N - 1] - f[N - 2]
        for ii in range(1, N - 1):
            phi[ii] = (f[ii - 1] - phi[ii - 1]) / phi_coefs[ii - 1]

        # Lax-Friedrichs time-stepping
        # Central differencing in space
        F_0 = f_n(n, u)
        Fnplus = np.roll(F_0, -1)
        Fnminus = np.roll(F_0, 1)
        nplus = np.roll(n, -1)
        nminus = np.roll(n, 1)
        n = 0.5 * (nplus + nminus) - 0.5 * dt * (Fnplus - Fnminus) / dx

        F_1 = f_u(n, u, phi)
        Fuplus = np.roll(F_1, -1)
        Fuminus = np.roll(F_1, 1)
        uplus = np.roll(u, -1)
        uminus = np.roll(u, 1)
        u = 0.5 * (uplus + uminus) - .5 * (dt / dx) * (Fuplus - Fuminus)

        # Measure integral over n(x,t), u(x,t), phi(x,t)
        # nTot[tt] = np.sum(n) * dx
        # uTot[tt] = np.sum(u) * dx
        # phiTot[tt] = np.sum(phi) * dx

        #### TODO: Figure out why n is constant`

    plt.plot(x,n,label="n")
    plt.legend()

    plt.figure()
    plt.plot(x,u,label="u")
    plt.legend()

    plt.show()
