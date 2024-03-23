import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

# Non-Dimensionalized Euler-Poisson Equations
# n_t + (nu)_x = 0
# n(u_t + u*u_x) = -n_x - gamma*n*phi_x + conv(n-n_mean,c_hat)
# phi_xx = f(X,t) = 3 - 4*pi*n


def test_corr():
    # TODO: HW: Change Gamma_0 and kappa_0
    #  Try solving separately, than third method to plot together. Sidestep simultaneous solve TypeError

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

    nc = np.zeros(N)
    n3 = np.zeros(3 * N)

    uc = np.zeros(N)

    phic = np.zeros(N)
    phi_coefs = np.ones(N)

    rho = np.zeros(N)
    f_corr = np.zeros(N)

    # Setting Initial Conditions
    nc = n_IC
    # nTot = np.zeros(nx)
    uc = u_IC
    # uTot = np.zeros(nx)
    # phiTot = np.zeros(nx)

    # Choose flux functions
    def f_n(n, u):
        return n * u

    def f_u(n, u, v, f_corr):
        return .5 * u * u + np.log(n) + Gamma_0 * v + f_corr

    # Solve
    for tt in range(T):
        # Phi
        # Define f(X)
        f = 3 - 4 * np.pi * dx * dx * nc
        f = f - np.mean(f)

        # First sweep
        phi_coefs[0] = -0.5
        f[0] = -0.5 * f[0]

        for ii in range(1, N - 1):
            phi_coefs[ii] = -1 / (2 + phi_coefs[ii - 1])
            f[ii] = (f[ii - 1] - f[ii]) / (2 + phi_coefs[ii - 1])

        # Second sweep
        phic[0] = f[N - 1] - f[N - 2]
        for ii in range(1, N - 1):
            phic[ii] = (f[ii - 1] - phic[ii - 1]) / phi_coefs[ii - 1]

        conc = nc/n_0
        Gamma = Gamma_0 * conc ** (1 / 3)
        kappa = kappa_0 * conc ** (1 / 6)

        n3[0:N] = nc[0:N]
        n3[N:2 * N] = nc[0:N]
        n3[2 * N:3 * N] = nc[0:N]
        for jj in range(N):
            xj = x[jj]
            rho_int = - 2 * np.pi * Gamma[jj] * np.exp(- kappa[jj] * np.abs(x3 - xj)) / kappa[jj]
            f_corr[jj] = dx * np.sum(n3 * rho_int)

        # Lax-Friedrichs time-stepping
        # Central differencing in space
        F_0 = f_n(nc, uc)
        Fnplus = np.roll(F_0, -1)
        Fnminus = np.roll(F_0, 1)
        nplus = np.roll(nc, -1)
        nminus = np.roll(nc, 1)
        nc = 0.5 * (nplus + nminus) - 0.5 * dt * (Fnplus - Fnminus) / dx

        F_1 = f_u(nc, uc, phic, f_corr)
        Fuplus = np.roll(F_1, -1)
        Fuminus = np.roll(F_1, 1)
        uplus = np.roll(uc, -1)
        uminus = np.roll(uc, 1)
        uc = 0.5 * (uplus + uminus) - .5 * (dt / dx) * (Fuplus - Fuminus)

        # Measure integral over n(X,t), u(X,t), phi(X,t)
        # nTot[tt] = np.sum(n) * dx
        # uTot[tt] = np.sum(u) * dx
        # phiTot[tt] = np.sum(phi) * dx

    plt.plot(x,nc,label="nc")
    plt.legend()

    plt.figure()
    plt.plot(x,uc,label="uc")
    plt.legend()

    plt.show()
