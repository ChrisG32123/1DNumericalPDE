import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

# Non-Dimensionalized Euler-Poisson Equations
# n_t + (nu)_x = 0
# n(u_t + u*u_x) = -n_x - gamma*n*phi_x + conv(n-n_mean,c_hat)
# phi_xx = f(x,t) = 3 - 4*pi*n


def phi_corr(N,T,L,x,x3,dx,dt, n_IC, u_IC, n_0, correlation, Gamma_0, kappa_0):
    nc = np.zeros(N)
    n3 = np.zeros(3 * N)

    uc = np.zeros(N)

    phic = np.zeros(N)
    phi_coefs = np.ones(N)

    rho = np.zeros(N)
    f_corr = np.zeros(N)

    # Setting Initial Conditions
    nc = n_IC
    # nTot = np.zeros(N)
    uc = u_IC

    # uTot = np.zeros(N)
    # phiTot = np.zeros(N)

    # Choose flux functions
    def f_n(n, u):
        return n * u

    def f_u(n, u, v, f_corr):
        return .5 * u * u + np.log(n) + Gamma_0 * v + f_corr

    # Solve
    for tt in range(T):
        # Phi
        # Define f(x)
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

        conc = nc / n_0
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

        # Measure integral over n(x,t), u(x,t), phi(x,t)
        # nTot[tt] = np.sum(n) * dx
        # uTot[tt] = np.sum(u) * dx
        # phiTot[tt] = np.sum(phi) * dx
    return phic



def phi_nocorr(N,T,Gamma_0,dx,dt,n_IC, u_IC):

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
    return u



    # Plotting
def plot_phi_solutions(dt,x, phi, phic, Gamma_0, kappa_0):

    # plt.plot(x, n_IC, label="n_IC")
    plt.plot(x, phi, label="u")
    plt.plot(x, phic, label="uc")
    plt.title("Electrostatic Potential: " + "Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0) + " dt = "+ str(dt))
    plt.legend()
    # plt.ylim(0, 2*rho_0)

    # plt.figure()
    # plt.plot(x, u_IC, label="u_IC")
    # plt.plot(x, u, label="u")
    # plt.title("Velocity: " + "Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0) + " Correlation = " + "true" if correlation else "false")
    # plt.legend()

    # Plot integral of phi(x,t), n(x,t), and u(x,t) over time
    # plt.plot(phiTot, label = "phiTot")
    # plt.plot(nTot, label = "nTot")
    # plt.plot(uTot, label = "uTot")

    plt.show(block=True)
