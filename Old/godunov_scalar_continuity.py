# Non-Dimensionalized Euler-Poisson Equations
# n_t + (nu)_x = 0
# n(u_t + u*u_x) = -n_x - gamma*n*phi_x + corr
# phi_xx = 3 - 4*pi*n

import numpy as np
import matplotlib.pyplot as plt


def godunov_scalar_continuity():

    # Memory Allocation and Initialization
    def memory_allocation(u_IC):
        u = np.copy(u_IC)
        uL = np.roll(u, 1)
        uR = np.copy(u)
        flux_or_matrix = np.zeros(N)
        snap_u = np.zeros((snaps + 1, N))
        snap_u[0] = np.copy(u_IC)
        return u, uL, uR, flux_or_matrix, snap_u

    # Flux Functions
    def f_n(n, v):
        return n * v

    # Numerical Scheme
    # def godunov_flux(fL, fR, uL, uR):
    #     s = 0.5 * (uL + uR)
    #     if uL >= uR:
    #         if s > 0.0:
    #             return fL
    #         else:
    #             return fR
    #     else:
    #         if uL > 0.0:
    #             return fL
    #         elif uR < 0.0:
    #             return fR
    #         else:
    #             return 0.0

    def godunov_flux(fL, fR, uL, uR):
        if uL > uR:
            return np.maximum(fL, fR)
        elif uL < uR:
            return np.minimum(fL, fR)
        else:
            return 0.0

    # Display
    def take_snapshot(tt, T, snaps, n_, snap_n_):
        snap_n_[int(tt / (T / snaps))] = n_

    def colormap(xx_, yy_, snap_u):
        plt.figure()
        clr = plt.contourf(xx_, yy_, snap_u)
        plt.colorbar()

    def plot(x_, snap_u):
        plt.figure()
        for ii in range(len(snap_u)):
            plt.plot(x_, snap_u[ii], label="T = " + str(ii * dt * T / snaps))
        plt.legend()

    # Parameters
    N = int(1e3)  # Grid Points 5e2
    T = int(1e3)  # Time Steps 5e3
    L = 10  # Domain Size
    x = np.linspace(0, L - L / N, N)  # Domain
    # x3 = np.linspace(-L, 2 * L - L / N, 3 * N)
    dx = x[2] - x[1]  # Grid Size
    dt = 1e-3  # Time Step Size
    lambda_ = dt / dx
    t = dt * T

    n_0 = 3 / (4 * np.pi)  # Mean Density
    snaps = int(input("Number of Snapshots "))  # Number of Snapshots
    Gamma_0 = float(input("Value of Gamma "))  # Coulomb Coupling Parameter
    kappa_0 = float(input("Value of kappa "))  # screening something
    # rho = np.zeros(N)
    # f_corr = np.zeros(N)

    # Initial Conditions
    n_IC = n_0 * np.ones(N)
    n_IC[0:int(N / 4)] = n_0 / 2
    n_IC[int(N / 4):int(3*N / 4)] = 3 * n_0 / 2
    n_IC[int(3*N / 4):N] = n_0 / 2
    v_IC = 2 + .2 * np.sin(2 * np.pi * x / L)

    n, nL, nR, flux_n, snap_n = memory_allocation(n_IC)
    v = np.copy(v_IC)
    vL = np.roll(v_IC,-1)
    vR = np.copy(v)

    FnL = np.zeros(N)
    FnR = np.zeros(N)

    # No Correlation
    for tt in range(T + 1):

        # Compute Fluxes
        for ii in range(N):
            FnL[ii] = f_n(nL[ii], vL[ii])
            FnR[ii] = f_n(nR[ii], vR[ii])
            flux_n[ii] = godunov_flux(FnL[ii],FnR[ii],nL[ii],nR[ii])

        # Godunov Scheme
        for ii in range(0, N-1):
            n[ii] = n[ii] - lambda_ * (flux_n[ii+1] - flux_n[ii])
        n[N-1] = n[N-1] - lambda_ * (flux_n[0] - flux_n[N - 1])

        nL = np.roll(n, 1)
        nR = np.copy(n)

        if tt % (T / snaps) == 0:
            take_snapshot(tt,T,snaps,n, snap_n)

    y = np.linspace(0, t, snaps + 1)
    xx, yy = np.meshgrid(x, y, sparse=False, indexing='xy')

    colormap(xx,yy,snap_n)
    plt.title("No Correlations: Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))

    plot(x,snap_n)
    plt.title("No Correlations: Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))

    plt.show()
