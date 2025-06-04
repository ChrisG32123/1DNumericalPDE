# Non-Dimensionalized Euler-Poisson Equations
# n_t + (nu)_x = 0
# n(u_t + u*u_x) = -n_x - gamma*n*phi_x + corr
# phi_xx = 3 - 4*pi*n

import numpy as np
import matplotlib.pyplot as plt


def organized_solve():
    # Parameters
    N = int(1e2)  # Grid Points
    T = int(1e3)  # Time Steps
    L = 10  # Domain Size
    x = np.linspace(0, L - L / N, N)  # Domain
    x3 = np.linspace(-L, 2 * L - L / N, 3 * N)
    dx = x[2] - x[1]  # Grid Size
    dt = 1e-3  # Time Step Size
    lambda_ = dt / dx
    t = dt * T

    n_0 = 3 / (4 * np.pi)  # Mean Density
    snaps = int(input("Number of Snapshots "))  # Number of Snapshots
    Gamma_0 = float(input("Value of Gamma "))  # Coulomb Coupling Parameter
    kappa_0 = float(input("Value of kappa "))  # screening something
    rho = np.zeros(N)
    f_corr = np.zeros(N)

    # Memory Allocation
    n = np.zeros(N)
    v = np.zeros(N)
    phi = np.zeros(N)
    A = np.zeros(N)
    snap_n = np.zeros((snaps + 1, N))
    snap_v = np.zeros((snaps + 1, N))
    snap_phi = np.zeros((snaps + 1, N))

    nc = np.zeros(N)
    n3 = np.zeros(3 * N)
    vc = np.zeros(N)
    phic = np.zeros(N)
    Ac = np.zeros(N)
    snap_nc = np.zeros((snaps + 1, N))
    snap_vc = np.zeros((snaps + 1, N))
    snap_phic = np.zeros((snaps + 1, N))

    # Initial Conditions
    n_IC = n_0 * np.ones(N)
    v_IC = np.zeros(N)
    for ii in range(N):
        if ii < N / 2:
            n_IC[ii] = 3 * n_0 / 2
            v_IC[ii] = 1
        else:
            n_IC[ii] = n_0 / 2
            v_IC[ii] = 0

    # Initialization
    n = n_IC
    v = v_IC
    snap_n[0] = n_IC
    snap_v[0] = v_IC

    # Define Flux Functions
    def f_n(n_, v_):
        return n_ * v_

    def f_v(n_, v_, phi_):
        return .5 * v_ * v_ + np.log(n_) + Gamma_0 * phi_

    def f_nc(nc_, vc_):
        return nc_ * vc_

    def f_vc(nc_, vc_, phic_, f_corr_):
        return .5 * vc_ * vc_ + np.log(nc_) + Gamma_0 * phic_ + f_corr_

    def compute_phi(n_):
        # Define b
        b = 3 - 4 * np.pi * dx * dx * n_
        b = b - np.mean(b)
        # First sweep
        A[0] = -0.5
        b[0] = -0.5 * b[0]
        for ii in range(1, N - 1):
            A[ii] = -1 / (2 + A[ii - 1])
            b[ii] = (b[ii - 1] - b[ii]) / (2 + A[ii - 1])
        # Second sweep
        phi[0] = b[N - 1] - b[N - 2]
        for ii in range(1, N - 1):
            phi[ii] = (b[ii - 1] - phi[ii - 1]) / A[ii - 1]
        return phi

    def LF_scheme(n_,v_,F_n_,F_v_):
        for ii in range(0, N - 1):
            n_[ii] = .5 * (n_[ii + 1] + n_[ii - 1]) - .5 * lambda_ * (F_n_[ii + 1] - F_n_[ii - 1])
            v_[ii] = .5 * (v_[ii + 1] + v_[ii - 1]) - .5 * lambda_ * (F_v_[ii + 1] - F_v_[ii - 1])
        n_[N - 1] = .5 * (n_[0] + n_[N - 2]) - .5 * lambda_ * (F_n_[0] - F_n_[N - 2])
        v_[N - 1] = .5 * (v_[0] + v_[N - 2]) - .5 * lambda_ * (F_v_[0] - F_v_[N - 2])
        return n_,v_

    # No Correlation
    for tt in range(T + 1):
        # Phi - A * phi = b
        phi = compute_phi(n)

        # Lax-Friedrichs
        F_n = f_n(n, v)
        F_v = f_v(n, v, phi)
        LF_scheme(n,v,F_n,F_v)

        if tt % (T / snaps) == 0:
            snap_n[int(tt / (T / snaps))] = n
            snap_v[int(tt / (T / snaps))] = v
            snap_phi[int(tt / (T / snaps))] = phi

    # Initial Conditions
    n_IC = n_0 * np.ones(N)
    v_IC = np.zeros(N)
    for ii in range(N):
        if ii < N / 2:
            n_IC[ii] = 3 * n_0 / 2
            v_IC[ii] = 1
        else:
            n_IC[ii] = n_0 / 2
            v_IC[ii] = 0

    # Initialization
    nc = n_IC
    vc = v_IC
    snap_nc[0] = n_IC
    snap_vc[0] = v_IC

    # Correlation
    for tt in range(T + 1):
        # Phi - A * phi = b
        phic = compute_phi(nc)

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

        F_nc = f_nc(nc, vc)
        F_vc = f_vc(nc, vc, phic, f_corr)

        # Lax-Friedrichs
        LF_scheme(nc,vc,F_nc,F_vc)

        if tt % (T / snaps) == 0:
            snap_nc[int(tt / (T / snaps))] = nc
            snap_vc[int(tt / (T / snaps))] = vc
            snap_phic[int(tt / (T / snaps))] = phic

    # Color Map
    y = np.linspace(0, t, snaps + 1)
    xx, yy = np.meshgrid(x, y, sparse=False, indexing='xy')
    def colormap(xx_,yy_,snap_u):
        plt.figure()
        clr = plt.contourf(xx_, yy_, snap_u)
        plt.colorbar()

    colormap(xx,yy,snap_n)
    plt.title("No Correlations: Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))

    colormap(xx, yy, snap_nc)
    plt.title("Correlations: Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))

    colormap(xx,yy,snap_n-snap_nc)
    plt.title("Difference: n-nc")

    # Plotting
    def plot(x_,snap_u):
        plt.figure()
        for ii in range(len(snap_u)):
            plt.plot(x_, snap_u[ii], label="T = " + str(ii * dt * T / snaps))
        plt.legend()

    plot(x,snap_n)
    plt.title("No Correlations: Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))

    plot(x,snap_nc)
    plt.title("Correlations: Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))

    plt.show()
