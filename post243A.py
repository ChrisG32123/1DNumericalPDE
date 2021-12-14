# Non-Dimensionalized Euler-Poisson Equations
# n_t + (nu)_x = 0
# n(u_t + u*u_x) = -n_x - gamma*n*phi_x + corr
# phi_xx = 3 - 4*pi*n

import numpy as np
import matplotlib.pyplot as plt


def solve():
    # Parameters
    N = int(5e2)  # Grid Points
    T = int(1e3)  # Time Steps
    L = 10  # Domain Size
    x = np.linspace(0, L - L / N, N)  # Domain
    x3 = np.linspace(-L, 2 * L - L / N, 3 * N)
    dx = x[2] - x[1]  # Grid Size
    dt = 1e-3  # Time Step Size
    lambda_ = dt / dx

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
        n_IC[ii] = n_IC[ii] + (n_0 / 2) * np.sin(2 * np.pi * ii * dx / L)
        v_IC[ii] = v_IC[ii] + np.sin(2 * np.pi * ii * dx / L)

    # Initialization
    n = n_IC
    v = v_IC
    snap_n[0] = n_IC
    snap_v[0] = v_IC
    nc = n_IC
    vc = v_IC
    snap_nc[0] = n_IC
    snap_vc[0] = v_IC

    # Define Flux Functions
    def f_n(n_, v_):
        return n_ * v_

    def f_v(n_, v_, phi_):
        return .5 * v_ * v_ + np.log(n_) + Gamma_0 * phi_

    def f_nc(nc_, vc_):
        return nc_ * vc_

    def f_vc(nc_, vc_, phic_):
        return .5 * vc_ * vc_ + np.log(nc_) + Gamma_0 * phic_ + f_corr

    # No Correlation
    for tt in range(T + 1):
        # Phi - A * phi = b
        # Define b
        b = 3 - 4 * np.pi * dx * dx * n
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

        F_n = f_n(n, v)
        F_v = f_v(n, v, phi)

        # Lax-Friedrichs
        n[0] = .5 * (n[1] + n[-1]) - .5 * lambda_ * (F_n[1] - F_n[-1])
        v[0] = .5 * (v[1] + v[-1]) - .5 * lambda_ * (F_v[1] - F_v[-1])
        for ii in range(1, N - 1):
            n[ii] = .5 * (n[ii + 1] + n[ii - 1]) - .5 * lambda_ * (F_n[ii + 1] - F_n[ii - 1])
            v[ii] = .5 * (v[ii + 1] + v[ii - 1]) - .5 * lambda_ * (F_v[ii + 1] - F_v[ii - 1])
        n[N - 1] = .5 * (n[0] + n[N - 2]) - .5 * lambda_ * (F_n[0] - F_n[N - 2])
        v[N - 1] = .5 * (v[0] + v[N - 2]) - .5 * lambda_ * (F_v[0] - F_v[N - 2])

        if tt % (T / snaps) == 0:
            snap_n[int(tt / (T / snaps))] = n
            snap_v[int(tt / (T / snaps))] = v
            snap_phi[int(tt / (T / snaps))] = phi

    # Correlation
    for tt in range(T + 1):
        # Phi - A * phi = b
        # Define b
        bc = 3 - 4 * np.pi * dx * dx * nc
        bc = bc - np.mean(bc)
        # First sweep
        Ac[0] = -0.5
        bc[0] = -0.5 * bc[0]
        for ii in range(1, N - 1):
            Ac[ii] = -1 / (2 + Ac[ii - 1])
            bc[ii] = (bc[ii - 1] - bc[ii]) / (2 + Ac[ii - 1])
        # Second sweep
        phic[0] = bc[N - 1] - bc[N - 2]
        for ii in range(1, N - 1):
            phic[ii] = (bc[ii - 1] - phic[ii - 1]) / Ac[ii - 1]

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
        F_vc = f_vc(nc, vc, phic)

        # Lax-Friedrichs
        nc[0] = .5 * (nc[1] + nc[-1]) - .5 * lambda_ * (F_nc[1] - F_nc[-1])
        vc[0] = .5 * (vc[1] + vc[-1]) - .5 * lambda_ * (F_vc[1] - F_vc[-1])
        for ii in range(1, N - 1):
            nc[ii] = .5 * (nc[ii + 1] + nc[ii - 1]) - .5 * lambda_ * (F_nc[ii + 1] - F_nc[ii - 1])
            vc[ii] = .5 * (vc[ii + 1] + vc[ii - 1]) - .5 * lambda_ * (F_vc[ii + 1] - F_vc[ii - 1])
        nc[N - 1] = .5 * (nc[0] + nc[N - 2]) - .5 * lambda_ * (F_nc[0] - F_nc[N - 2])
        vc[N - 1] = .5 * (vc[0] + vc[N - 2]) - .5 * lambda_ * (F_vc[0] - F_vc[N - 2])

        if tt % (T / snaps) == 0:
            snap_nc[int(tt / (T / snaps))] = nc
            snap_vc[int(tt / (T / snaps))] = vc
            snap_phic[int(tt / (T / snaps))] = phic

    # Plotting
    for ii in range(len(snap_n)):
        plt.plot(x, snap_n[ii], label="n @ T = " + str(ii * dt * T / snaps))
        plt.plot(x, snap_nc[ii], label="nc @ T = " + str(ii * dt * T / snaps))
    plt.title("Density: Gamma_0 = " + str(Gamma_0))
    plt.legend()



    # plt.figure()
    # for ii in range(len(snap_v)):
    #     plt.plot(x, snap_v[ii], label="v @ T = " + str(ii * dt * T / snaps))
    # plt.title("Velocity: Gamma = " + str(Gamma))
    # plt.legend()
    #
    # plt.figure()
    # for ii in range(len(snap_phi)):
    #     plt.plot(x, snap_phi[ii], label="phi @ T = " + str(ii * dt * T / snaps))
    # plt.title("Electrostatic Potential: Gamma = " + str(Gamma))
    # plt.legend()

    plt.show(block=True)
