# Non-Dimensionalized Euler-Poisson Equations
# n_t + (nu)_x = 0
# n(u_t + u*u_x) = -n_x - gamma*n*phi_x + corr
# phi_xx = 3 - 4*pi*n

import numpy as np
import matplotlib.pyplot as plt


def godunov_solve():
    # Memory Allocation and Initialization
    def memory_allocation(u_IC):
        u = np.copy(u_IC)
        uL = np.roll(u, 1)
        uR = np.copy(u)
        flux_or_matrix = np.zeros(N)
        snap_u = np.zeros((snaps + 1, N))
        snap_u[0] = np.copy(u_IC)
        return u, uL, uR, flux_or_matrix, snap_u

    # Define Flux Functions
    def f_n(n, v):
        return n * v

    def f_v(n, v, phi, f_corr):
        flux = .5 * v * v + np.log(n) + Gamma_0 * phi
        if f_corr != 0:
            flux += f_corr
        return

    def compute_phi(n, phi):
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
        return phi

    def take_snapshot(tt, T, snaps, n, v, phi, snap_n, snap_v, snap_phi):
        snap_n[int(tt / (T / snaps))] = n
        snap_v[int(tt / (T / snaps))] = v
        snap_phi[int(tt / (T / snaps))] = phi

    # Define Numerical Scheme
    def godunov_flux(fL, fR, uL, uR):
        s = 0.5 * (uL + uR)
        if uL >= uR:
            if s > 0.0:
                return fL
            else:
                return fR
        else:
            if uL > 0.0:
                return fL
            elif uR < 0.0:
                return fR
            else:
                return 0.0

    def colormap(xx, yy, snap_u):
        plt.figure()
        clr = plt.contourf(xx, yy, snap_u)
        plt.colorbar()

    def plot(x, snap_u):
        plt.figure()
        for ii in range(len(snap_u)):
            plt.plot(x, snap_u[ii], label="T = " + str(ii * dt * T / snaps))
        plt.legend()

    # Parameters
    N = int(1e3)  # Grid Points 5e2
    T = int(1e3)  # Time Steps 5e3
    L = 10  # Domain Size
    x = np.linspace(0, L - L / N, N)  # Domain
    # x3 = np.linspace(-Xlngth, 2 * Xlngth - Xlngth / nx, 3 * nx)
    dx = x[2] - x[1]  # Grid Size
    dt = 1e-3  # Time Step Size
    lambda_ = dt / dx
    t = dt * T

    n_0 = 3 / (4 * np.pi)  # Mean Density
    snaps = int(input("Number of Snapshots "))  # Number of Snapshots
    Gamma_0 = float(input("Value of Gamma "))  # Coulomb Coupling Parameter
    kappa_0 = float(input("Value of kappa "))  # screening something
    # rho = np.zeros(nx)
    # f_corr = np.zeros(nx)

    # Initial Conditions
    n_IC = n_0 * np.ones(N)
    n_IC[0:int(N / 4)] = n_0 / 2
    n_IC[int(N / 4):int(3 * N / 4)] = 3 * n_0 / 2
    n_IC[int(3 * N / 4):N] = n_0 / 2
    v_IC = .2 * np.sin(2 * np.pi * x / L)

    n, nL, nR, flux_n, snap_n = memory_allocation()
    v, vL, vR, flux_v, snap_v = memory_allocation(v_IC)
    # phi, phiL, phiR, A, snap_phi = memory_allocation()
    # nc, ncL, ncR, flux_nc, snap_nc = memory_allocation()
    # n3 = np.zeros(3 * nx)
    # vc, vcL, vcR, flux_vc, snap_vc = memory_allocation()
    # phic, phicL, phicR, Ac, snap_phic = memory_allocation()

    for tt in range(T + 1):
        if tt < 10:
            print(n)  # TODO: debug line
        # Phi - A * phi = b
        phi = compute_phi(n, phi)

        # Compute Fluxes
        # TODO: Check left and right boundary conditions
        FnL = f_n(nL, vL)
        FnR = f_n(nR, vR)
        FvL = f_v(nL, vL, phiL)
        FvR = f_v(nR, vR, phiR)
        for ii in range(N):
            flux_n[ii - 1] = godunov_flux(FnL[ii], FnR[ii], nL[ii], nR[ii])  # TODO: flux is one index behind
            flux_v[ii - 1] = godunov_flux(FvL[ii], FvR[ii], vL[ii], vR[ii])

        # Godunov Scheme
        # TODO: Check left and right boundary conditions
        for ii in range(0, N - 1):
            n[ii] = n[ii - 1] - lambda_ * (flux_n[ii + 1] - flux_n[ii])
            v[ii] = v[ii - 1] - lambda_ * (flux_v[ii + 1] - flux_v[ii])
        n[N - 1] = n[N - 2] - lambda_ * (flux_n[0] - flux_n[N - 1])
        v[N - 1] = v[N - 2] - lambda_ * (flux_v[0] - flux_v[N - 1])

        if tt % (T / snaps) == 0:
            take_snapshot(tt, T, snaps, n, v, phi, snap_n, snap_v, snap_phi)

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
        phic = compute_phi(nc, phic)

        conc = nc / n_0
        Gamma = Gamma_0 * conc ** (1 / 3)
        kappa = kappa_0 * conc ** (1 / 6)

        n3[0:N] = nc[0:N]
        n3[N:2 * N] = nc[0:N]
        n3[2 * N:3 * N] = nc[0:N]
        for jj in range(N):
            xj = x[jj]
            # TODO: rho not used, using rho_int instead?
            rho_int = - 2 * np.pi * Gamma[jj] * np.exp(- kappa[jj] * np.abs(x3 - xj)) / kappa[jj]
            f_corr[jj] = dx * np.sum(n3 * rho_int)

        F_nc = f_n(nc, vc)
        F_vc = f_v(nc, vc, phic, f_corr)

        # Lax-Friedrichs
        LF_scheme(nc, vc, F_nc, F_vc)

        if tt % (T / snaps) == 0:
            take_snapshot(tt, T, snaps, nc, vc, phic, snap_nc, snap_vc, snap_phic)

    # Color Map
    y = np.linspace(0, t, snaps + 1)
    xx, yy = np.meshgrid(x, y, sparse=False, indexing='xy')

    # colormap(xx,yy,snap_n) # TODO: Debug
    # plt.title("No Correlations: Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))
    #
    # colormap(xx, yy, snap_nc)
    # plt.title("Correlations: Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))
    #
    # colormap(xx,yy,snap_n-snap_nc)
    # plt.title("Difference: n-nc")

    plot(x, snap_n)
    plt.title("No Correlations: Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))

    plot(x, snap_nc)
    plt.title("Correlations: Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))

    plt.show()
