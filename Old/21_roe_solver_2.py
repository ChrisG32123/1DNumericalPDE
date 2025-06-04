# Non-Dimensionalized Euler-Poisson Equations
# n_t + (nu)_x = 0
# n(u_t + u*u_x) = -n_x - gamma*n*phi_x + corr
# phi_xx = 3 - 4*pi*n

import numpy as np
import matplotlib.pyplot as plt


def roe_solve_2():
    # Parameters
    N = int(1e2)  # Grid Points 5e2
    T = int(1e3)  # Time Steps 5e3
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
    def memory_allocation():
        u = np.zeros(N)
        uL = np.roll(u, -1)
        uR = u
        flux_or_matrix = np.zeros(N)
        snap_u = np.zeros((snaps + 1, N))
        return u, uL, uR, flux_or_matrix, snap_u

    n, nL, nR, flux_n, snap_n = memory_allocation()
    nc, ncL, ncR, flux_nc, snap_nc = memory_allocation()
    n3 = np.zeros(3 * N)

    v, vL, vR, flux_v, snap_v = memory_allocation()
    vc, vcL, vcR, flux_vc, snap_vc = memory_allocation()

    phi, phiL, phiR, A, snap_phi = memory_allocation()
    phic, phicL, phicR, Ac, snap_phic = memory_allocation()

    # TODO: Explicit Memory Allocation
    # n = np.zeros(nx)
    # nL = np.roll(n,-1)
    # nR = n
    # flux_n = np.zeros(nx)
    # v = np.zeros(nx)
    # flux_v = np.zeros(nx)
    # phi = np.zeros(nx)
    # A = np.zeros(nx)
    # snap_n = np.zeros((snaps + 1, nx))
    # snap_v = np.zeros((snaps + 1, nx))
    # snap_phi = np.zeros((snaps + 1, nx))
    #
    # nc = np.zeros(nx)
    # flux_nc = np.zeros(nx)
    # n3 = np.zeros(3 * nx)
    # vc = np.zeros(nx)
    # flux_vc = np.zeros(nx)
    # phic = np.zeros(nx)
    # Ac = np.zeros(nx)
    # snap_nc = np.zeros((snaps + 1, nx))
    # snap_vc = np.zeros((snaps + 1, nx))
    # snap_phic = np.zeros((snaps + 1, nx))

    # Initial Conditions
    n_IC = n_0 * np.ones(N)
    v_IC = np.zeros(N)
    n_IC[0:int(N / 2)] = 3 * n_0 / 2
    v_IC[0:int(N / 2)] = 1
    n_IC[int(N/2):N] = n_0 / 2
    v_IC[int(N/2):N] = 0

    # Initialization
    n = n_IC
    v = v_IC
    snap_n[0] = n_IC
    snap_v[0] = v_IC

    # Define Flux Functions
    def f_n(n_, v_):
        return n_ * v_

    def f_v(n_, v_):
        return .5 * v_ * v_ + np.log(n_)

    def f_nc(nc_, vc_):
        return nc_ * vc_

    def f_vc(nc_, vc_, phic_,f_corr_):
        return .5 * vc_ * vc_ + np.log(nc_) + Gamma_0 * phic_ + f_corr_

    def compute_phi(n_,phi_):
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
        phi_[0] = b[N - 1] - b[N - 2]
        for ii in range(1, N - 1):
            phi_[ii] = (b[ii - 1] - phi_[ii - 1]) / A[ii - 1]
        return phi_

    def LF_scheme(n_,v_,F_n_,F_v_):
        for ii in range(0, N - 1):
            n_[ii] = .5 * (n_[ii + 1] + n_[ii - 1]) - .5 * lambda_ * (F_n_[ii + 1] - F_n_[ii - 1])
            v_[ii] = .5 * (v_[ii + 1] + v_[ii - 1]) - .5 * lambda_ * (F_v_[ii + 1] - F_v_[ii - 1])
        n_[N - 1] = .5 * (n_[0] + n_[N - 2]) - .5 * lambda_ * (F_n_[0] - F_n_[N - 2])
        v_[N - 1] = .5 * (v_[0] + v_[N - 2]) - .5 * lambda_ * (F_v_[0] - F_v_[N - 2])
        return n_,v_

    def Jn(n_,v_,nr_,nl_,vr_,vl_):
        matrix_00 = np.abs(v_+1) / 2 + np.abs(v_-1) / 2
        matrix_01 = np.abs(v_+1) * n_ / 2 - np.abs(v_-1) * n_ / 2
        diff_n = nr_ - nl_
        diff_v = vr_ - vl_
        Jn = matrix_00 * diff_n + matrix_01 * diff_v
        return Jn

    def Jv(n_,v_,nr_,nl_,vr_,vl_):
        matrix_10 = np.abs(v_+1) / (2*n_) - np.abs(v_-1) / (2*n_)
        matrix_11 = np.abs(v_+1) / 2 + np.abs(v_-1) / 2
        diff_n = nr_ - nl_
        diff_v = vr_ - vl_
        Jv = matrix_10 * diff_n + matrix_11 * diff_v
        return Jv

    def take_snapshot(tt, T, snaps, n_, v_, phi_, snap_n_, snap_v_, snap_phi_):
        snap_n_[int(tt / (T / snaps))] = n_
        snap_v_[int(tt / (T / snaps))] = v_
        snap_phi_[int(tt / (T / snaps))] = phi_

    # Define Numerical Scheme

    # No Correlation
    for tt in range(T + 1):
        if tt < 10:
            print("n",n)  # TODO: debug line
        # Phi - A * phi = b
        phi = compute_phi(n,phi)

        rhs = f_corr - Gamma_0 * phi

        # Compute Fluxes
        FnL = f_n(nL, vL)
        FnR = f_n(nR, vR)
        FvL = f_v(nL, vL)
        FvR = f_v(nR, vR)
        jn = Jn(n,v,nR,nL,vR,vL)
        jv = Jv(n,v,nR,nL,vR,vL)

        flux_n = .5 * (FnR + FnL) - .5 * jn
        flux_v = .5 * (FvR + FvL) - .5 * jv

        print("Jv",jv)
        print("Jn",jn)

        for ii in range(0, N-1):
            n[ii] = n[ii] - lambda_ * (.5 * (FnR[ii+1] - FnR[ii-1]) - .5 * (jn[ii]-jn[ii-1]))
            v[ii] = v[ii] + .5 * lambda_ * (rhs[ii+1] - rhs[ii-1]) - lambda_ * (.5 * (FvR[ii+1] - FvR[ii-1]) - .5 * (jv[ii]-jv[ii-1]))
        n[N-1] = n[N-1] - lambda_ * (.5 * (FnR[0] - FnR[N-2]) - .5 * (jn[N-1] - jn[N-2]))
        v[N-1] = v[N-1] + .5 * lambda_ * (rhs[0] - rhs[N-2]) - lambda_ * (.5 * (FvR[0] - FvR[N-2]) - .5 * (jv[N-1] - jv[N-2]))

        if tt % (T / snaps) == 0:
            take_snapshot(tt,T,snaps,n, v, phi,snap_n,snap_v,snap_phi)

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
        phic = compute_phi(nc,phic)

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

        F_nc = f_nc(nc, vc)
        F_vc = f_vc(nc, vc, phic,f_corr)

        # Lax-Friedrichs
        LF_scheme(nc, vc, F_nc, F_vc)

        if tt % (T / snaps) == 0:
            take_snapshot(tt,T,snaps,nc, vc, phic,snap_nc,snap_vc,snap_phic)

    # Color Map
    y = np.linspace(0, t, snaps + 1)
    xx, yy = np.meshgrid(x, y, sparse=False, indexing='xy')

    def colormap(xx_,yy_,snap_u):
        plt.figure()
        clr = plt.contourf(xx_, yy_, snap_u)
        plt.colorbar()

    # colormap(xx,yy,snap_n) # TODO: Debug
    # plt.title("No Correlations: Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))
    #
    # colormap(xx, yy, snap_nc)
    # plt.title("Correlations: Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))
    #
    # colormap(xx,yy,snap_n-snap_nc)
    # plt.title("Difference: n-nc")

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
