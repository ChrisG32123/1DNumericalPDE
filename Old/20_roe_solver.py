# Non-Dimensionalized Euler-Poisson Equations
# n_t + (nu)_x = 0
# n(u_t + u*u_x) = -n_x - gamma*n*phi_x + corr
# phi_xx = 3 - 4*pi*n

import numpy as np
import matplotlib.pyplot as plt

# TODO: Changes
#   A=P|D|P^-1 not A=PDP^-1
#   (1,nx) v. nx
#   vector PDE instead of n and v system
#   phi and f_c on rhs


def roe_solve():
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

    # Initial Conditions
    n_IC = n_0 * np.ones(N)
    v_IC = np.zeros(N)
    n_IC[0:int(N / 2)] = 3 * n_0 / 2
    v_IC[0:int(N / 2)] = 1
    n_IC[int(N / 2):N] = n_0 / 2
    v_IC[int(N / 2):N] = 0

    # Memory Allocation
    def memory_allocation_pde(u_IC):
        # u = np.zeros((1, nx))
        # u[0] = u_IC
        u = np.copy(u_IC)
        uL = np.roll(u, -1)
        uR = np.copy(u)
        flux = np.zeros((1, N))
        fuL = np.zeros(N)
        fuR = np.zeros(N)
        snap_u = np.zeros((snaps + 1, N))
        return u, uL, uR, flux, fuL, fuR, snap_u

    def memory_allocation_phi():
        u = np.zeros(N)
        uL = np.roll(u, -1)
        uR = np.copy(u)
        matrix = np.zeros(N)
        snap_u = np.zeros((snaps + 1, N))
        return u, uL, uR, matrix, snap_u

    n, nL, nR, flux_n, FnL, FnR, snap_n = memory_allocation_pde(n_IC)
    nc, ncL, ncR, flux_nc, FncL, FncR, snap_nc = memory_allocation_pde(n_IC)

    v, vL, vR, flux_v, FvL, FvR, snap_v = memory_allocation_pde(v_IC)
    vc, vcL, vcR, flux_vc, FvcL, FvcR, snap_vc = memory_allocation_pde(v_IC)

    phi, phiL, phiR, A, snap_phi = memory_allocation_phi()
    phic, phicL, phicR, Ac, snap_phic = memory_allocation_phi()

    n3 = np.zeros(3 * N)
    V = np.array([n,v])
    VR = np.array([nR,vR])
    VL = np.array([nL,vL])
    flux_V = np.zeros((2, N))

    linearized_matrix = np.zeros((2, 2))
    rhs = np.zeros((2, N))

    rho = np.zeros(N)
    f_corr = np.zeros(N)

    # Initialization
    n = n_IC
    v = v_IC
    snap_n[0] = n_IC
    snap_v[0] = v_IC

    # Define Flux Functions
    def f_n(n_, v_):
        return n_ * v_

    def f_v(n_, v_):
        flux = .5 * v_ * v_ + np.log(n_)
        return flux

    # Define Numerical Scheme
    def compute_phi(n_, phi_):
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

    def LF_scheme(n_, v_, F_n_, F_v_):
        for ii in range(0, N - 1):
            n_[ii] = .5 * (n_[ii + 1] + n_[ii - 1]) - .5 * lambda_ * (F_n_[ii + 1] - F_n_[ii - 1])
            v_[ii] = .5 * (v_[ii + 1] + v_[ii - 1]) - .5 * lambda_ * (F_v_[ii + 1] - F_v_[ii - 1])
        n_[N - 1] = .5 * (n_[0] + n_[N - 2]) - .5 * lambda_ * (F_n_[0] - F_n_[N - 2])
        v_[N - 1] = .5 * (v_[0] + v_[N - 2]) - .5 * lambda_ * (F_v_[0] - F_v_[N - 2])
        return n_, v_

    def compute_linearization(n_,v_):
        # f'(V) = J = PDP^-1. Solve for A = P|D|P^-1
        P = np.array([[n_,-n_],[1,1]])
        abs_D = np.array([[np.abs(v_+1),0],[0,np.abs(v_-1)]])
        Pinv = np.array([[1/(2*n_),.5],[-1/(2*n_),.5]])
        lin_matrix = P.dot(abs_D.dot(Pinv))
        return lin_matrix

    def roe_flux(FVR_, FVL_, lin_matrix_, VR_, VL_):
        flux = .5 * (FVR_ + FVL_) - .5 * lin_matrix_.dot(VR_ - VL_)
        return flux

    def take_snapshot(tt, T, snaps, n_, v_, phi_, snap_n_, snap_v_, snap_phi_):
        snap_n_[int(tt / (T / snaps))] = n_
        snap_v_[int(tt / (T / snaps))] = v_
        snap_phi_[int(tt / (T / snaps))] = phi_

    # No Correlation
    for tt in range(T + 1):
        # Phi - A * phi = b
        phi = compute_phi(n, phi)

        # Compute Fluxes
        for ii in range(N):
            FnL[ii] = f_n(nL[ii], vL[ii])
            FnR[ii] = f_n(nR[ii], vR[ii])
            FvL[ii] = f_v(nL[ii], vL[ii])
            FvR[ii] = f_v(nR[ii], vR[ii])
        FVL = np.vstack((FnL,FvL))
        FVR = np.vstack((FnR,FvR))

        linearized_matrix = compute_linearization(n[ii],v[ii])

        flux_V = roe_flux(FVR,FVL,linearized_matrix,VR,VL)

        rhs[1] = f_corr - Gamma_0 * phi

        if tt < 3:
            print("n", n)
            print("v", v)
            print("FnL", FnL)
            print("FnR", FnR)
            # print("FvL", FvL)
            # print("FvR", FvR)
        # Solve
        arrayRow, arrayColumn = V.shape
        for row in range(arrayRow):
            Vrow = V[row]
            rhsrow = rhs[row]
            flux_Vrow = flux_V[row]
            for ii in range(0, N - 1):
                Vrow[ii] = Vrow[ii - 1] + .5 * lambda_ * (rhsrow[ii + 1] - rhsrow[ii-1]) - lambda_ * (flux_Vrow[ii] - flux_Vrow[ii-1])
        Vrow[N - 1] = Vrow[N - 2] + .5 * lambda_ * (rhsrow[0] - rhsrow[N-2]) - lambda_ * (flux_Vrow[N-1] - flux_Vrow[N - 2])
        n = V[0]
        v = V[1]

        if tt % (T / snaps) == 0:
            print("n", n)
            print("v", v)
            print("FnL", FnL)
            print("FnR", FnR)
            print("FvL", FvL)
            print("FvR", FvR)
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

        # F_nc = f_n(nc, vc)
        # F_vc = f_v(nc, vc, phic, f_corr,False)
        F_nc = f_n(nc, vc)
        F_vc = f_v(nc, vc)

        # Lax-Friedrichs
        LF_scheme(nc, vc, F_nc, F_vc)

        if tt % (T / snaps) == 0:
            take_snapshot(tt, T, snaps, nc, vc, phic, snap_nc, snap_vc, snap_phic)

    # Color Map
    y = np.linspace(0, t, snaps + 1)
    xx, yy = np.meshgrid(x, y, sparse=False, indexing='xy')

    def colormap(xx_, yy_, snap_u):
        plt.figure()
        clr = plt.contourf(xx_, yy_, snap_u)
        plt.colorbar()

    colormap(xx,yy,snap_n) # TODO: Debug
    plt.title("No Correlations: Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))

    colormap(xx, yy, snap_nc)
    plt.title("Correlations: Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))

    colormap(xx,yy,snap_n-snap_nc)
    plt.title("Difference: n-nc")

    # Plotting
    def plot(x_, snap_u):
        plt.figure()
        for ii in range(len(snap_u)):
            plt.plot(x_, snap_u[ii], label="T = " + str(ii * dt * T / snaps))
        plt.legend()

    plot(x, snap_n)
    plt.title("No Correlations: Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))

    plot(x, snap_nc)
    plt.title("Correlations: Gamma_0 = " + str(Gamma_0) + " kappa_0 = " + str(kappa_0))

    plt.show()
